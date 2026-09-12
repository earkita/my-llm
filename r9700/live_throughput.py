from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any
import urllib.request

from .config import ConfigurationError, load_profile
from .service import WORKER_POOL_STATE_PATH, managed_state


KV_CAPACITY_RE = re.compile(r"GPU KV cache size: ([0-9,]+) tokens")
DP_ENGINE_RE = re.compile(r"EngineCore_DP(?P<engine>[0-9]+)")
PROMETHEUS_ENGINE_RE = re.compile(r'(?:^|,)engine="(?P<engine>[^"]+)"(?:,|$)')
METRICS = {
    "running_requests": "vllm:num_requests_running",
    "waiting_requests": "vllm:num_requests_waiting",
    "kv_cache_fraction": "vllm:kv_cache_usage_perc",
    "prompt_tokens": "vllm:prompt_tokens_total",
    "prefix_cache_queries": "vllm:prefix_cache_queries_total",
    "prefix_cache_hits": "vllm:prefix_cache_hits_total",
    "generation_tokens": "vllm:generation_tokens_total",
    "completed_requests": "vllm:request_success_total",
    "prefill_seconds": "vllm:request_prefill_time_seconds_sum",
    "decode_seconds": "vllm:request_decode_time_seconds_sum",
    "ttft_seconds": "vllm:time_to_first_token_seconds_sum",
    "e2e_seconds": "vllm:e2e_request_latency_seconds_sum",
}
OPTIONAL_METRICS = {
    "spec_draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "spec_accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
}


@dataclass(frozen=True)
class Endpoint:
    label: str
    url: str
    profile: str
    runtime: str
    log_path: Path
    capacities: dict[str, int]
    gpu_ids: tuple[int, ...]

    def gpu_ids_for_engine(self, engine: str) -> tuple[int, ...]:
        if len(self.capacities) == 1:
            return self.gpu_ids
        try:
            return (self.gpu_ids[int(engine)],)
        except (IndexError, ValueError):
            return ()


def _cache_capacities(path: Path) -> dict[str, int]:
    capacities: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        capacity_match = KV_CAPACITY_RE.search(line)
        if capacity_match is None:
            continue
        engine_match = DP_ENGINE_RE.search(line)
        engine = engine_match.group("engine") if engine_match else "0"
        capacities[engine] = int(capacity_match.group(1).replace(",", ""))
    if not capacities:
        raise ConfigurationError(f"GPU KV cache capacity is absent from {path}")
    return capacities


def _cache_capacity(path: Path) -> int:
    """Return aggregate capacity for the legacy single-endpoint interface."""

    return sum(_cache_capacities(path).values())


def _parse_metrics(text: str) -> dict[str, dict[str, float]]:
    names = {value: key for key, value in {**METRICS, **OPTIONAL_METRICS}.items()}
    engines: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        if not line.startswith("vllm:") or line.startswith("#"):
            continue
        try:
            identity, raw_value = line.rsplit(None, 1)
            value = float(raw_value)
        except ValueError:
            continue
        metric_name, separator, labels = identity.partition("{")
        key = names.get(metric_name)
        if key is None:
            continue
        engine = "0"
        if separator:
            match = PROMETHEUS_ENGINE_RE.search(labels.removesuffix("}"))
            if match is not None:
                engine = match.group("engine")
        bucket = engines.setdefault(engine, {})
        bucket[key] = bucket.get(key, 0.0) + value

    if not engines:
        raise ConfigurationError("vLLM metrics contain no engine samples")
    for engine, values in engines.items():
        missing = sorted(set(METRICS) - set(values))
        if missing:
            raise ConfigurationError(
                f"vLLM engine {engine} is missing metrics: {', '.join(missing)}"
            )
        for key in OPTIONAL_METRICS:
            values.setdefault(key, 0.0)
    return engines


def _metrics_by_engine(url: str) -> dict[str, dict[str, float]]:
    with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=5) as response:
        return _parse_metrics(response.read().decode(errors="replace"))


def _metrics(url: str) -> dict[str, float]:
    """Return sums for compatibility with the original single-engine monitor."""

    engines = _metrics_by_engine(url)
    return {
        key: sum(values[key] for values in engines.values())
        for key in {**METRICS, **OPTIONAL_METRICS}
    }


def _delta(current: dict[str, float], previous: dict[str, float], key: str) -> float:
    return max(current.get(key, 0.0) - previous.get(key, 0.0), 0.0)


def _prefix_cache_delta(
    current: dict[str, float], previous: dict[str, float]
) -> dict[str, float | int]:
    queries = round(_delta(current, previous, "prefix_cache_queries"))
    hits = min(round(_delta(current, previous, "prefix_cache_hits")), queries)
    return {
        "cached_tokens": hits,
        "cache_query_tokens": queries,
        "cache_hit_percent": hits / queries * 100 if queries else 0.0,
    }


def _speculative_delta(
    current: dict[str, float], previous: dict[str, float]
) -> dict[str, float | int]:
    drafts = round(_delta(current, previous, "spec_draft_tokens"))
    accepted = min(round(_delta(current, previous, "spec_accepted_tokens")), drafts)
    return {
        "spec_draft_tokens": drafts,
        "spec_accepted_tokens": accepted,
        "spec_acceptance_percent": accepted / drafts * 100 if drafts else 0.0,
    }


def _completion(
    current: dict[str, float], previous: dict[str, float]
) -> dict[str, float | int] | None:
    completed = round(_delta(current, previous, "completed_requests"))
    if completed < 1:
        return None
    prompt_tokens = _delta(current, previous, "prompt_tokens")
    output_tokens = _delta(current, previous, "generation_tokens")
    prefill_seconds = _delta(current, previous, "prefill_seconds")
    decode_seconds = _delta(current, previous, "decode_seconds")
    cache = _prefix_cache_delta(current, previous)
    uncached_tokens = max(prompt_tokens - cache["cached_tokens"], 0.0)
    return {
        "completed_requests": completed,
        "prompt_tokens": round(prompt_tokens),
        "output_tokens": round(output_tokens),
        "prefill_tokens_per_second": prompt_tokens / max(prefill_seconds, 1e-9),
        "uncached_prefill_tokens_per_second": uncached_tokens
        / max(prefill_seconds, 1e-9),
        "decode_tokens_per_second": max(output_tokens - completed, 0.0)
        / max(decode_seconds, 1e-9),
        "mean_ttft_seconds": _delta(current, previous, "ttft_seconds") / completed,
        "mean_e2e_seconds": _delta(current, previous, "e2e_seconds") / completed,
        **cache,
        **_speculative_delta(current, previous),
    }


def _active(metrics: dict[str, float]) -> bool:
    return bool(metrics["running_requests"] or metrics["waiting_requests"])


class CompletionAccumulator:
    """Retain counter baselines across prefill and decode metric updates."""

    def __init__(self, initial: dict[str, float]) -> None:
        self.previous = initial
        self.previous_active = _active(initial)
        self.request_baseline: dict[str, float] | None = None

    def observe(
        self, current: dict[str, float]
    ) -> tuple[dict[str, float | int] | None, int, bool]:
        active = _active(current)
        completed_now = round(_delta(current, self.previous, "completed_requests"))
        if self.request_baseline is None and not self.previous_active and (
            active or completed_now
        ):
            self.request_baseline = self.previous
        completion = (
            _completion(current, self.request_baseline)
            if completed_now and self.request_baseline is not None
            else None
        )
        completion_unavailable = (
            completed_now if completed_now and completion is None else 0
        )
        if completed_now or not active:
            self.request_baseline = None
        self.previous = current
        self.previous_active = active
        return completion, completion_unavailable, active

    def request_cache(
        self, current: dict[str, float]
    ) -> dict[str, float | int] | None:
        if self.request_baseline is None:
            return None
        return _prefix_cache_delta(current, self.request_baseline)

    def request_speculative(
        self, current: dict[str, float]
    ) -> dict[str, float | int] | None:
        if self.request_baseline is None:
            return None
        return _speculative_delta(current, self.request_baseline)


class LiveDecodeRate:
    """Track wall-clock output-token throughput while the engine is running."""

    def __init__(
        self,
        initial_tokens: float,
        initial_time: float,
        *,
        active: bool,
        window: float,
    ) -> None:
        self.previous_tokens = initial_tokens
        self.previous_time = initial_time
        self.previous_active = active
        self.window = window
        self.samples: deque[tuple[float, float]] = deque()

    def observe(
        self,
        now: float,
        tokens: float,
        *,
        active: bool,
        request_completed: bool = False,
    ) -> float:
        if not active or request_completed:
            self.samples.clear()
            self.previous_tokens = tokens
            self.previous_time = now
            self.previous_active = active
            return 0.0

        if not self.previous_active:
            self.samples.clear()

        token_delta = max(tokens - self.previous_tokens, 0.0)
        if token_delta and not self.samples:
            self.samples.append((self.previous_time, self.previous_tokens))
        if self.samples:
            self.samples.append((now, tokens))
            while len(self.samples) > 2 and now - self.samples[1][0] >= self.window:
                self.samples.popleft()

        self.previous_tokens = tokens
        self.previous_time = now
        self.previous_active = True
        if len(self.samples) < 2:
            return 0.0
        oldest_time, oldest_tokens = self.samples[0]
        elapsed = max(now - oldest_time, 1e-9)
        return max(tokens - oldest_tokens, 0.0) / elapsed


class EngineMonitor:
    def __init__(
        self,
        endpoint: Endpoint,
        engine: str,
        initial: dict[str, float],
        started_at: float,
        *,
        window: float,
    ) -> None:
        self.endpoint = endpoint
        self.engine = engine
        self.capacity = endpoint.capacities.get(engine)
        if self.capacity is None and len(endpoint.capacities) == 1:
            self.capacity = next(iter(endpoint.capacities.values()))
        if self.capacity is None:
            raise ConfigurationError(
                f"{endpoint.label} engine {engine} has no KV cache capacity"
            )
        self.history: deque[tuple[float, float]] = deque(
            [(started_at, initial["kv_cache_fraction"])]
        )
        self.accumulator = CompletionAccumulator(initial)
        self.decode_rate = LiveDecodeRate(
            initial["generation_tokens"],
            started_at,
            active=bool(initial["running_requests"]),
            window=window,
        )
        self.window = window

    def observe(self, current: dict[str, float], now: float) -> dict[str, Any]:
        self.history.append((now, current["kv_cache_fraction"]))
        while len(self.history) > 1 and now - self.history[0][0] > self.window:
            self.history.popleft()
        oldest_time, oldest_kv = self.history[0]
        elapsed = max(now - oldest_time, 1e-9)
        kv_growth = max(current["kv_cache_fraction"] - oldest_kv, 0.0)
        completion, unavailable, active = self.accumulator.observe(current)
        request_cache = self.accumulator.request_cache(current)
        request_speculative = self.accumulator.request_speculative(current)
        live_decode = self.decode_rate.observe(
            now,
            current["generation_tokens"],
            active=bool(current["running_requests"]),
            request_completed=bool(completion or unavailable),
        )
        if current["running_requests"]:
            phase = "decode" if live_decode > 0 else "prefill"
        elif current["waiting_requests"]:
            phase = "queued"
        else:
            phase = "idle"
        return {
            "timestamp": datetime.now().astimezone().isoformat(),
            "source": self.endpoint.label,
            "engine": self.engine,
            "phase": phase,
            "running_requests": round(current["running_requests"]),
            "waiting_requests": round(current["waiting_requests"]),
            "kv_cache_percent": current["kv_cache_fraction"] * 100,
            "kv_growth_tokens_per_second": kv_growth * self.capacity / elapsed,
            "kv_growth_window_seconds": elapsed,
            "live_decode_tokens_per_second": live_decode,
            "completion": completion,
            "completion_unavailable": unavailable,
            "request_cache": request_cache,
            "request_speculative": request_speculative,
            "active": active,
        }


def _number(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _gpu_snapshot(gpu_ids: tuple[int, ...]) -> dict[int, dict[str, float]]:
    if not gpu_ids:
        return {}
    executable = shutil.which("amd-smi")
    if executable is None:
        raise ConfigurationError("--gpu requires amd-smi")
    command = [
        executable,
        "metric",
        "--usage",
        "--clock",
        "--power",
        "--json",
        "--gpu",
        *(str(gpu_id) for gpu_id in gpu_ids),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise ConfigurationError(
            f"amd-smi metric failed: {(completed.stderr or completed.stdout).strip()}"
        )
    try:
        rows = json.loads(completed.stdout)["gpu_data"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ConfigurationError("amd-smi emitted invalid metric JSON") from exc
    result: dict[int, dict[str, float]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("gpu"), int):
            continue
        values = {
            "gfx_percent": _number(row.get("usage", {}).get("gfx_activity")),
            "umc_percent": _number(row.get("usage", {}).get("umc_activity")),
            "clock_mhz": _number(row.get("clock", {}).get("gfx_0", {}).get("clk")),
            "power_watts": _number(row.get("power", {}).get("socket_power")),
        }
        result[row["gpu"]] = {
            key: value for key, value in values.items() if value is not None
        }
    return result


def _gpu_summary(
    gpu_ids: tuple[int, ...], snapshot: dict[int, dict[str, float]]
) -> dict[str, Any] | None:
    rows = [snapshot[gpu_id] for gpu_id in gpu_ids if gpu_id in snapshot]
    if not rows:
        return None

    def values(key: str) -> list[float]:
        return [row[key] for row in rows if key in row]

    gfx = values("gfx_percent")
    umc = values("umc_percent")
    clocks = values("clock_mhz")
    power = values("power_watts")
    return {
        "gpu_ids": list(gpu_ids),
        "gfx_percent_mean": sum(gfx) / len(gfx) if gfx else None,
        "gfx_percent_min": min(gfx) if gfx else None,
        "gfx_percent_max": max(gfx) if gfx else None,
        "umc_percent_mean": sum(umc) / len(umc) if umc else None,
        "clock_mhz_mean": sum(clocks) / len(clocks) if clocks else None,
        "power_watts_total": sum(power) if power else None,
    }


def _amd_bdf_map() -> dict[str, int]:
    executable = shutil.which("amd-smi")
    if executable is None:
        raise ConfigurationError("--gpu requires amd-smi")
    completed = subprocess.run(
        [executable, "list", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ConfigurationError(
            f"amd-smi list failed: {(completed.stderr or completed.stdout).strip()}"
        )
    try:
        rows = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ConfigurationError("amd-smi emitted invalid device JSON") from exc
    return {
        str(row["bdf"]).lower(): int(row["gpu"])
        for row in rows
        if isinstance(row, dict) and "bdf" in row and "gpu" in row
    }


def _endpoint(label: str, state: dict[str, Any], *, with_gpu: bool) -> Endpoint:
    log_path = Path(str(state.get("log", "")))
    if not log_path.is_file():
        raise ConfigurationError(f"managed runtime log is unavailable: {log_path}")
    profile = load_profile(str(state.get("profile", "")))
    gpu_ids: tuple[int, ...] = ()
    if with_gpu:
        bdf_map = _amd_bdf_map()
        bdfs = profile["runtime"].get("gpu_bdfs", [])
        missing = [bdf for bdf in bdfs if str(bdf).lower() not in bdf_map]
        if missing:
            raise ConfigurationError(
                f"amd-smi cannot resolve profile GPUs: {', '.join(missing)}"
            )
        gpu_ids = tuple(bdf_map[str(bdf).lower()] for bdf in bdfs)
    return Endpoint(
        label=label,
        url=str(state["url"]),
        profile=str(state["profile"]),
        runtime=str(state["runtime"]),
        log_path=log_path,
        capacities=_cache_capacities(log_path),
        gpu_ids=gpu_ids,
    )


def _format_gpu(summary: dict[str, Any] | None) -> str:
    if not summary:
        return ""
    gpu_ids = ",".join(str(value) for value in summary["gpu_ids"])
    parts = [f"gpu={gpu_ids}"]
    if summary["gfx_percent_mean"] is not None:
        if len(summary["gpu_ids"]) > 1:
            parts.append(
                f"gfx={summary['gfx_percent_mean']:.0f}%"
                f"[{summary['gfx_percent_min']:.0f}-{summary['gfx_percent_max']:.0f}]"
            )
        else:
            parts.append(f"gfx={summary['gfx_percent_mean']:.0f}%")
    if summary["umc_percent_mean"] is not None:
        parts.append(f"umc={summary['umc_percent_mean']:.0f}%")
    if summary["clock_mhz_mean"] is not None:
        parts.append(f"clk={summary['clock_mhz_mean']:.0f}MHz")
    if summary["power_watts_total"] is not None:
        parts.append(f"pwr={summary['power_watts_total']:.0f}W")
    return "  " + " ".join(parts)


def _print(payload: dict[str, Any], *, json_lines: bool) -> None:
    if json_lines:
        serializable = {
            key: value for key, value in payload.items() if key != "active"
        }
        print(json.dumps(serializable, separators=(",", ":")), flush=True)
        return
    identity = f"{payload['source']}/e{payload['engine']}"
    completion = payload.get("completion")
    if isinstance(completion, dict):
        cache_text = (
            f"cached={completion['cached_tokens']}/"
            f"{completion['cache_query_tokens']} "
            f"({completion['cache_hit_percent']:.1f}%)"
        )
        spec_text = ""
        if completion["spec_draft_tokens"]:
            spec_text = (
                f"  spec={completion['spec_accepted_tokens']}/"
                f"{completion['spec_draft_tokens']} "
                f"({completion['spec_acceptance_percent']:.1f}%)"
            )
        print(
            f"{payload['timestamp']}  {identity} COMPLETE "
            f"n={completion['completed_requests']} "
            f"prompt={completion['prompt_tokens']} output={completion['output_tokens']}  "
            f"{cache_text}  "
            f"prefill={completion['prefill_tokens_per_second']:.1f} tok/s "
            f"uncached={completion['uncached_prefill_tokens_per_second']:.1f} tok/s  "
            f"decode={completion['decode_tokens_per_second']:.1f} tok/s  "
            f"TTFT={completion['mean_ttft_seconds']:.3f}s "
            f"E2E={completion['mean_e2e_seconds']:.3f}s{spec_text}",
            flush=True,
        )
    elif payload.get("completion_unavailable"):
        print(
            f"{payload['timestamp']}  {identity} COMPLETE "
            f"n={payload['completion_unavailable']} "
            "metrics=unavailable (monitor attached after request start)",
            flush=True,
        )
    live_cache = payload.get("request_cache")
    cache_text = ""
    if isinstance(live_cache, dict) and live_cache["cache_query_tokens"]:
        cache_text = (
            f"  cached={live_cache['cached_tokens']}/"
            f"{live_cache['cache_query_tokens']} "
            f"({live_cache['cache_hit_percent']:.1f}%)"
        )
    live_spec = payload.get("request_speculative")
    spec_text = ""
    if isinstance(live_spec, dict) and live_spec["spec_draft_tokens"]:
        spec_text = (
            f"  spec={live_spec['spec_accepted_tokens']}/"
            f"{live_spec['spec_draft_tokens']} "
            f"({live_spec['spec_acceptance_percent']:.1f}%)"
        )
    print(
        f"{payload['timestamp']}  {identity} LIVE phase={payload['phase']:<7} "
        f"run={payload['running_requests']} wait={payload['waiting_requests']} "
        f"kv={payload['kv_cache_percent']:6.2f}%  "
        f"kv-growth~={payload['kv_growth_tokens_per_second']:8.1f} tok/s  "
        f"decode~={payload['live_decode_tokens_per_second']:6.1f} tok/s"
        f"{cache_text}{spec_text}{_format_gpu(payload.get('gpu'))}",
        flush=True,
    )


def follow(
    endpoints: list[Endpoint],
    *,
    interval: float,
    window: float,
    include_idle: bool,
    json_lines: bool,
    with_gpu: bool,
    samples: int,
) -> None:
    started_at = time.perf_counter()
    initial = {
        endpoint.label: _metrics_by_engine(endpoint.url) for endpoint in endpoints
    }
    monitors = {
        (endpoint.label, engine): EngineMonitor(
            endpoint, engine, values, started_at, window=window
        )
        for endpoint in endpoints
        for engine, values in initial[endpoint.label].items()
    }
    all_gpu_ids = tuple(
        sorted({gpu_id for endpoint in endpoints for gpu_id in endpoint.gpu_ids})
    )
    deadline = started_at
    iteration = 0
    while samples == 0 or iteration < samples:
        gpu_metrics = _gpu_snapshot(all_gpu_ids) if with_gpu else {}
        for endpoint in endpoints:
            current = _metrics_by_engine(endpoint.url)
            for engine in sorted(current, key=lambda value: int(value)):
                key = (endpoint.label, engine)
                if key not in monitors:
                    monitors[key] = EngineMonitor(
                        endpoint,
                        engine,
                        current[engine],
                        time.perf_counter(),
                        window=window,
                    )
                payload = monitors[key].observe(current[engine], time.perf_counter())
                payload["gpu"] = _gpu_summary(
                    endpoint.gpu_ids_for_engine(engine), gpu_metrics
                )
                if (
                    include_idle
                    or payload["active"]
                    or payload["completion"] is not None
                    or payload["completion_unavailable"]
                ):
                    _print(payload, json_lines=json_lines)
        iteration += 1
        deadline += interval
        if samples == 0 or iteration < samples:
            time.sleep(max(deadline - time.perf_counter(), 0.0))


def _endpoints(target: str, *, with_gpu: bool) -> list[Endpoint]:
    endpoints = []
    if target in {"main", "all"}:
        endpoints.append(_endpoint("MAIN", managed_state(), with_gpu=with_gpu))
    if target in {"workers", "all"}:
        endpoints.append(
            _endpoint(
                "WORKER",
                managed_state(state_path=WORKER_POOL_STATE_PATH),
                with_gpu=with_gpu,
            )
        )
    return endpoints


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Passively follow live vLLM throughput for the main Qwen, its four "
            "worker replicas, or the complete qwen-multi stack. No requests are sent."
        )
    )
    parser.add_argument(
        "--target",
        choices=("main", "workers", "all"),
        default="main",
        help="runtime component to follow (default: main)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="include passive amd-smi utilization, clock, and power samples",
    )
    parser.add_argument(
        "--include-idle", action="store_true", help="also print zero-request windows"
    )
    parser.add_argument(
        "--json-lines", action="store_true", help="emit machine-readable JSONL"
    )
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--window", type=float, default=10.0)
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="stop after N samples; zero follows until Ctrl-C",
    )
    args = parser.parse_args()
    if args.interval <= 0 or args.window <= 0:
        parser.error("--interval and --window must be positive")
    if args.samples < 0:
        parser.error("--samples cannot be negative")

    try:
        endpoints = _endpoints(args.target, with_gpu=args.gpu)
        for endpoint in endpoints:
            capacities = ",".join(
                f"e{engine}:{capacity}"
                for engine, capacity in sorted(
                    endpoint.capacities.items(), key=lambda item: int(item[0])
                )
            )
            print(
                f"following={endpoint.label} url={endpoint.url} "
                f"profile={endpoint.profile} runtime={endpoint.runtime} "
                f"kv_capacity_tokens={capacities}",
                file=sys.stderr if args.json_lines else sys.stdout,
                flush=True,
            )
        follow(
            endpoints,
            interval=args.interval,
            window=args.window,
            include_idle=args.include_idle,
            json_lines=args.json_lines,
            with_gpu=args.gpu,
            samples=args.samples,
        )
    except KeyboardInterrupt:
        print("monitor stopped", flush=True)
    except (ConfigurationError, OSError) as error:
        parser.error(str(error))
    return 0
