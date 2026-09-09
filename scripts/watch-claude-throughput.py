#!/usr/bin/env python3
"""Show live vLLM throughput while Claude Code uses the local LiteLLM route."""

from __future__ import annotations

import argparse
from collections import deque
import json
import re
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from r9700.config import ConfigurationError  # noqa: E402
from r9700.service import managed_state  # noqa: E402


KV_CAPACITY_RE = re.compile(r"GPU KV cache size: ([0-9,]+) tokens")
METRICS = {
    "running_requests": "vllm:num_requests_running",
    "waiting_requests": "vllm:num_requests_waiting",
    "kv_cache_fraction": "vllm:kv_cache_usage_perc",
    "prompt_tokens": "vllm:prompt_tokens_total",
    "generation_tokens": "vllm:generation_tokens_total",
    "completed_requests": "vllm:request_success_total",
    "prefill_seconds": "vllm:request_prefill_time_seconds_sum",
    "decode_seconds": "vllm:request_decode_time_seconds_sum",
    "ttft_seconds": "vllm:time_to_first_token_seconds_sum",
    "e2e_seconds": "vllm:e2e_request_latency_seconds_sum",
}


def _cache_capacity(path: Path) -> int:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := KV_CAPACITY_RE.search(line):
            return int(match.group(1).replace(",", ""))
    raise ConfigurationError(f"GPU KV cache capacity is absent from {path}")


def _metric_sum(text: str, name: str) -> float:
    values = []
    for line in text.splitlines():
        if not (line.startswith(name + "{") or line.startswith(name + " ")):
            continue
        try:
            values.append(float(line.rsplit(" ", 1)[1]))
        except ValueError:
            continue
    if not values:
        raise ConfigurationError(f"metric {name!r} is absent")
    return sum(values)


def _metrics(url: str) -> dict[str, float]:
    with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=5) as response:
        text = response.read().decode(errors="replace")
    return {key: _metric_sum(text, name) for key, name in METRICS.items()}


def _delta(current: dict[str, float], previous: dict[str, float], key: str) -> float:
    return max(current[key] - previous[key], 0.0)


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
    return {
        "completed_requests": completed,
        "prompt_tokens": round(prompt_tokens),
        "output_tokens": round(output_tokens),
        "prefill_tokens_per_second": prompt_tokens / max(prefill_seconds, 1e-9),
        "decode_tokens_per_second": max(output_tokens - completed, 0.0)
        / max(decode_seconds, 1e-9),
        "mean_ttft_seconds": _delta(current, previous, "ttft_seconds") / completed,
        "mean_e2e_seconds": _delta(current, previous, "e2e_seconds") / completed,
    }


def _print(payload: dict[str, Any], *, json_lines: bool) -> None:
    if json_lines:
        print(json.dumps(payload, separators=(",", ":")), flush=True)
        return
    completion = payload.get("completion")
    if isinstance(completion, dict):
        print(
            f"{payload['timestamp']}  COMPLETE n={completion['completed_requests']}  "
            f"prompt={completion['prompt_tokens']}  output={completion['output_tokens']}  "
            f"prefill={completion['prefill_tokens_per_second']:.1f} tok/s  "
            f"decode={completion['decode_tokens_per_second']:.1f} tok/s  "
            f"TTFT={completion['mean_ttft_seconds']:.3f}s  "
            f"E2E={completion['mean_e2e_seconds']:.3f}s",
            flush=True,
        )
    print(
        f"{payload['timestamp']}  LIVE running={payload['running_requests']}  "
        f"waiting={payload['waiting_requests']}  kv={payload['kv_cache_percent']:6.2f}%  "
        f"kv-growth~={payload['kv_growth_tokens_per_second']:8.1f} tok/s",
        flush=True,
    )


def follow(
    url: str,
    *,
    cache_capacity: int,
    interval: float,
    window: float,
    include_idle: bool,
    json_lines: bool,
) -> None:
    history: deque[tuple[float, float]] = deque()
    previous = _metrics(url)
    deadline = time.perf_counter()
    while True:
        now = time.perf_counter()
        current = _metrics(url)
        history.append((now, current["kv_cache_fraction"]))
        while len(history) > 1 and now - history[0][0] > window:
            history.popleft()
        oldest_time, oldest_kv = history[0]
        elapsed = max(now - oldest_time, 1e-9)
        kv_growth = max(current["kv_cache_fraction"] - oldest_kv, 0.0)
        completion = _completion(current, previous)
        active = current["running_requests"] or current["waiting_requests"]
        if include_idle or active or completion is not None:
            _print(
                {
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "running_requests": round(current["running_requests"]),
                    "waiting_requests": round(current["waiting_requests"]),
                    "kv_cache_percent": current["kv_cache_fraction"] * 100,
                    "kv_growth_tokens_per_second": kv_growth
                    * cache_capacity
                    / elapsed,
                    "kv_growth_window_seconds": elapsed,
                    "completion": completion,
                },
                json_lines=json_lines,
            )
        previous = current
        deadline += interval
        time.sleep(max(deadline - time.perf_counter(), 0.0))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Follow vLLM's live engine throughput while requests arrive through "
            "Claude Code and LiteLLM. Values are aggregate engine-window metrics."
        )
    )
    parser.add_argument(
        "--include-idle", action="store_true", help="also print zero-request windows"
    )
    parser.add_argument(
        "--json-lines", action="store_true", help="emit machine-readable JSONL"
    )
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--window", type=float, default=10.0)
    args = parser.parse_args()
    if args.interval <= 0 or args.window <= 0:
        parser.error("--interval and --window must be positive")

    try:
        state = managed_state()
    except ConfigurationError as error:
        parser.error(str(error))
    log_path = Path(str(state.get("log", "")))
    if not log_path.is_file():
        parser.error(f"managed runtime log is unavailable: {log_path}")
    cache_capacity = _cache_capacity(log_path)

    print(
        f"following={log_path} profile={state.get('profile')} "
        f"runtime={state.get('runtime')} kv_capacity_tokens={cache_capacity}",
        flush=True,
    )
    try:
        follow(
            str(state["url"]),
            cache_capacity=cache_capacity,
            interval=args.interval,
            window=args.window,
            include_idle=args.include_idle,
            json_lines=args.json_lines,
        )
    except KeyboardInterrupt:
        print("monitor stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
