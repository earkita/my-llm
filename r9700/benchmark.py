from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import statistics
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from .api import get_json, post_json, served_context_length
from .config import (
    ConfigurationError,
    load_model,
    load_runtime,
    validate_compatibility,
)
from .backends import backend_manifest_sha256, runtime_backend
from .service import managed_state


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def _scheduler_session_capacity(runtime: dict[str, Any]) -> int:
    return runtime["limits"]["max_num_seqs"] * runtime["parallel"].get("data", 1)


def _data_parallel_gpu_groups(runtime: dict[str, Any]) -> list[list[int]]:
    parallel = runtime["parallel"]
    data_parallel_size = parallel.get("data", 1)
    devices_per_instance = parallel["tensor"] * parallel["pipeline"]
    gpu_order = runtime["gpu_order"]
    return [
        gpu_order[
            rank * devices_per_instance : (rank + 1) * devices_per_instance
        ]
        for rank in range(data_parallel_size)
    ]


def _prompt_seed_ids(url: str, model: str, backend: str) -> list[int]:
    seed = "amber cedar delta granite harbor juniper"
    tokenize_body = (
        {"content": seed, "add_special": False}
        if backend == "llama-cpp"
        else {"model": model, "prompt": seed, "add_special_tokens": False}
    )
    payload, _elapsed = post_json(
        url.rstrip("/") + "/tokenize",
        tokenize_body,
        timeout=120,
    )
    ids = payload.get("tokens")
    if not isinstance(ids, list) or not ids or not all(isinstance(item, int) for item in ids):
        raise ConfigurationError("/tokenize did not return token ids")
    return ids


def _prompt_ids(seed_ids: list[int], count: int, variant: int) -> list[int]:
    """Tile the benchmark seed with a distinct first token for each wave.

    A different first token prevents a warmup request from turning a measured
    prefill into a prefix-cache hit.  Rotations remain deterministic so the
    same wave is directly comparable across runtime profiles.
    """
    offsets: list[int] = []
    seen: set[int] = set()
    for offset, token_id in enumerate(seed_ids):
        if token_id not in seen:
            seen.add(token_id)
            offsets.append(offset)
    if variant >= len(offsets):
        raise ConfigurationError(
            "benchmark has more waves than distinct prompt-seed variants"
        )
    offset = offsets[variant]
    return [seed_ids[(offset + index) % len(seed_ids)] for index in range(count)]


def _stream_request(
    url: str,
    model: str,
    prompt: list[int],
    output_tokens: int,
    barrier: threading.Barrier,
    timeout: float,
    data_parallel_rank: int | None,
) -> dict[str, Any]:
    expected_prompt_tokens = len(prompt)
    body = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": output_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Content-Type": "application/json"}
    if data_parallel_rank is not None:
        headers["X-data-parallel-rank"] = str(data_parallel_rank)
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/completions",
        data=json.dumps(body).encode(),
        headers=headers,
    )
    barrier.wait(timeout=30)
    started = time.perf_counter()
    first_content = None
    usage = None
    finish_reason = None
    event_count = 0
    output_hasher = hashlib.sha256()
    output_characters = 0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            event_count += 1
            choices = event.get("choices") or []
            if choices:
                text = choices[0].get("text")
                if isinstance(text, str) and text:
                    if first_content is None:
                        first_content = time.perf_counter()
                    encoded = text.encode("utf-8")
                    output_hasher.update(encoded)
                    output_characters += len(text)
                finish_reason = choices[0].get("finish_reason") or finish_reason
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
    ended = time.perf_counter()
    if first_content is None or not isinstance(usage, dict):
        raise ConfigurationError("stream omitted content timing or final usage")
    if usage.get("prompt_tokens") != expected_prompt_tokens:
        raise ConfigurationError("stream prompt usage differs from requested token ids")
    if usage.get("completion_tokens") != output_tokens or finish_reason != "length":
        raise ConfigurationError(
            "benchmark completion was not exact length: "
            f"completion_tokens={usage.get('completion_tokens')!r}, "
            f"expected={output_tokens}, finish_reason={finish_reason!r}"
        )
    ttft = first_content - started
    e2e = ended - started
    decode_time = max(ended - first_content, 1e-9)
    return {
        "started_perf": started,
        "ended_perf": ended,
        "ttft_seconds": ttft,
        "e2e_seconds": e2e,
        "observed_prefill_tokens_per_second": expected_prompt_tokens / ttft,
        "decode_tokens_per_second": max(output_tokens - 1, 0) / decode_time,
        "event_count": event_count,
        "output_sha256": output_hasher.hexdigest(),
        "output_characters": output_characters,
        "usage": usage,
        "finish_reason": finish_reason,
    }


def benchmark(
    *,
    url: str,
    model_name: str,
    runtime_name: str,
    runtime_mode: str | None = None,
    prompt_tokens: int,
    output_tokens: int,
    concurrency: int,
    repetitions: int,
    warmup: int,
    timeout: float,
    output: Path,
    pin_data_parallel: bool = False,
    data_parallel_rank: int | None = None,
    prompt_variant_offset: int = 0,
) -> dict[str, Any]:
    if min(prompt_tokens, output_tokens, concurrency, repetitions) < 1 or warmup < 0:
        raise ConfigurationError("benchmark dimensions must be positive")
    if prompt_variant_offset < 0:
        raise ConfigurationError("prompt variant offset must be non-negative")
    model = load_model(model_name)
    runtime = load_runtime(runtime_name, runtime_mode)
    backend = runtime_backend(runtime)
    selected_manifest_sha256 = backend_manifest_sha256(runtime)
    validate_compatibility(model, runtime)
    if prompt_tokens + output_tokens > runtime["limits"]["max_model_len"]:
        raise ConfigurationError("benchmark exceeds the runtime context boundary")
    scheduler_sessions = _scheduler_session_capacity(runtime)
    if concurrency > scheduler_sessions:
        raise ConfigurationError("benchmark exceeds runtime scheduler sessions")
    data_parallel_size = runtime["parallel"].get("data", 1)
    if pin_data_parallel and data_parallel_rank is not None:
        raise ConfigurationError(
            "choose either balanced DP pinning or one explicit data parallel rank"
        )
    if data_parallel_rank is not None and not (
        0 <= data_parallel_rank < data_parallel_size
    ):
        raise ConfigurationError("data parallel rank is outside the runtime")
    if pin_data_parallel and concurrency % data_parallel_size:
        raise ConfigurationError(
            "pinned benchmark concurrency must be divisible by data parallel size"
        )
    if (
        data_parallel_rank is not None
        and concurrency > runtime["limits"]["max_num_seqs"]
    ):
        raise ConfigurationError(
            "benchmark exceeds scheduler sessions of the selected DP rank"
        )
    url = url.rstrip("/")
    state = managed_state()
    if (
        state.get("url") != url
        or state.get("model") != model["name"]
        or state.get("backend", "vllm") != backend
        or state.get("recipe") != runtime["recipe"]
        or state.get("runtime") != runtime["name"]
        or state.get("runtime_mode") != runtime.get("active_experimental_mode")
        or state.get("runtime_profile_sha256") != runtime["_sha256"]
    ):
        raise ConfigurationError("managed service identity differs from benchmark profiles")
    models = get_json(url + "/v1/models")
    served = next(
        (
            row
            for row in models.get("data") or []
            if isinstance(row, dict) and row.get("id") == model["served_name"]
        ),
        None,
    )
    if (
        not isinstance(served, dict)
        or served_context_length(served) != runtime["limits"]["max_model_len"]
    ):
        raise ConfigurationError("API model/context identity differs from benchmark profile")
    seed_ids = _prompt_seed_ids(url, model["served_name"], backend)
    rows: list[dict[str, Any]] = []
    waves: list[dict[str, Any]] = []
    for wave in range(warmup + repetitions):
        prompt_variant = prompt_variant_offset + wave
        prompt = _prompt_ids(seed_ids, prompt_tokens, prompt_variant)
        barrier = threading.Barrier(concurrency)
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = []
            for session in range(concurrency):
                request_rank = (
                    session % data_parallel_size
                    if pin_data_parallel
                    else data_parallel_rank
                )
                futures.append(
                    pool.submit(
                        _stream_request,
                        url,
                        model["served_name"],
                        prompt,
                        output_tokens,
                        barrier,
                        timeout,
                        request_rank,
                    )
                )
            wave_rows = [future.result() for future in futures]
        if wave >= warmup:
            for session, row in enumerate(wave_rows):
                request_rank = (
                    session % data_parallel_size
                    if pin_data_parallel
                    else data_parallel_rank
                )
                row.update(
                    {
                        "wave": wave - warmup,
                        "prompt_variant": prompt_variant,
                        "prompt_first_token_id": prompt[0],
                        "session": session,
                        "data_parallel_rank": request_rank,
                        "instance_session": (
                            session // data_parallel_size
                            if pin_data_parallel
                            else session if request_rank is not None else None
                        ),
                    }
                )
                rows.append(row)
            start = min(row["started_perf"] for row in wave_rows)
            end = max(row["ended_perf"] for row in wave_rows)
            rates = [row["decode_tokens_per_second"] for row in wave_rows]
            fairness = (sum(rates) ** 2) / (len(rates) * sum(rate * rate for rate in rates))
            waves.append(
                {
                    "wave": wave - warmup,
                    "wall_seconds": end - start,
                    "start_skew_seconds": max(row["started_perf"] for row in wave_rows) - start,
                    "aggregate_output_tokens_per_second": (
                        concurrency * output_tokens / max(end - start, 1e-9)
                    ),
                    "decode_jain_fairness": fairness,
                }
            )
    ttft = [row["ttft_seconds"] for row in rows]
    e2e = [row["e2e_seconds"] for row in rows]
    decode = [row["decode_tokens_per_second"] for row in rows]
    prefill = [row["observed_prefill_tokens_per_second"] for row in rows]
    instance_summaries = []
    gpu_groups = _data_parallel_gpu_groups(runtime)
    measured_ranks: list[int] = []
    if pin_data_parallel:
        measured_ranks = list(range(data_parallel_size))
    elif data_parallel_rank is not None:
        measured_ranks = [data_parallel_rank]
    for rank in measured_ranks:
            gpu_group = gpu_groups[rank]
            instance_rows = [
                row for row in rows if row["data_parallel_rank"] == rank
            ]
            instance_ttft = [row["ttft_seconds"] for row in instance_rows]
            instance_e2e = [row["e2e_seconds"] for row in instance_rows]
            instance_decode = [
                row["decode_tokens_per_second"] for row in instance_rows
            ]
            instance_prefill = [
                row["observed_prefill_tokens_per_second"] for row in instance_rows
            ]
            wave_rates = []
            for wave in range(repetitions):
                wave_instance_rows = [
                    row for row in instance_rows if row["wave"] == wave
                ]
                start = min(row["started_perf"] for row in wave_instance_rows)
                end = max(row["ended_perf"] for row in wave_instance_rows)
                wave_rates.append(
                    len(wave_instance_rows)
                    * output_tokens
                    / max(end - start, 1e-9)
                )
            instance_summaries.append(
                {
                    "data_parallel_rank": rank,
                    "gpu_order": gpu_group,
                    "requests": len(instance_rows),
                    "sessions_per_wave": (
                        concurrency // data_parallel_size
                        if pin_data_parallel
                        else concurrency
                    ),
                    "ttft_mean_seconds": statistics.mean(instance_ttft),
                    "ttft_p95_seconds": _percentile(instance_ttft, 0.95),
                    "e2e_mean_seconds": statistics.mean(instance_e2e),
                    "e2e_p95_seconds": _percentile(instance_e2e, 0.95),
                    "observed_prefill_mean_tokens_per_second": statistics.mean(
                        instance_prefill
                    ),
                    "decode_mean_tokens_per_second": statistics.mean(
                        instance_decode
                    ),
                    "decode_min_tokens_per_second": min(instance_decode),
                    "aggregate_output_mean_tokens_per_second": statistics.mean(
                        wave_rates
                    ),
                }
            )
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "configuration": {
            "url": url,
            "model": model["name"],
            "model_profile_sha256": model["_sha256"],
            "runtime": runtime["name"],
            "runtime_profile_sha256": runtime["_sha256"],
            "backend": backend,
            "backend_manifest_sha256": selected_manifest_sha256,
            "runtime_manifest_sha256": selected_manifest_sha256,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "concurrency": concurrency,
            "data_parallel_size": data_parallel_size,
            "pin_data_parallel": pin_data_parallel,
            "data_parallel_rank": data_parallel_rank,
            "data_parallel_gpu_groups": gpu_groups,
            "scheduler_sessions": scheduler_sessions,
            "repetitions": repetitions,
            "warmup": warmup,
            "prompt_variant_offset": prompt_variant_offset,
            "prompt_variant_policy": "unique-first-token rotation per wave",
        },
        "rows": rows,
        "instances": instance_summaries,
        "waves": waves,
        "summary": {
            "requests": len(rows),
            "ttft_mean_seconds": statistics.mean(ttft),
            "ttft_p50_seconds": _percentile(ttft, 0.5),
            "ttft_p95_seconds": _percentile(ttft, 0.95),
            "e2e_mean_seconds": statistics.mean(e2e),
            "e2e_p95_seconds": _percentile(e2e, 0.95),
            "observed_prefill_mean_tokens_per_second": statistics.mean(prefill),
            "decode_mean_tokens_per_second": statistics.mean(decode),
            "decode_min_tokens_per_second": min(decode),
            "aggregate_output_mean_tokens_per_second": statistics.mean(
                row["aggregate_output_tokens_per_second"] for row in waves
            ),
            "decode_jain_fairness_min": min(row["decode_jain_fairness"] for row in waves),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps(payload["summary"], indent=2))
    return payload
