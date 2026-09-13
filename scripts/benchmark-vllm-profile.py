#!/usr/bin/env python3
"""Run reproducible client-side vLLM serving benchmarks for a profile."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Scenario:
    name: str
    input_tokens: int
    output_tokens: int
    requests: int
    concurrency: int
    warmups: int
    seed: int
    dataset: str = "random"


SCENARIOS = {
    "interactive": Scenario("interactive", 1024, 256, 5, 1, 1, 3101),
    "prefill": Scenario("prefill", 16384, 32, 3, 1, 1, 3201),
    "decode": Scenario("decode", 128, 512, 5, 1, 1, 3301),
    "prefix": Scenario("prefix", 8256, 32, 4, 1, 0, 3401, "prefix_repetition"),
    "pool": Scenario("pool", 1024, 256, 12, 4, 1, 3501),
}


@dataclass(frozen=True)
class Target:
    requested_profile: str
    component: str
    profile_path: Path
    state_path: Path
    endpoint: str
    model_name: str
    served_name: str
    model_directory: Path
    model_revision: str
    runtime_name: str
    recipe: str
    max_model_len: int
    max_num_seqs: int
    data_parallel_size: int
    cache_dtype: str
    prefix_cache: bool
    profile_sha256: str
    runtime_profile_sha256: str | None
    access_mode: str
    alias: str | None
    gateway_state_path: Path | None
    request_overrides: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise RuntimeError(f"missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if not key.startswith("_")}
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _profile_path(repo_root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.suffix == ".json" or candidate.is_absolute() or "/" in value:
        path = candidate if candidate.is_absolute() else repo_root / candidate
    else:
        path = repo_root / "profiles" / "production" / f"{value}.json"
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"profile does not exist: {path}")
    return path


def _resolve_component_profile(
    repo_root: Path,
    profile: dict[str, Any],
    component: str,
) -> tuple[Path, dict[str, Any], Path, str]:
    if component == "primary":
        return (
            Path(profile["_path"]),
            profile,
            repo_root / ".runtime" / "service.json",
            "http://127.0.0.1:8000",
        )
    worker = profile.get("components", {}).get("worker_pool", {})
    worker_name = worker.get("profile")
    if not isinstance(worker_name, str) or not worker_name:
        raise RuntimeError("profile does not define components.worker_pool.profile")
    worker_path = _profile_path(repo_root, worker_name)
    worker_profile = _load_json(worker_path)
    worker_profile["_path"] = str(worker_path)
    endpoint = worker.get("url", "http://127.0.0.1:8100")
    return (
        worker_path,
        worker_profile,
        repo_root / ".runtime" / "worker-pool" / "service.json",
        endpoint,
    )


def resolve_target(
    repo_root: Path,
    profile_name: str,
    component: str,
    *,
    require_active: bool,
) -> Target:
    requested_path = _profile_path(repo_root, profile_name)
    requested = _load_json(requested_path)
    requested["_path"] = str(requested_path)
    requested_name = requested.get("name", requested_path.stem)
    if not isinstance(requested_name, str) or not requested_name:
        raise RuntimeError("profile name must be a non-empty string")
    profile_path, profile, state_path, default_endpoint = _resolve_component_profile(
        repo_root, requested, component
    )
    model = profile.get("model", {})
    runtime = profile.get("runtime", {})
    if not isinstance(model, dict) or not isinstance(runtime, dict):
        raise RuntimeError("profile must embed model and runtime objects")
    if runtime.get("recipe") is None:
        raise RuntimeError("profile runtime has no recipe")
    state: dict[str, Any] = {}
    if require_active:
        state = _load_json(state_path)
        expected = {
            "model": model.get("name"),
            "runtime": runtime.get("name"),
            "recipe": runtime.get("recipe"),
            "runtime_profile_sha256": _canonical_sha256(runtime),
        }
        mismatches = [
            f"{key}={state.get(key)!r}, expected {value!r}"
            for key, value in expected.items()
            if state.get(key) != value
        ]
        pid = state.get("pid")
        if not isinstance(pid, int) or not Path(f"/proc/{pid}").exists():
            mismatches.append(f"pid={pid!r} is not live")
        if mismatches:
            raise RuntimeError(
                "managed service does not match the benchmark target: "
                + "; ".join(mismatches)
            )
    limits = runtime.get("limits", {})
    parallel = runtime.get("parallel", {})
    cache = runtime.get("cache", {})
    return Target(
        requested_profile=requested_name,
        component=component,
        profile_path=profile_path,
        state_path=state_path,
        endpoint=str(state.get("url", default_endpoint)).rstrip("/"),
        model_name=str(model["name"]),
        served_name=str(model["served_name"]),
        model_directory=Path(model["default_directory"]),
        model_revision=str(model.get("revision", "unknown")),
        runtime_name=str(runtime["name"]),
        recipe=str(runtime["recipe"]),
        max_model_len=int(limits["max_model_len"]),
        max_num_seqs=int(limits["max_num_seqs"]),
        data_parallel_size=int(parallel.get("data", 1)),
        cache_dtype=str(cache.get("dtype", "auto")),
        prefix_cache=bool(cache.get("prefix_cache", False)),
        profile_sha256=_canonical_sha256(profile),
        runtime_profile_sha256=(
            str(state["runtime_profile_sha256"])
            if state.get("runtime_profile_sha256")
            else None
        ),
        access_mode="direct",
        alias=None,
        gateway_state_path=None,
        request_overrides={},
    )


def resolve_alias_target(
    repo_root: Path,
    alias: str,
    *,
    access_mode: str,
    require_active: bool,
    requested_profile: str | None = None,
) -> Target:
    if access_mode not in {"litellm", "direct", "raw"}:
        raise RuntimeError(f"unknown access mode: {access_mode}")
    gateway_state_path = repo_root / ".runtime" / "litellm" / "service.json"
    routing_state_path = (
        repo_root / ".runtime" / "service.json"
        if access_mode != "litellm"
        else gateway_state_path
    )
    routing_state = _load_json(routing_state_path)
    profile_name = routing_state.get("profile")
    if not isinstance(profile_name, str) or not profile_name:
        raise RuntimeError("LiteLLM state does not identify its active profile")
    if requested_profile is not None:
        requested_name = _load_json(_profile_path(repo_root, requested_profile)).get(
            "name"
        )
        if requested_name != profile_name:
            raise RuntimeError(
                f"LiteLLM is bound to {profile_name}, not {requested_name}"
            )
    if require_active:
        pid = routing_state.get("pid")
        if not isinstance(pid, int) or not Path(f"/proc/{pid}").exists():
            raise RuntimeError(
                f"{routing_state_path} does not point to a live process"
            )
    profile = _load_json(_profile_path(repo_root, profile_name))
    active_aliases = profile.get("stack", {}).get("litellm_aliases", [])
    if alias not in active_aliases:
        raise RuntimeError(
            f"model {alias!r} is not active for profile {profile_name}; "
            f"available: {', '.join(sorted(active_aliases))}"
        )
    component = "primary"
    worker = profile.get("components", {}).get("worker_pool", {})
    worker_profile_name = worker.get("profile") if isinstance(worker, dict) else None
    if isinstance(worker_profile_name, str):
        worker_profile = _load_json(_profile_path(repo_root, worker_profile_name))
        worker_aliases = worker_profile.get("stack", {}).get("litellm_aliases", [])
        if alias in worker_aliases:
            component = "worker-pool"
    target = resolve_target(
        repo_root, profile_name, component, require_active=require_active
    )
    if access_mode != "litellm":
        return replace(target, access_mode=access_mode, alias=alias)
    gateway_state = routing_state
    endpoint = gateway_state.get("probe_url") or gateway_state.get("url")
    if not isinstance(endpoint, str) or not endpoint:
        raise RuntimeError("LiteLLM state does not contain a usable URL")
    return replace(
        target,
        endpoint=endpoint.rstrip("/"),
        served_name=alias,
        access_mode="litellm",
        alias=alias,
        gateway_state_path=gateway_state_path,
    )


def _litellm_key(repo_root: Path) -> str:
    helper = repo_root / "scripts" / "claude-litellm-key.sh"
    completed = subprocess.run(
        [str(helper)],
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    key = completed.stdout
    if completed.returncode or not key:
        detail = completed.stderr.strip() or "key helper returned no value"
        raise RuntimeError(f"cannot obtain LiteLLM key: {detail}")
    return key


def _models(endpoint: str, api_key: str | None = None) -> list[str]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    request = urllib.request.Request(endpoint + "/v1/models", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.load(response)
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot query {endpoint}/v1/models: {exc}") from exc
    return [
        row["id"]
        for row in payload.get("data", [])
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    ]


DIRECT_ALIAS_PARAMETER_KEYS = frozenset(
    {
        "chat_template_kwargs",
        "frequency_penalty",
        "min_p",
        "presence_penalty",
        "reasoning_effort",
        "repetition_penalty",
        "temperature",
        "top_k",
        "top_p",
    }
)


def _litellm_alias_overrides(repo_root: Path, alias: str) -> dict[str, Any]:
    state_path = repo_root / ".runtime" / "litellm" / "service.json"
    state = _load_json(state_path)
    endpoint = state.get("probe_url") or state.get("url")
    if not isinstance(endpoint, str) or not endpoint:
        raise RuntimeError("LiteLLM state does not contain a usable URL")
    pid = state.get("pid")
    if not isinstance(pid, int) or not Path(f"/proc/{pid}").exists():
        raise RuntimeError(
            "--direct needs the active LiteLLM model metadata; "
            "use --raw when the proxy is stopped"
        )
    headers = {"Authorization": f"Bearer {_litellm_key(repo_root)}"}
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/model/info", headers=headers
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.load(response)
    except (urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot query LiteLLM model metadata: {exc}") from exc
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    matches = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("model_name") == alias
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one LiteLLM deployment for {alias!r}, found {len(matches)}"
        )
    parameters = matches[0].get("litellm_params", {})
    if not isinstance(parameters, dict):
        raise RuntimeError(f"LiteLLM parameters for {alias!r} are invalid")
    overrides: dict[str, Any] = {}
    extra_body = parameters.get("extra_body")
    if isinstance(extra_body, dict):
        overrides.update(extra_body)
    overrides.update(
        {
            key: parameters[key]
            for key in DIRECT_ALIAS_PARAMETER_KEYS
            if key in parameters
        }
    )
    return overrides


def _recipe_paths(repo_root: Path, recipe: str) -> tuple[Path, Path, Path]:
    recipe_root = repo_root / ".runtime" / "recipes" / recipe
    vllm = recipe_root / "venv" / "bin" / "vllm"
    candidates = sorted(
        (recipe_root / "venv" / "lib").glob(
            "python*/site-packages/_rocm_sdk_devel"
        )
    )
    if not vllm.is_file():
        raise RuntimeError(f"profile recipe is not installed: {vllm}")
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one packaged ROCm root for {recipe}, found {len(candidates)}"
        )
    return recipe_root, vllm, candidates[0]


def benchmark_environment(
    repo_root: Path, recipe_root: Path, rocm_root: Path
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": (
                f"{rocm_root / 'bin'}:{recipe_root / 'venv' / 'bin'}:"
                + environment.get("PATH", "")
            ),
            "LD_LIBRARY_PATH": (
                f"{rocm_root / 'lib'}:" + environment.get("LD_LIBRARY_PATH", "")
            ),
            "PYTHONPATH": f"{repo_root / 'r9700' / 'vllm_bootstrap'}:{repo_root}",
            # `vllm bench serve` is only an HTTP client. Force its platform to
            # CPU so mixed AMD/NVIDIA hosts cannot activate both built-in GPU
            # plugins while importing the ROCm recipe's vLLM CLI.
            "VLLM_TARGET_DEVICE": "cpu",
            "HIP_VISIBLE_DEVICES": "",
            "CUDA_VISIBLE_DEVICES": "",
            "VLLM_PLUGINS": "",
            "VLLM_ROCM_USE_AITER": "0",
            "VLLM_ROCM_USE_AITER_MOE": "0",
            "VLLM_LOGGING_LEVEL": "ERROR",
        }
    )
    return environment


def _selected_scenarios(component: str, values: list[str] | None) -> list[Scenario]:
    names = values or ["all"]
    if "all" in names and len(names) != 1:
        raise RuntimeError("--scenario all cannot be combined with another scenario")
    if names == ["all"]:
        names = ["interactive", "prefill", "decode", "prefix"]
        if component == "worker-pool":
            names.append("pool")
    if component != "worker-pool" and "pool" in names:
        raise RuntimeError("the pool scenario is only valid for worker-pool")
    return [SCENARIOS[name] for name in names]


def _apply_overrides(args: argparse.Namespace, scenarios: list[Scenario]) -> list[Scenario]:
    overrides = (args.input_tokens, args.output_tokens, args.requests, args.concurrency)
    if any(value is not None for value in overrides) and len(scenarios) != 1:
        raise RuntimeError(
            "dimension overrides require exactly one explicit --scenario"
        )
    if not scenarios:
        raise RuntimeError("no benchmark scenarios selected")
    if len(scenarios) != 1:
        return scenarios
    scenario = scenarios[0]
    def selected(value: int | None, default: int) -> int:
        return default if value is None else value

    return [
        replace(
            scenario,
            input_tokens=selected(args.input_tokens, scenario.input_tokens),
            output_tokens=selected(args.output_tokens, scenario.output_tokens),
            requests=selected(args.requests, scenario.requests),
            concurrency=selected(args.concurrency, scenario.concurrency),
        )
    ]


def validate_scenario(
    target: Target, scenario: Scenario, worker_rank: int | None
) -> None:
    dimensions = (
        scenario.input_tokens,
        scenario.output_tokens,
        scenario.requests,
        scenario.concurrency,
    )
    if any(value < 1 for value in dimensions):
        raise RuntimeError(f"invalid dimensions for scenario {scenario.name}")
    if scenario.input_tokens + scenario.output_tokens > target.max_model_len:
        raise RuntimeError(
            f"scenario {scenario.name} exceeds context {target.max_model_len}"
        )
    if worker_rank is not None and not 0 <= worker_rank < target.data_parallel_size:
        raise RuntimeError(
            f"worker rank {worker_rank} is outside DP{target.data_parallel_size}"
        )
    capacity = target.max_num_seqs * target.data_parallel_size
    if scenario.concurrency > capacity:
        raise RuntimeError(
            f"scenario concurrency {scenario.concurrency} exceeds capacity {capacity}"
        )
    if worker_rank is not None and scenario.concurrency > target.max_num_seqs:
        raise RuntimeError("a rank-pinned benchmark exceeds per-worker capacity")


def build_command(
    vllm: Path,
    target: Target,
    scenario: Scenario,
    output_dir: Path,
    worker_rank: int | None,
) -> list[str]:
    via_litellm = target.access_mode == "litellm"
    raw_engine = target.access_mode == "raw"
    chat_api = not raw_engine
    command = [
        str(vllm.parent / "python"),
        str(REPO_ROOT / "scripts" / "vllm-bench-client.py"),
        "bench",
        "serve",
        "--backend",
        "openai-chat" if chat_api else "openai",
        "--base-url",
        target.endpoint,
        "--endpoint",
        "/v1/chat/completions" if chat_api else "/v1/completions",
        "--model",
        str(target.model_directory),
        "--served-model-name",
        target.served_name,
        "--tokenizer",
        str(target.model_directory),
        "--num-prompts",
        str(scenario.requests),
        "--num-warmups",
        str(scenario.warmups),
        "--max-concurrency",
        str(scenario.concurrency),
        "--request-rate",
        "inf",
        "--seed",
        str(scenario.seed),
        "--disable-tqdm",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,90,95,99",
        "--ready-check-timeout-sec",
        "30",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(output_dir),
        "--result-filename",
        f"{scenario.name}.json",
        "--label",
        f"{target.requested_profile}-{target.component}-{scenario.name}",
        "--metadata",
        f"profile={target.requested_profile}",
        f"component={target.component}",
        f"scenario={scenario.name}",
        f"model_revision={target.model_revision}",
        f"runtime={target.runtime_name}",
        f"recipe={target.recipe}",
        f"profile_sha256={target.profile_sha256}",
        f"access_mode={target.access_mode}",
    ]
    if raw_engine:
        command.extend(["--ignore-eos", "--temperature", "0"])
    elif target.access_mode == "direct" and target.request_overrides:
        command.extend(
            [
                "--extra-body",
                json.dumps(
                    target.request_overrides,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )
    if scenario.dataset == "prefix_repetition":
        command.extend(
            [
                "--dataset-name",
                "prefix_repetition",
                "--prefix-repetition-num-prefixes",
                "1",
                "--prefix-repetition-prefix-len",
                str(scenario.input_tokens - 64),
                "--prefix-repetition-suffix-len",
                "64",
                "--prefix-repetition-output-len",
                str(scenario.output_tokens),
            ]
        )
    else:
        command.extend(
            [
                "--dataset-name",
                "random",
                "--random-input-len",
                str(scenario.input_tokens),
                "--random-output-len",
                str(scenario.output_tokens),
                "--random-range-ratio",
                "0",
            ]
        )
    if worker_rank is not None and not via_litellm:
        command.extend(["--header", f"X-data-parallel-rank={worker_rank}"])
    return command


def summarize_result(scenario: Scenario, payload: dict[str, Any]) -> dict[str, Any]:
    completed = int(payload.get("completed", 0))
    failed = int(payload.get("failed", 0))
    input_lens = [int(value) for value in payload.get("input_lens", [])]
    output_lens = [int(value) for value in payload.get("output_lens", [])]
    ttfts = [float(value) for value in payload.get("ttfts", [])]
    effective_prefill = [
        input_len / ttft
        for input_len, ttft in zip(input_lens, ttfts)
        if input_len > 0 and ttft > 0
    ]
    mean_tpot_ms = float(payload.get("mean_tpot_ms", 0) or 0)
    summary: dict[str, Any] = {
        "scenario": scenario.name,
        "requests": scenario.requests,
        "completed": completed,
        "failed": failed,
        "concurrency": scenario.concurrency,
        "mean_input_tokens": statistics.mean(input_lens) if input_lens else None,
        "mean_output_tokens": statistics.mean(output_lens) if output_lens else None,
        "requested_output_tokens": scenario.output_tokens,
        "output_completion_percent": (
            100.0 * statistics.mean(output_lens) / scenario.output_tokens
            if output_lens and scenario.output_tokens > 0
            else None
        ),
        "ttft_mean_ms": payload.get("mean_ttft_ms"),
        "ttft_p95_ms": payload.get("p95_ttft_ms"),
        "effective_prefill_tokens_per_second": (
            statistics.mean(effective_prefill) if effective_prefill else None
        ),
        "tpot_mean_ms": payload.get("mean_tpot_ms"),
        "decode_tokens_per_second": (
            1000.0 / mean_tpot_ms if mean_tpot_ms > 0 else None
        ),
        "itl_mean_ms": payload.get("mean_itl_ms"),
        "e2el_mean_ms": payload.get("mean_e2el_ms"),
        "output_throughput_tokens_per_second": payload.get("output_throughput"),
        "total_throughput_tokens_per_second": payload.get(
            "total_token_throughput"
        ),
        "spec_decode_acceptance_percent": payload.get(
            "spec_decode_acceptance_rate"
        ),
    }
    if scenario.dataset == "prefix_repetition" and ttfts:
        warm = ttfts[1:]
        cold_ms = ttfts[0] * 1000
        warm_ms = statistics.mean(warm) * 1000 if warm else None
        summary.update(
            {
                "prefix_cold_ttft_ms": cold_ms,
                "prefix_warm_ttft_ms": warm_ms,
                "prefix_ttft_speedup": (
                    cold_ms / warm_ms if warm_ms and warm_ms > 0 else None
                ),
            }
        )
    return summary


def _number(value: Any, digits: int = 1) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{value:.{digits}f}"


def print_summary(rows: list[dict[str, Any]]) -> None:
    headings = (
        "SCENARIO",
        "IN",
        "OUT/TARGET",
        "C",
        "TTFT ms",
        "PREFILL tok/s*",
        "TPOT ms",
        "DECODE tok/s",
        "OUT tok/s",
    )
    rendered = []
    for row in rows:
        rendered.append(
            (
                str(row["scenario"]),
                _number(row["mean_input_tokens"], 0),
                (
                    f"{_number(row['mean_output_tokens'], 0)}/"
                    f"{row['requested_output_tokens']}"
                ),
                str(row["concurrency"]),
                _number(row["ttft_mean_ms"]),
                _number(row["effective_prefill_tokens_per_second"]),
                _number(row["tpot_mean_ms"], 2),
                _number(row["decode_tokens_per_second"]),
                _number(row["output_throughput_tokens_per_second"]),
            )
        )
    widths = [
        max(len(headings[index]), *(len(row[index]) for row in rendered))
        for index in range(len(headings))
    ]
    print()
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(headings)))
    print("  ".join("-" * width for width in widths))
    for row in rendered:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    print("\n* effective client-observed prefill = actual input tokens / TTFT")


def render_html(target: Target, rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['scenario']))}</td>"
            f"<td>{_number(row['mean_input_tokens'], 0)}</td>"
            f"<td>{_number(row['mean_output_tokens'], 0)}/{row['requested_output_tokens']}</td>"
            f"<td>{row['concurrency']}</td>"
            f"<td>{_number(row['ttft_mean_ms'])}</td>"
            f"<td>{_number(row['ttft_p95_ms'])}</td>"
            f"<td>{_number(row['effective_prefill_tokens_per_second'])}</td>"
            f"<td>{_number(row['tpot_mean_ms'], 2)}</td>"
            f"<td>{_number(row['decode_tokens_per_second'])}</td>"
            f"<td>{_number(row['output_throughput_tokens_per_second'])}</td>"
            f"<td>{_number(row['spec_decode_acceptance_percent'])}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(target.requested_profile)} benchmark</title>
<style>
body {{ background:#0b1020; color:#e7ecf7; font:15px system-ui; margin:2rem; }}
h1 {{ margin-bottom:.25rem; }}
.meta {{ color:#a8b3cf; margin-bottom:1.5rem; }}
table {{ border-collapse:collapse; width:100%; background:#121a2d; }}
th,td {{ border-bottom:1px solid #26324d; padding:.7rem; text-align:right; }}
th:first-child,td:first-child {{ text-align:left; }}
th {{ color:#8bd5ff; }}
.note {{ color:#a8b3cf; margin-top:1rem; }}
</style>
</head>
<body>
<h1>{html.escape(target.requested_profile)} / {html.escape(target.component)}</h1>
<div class="meta">{html.escape(target.served_name)} · {html.escape(target.access_mode)} · {html.escape(target.runtime_name)} · KV {html.escape(target.cache_dtype)}</div>
<table>
<thead><tr><th>scenario</th><th>input</th><th>output/target</th><th>C</th><th>TTFT ms</th><th>P95 TTFT</th><th>effective prefill tok/s*</th><th>TPOT ms</th><th>decode tok/s</th><th>output tok/s</th><th>spec accept %</th></tr></thead>
<tbody>{''.join(body)}</tbody>
</table>
<div class="note">* Client-observed effective prefill = actual input tokens / TTFT. It includes HTTP and scheduling overhead; it is not a kernel-only counter.</div>
</body>
</html>
"""


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--profile")
    result.add_argument(
        "--component", choices=("primary", "worker-pool")
    )
    result.add_argument("--litellm-model")
    access = result.add_mutually_exclusive_group()
    access.add_argument(
        "--direct",
        action="store_true",
        help=(
            "send the same workload dimensions directly through vLLM's "
            "OpenAI Chat API, using the active alias sampling parameters"
        ),
    )
    access.add_argument(
        "--raw",
        action="store_true",
        help=(
            "measure raw vLLM completions with temperature=0 and ignore_eos"
        ),
    )
    result.add_argument("--list-litellm-models", action="store_true")
    result.add_argument(
        "--scenario",
        action="append",
        choices=("all", *SCENARIOS),
        help="repeat to run selected scenarios; default: all",
    )
    result.add_argument("--worker-rank", type=int, default=0)
    result.add_argument("--input-tokens", type=int)
    result.add_argument("--output-tokens", type=int)
    result.add_argument("--requests", type=int)
    result.add_argument("--concurrency", type=int)
    result.add_argument(
        "--seed-base",
        type=int,
        help="base for reproducible prompts; default changes on every run",
    )
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.list_litellm_models:
            gateway_state = _load_json(
                REPO_ROOT / ".runtime" / "litellm" / "service.json"
            )
            endpoint = gateway_state.get("probe_url") or gateway_state.get("url")
            if not isinstance(endpoint, str) or not endpoint:
                raise RuntimeError("LiteLLM state does not contain a usable URL")
            profile_name = gateway_state.get("profile")
            if not isinstance(profile_name, str) or not profile_name:
                raise RuntimeError("LiteLLM state does not identify its active profile")
            profile = _load_json(_profile_path(REPO_ROOT, profile_name))
            configured = set(profile.get("stack", {}).get("litellm_aliases", []))
            exposed = set(
                _models(endpoint.rstrip("/"), _litellm_key(REPO_ROOT))
            )
            for model in sorted(configured & exposed):
                print(model)
            return 0
        if args.litellm_model:
            access_mode = "raw" if args.raw else "direct" if args.direct else "litellm"
            target = resolve_alias_target(
                REPO_ROOT,
                args.litellm_model,
                access_mode=access_mode,
                require_active=not args.dry_run,
                requested_profile=args.profile,
            )
            if access_mode == "direct":
                target = replace(
                    target,
                    request_overrides=_litellm_alias_overrides(
                        REPO_ROOT, args.litellm_model
                    ),
                    gateway_state_path=(
                        REPO_ROOT / ".runtime" / "litellm" / "service.json"
                    ),
                )
            if args.component is not None and args.component != target.component:
                raise RuntimeError(
                    f"alias {args.litellm_model} resolves to {target.component}, "
                    f"not {args.component}"
                )
        else:
            if not args.profile:
                raise RuntimeError("--profile or --litellm-model is required")
            target = resolve_target(
                REPO_ROOT,
                args.profile,
                args.component or "primary",
                require_active=not args.dry_run,
            )
        if target.component != "worker-pool" and args.worker_rank != 0:
            raise RuntimeError("--worker-rank is only valid for worker-pool")
        scenarios = _apply_overrides(
            args, _selected_scenarios(target.component, args.scenario)
        )
        seed_base = (
            args.seed_base
            if args.seed_base is not None
            else time.time_ns() % 1_000_000_000
        )
        scenarios = [
            replace(scenario, seed=seed_base + scenario.seed)
            for scenario in scenarios
        ]
        if not args.dry_run:
            if not target.model_directory.is_dir():
                raise RuntimeError(f"model directory is missing: {target.model_directory}")
            api_key = (
                _litellm_key(REPO_ROOT)
                if target.access_mode == "litellm"
                else None
            )
            models = _models(target.endpoint, api_key)
            if target.served_name not in models:
                raise RuntimeError(
                    f"{target.endpoint} does not serve {target.served_name}; found {models}"
                )
        recipe_root, vllm, rocm_root = _recipe_paths(REPO_ROOT, target.recipe)
        timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
        output_scope = target.component
        if target.alias:
            alias_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", target.alias)
            output_scope = f"{target.access_mode}/{alias_slug}"
        output_dir = (
            args.output_dir
            or REPO_ROOT
            / "logs"
            / "benchmarks"
            / target.requested_profile
            / output_scope
            / timestamp
        ).resolve()
        commands: list[tuple[Scenario, list[str]]] = []
        for scenario in scenarios:
            worker_rank = (
                None
                if (
                    target.component != "worker-pool"
                    or scenario.name == "pool"
                    or target.access_mode == "litellm"
                )
                else args.worker_rank
            )
            validate_scenario(target, scenario, worker_rank)
            commands.append(
                (
                    scenario,
                    build_command(vllm, target, scenario, output_dir, worker_rank),
                )
            )
        if args.dry_run:
            print(
                "Environment: packaged runtime; VLLM_TARGET_DEVICE=cpu; "
                "GPU and external vLLM plugins disabled (HTTP client only)"
            )
            for scenario, command in commands:
                print(f"[{scenario.name}] {shlex.join(command)}")
            return 0
        output_dir.mkdir(parents=True, exist_ok=True)
        environment = benchmark_environment(REPO_ROOT, recipe_root, rocm_root)
        if target.access_mode == "litellm":
            environment["OPENAI_API_KEY"] = _litellm_key(REPO_ROOT)
        summaries = []
        raw_results = []
        for scenario, command in commands:
            raw_path = output_dir / f"{scenario.name}.json"
            if raw_path.exists() and not args.overwrite:
                raise RuntimeError(f"refusing to overwrite {raw_path}")
            print(
                f"\n=== {target.access_mode}/{target.component}: "
                f"{scenario.name} ===",
                flush=True,
            )
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            tool_log_path = output_dir / f"{scenario.name}.stdout.log"
            tool_log_path.write_text(completed.stdout)
            if completed.returncode:
                tail = "\n".join(completed.stdout.splitlines()[-80:])
                if tail:
                    print(tail, file=sys.stderr)
                raise RuntimeError(
                    f"vllm bench serve failed for {scenario.name}: "
                    f"exit {completed.returncode}"
                )
            payload = _load_json(raw_path)
            summary = summarize_result(scenario, payload)
            if summary["failed"] or summary["completed"] != scenario.requests:
                raise RuntimeError(
                    f"scenario {scenario.name} was incomplete: "
                    f"completed={summary['completed']} failed={summary['failed']}"
                )
            summaries.append(summary)
            raw_results.append(
                {
                    "scenario": asdict(scenario),
                    "path": str(raw_path),
                    "sha256": _sha256(raw_path),
                    "tool_log": str(tool_log_path),
                    "tool_log_sha256": _sha256(tool_log_path),
                    "summary": summary,
                }
            )
        manifest_path = REPO_ROOT / "manifest" / f"{target.recipe}.json"
        manifest = _load_json(manifest_path)
        report = {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(),
            "target": {
                **asdict(target),
                "profile_path": str(target.profile_path),
                "state_path": str(target.state_path),
                "model_directory": str(target.model_directory),
                "gateway_state_path": (
                    str(target.gateway_state_path)
                    if target.gateway_state_path is not None
                    else None
                ),
            },
            "backend": {
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "vllm_version": manifest["sources"]["vllm"]["version"],
                "vllm_commit": manifest["sources"]["vllm"]["commit"],
                "rocm_version": manifest["environment"]["rocm_version"],
            },
            "results": raw_results,
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(report, indent=2) + "\n")
        html_path = output_dir / "summary.html"
        html_path.write_text(render_html(target, summaries))
        print_summary(summaries)
        print(f"\nJSON: {summary_path}")
        print(f"HTML: {html_path}")
        return 0
    except (RuntimeError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
