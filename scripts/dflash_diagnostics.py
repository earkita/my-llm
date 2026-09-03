#!/usr/bin/env python3
"""Capture and compare bounded, deterministic DFlash API diagnostics.

This client deliberately does not manage the runtime.  Run ``prepare`` once
against either mode to freeze an exact token-id prompt, then run ``capture``
after explicitly starting each target/DFlash configuration.  ``compare``
produces both machine-readable evidence and a small Markdown summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
DEFAULT_PROMPT = (
    "A deterministic decoder check: list the first eight prime numbers in "
    "ascending order, separated only by commas."
)
MAX_PROMPT_TOKENS = 4096
MAX_PROMPT_CHARACTERS = 65_536
MAX_OUTPUT_TOKENS = 256
MAX_LOGPROBS = 20
LABEL = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
PROM_METRICS = {
    "num_drafts": "vllm:spec_decode_num_drafts_total",
    "num_draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "num_accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
}
PER_POSITION_METRIC = "vllm:spec_decode_num_accepted_tokens_per_pos_total"


class DiagnosticError(RuntimeError):
    """A deterministic diagnostic invariant was not satisfied."""


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _headers(api_key_env: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get(api_key_env)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _get(url: str, *, timeout: float, api_key_env: str) -> tuple[bytes, str]:
    request = urllib.request.Request(url, headers=_headers(api_key_env))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read(), response.headers.get("Content-Type", "")
    except (OSError, urllib.error.URLError) as exc:
        raise DiagnosticError(f"GET failed for {url}: {exc}") from exc


def _get_json(url: str, *, timeout: float, api_key_env: str) -> dict[str, Any]:
    raw, _content_type = _get(url, timeout=timeout, api_key_env=api_key_env)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DiagnosticError(f"GET returned invalid JSON for {url}") from exc
    if not isinstance(payload, dict):
        raise DiagnosticError(f"GET returned a non-object for {url}")
    return payload


def _post_json(
    url: str,
    body: dict[str, Any],
    *,
    timeout: float,
    api_key_env: str,
) -> tuple[dict[str, Any], float]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, separators=(",", ":")).encode(),
        headers=_headers(api_key_env),
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode(errors="replace")
        raise DiagnosticError(f"POST {url} returned HTTP {exc.code}: {detail}") from exc
    except (OSError, urllib.error.URLError) as exc:
        raise DiagnosticError(f"POST failed for {url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise DiagnosticError(f"POST returned a non-object for {url}")
    return payload, time.perf_counter() - started


def parse_prometheus(text: str) -> dict[str, Any]:
    """Return summed speculative counters from Prometheus exposition text."""
    totals = {name: 0.0 for name in PROM_METRICS}
    per_position: dict[str, float] = {}
    found = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        metric_and_labels, raw_value = parts
        metric = metric_and_labels.split("{", 1)[0]
        try:
            value = float(raw_value)
        except ValueError:
            continue
        for short_name, expected in PROM_METRICS.items():
            if metric == expected:
                totals[short_name] += value
                found = True
                break
        if metric == PER_POSITION_METRIC:
            match = re.search(r'(?:^|[,{])position="([^"\\]+)"', metric_and_labels)
            if match:
                position = match.group(1)
                per_position[position] = per_position.get(position, 0.0) + value
                found = True
    return {
        "available": found,
        **{name: int(value) for name, value in totals.items()},
        "accepted_tokens_per_position": {
            key: int(value)
            for key, value in sorted(
                per_position.items(), key=lambda item: int(item[0])
            )
        },
    }


def _metrics_snapshot(
    base_url: str, *, timeout: float, api_key_env: str
) -> dict[str, Any]:
    try:
        raw, _content_type = _get(
            base_url.rstrip("/") + "/metrics",
            timeout=timeout,
            api_key_env=api_key_env,
        )
    except DiagnosticError as exc:
        return {"available": False, "error": str(exc)}
    return parse_prometheus(raw.decode(errors="replace"))


def metric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    if not before.get("available") or not after.get("available"):
        return {"available": False}
    result: dict[str, Any] = {"available": True}
    for name in PROM_METRICS:
        result[name] = int(after.get(name, 0)) - int(before.get(name, 0))
    positions = set(before.get("accepted_tokens_per_position", {})) | set(
        after.get("accepted_tokens_per_position", {})
    )
    result["accepted_tokens_per_position"] = {
        position: int(after.get("accepted_tokens_per_position", {}).get(position, 0))
        - int(before.get("accepted_tokens_per_position", {}).get(position, 0))
        for position in sorted(positions, key=int)
    }
    drafts = result["num_drafts"]
    draft_tokens = result["num_draft_tokens"]
    accepted = result["num_accepted_tokens"]
    result["draft_acceptance_rate"] = (
        accepted / draft_tokens if draft_tokens > 0 else None
    )
    result["mean_acceptance_length"] = (
        1.0 + accepted / drafts if drafts > 0 else None
    )
    return result


def _validate_prompt_ids(token_ids: Any) -> list[int]:
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or not all(isinstance(value, int) and value >= 0 for value in token_ids)
    ):
        raise DiagnosticError("prompt token IDs must be a non-empty integer list")
    if len(token_ids) > MAX_PROMPT_TOKENS:
        raise DiagnosticError(
            f"prompt has {len(token_ids)} tokens; cap is {MAX_PROMPT_TOKENS}"
        )
    return token_ids


def prepare_case(args: argparse.Namespace) -> dict[str, Any]:
    prompt = args.prompt
    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding="utf-8")
    if len(prompt) > MAX_PROMPT_CHARACTERS:
        raise DiagnosticError(
            f"source prompt has {len(prompt)} characters; cap is "
            f"{MAX_PROMPT_CHARACTERS}"
        )
    body = {
        "model": args.model,
        "prompt": prompt,
        "add_special_tokens": True,
    }
    response, elapsed = _post_json(
        args.url.rstrip("/") + "/tokenize",
        body,
        timeout=args.timeout,
        api_key_env=args.api_key_env,
    )
    token_ids = _validate_prompt_ids(response.get("tokens"))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "dflash_case",
        "created_at": datetime.now().astimezone().isoformat(),
        "model": args.model,
        "source_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_token_ids": token_ids,
        "prompt_token_ids_sha256": _sha256_json(token_ids),
        "tokenize_elapsed_seconds": elapsed,
    }
    _atomic_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "prompt_tokens": len(token_ids)}))
    return payload


def _load_case(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"cannot read diagnostic case {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("artifact_type") != "dflash_case":
        raise DiagnosticError(f"not a DFlash diagnostic case: {path}")
    payload["prompt_token_ids"] = _validate_prompt_ids(payload.get("prompt_token_ids"))
    return payload


def _first_choice(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise DiagnosticError("completion response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise DiagnosticError("completion choice is not an object")
    return choice


def capture(args: argparse.Namespace) -> dict[str, Any]:
    if not LABEL.fullmatch(args.label):
        raise DiagnosticError("label must contain only lowercase ASCII, digits, ._- ")
    if not 1 <= args.max_tokens <= MAX_OUTPUT_TOKENS:
        raise DiagnosticError(f"max_tokens must be within 1..{MAX_OUTPUT_TOKENS}")
    if not 0 <= args.logprobs <= MAX_LOGPROBS:
        raise DiagnosticError(f"logprobs must be within 0..{MAX_LOGPROBS}")

    case = _load_case(args.case)
    model = args.model or case["model"]
    base_url = args.url.rstrip("/")
    models = _get_json(
        base_url + "/v1/models", timeout=args.timeout, api_key_env=args.api_key_env
    )
    served_ids = [
        row.get("id")
        for row in models.get("data", [])
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    ]
    if model not in served_ids:
        raise DiagnosticError(f"served model list does not contain {model!r}")

    request_id = f"dflash-diag-{args.label}-{case['prompt_token_ids_sha256'][:12]}"
    request_body: dict[str, Any] = {
        "model": model,
        "prompt": case["prompt_token_ids"],
        "add_special_tokens": False,
        "temperature": 0,
        "top_p": 1,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "ignore_eos": args.ignore_eos,
        "n": 1,
        "request_id": request_id,
        "return_token_ids": True,
    }
    if args.logprobs:
        request_body["logprobs"] = args.logprobs
        request_body["return_tokens_as_token_ids"] = True

    metrics_before = _metrics_snapshot(
        base_url, timeout=min(args.timeout, 30), api_key_env=args.api_key_env
    )
    response, elapsed = _post_json(
        base_url + "/v1/completions",
        request_body,
        timeout=args.timeout,
        api_key_env=args.api_key_env,
    )
    metrics_after = _metrics_snapshot(
        base_url, timeout=min(args.timeout, 30), api_key_env=args.api_key_env
    )
    choice = _first_choice(response)
    output_token_ids = choice.get("token_ids")
    if not isinstance(output_token_ids, list) or not all(
        isinstance(value, int) for value in output_token_ids
    ):
        raise DiagnosticError(
            "server omitted exact output token IDs; this vLLM build must support "
            "return_token_ids"
        )
    text = choice.get("text")
    if not isinstance(text, str):
        raise DiagnosticError("server omitted completion text")
    returned_prompt_ids = choice.get("prompt_token_ids")
    if returned_prompt_ids != case["prompt_token_ids"]:
        raise DiagnosticError(
            "server returned prompt token IDs different from the frozen case"
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "dflash_capture",
        "captured_at": datetime.now().astimezone().isoformat(),
        "label": args.label,
        "url": base_url,
        "case_sha256": _sha256_json(
            {
                "model": case["model"],
                "prompt_token_ids": case["prompt_token_ids"],
            }
        ),
        "request": request_body,
        "request_sha256": _sha256_json(request_body),
        "response": response,
        "result": {
            "prompt_token_ids": returned_prompt_ids,
            "output_token_ids": output_token_ids,
            "output_token_ids_sha256": _sha256_json(output_token_ids),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "finish_reason": choice.get("finish_reason"),
            "usage": response.get("usage"),
            "elapsed_seconds": elapsed,
            "speculative_decoding": (
                response.get("metrics", {}).get("speculative_decoding")
                if isinstance(response.get("metrics"), dict)
                else None
            ),
        },
        "prometheus": {
            "before": metrics_before,
            "after": metrics_after,
            "delta": metric_delta(metrics_before, metrics_after),
            "warning": (
                "Prometheus deltas are process-wide; use per-request response "
                "metrics when other traffic is present."
            ),
        },
    }
    _atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "label": args.label,
                "output_tokens": len(output_token_ids),
                "output_token_ids_sha256": payload["result"][
                    "output_token_ids_sha256"
                ],
                "speculative_decoding": payload["result"]["speculative_decoding"],
            },
            indent=2,
        )
    )
    return payload


def _load_capture(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"cannot read capture {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("artifact_type") != "dflash_capture":
        raise DiagnosticError(f"not a DFlash capture: {path}")
    return payload


def _first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _sampled_logprobs(capture_payload: dict[str, Any]) -> list[float | None]:
    choice = _first_choice(capture_payload["response"])
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return []
    values = logprobs.get("token_logprobs")
    if not isinstance(values, list):
        return []
    return [float(value) if isinstance(value, int | float) else None for value in values]


def _max_common_logprob_delta(
    baseline: dict[str, Any], candidate: dict[str, Any], common_tokens: int
) -> float | None:
    left = _sampled_logprobs(baseline)
    right = _sampled_logprobs(candidate)
    deltas = [
        abs(left[index] - right[index])
        for index in range(min(common_tokens, len(left), len(right)))
        if left[index] is not None
        and right[index] is not None
        and math.isfinite(left[index])
        and math.isfinite(right[index])
    ]
    return max(deltas) if deltas else None


def compare_payloads(captures: list[dict[str, Any]]) -> dict[str, Any]:
    if len(captures) < 2:
        raise DiagnosticError("compare requires a baseline and at least one candidate")
    baseline = captures[0]
    baseline_ids = baseline["result"]["output_token_ids"]
    baseline_prompt = baseline["request"]["prompt"]
    rows = []
    for candidate in captures[1:]:
        if candidate["request"]["prompt"] != baseline_prompt:
            raise DiagnosticError(
                f"{candidate['label']!r} uses different prompt token IDs"
            )
        comparable_keys = (
            "temperature",
            "top_p",
            "seed",
            "max_tokens",
            "ignore_eos",
            "n",
        )
        mismatches = [
            key
            for key in comparable_keys
            if candidate["request"].get(key) != baseline["request"].get(key)
        ]
        if mismatches:
            raise DiagnosticError(
                f"{candidate['label']!r} request differs in: {', '.join(mismatches)}"
            )
        candidate_ids = candidate["result"]["output_token_ids"]
        divergence = _first_divergence(baseline_ids, candidate_ids)
        common_tokens = divergence if divergence is not None else len(baseline_ids)
        rows.append(
            {
                "label": candidate["label"],
                "greedy_equivalent": divergence is None,
                "first_divergent_output_token": divergence,
                "baseline_token_id_at_divergence": (
                    baseline_ids[divergence]
                    if divergence is not None and divergence < len(baseline_ids)
                    else None
                ),
                "candidate_token_id_at_divergence": (
                    candidate_ids[divergence]
                    if divergence is not None and divergence < len(candidate_ids)
                    else None
                ),
                "common_prefix_tokens": common_tokens,
                "max_sampled_logprob_delta_on_common_prefix": (
                    _max_common_logprob_delta(baseline, candidate, common_tokens)
                ),
                "output_tokens": len(candidate_ids),
                "output_token_ids_sha256": candidate["result"][
                    "output_token_ids_sha256"
                ],
                "speculative_decoding": candidate["result"].get(
                    "speculative_decoding"
                ),
                "prometheus_delta": candidate.get("prometheus", {}).get("delta"),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "dflash_comparison",
        "generated_at": datetime.now().astimezone().isoformat(),
        "baseline": {
            "label": baseline["label"],
            "output_tokens": len(baseline_ids),
            "output_token_ids_sha256": baseline["result"][
                "output_token_ids_sha256"
            ],
        },
        "prompt_token_ids_sha256": _sha256_json(baseline_prompt),
        "request": {
            key: baseline["request"].get(key)
            for key in (
                "temperature",
                "top_p",
                "seed",
                "max_tokens",
                "ignore_eos",
                "n",
            )
        },
        "comparisons": rows,
        "all_greedy_equivalent": all(row["greedy_equivalent"] for row in rows),
    }


def render_markdown(comparison: dict[str, Any]) -> str:
    baseline = comparison["baseline"]
    lines = [
        "## GLM DFlash deterministic A/B",
        "",
        f"Baseline: `{baseline['label']}` ({baseline['output_tokens']} output tokens).",
        "",
        "| Candidate | Greedy-equivalent | First divergence | Acceptance | Mean length |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparison["comparisons"]:
        metrics = row.get("speculative_decoding") or {}
        rate = metrics.get("draft_acceptance_rate")
        mean = metrics.get("mean_acceptance_length")
        if rate is None:
            rate = (row.get("prometheus_delta") or {}).get("draft_acceptance_rate")
        if mean is None:
            mean = (row.get("prometheus_delta") or {}).get("mean_acceptance_length")
        rate_text = "n/a" if rate is None else f"{100 * float(rate):.2f}%"
        mean_text = "n/a" if mean is None else f"{float(mean):.3f}"
        divergence = row["first_divergent_output_token"]
        lines.append(
            "| `{label}` | {same} | {divergence} | {rate} | {mean} |".format(
                label=row["label"],
                same="yes" if row["greedy_equivalent"] else "no",
                divergence="—" if divergence is None else divergence,
                rate=rate_text,
                mean=mean_text,
            )
        )
    lines.extend(
        [
            "",
            "Greedy equivalence is an exact comparison of returned output token IDs; "
            "it is the losslessness gate for speculative decoding.",
            "",
        ]
    )
    return "\n".join(lines)


def compare(args: argparse.Namespace) -> dict[str, Any]:
    captures = [_load_capture(path) for path in args.captures]
    payload = compare_payloads(captures)
    _atomic_json(args.output, payload)
    markdown = render_markdown(payload)
    if args.summary_output:
        _atomic_text(args.summary_output, markdown)
    print(markdown, end="")
    return payload


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Deterministic, bounded GLM DFlash A/B diagnostics"
    )
    commands = root.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="freeze a prompt as exact token IDs")
    prepare.add_argument("--url", default="http://127.0.0.1:8000")
    prepare.add_argument("--model", required=True)
    prompt_source = prepare.add_mutually_exclusive_group()
    prompt_source.add_argument("--prompt", default=DEFAULT_PROMPT)
    prompt_source.add_argument("--prompt-file", type=Path)
    prepare.add_argument("--timeout", type=float, default=120)
    prepare.add_argument("--api-key-env", default="VLLM_API_KEY")
    prepare.add_argument("--output", type=Path, required=True)

    capture_parser = commands.add_parser(
        "capture", help="capture one explicitly selected runtime mode"
    )
    capture_parser.add_argument("--case", type=Path, required=True)
    capture_parser.add_argument("--label", required=True)
    capture_parser.add_argument("--url", default="http://127.0.0.1:8000")
    capture_parser.add_argument("--model")
    capture_parser.add_argument("--seed", type=int, default=0)
    capture_parser.add_argument("--max-tokens", type=int, default=64)
    capture_parser.add_argument("--logprobs", type=int, default=5)
    capture_parser.add_argument(
        "--honor-eos",
        action="store_false",
        dest="ignore_eos",
        help="allow EOS before max-tokens (the default captures an exact length)",
    )
    capture_parser.set_defaults(ignore_eos=True)
    capture_parser.add_argument("--timeout", type=float, default=600)
    capture_parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    capture_parser.add_argument("--output", type=Path, required=True)

    compare_parser = commands.add_parser(
        "compare", help="compare baseline followed by one or more DFlash captures"
    )
    compare_parser.add_argument("captures", type=Path, nargs="+")
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.add_argument("--summary-output", type=Path)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare":
            prepare_case(args)
        elif args.command == "capture":
            capture(args)
        else:
            compare(args)
    except DiagnosticError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
