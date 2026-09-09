#!/usr/bin/env python3
"""Replay immutable NIAH request fixtures against an already-running API."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _load_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _encoded_request(request: dict[str, Any]) -> bytes:
    return json.dumps(request, separators=(",", ":")).encode()


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object from {url}")
    return payload


def _post_json(url: str, body: bytes, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object from {url}")
    return payload


def _content(response: dict[str, Any]) -> tuple[str, str | None, int]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError("response choice is not an object")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("response choice has no message object")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("response message content is not text")
    reasoning = message.get("reasoning_content")
    reasoning_characters = len(reasoning) if isinstance(reasoning, str) else 0
    finish_reason = choice.get("finish_reason")
    return content.strip(), finish_reason, reasoning_characters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--label", required=True)
    parser.add_argument("--timeout", type=float, default=1200.0)
    args = parser.parse_args()

    requests_dir = args.requests_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        parser.error(f"output directory already exists: {output_dir}")
    fixtures = sorted(requests_dir.glob("request-depth-*.json"))
    if not fixtures:
        parser.error(f"no request-depth-*.json fixtures in {requests_dir}")

    try:
        models = _get_json(f"{args.base_url.rstrip('/')}/v1/models", 5.0)
        model_ids = {
            item.get("id")
            for item in models.get("data", [])
            if isinstance(item, dict)
        }
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"API validation failed: {error}", file=sys.stderr)
        return 2

    prepared: list[tuple[Path, dict[str, Any], bytes]] = []
    for path in fixtures:
        try:
            fixture = _load_object(path)
            request = fixture["request"]
            if not isinstance(request, dict):
                raise ValueError("request is not an object")
            body = _encoded_request(request)
            digest = hashlib.sha256(body).hexdigest()
            if digest != fixture.get("request_sha256"):
                raise ValueError(
                    f"request hash differs: expected {fixture.get('request_sha256')}, "
                    f"got {digest}"
                )
            if request.get("model") not in model_ids:
                raise ValueError(
                    f"model {request.get('model')!r} is not served; available={model_ids}"
                )
            prepared.append((path, fixture, body))
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
            print(f"invalid fixture {path}: {error}", file=sys.stderr)
            return 2

    output_dir.mkdir(parents=True)
    results: list[dict[str, Any]] = []
    endpoint = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    for path, fixture, body in prepared:
        depth = float(fixture["depth_requested"])
        expected = str(fixture["secret"])
        started_at = _now()
        started = time.monotonic()
        print(f"depth={depth:.2f} started_at={started_at}", flush=True)
        try:
            response = _post_json(endpoint, body, args.timeout)
            elapsed = time.monotonic() - started
            answer, finish_reason, reasoning_characters = _content(response)
            result = {
                "schema_version": 1,
                "label": args.label,
                "started_at": started_at,
                "finished_at": _now(),
                "elapsed_seconds": elapsed,
                "depth_requested": depth,
                "needle_prompt_depth": fixture["needle_prompt_depth"],
                "request_path": str(path),
                "request_sha256": fixture["request_sha256"],
                "prompt_token_ids_sha256": fixture["prompt_token_ids_sha256"],
                "expected": expected,
                "answer": answer,
                "exact_match": answer == expected,
                "contains_match": expected in answer,
                "finish_reason": finish_reason,
                "reasoning_characters": reasoning_characters,
                "system_fingerprint": response.get("system_fingerprint"),
                "usage": response.get("usage"),
            }
        except (OSError, ValueError, json.JSONDecodeError) as error:
            elapsed = time.monotonic() - started
            result = {
                "schema_version": 1,
                "label": args.label,
                "started_at": started_at,
                "finished_at": _now(),
                "elapsed_seconds": elapsed,
                "depth_requested": depth,
                "request_path": str(path),
                "request_sha256": fixture["request_sha256"],
                "expected": expected,
                "error": f"{type(error).__name__}: {error}",
                "exact_match": False,
                "contains_match": False,
            }
        result_path = output_dir / f"depth-{round(depth * 100):03d}.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        results.append(result)
        print(
            f"depth={depth:.2f} exact_match={result['exact_match']} "
            f"elapsed_seconds={elapsed:.3f}",
            flush=True,
        )
        if "error" in result:
            break

    summary = {
        "schema_version": 1,
        "label": args.label,
        "requests_dir": str(requests_dir),
        "base_url": args.base_url,
        "cases_planned": len(prepared),
        "cases_completed": len(results),
        "exact_matches": sum(bool(result["exact_match"]) for result in results),
        "all_exact": len(results) == len(prepared)
        and all(bool(result["exact_match"]) for result in results),
        "results": [
            {
                key: result.get(key)
                for key in (
                    "depth_requested",
                    "exact_match",
                    "elapsed_seconds",
                    "error",
                )
                if result.get(key) is not None
            }
            for result in results
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["all_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
