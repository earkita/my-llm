#!/usr/bin/env python3
"""Exercise Qwen3.8 worker reasoning, tools, speculative decode, and caching."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import time
from typing import Any
import urllib.error
import urllib.request


METRIC_RE = re.compile(
    r'^vllm:(?P<name>prefix_cache_queries_total|prefix_cache_hits_total|'
    r'spec_decode_num_draft_tokens_total|spec_decode_num_accepted_tokens_total)'
    r'\{(?P<labels>[^}]*)\} (?P<value>[0-9.eE+-]+)$'
)


def _request_json(
    url: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 300,
) -> dict[str, Any]:
    data = json.dumps(payload).encode() if payload is not None else None
    request_headers = {"Content-Type": "application/json"}
    request_headers.update(headers or {})
    request = urllib.request.Request(
        url.rstrip("/") + path,
        data=data,
        headers=request_headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"{path} returned HTTP {exc.code}: {detail}") from exc


def _metrics(url: str, model: str, engine: str) -> dict[str, float]:
    with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=30) as response:
        lines = response.read().decode().splitlines()
    values: dict[str, float] = {}
    for line in lines:
        match = METRIC_RE.match(line)
        if not match:
            continue
        labels = match.group("labels")
        if f'engine="{engine}"' not in labels or f'model_name="{model}"' not in labels:
            continue
        values[match.group("name")] = float(match.group("value"))
    missing = sorted(
        {
            "prefix_cache_queries_total",
            "prefix_cache_hits_total",
            "spec_decode_num_draft_tokens_total",
            "spec_decode_num_accepted_tokens_total",
        }
        - values.keys()
    )
    if missing:
        raise RuntimeError(f"missing metrics for engine {engine}: {', '.join(missing)}")
    return values


def _chat(
    url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    rank: int,
    max_tokens: int,
    thinking: bool,
    temperature: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {
            "enable_thinking": thinking,
            "preserve_thinking": thinking,
        },
    }
    payload.update(extra or {})
    started = time.perf_counter()
    response = _request_json(
        url,
        "/v1/chat/completions",
        payload=payload,
        headers={"X-data-parallel-rank": str(rank)},
    )
    elapsed = time.perf_counter() - started
    choice = response["choices"][0]
    message = choice["message"]
    return {
        "elapsed_seconds": elapsed,
        "finish_reason": choice.get("finish_reason"),
        "message": message,
        "usage": response.get("usage"),
        "system_fingerprint": response.get("system_fingerprint"),
    }


def qualify(args: argparse.Namespace) -> dict[str, Any]:
    url = args.url.rstrip("/")
    started = datetime.now().astimezone()
    before = _metrics(url, args.model, str(args.rank))
    listed = _request_json(url, "/v1/models", timeout=30)

    reasoning = _chat(
        url,
        args.model,
        [
            {
                "role": "user",
                "content": (
                    "Rozwiąż 17x + 29 = 199. Ostatnia linia odpowiedzi ma "
                    "brzmieć dokładnie: X=10"
                ),
            }
        ],
        rank=args.rank,
        max_tokens=1024,
        thinking=True,
        temperature=0.6,
        extra={"top_p": 0.95, "top_k": 20},
    )
    fast = _chat(
        url,
        args.model,
        [{"role": "user", "content": "Odpowiedz dokładnie: FAST_OK"}],
        rank=args.rank,
        max_tokens=64,
        thinking=False,
        temperature=0,
    )
    tool = _chat(
        url,
        args.model,
        [
            {
                "role": "user",
                "content": (
                    "Jaka jest teraz pogoda w Warszawie? Użyj dostępnego "
                    "narzędzia get_weather."
                ),
            }
        ],
        rank=args.rank,
        max_tokens=512,
        thinking=False,
        temperature=0,
        extra={
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Pobiera bieżącą pogodę dla miasta",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_weather"},
            },
        },
    )

    prefix_prompt = (
        "Powtarzalny prefiks testowy cache Qwen trzy osiem. "
        * args.prefix_repetitions
    ) + " Odpowiedz jednym słowem: CACHE_OK"
    prefix_before = _metrics(url, args.model, str(args.rank))
    prefix_first = _chat(
        url,
        args.model,
        [{"role": "user", "content": prefix_prompt}],
        rank=args.rank,
        max_tokens=8,
        thinking=False,
        temperature=0,
    )
    prefix_middle = _metrics(url, args.model, str(args.rank))
    prefix_second = _chat(
        url,
        args.model,
        [{"role": "user", "content": prefix_prompt}],
        rank=args.rank,
        max_tokens=8,
        thinking=False,
        temperature=0,
    )
    after = _metrics(url, args.model, str(args.rank))

    reasoning_message = reasoning["message"]
    fast_message = fast["message"]
    tool_calls = tool["message"].get("tool_calls") or []
    tool_arguments: dict[str, Any] = {}
    if tool_calls:
        tool_arguments = json.loads(tool_calls[0]["function"]["arguments"])
    prefix_hits = (
        after["prefix_cache_hits_total"]
        - prefix_middle["prefix_cache_hits_total"]
    )
    draft_delta = (
        after["spec_decode_num_draft_tokens_total"]
        - before["spec_decode_num_draft_tokens_total"]
    )
    accepted_delta = (
        after["spec_decode_num_accepted_tokens_total"]
        - before["spec_decode_num_accepted_tokens_total"]
    )
    checks = {
        "health_and_model_identity": any(
            item.get("id") == args.model for item in listed.get("data", [])
        ),
        "reasoning_present": bool(reasoning_message.get("reasoning")),
        "reasoning_answer": str(reasoning_message.get("content") or "")
        .strip()
        .endswith("X=10"),
        "fast_answer": str(fast_message.get("content") or "").strip()
        == "FAST_OK",
        "fast_without_reasoning": not fast_message.get("reasoning"),
        "tool_name": bool(tool_calls)
        and tool_calls[0]["function"].get("name") == "get_weather",
        "tool_arguments": tool_arguments.get("city") == "Warszawa",
        "prefix_cache_hit": prefix_hits > 0,
        "prefix_answer": all(
            str(case["message"].get("content") or "").strip() == "CACHE_OK"
            for case in (prefix_first, prefix_second)
        ),
        "speculative_drafted": draft_delta > 0,
        "speculative_accepted": accepted_delta > 0,
    }
    payload = {
        "schema_version": 1,
        "started_at": started.isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "url": url,
        "model": args.model,
        "data_parallel_rank": args.rank,
        "checks": checks,
        "passed": all(checks.values()),
        "reasoning": reasoning,
        "fast": fast,
        "tool_call": tool,
        "prefix_cache": {
            "prompt_tokens": prefix_first.get("usage", {}).get("prompt_tokens"),
            "first_elapsed_seconds": prefix_first["elapsed_seconds"],
            "second_elapsed_seconds": prefix_second["elapsed_seconds"],
            "second_request_hit_tokens": prefix_hits,
            "metrics_before": prefix_before,
            "metrics_after_first": prefix_middle,
            "metrics_after_second": after,
        },
        "speculative_decode": {
            "draft_tokens": draft_delta,
            "accepted_tokens": accepted_delta,
            "acceptance_rate": accepted_delta / draft_delta if draft_delta else 0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "passed": payload["passed"],
                "checks": checks,
                "prefix_cache_hit_tokens": prefix_hits,
                "speculative_decode": payload["speculative_decode"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    if not payload["passed"]:
        raise RuntimeError("worker qualification failed")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8100")
    parser.add_argument("--model", default="qwen3.8-27b-worker")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--prefix-repetitions", type=int, default=1200)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/validation/qwen38-4x27b-features.json"),
    )
    qualify(parser.parse_args())


if __name__ == "__main__":
    main()
