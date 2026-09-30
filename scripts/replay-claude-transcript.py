#!/usr/bin/env python3
"""Replay a bounded suffix of a Claude Code JSONL transcript through vLLM.

This is a diagnostic client only. It never executes returned tool calls and
does not manage the runtime. The selected suffix starts at a user message so a
tool result is never detached from its originating assistant turn.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any


def _post_json(
    url: str,
    body: dict[str, Any],
    timeout: float,
    api_key: str | None = None,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(body, separators=(",", ":")).encode(),
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"{url} returned a non-object")
    return payload


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    parts: list[str] = []
    for item in value:
        if isinstance(item, dict):
            rendered = item.get("text", item.get("content", item))
            parts.append(_text(rendered))
        else:
            parts.append(str(item))
    return "\n".join(parts)


def _openai_tools(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    snapshots = [
        row["attachment"]
        for row in rows
        if row.get("attachment", {}).get("type") == "prompt_snapshot"
        and row["attachment"].get("tools")
    ]
    if not snapshots:
        raise ValueError("transcript has no prompt snapshot with tools")
    result: list[dict[str, Any]] = []
    for tool in snapshots[-1]["tools"]:
        name = tool.get("name")
        schema = tool.get("input_schema", tool.get("schema", {}).get("input_schema"))
        if not isinstance(name, str) or not isinstance(schema, dict):
            continue
        result.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(tool.get("description", "")),
                    "parameters": schema,
                },
            }
        )
    return result


def _system_message(rows: list[dict[str, Any]]) -> dict[str, str]:
    snapshots = [
        row["attachment"]
        for row in rows
        if row.get("attachment", {}).get("type") == "prompt_snapshot"
        and row["attachment"].get("systemPrompt")
    ]
    if not snapshots:
        return {"role": "system", "content": "You are a coding agent."}
    return {
        "role": "system",
        "content": "\n\n".join(str(item) for item in snapshots[-1]["systemPrompt"]),
    }


def _messages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    pending_id: str | None = None
    pending_blocks: list[dict[str, Any]] = []

    def flush_assistant() -> None:
        nonlocal pending_id, pending_blocks
        if not pending_blocks:
            return
        content = "\n".join(
            str(block.get("text", ""))
            for block in pending_blocks
            if block.get("type") == "text"
        ).strip()
        reasoning = "\n".join(
            str(block.get("thinking", ""))
            for block in pending_blocks
            if block.get("type") == "thinking"
        ).strip()
        tool_calls = [
            {
                "id": block["id"],
                "type": "function",
                "function": {
                    "name": block["name"],
                    "arguments": json.dumps(
                        block.get("input", {}), separators=(",", ":")
                    ),
                },
            }
            for block in pending_blocks
            if block.get("type") == "tool_use"
        ]
        message: dict[str, Any] = {"role": "assistant", "content": content or None}
        if reasoning:
            message["reasoning_content"] = reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls
        messages.append(message)
        pending_id = None
        pending_blocks = []

    for row in rows:
        row_type = row.get("type")
        if row_type == "assistant":
            message = row.get("message", {})
            message_id = message.get("id")
            if pending_id is not None and message_id != pending_id:
                flush_assistant()
            pending_id = message_id
            pending_blocks.extend(message.get("content") or [])
            continue

        flush_assistant()
        if row_type == "user" and not row.get("isMeta"):
            content = row.get("message", {}).get("content")
            if isinstance(content, list) and any(
                isinstance(item, dict) and item.get("type") == "tool_result"
                for item in content
            ):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": item.get("tool_use_id", "unknown"),
                                "content": _text(item.get("content", "")),
                            }
                        )
            elif content:
                messages.append({"role": "user", "content": _text(content)})
        elif row_type == "attachment":
            rendered = row.get("rendered") or []
            content = "\n".join(
                str(item.get("content", ""))
                for item in rendered
                if isinstance(item, dict)
            ).strip()
            if content:
                messages.append({"role": "user", "content": content})

    flush_assistant()
    return messages


def _prompt_tokens(
    base_url: str,
    model: str,
    system: dict[str, str],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    timeout: float,
) -> int:
    payload = _post_json(
        base_url.rstrip("/") + "/tokenize",
        {
            "model": model,
            "messages": [system, *messages],
            "tools": tools,
            "add_generation_prompt": True,
        },
        timeout,
    )
    tokens = payload.get("tokens")
    if not isinstance(tokens, list):
        raise ValueError("/tokenize did not return tokens")
    return len(tokens)


def _bounded_suffix(
    base_url: str,
    model: str,
    system: dict[str, str],
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    target_tokens: int,
    timeout: float,
) -> tuple[list[dict[str, Any]], int, int]:
    starts = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user"
    ]
    if not starts:
        raise ValueError("transcript contains no replayable user message")
    low = 0
    high = len(starts) - 1
    best = starts[-1]
    best_count = _prompt_tokens(
        base_url, model, system, messages[best:], tools, timeout
    )
    while low <= high:
        middle = (low + high) // 2
        start = starts[middle]
        count = _prompt_tokens(
            base_url, model, system, messages[start:], tools, timeout
        )
        if count <= target_tokens:
            best = start
            best_count = count
            high = middle - 1
        else:
            low = middle + 1
    return messages[best:], best_count, best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--target-prompt-tokens", type=int, default=88_000)
    parser.add_argument(
        "--suffix-start",
        type=int,
        help="use an explicit converted-message index instead of /tokenize sizing",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--label", required=True)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--api-key-env")
    parser.add_argument("--skip-tokenize", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for line in args.transcript.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        timestamp = row.get("timestamp", "")
        if timestamp and timestamp > args.cutoff:
            break
        rows.append(row)

    system = _system_message(rows)
    tools = _openai_tools(rows)
    converted_messages = _messages(rows)
    if args.suffix_start is None:
        messages, prompt_tokens, suffix_start = _bounded_suffix(
            args.url,
            args.model,
            system,
            converted_messages,
            tools,
            args.target_prompt_tokens,
            args.timeout,
        )
    else:
        suffix_start = args.suffix_start
        if not 0 <= suffix_start < len(converted_messages):
            raise ValueError("--suffix-start is outside the converted transcript")
        if converted_messages[suffix_start].get("role") != "user":
            raise ValueError("--suffix-start must select a user message")
        messages = converted_messages[suffix_start:]
        prompt_tokens = (
            -1
            if args.skip_tokenize
            else _prompt_tokens(
                args.url, args.model, system, messages, tools, args.timeout
            )
        )
    body = {
        "model": args.model,
        "messages": [system, *messages],
        "tools": tools,
        "tool_choice": "auto",
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    if args.repetition_penalty is not None:
        body["repetition_penalty"] = args.repetition_penalty
    if args.disable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    started = time.perf_counter()
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    output = _post_json(
        args.url.rstrip("/") + "/v1/chat/completions",
        body,
        args.timeout,
        api_key,
    )
    elapsed = time.perf_counter() - started
    choice = output["choices"][0]
    message = choice["message"]
    reasoning = message.get("reasoning_content", message.get("reasoning", "")) or ""
    content = message.get("content") or ""
    calls = message.get("tool_calls") or []
    print(
        json.dumps(
            {
                "label": args.label,
                "suffix_start": suffix_start,
                "history_messages": len(messages),
                "tokenized_prompt_tokens": prompt_tokens,
                "elapsed_seconds": round(elapsed, 3),
                "usage": output.get("usage"),
                "finish_reason": choice.get("finish_reason"),
                "message_keys": sorted(message),
                "reasoning_characters": len(reasoning),
                "content_characters": len(content),
                "tool_calls": [
                    {
                        "name": call.get("function", {}).get("name"),
                        "arguments": call.get("function", {}).get("arguments"),
                    }
                    for call in calls
                ],
                "reasoning_start": reasoning[:500],
                "reasoning_end": reasoning[-800:],
                "content_end": content[-1000:],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
