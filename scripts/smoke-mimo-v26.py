from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any


def default_api_key() -> str:
    api_key = os.environ.get("LITELLM_MASTER_KEY", "")
    if api_key:
        return api_key

    env_file = Path(__file__).resolve().parents[1] / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            if line.startswith("LITELLM_MASTER_KEY="):
                return line.removeprefix("LITELLM_MASTER_KEY=")

    raise RuntimeError(
        "LITELLM_MASTER_KEY is missing from the environment and repository .env"
    )


def request(base_url: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.load(response)


def stream_request(base_url: str, api_key: str, payload: dict[str, Any]) -> str:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps({**payload, "stream": True}).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    chunks: list[str] = []
    with urllib.request.urlopen(req, timeout=300) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                break
            event = json.loads(data)
            delta = event["choices"][0].get("delta", {})
            if delta.get("content"):
                chunks.append(delta["content"])
    return "".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description="MiMo v2.6 OpenAI API smoke test")
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:4000/v1", help="OpenAI API root"
    )
    parser.add_argument("--model", default="mimo-v2.6-flash")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()
    api_key = args.api_key or default_api_key()

    base = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Reply with exactly: MIMO_OK"}],
        "max_tokens": 32,
    }
    answer = request(args.base_url, api_key, base)
    content = answer["choices"][0]["message"].get("content", "")
    if "MIMO_OK" not in content:
        raise RuntimeError(f"basic generation failed: {content!r}")
    print("basic: ok")

    streamed = stream_request(args.base_url, api_key, base)
    if "MIMO_OK" not in streamed:
        raise RuntimeError(f"streaming failed: {streamed!r}")
    print("streaming: ok")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_temperature",
                "description": "Return the current temperature for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    first = request(
        args.base_url,
        api_key,
        {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": "Use get_temperature for Warsaw, then report it.",
                }
            ],
            "tools": tools,
            "tool_choice": "required",
            "max_tokens": 256,
        },
    )
    assistant = first["choices"][0]["message"]
    calls = assistant.get("tool_calls") or []
    if not calls:
        raise RuntimeError(f"tool parser produced no call: {assistant!r}")
    call = calls[0]
    final = request(
        args.base_url,
        api_key,
        {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": "Use get_temperature for Warsaw, then report it.",
                },
                assistant,
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": '{"city":"Warsaw","temperature_c":21}',
                },
            ],
            "tools": tools,
            "max_tokens": 256,
        },
    )
    final_content = final["choices"][0]["message"].get("content", "")
    if "21" not in final_content:
        raise RuntimeError(f"multi-turn tool result was not used: {final_content!r}")
    print("tool-loop: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
