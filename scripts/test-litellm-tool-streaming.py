from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_COMMAND = (
    'grep -n "const mediaMode\\|setAudioOnly\\|RoomControls\\|DemoController\\|'
    '</main>\\|import { useEffect\\|" games/neon-snake/index.html'
)


def default_api_key() -> str:
    api_key = os.environ.get("LITELLM_MASTER_KEY", "")
    if api_key:
        return api_key
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            if line.startswith("LITELLM_MASTER_KEY="):
                return line.removeprefix("LITELLM_MASTER_KEY=")
    raise RuntimeError("LITELLM_MASTER_KEY is missing")


def bash_tool_openai() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    }


def bash_tool_anthropic() -> dict[str, Any]:
    function = bash_tool_openai()["function"]
    return {
        "name": function["name"],
        "description": function["description"],
        "input_schema": function["parameters"],
    }


def request(url: str, api_key: str, payload: dict[str, Any]) -> Any:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if "/messages" in url:
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    return urllib.request.urlopen(req, timeout=300)


def anthropic_stream(
    base_url: str,
    api_key: str,
    model: str,
    command: str,
    temperature: float | None,
) -> list[tuple[str, str]]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"Call Bash exactly once with this command: {command}",
            }
        ],
        "tools": [bash_tool_anthropic()],
        "tool_choice": {"type": "tool", "name": "Bash"},
        "max_tokens": 512,
        "stream": True,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    names: dict[int, str] = {}
    arguments: dict[int, list[str]] = {}
    with request(
        base_url.rstrip("/") + "/v1/messages?beta=true", api_key, payload
    ) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                break
            event = json.loads(data)
            index = event.get("index")
            block = event.get("content_block") or {}
            if (
                event.get("type") == "content_block_start"
                and block.get("type") == "tool_use"
                and isinstance(index, int)
            ):
                names[index] = str(block.get("name", ""))
                arguments.setdefault(index, [])
            delta = event.get("delta") or {}
            if (
                delta.get("type") == "input_json_delta"
                and isinstance(index, int)
            ):
                arguments.setdefault(index, []).append(delta.get("partial_json", ""))
    return [(names[index], "".join(arguments[index])) for index in sorted(names)]


def openai_stream(
    base_url: str,
    api_key: str,
    model: str,
    command: str,
    temperature: float | None,
) -> list[tuple[str, str]]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"Call Bash exactly once with this command: {command}",
            }
        ],
        "tools": [bash_tool_openai()],
        "tool_choice": {"type": "function", "function": {"name": "Bash"}},
        "max_tokens": 512,
        "stream": True,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    names: dict[int, str] = {}
    arguments: dict[int, list[str]] = {}
    with request(
        base_url.rstrip("/") + "/v1/chat/completions", api_key, payload
    ) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                break
            event = json.loads(data)
            choices = event.get("choices") or []
            if not choices:
                continue
            for call in (choices[0].get("delta") or {}).get("tool_calls") or []:
                index = int(call.get("index", 0))
                function = call.get("function") or {}
                if function.get("name"):
                    names[index] = function["name"]
                if function.get("arguments") is not None:
                    arguments.setdefault(index, []).append(function["arguments"])
    return [(names.get(index, ""), "".join(parts)) for index, parts in sorted(arguments.items())]


def validate(calls: list[tuple[str, str]], command: str) -> str | None:
    if len(calls) != 1:
        return f"expected one tool call, received {len(calls)}: {calls!r}"
    name, raw_arguments = calls[0]
    if name != "Bash":
        return f"expected Bash, received {name!r}"
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as error:
        return f"invalid JSON ({error}): {raw_arguments!r}"
    if arguments != {"command": command}:
        return f"unexpected arguments: {arguments!r}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regression test LiteLLM streaming tool-call arguments"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:4000")
    parser.add_argument("--model", default="mimo-v2.6-flash")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--protocol", choices=("anthropic", "openai", "both"), default="both"
    )
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    api_key = args.api_key or default_api_key()
    protocols = (
        ("anthropic", anthropic_stream),
        ("openai", openai_stream),
    )
    failures: list[str] = []
    for protocol, operation in protocols:
        if args.protocol not in (protocol, "both"):
            continue
        valid = 0
        for iteration in range(1, args.iterations + 1):
            error = validate(
                operation(
                    args.base_url,
                    api_key,
                    args.model,
                    DEFAULT_COMMAND,
                    args.temperature,
                ),
                DEFAULT_COMMAND,
            )
            if error is None:
                valid += 1
            else:
                failures.append(f"{protocol} iteration {iteration}: {error}")
        print(
            f"{protocol}: valid={valid}/{args.iterations} "
            f"invalid={args.iterations - valid}/{args.iterations}"
        )
    for failure in failures:
        print(f"FAIL: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
