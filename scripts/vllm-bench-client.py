#!/usr/bin/env python3
"""Run vLLM's benchmark CLI with correct combined SSE usage handling."""

from __future__ import annotations

import json
from typing import Any


def split_combined_usage_chunks(messages: list[str]) -> list[str]:
    """Split LiteLLM's final choices+usage event for vLLM's benchmark client.

    The current vLLM OpenAI chat benchmark handles ``choices`` and ``usage``
    with an if/elif pair. LiteLLM legally emits both in one final SSE event,
    causing vLLM to discard the exact usage counters and re-tokenize only the
    visible response. Presenting the same event as two messages keeps the
    upstream benchmark implementation and its metrics intact.
    """

    result: list[str] = []
    for message in messages:
        prefix = "data: " if message.startswith("data: ") else ""
        raw = message[len(prefix) :]
        try:
            payload: Any = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            result.append(message)
            continue
        if not (
            isinstance(payload, dict)
            and payload.get("choices")
            and isinstance(payload.get("usage"), dict)
        ):
            result.append(message)
            continue

        choice_payload = dict(payload)
        choice_payload.pop("usage", None)
        usage_payload = dict(payload)
        usage_payload["choices"] = []
        result.extend(
            [
                prefix
                + json.dumps(choice_payload, ensure_ascii=False, separators=(",", ":")),
                prefix
                + json.dumps(usage_payload, ensure_ascii=False, separators=(",", ":")),
            ]
        )
    return result


def main() -> None:
    from vllm.benchmarks.lib import endpoint_request_func

    original_handler = endpoint_request_func.StreamedResponseHandler

    class CombinedUsageHandler(original_handler):
        def add_chunk(self, chunk: bytes) -> list[str]:
            return split_combined_usage_chunks(super().add_chunk(chunk))

    endpoint_request_func.StreamedResponseHandler = CombinedUsageHandler

    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    main()
