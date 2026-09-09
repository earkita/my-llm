#!/usr/bin/env python3
"""Run a deterministic in-memory PNG smoke test against a vision chat API."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import struct
import sys
import urllib.request
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any


def _chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = binascii.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _red_png(width: int = 32, height: int = 32) -> bytes:
    signature = b"\x89PNG\r\n\x1a\n"
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    scanlines = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    return signature + _chunk(b"IHDR", header) + _chunk(
        b"IDAT", zlib.compress(scanlines)
    ) + _chunk(b"IEND", b"")


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError("vision API response is not an object")
    return result


def _answer(response: dict[str, Any]) -> tuple[str, str | None]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("vision API must return exactly one choice")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise ValueError("vision API choice has no message")
    content = choice["message"].get("content")
    if not isinstance(content, str):
        raise ValueError("vision API response content is not text")
    return content.strip(), choice.get("finish_reason")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="glm-5.3-flash-quark-mxfp4")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    png = _red_png()
    image_url = "data:image/png;base64," + base64.b64encode(png).decode()
    request_body = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Reply with only the dominant color."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 32,
        "reasoning_effort": "low",
    }
    try:
        response = _post_json(
            f"{args.base_url.rstrip('/')}/v1/chat/completions",
            request_body,
            args.timeout,
        )
        answer, finish_reason = _answer(response)
        passed = answer.lower().strip(" .") == "red"
        result = {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(),
            "base_url": args.base_url,
            "model": args.model,
            "image": {
                "format": "png",
                "width": 32,
                "height": 32,
                "rgb": [255, 0, 0],
                "sha256": hashlib.sha256(png).hexdigest(),
            },
            "answer": answer,
            "expected": "red",
            "exact_match_case_insensitive": passed,
            "finish_reason": finish_reason,
            "usage": response.get("usage"),
            "system_fingerprint": response.get("system_fingerprint"),
        }
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 2

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
