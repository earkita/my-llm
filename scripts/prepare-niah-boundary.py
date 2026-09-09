#!/usr/bin/env python3
"""Derive an exact-size chat NIAH fixture using a running tokenizer API."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


NEEDLE_MARKER = "CRITICAL NEEDLE RECORD."


def _load_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, separators=(",", ":")).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise ValueError(f"expected a JSON object from {url}")
    return result


def _tokenize(
    base_url: str,
    model: str,
    payload: dict[str, Any],
    timeout: float,
) -> list[int]:
    response = _post_json(
        f"{base_url.rstrip('/')}/tokenize",
        {"model": model, **payload},
        timeout,
    )
    token_ids = response.get("tokens")
    if not isinstance(token_ids, list) or not all(
        isinstance(token_id, int) for token_id in token_ids
    ):
        raise ValueError("/tokenize did not return integer token ids")
    return token_ids


def _detokenize(
    base_url: str,
    model: str,
    token_ids: list[int],
    timeout: float,
) -> str:
    response = _post_json(
        f"{base_url.rstrip('/')}/detokenize",
        {"model": model, "tokens": token_ids},
        timeout,
    )
    prompt = response.get("prompt")
    if not isinstance(prompt, str):
        raise ValueError("/detokenize did not return text")
    return prompt


def _token_ids_sha256(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for start in range(0, len(token_ids), 65536):
        chunk = token_ids[start : start + 65536]
        digest.update(struct.pack(f"<{len(chunk)}I", *chunk))
    return digest.hexdigest()


def _request_sha256(request: dict[str, Any]) -> str:
    encoded = json.dumps(request, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--target-prompt-tokens", required=True, type=int)
    parser.add_argument("--depth", type=float, default=0.95)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--preserve-start-tokens", type=int, default=64)
    parser.add_argument("--preserve-after-needle-tokens", type=int, default=64)
    parser.add_argument("--tail-guard-tokens", type=int, default=128)
    args = parser.parse_args()

    source_path = args.source.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        parser.error(f"output already exists: {output_path}")
    if args.target_prompt_tokens < 1:
        parser.error("--target-prompt-tokens must be positive")
    if not 0.0 < args.depth < 1.0:
        parser.error("--depth must be between zero and one")
    if min(
        args.preserve_start_tokens,
        args.preserve_after_needle_tokens,
        args.tail_guard_tokens,
    ) < 1:
        parser.error("token preservation values must be positive")

    try:
        source = _load_object(source_path)
        request = source["request"]
        if not isinstance(request, dict):
            raise ValueError("source request is not an object")
        model = request["model"]
        messages = request["messages"]
        if not isinstance(model, str) or not model:
            raise ValueError("source request has no model")
        if not isinstance(messages, list):
            raise ValueError("source request has no messages list")
        user_indexes = [
            index
            for index, message in enumerate(messages)
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        if len(user_indexes) != 1:
            raise ValueError("source must contain exactly one user message")
        user_index = user_indexes[0]
        user_text = messages[user_index].get("content")
        if not isinstance(user_text, str):
            raise ValueError("source user message is not text")
        if user_text.count(NEEDLE_MARKER) != 1:
            raise ValueError("source must contain exactly one needle marker")
        secret = source["secret"]
        if not isinstance(secret, str) or user_text.count(secret) != 1:
            raise ValueError("source must contain its secret exactly once")

        full_ids = _tokenize(
            args.base_url,
            model,
            {"messages": messages, "add_generation_prompt": True},
            args.timeout,
        )
        expected_source_hash = source.get("prompt_token_ids_sha256")
        source_hash = _token_ids_sha256(full_ids)
        if source_hash != expected_source_hash:
            raise ValueError(
                f"source token hash differs: expected {expected_source_hash}, "
                f"got {source_hash}"
            )
        if args.target_prompt_tokens >= len(full_ids):
            raise ValueError("target must be shorter than the source prompt")

        user_ids = _tokenize(
            args.base_url,
            model,
            {"prompt": user_text, "add_special_tokens": False},
            args.timeout,
        )
        marker_character_index = user_text.index(NEEDLE_MARKER)
        user_needle_index = len(
            _tokenize(
                args.base_url,
                model,
                {
                    "prompt": user_text[:marker_character_index],
                    "add_special_tokens": False,
                },
                args.timeout,
            )
        )
        source_needle_index = int(source["needle_prompt_token_index"])
        prompt_prefix_tokens = source_needle_index - user_needle_index
        desired_needle_index = round(args.target_prompt_tokens * args.depth)
        delete_total = len(full_ids) - args.target_prompt_tokens
        delete_before = source_needle_index - desired_needle_index
        delete_after = delete_total - delete_before
        if min(prompt_prefix_tokens, delete_before, delete_after) < 0:
            raise ValueError("source cannot satisfy the requested target and depth")

        first_cut_end = args.preserve_start_tokens + delete_before
        second_cut_start = user_needle_index + args.preserve_after_needle_tokens
        second_cut_end = second_cut_start + delete_after
        if first_cut_end >= user_needle_index:
            raise ValueError("first cut would overlap the needle")
        if second_cut_end >= len(user_ids) - args.tail_guard_tokens:
            raise ValueError("second cut would overlap the guarded prompt tail")
        new_user_ids = (
            user_ids[: args.preserve_start_tokens]
            + user_ids[first_cut_end:second_cut_start]
            + user_ids[second_cut_end:]
        )
        new_user_text = _detokenize(
            args.base_url, model, new_user_ids, args.timeout
        )
        if new_user_text.count(NEEDLE_MARKER) != 1:
            raise ValueError("derived prompt did not preserve exactly one needle")
        if new_user_text.count(secret) != 1:
            raise ValueError("derived prompt did not preserve exactly one secret")

        new_messages = copy.deepcopy(messages)
        new_messages[user_index]["content"] = new_user_text
        new_full_ids = _tokenize(
            args.base_url,
            model,
            {"messages": new_messages, "add_generation_prompt": True},
            args.timeout,
        )
        if len(new_full_ids) != args.target_prompt_tokens:
            raise ValueError(
                f"derived prompt has {len(new_full_ids)} tokens, expected "
                f"{args.target_prompt_tokens}"
            )
        new_marker_character_index = new_user_text.index(NEEDLE_MARKER)
        new_user_needle_index = len(
            _tokenize(
                args.base_url,
                model,
                {
                    "prompt": new_user_text[:new_marker_character_index],
                    "add_special_tokens": False,
                },
                args.timeout,
            )
        )
        new_needle_index = new_user_needle_index + prompt_prefix_tokens
        if new_needle_index != desired_needle_index:
            raise ValueError(
                f"derived needle index is {new_needle_index}, expected "
                f"{desired_needle_index}"
            )
        preserved_before = min(32, new_needle_index, source_needle_index)
        # The cut operates on user-message token ids while this comparison
        # uses the fully rendered chat prompt. Token-boundary normalization at
        # the later filler seam can change a few rendered tokens before the
        # nominal cut offset. Forty-eight tokens cover the complete needle
        # record and its immediate context without reaching that seam.
        preserved_after = min(48, args.preserve_after_needle_tokens)
        if (
            new_full_ids[
                new_needle_index - preserved_before : new_needle_index
                + preserved_after
            ]
            != full_ids[
                source_needle_index - preserved_before : source_needle_index
                + preserved_after
            ]
        ):
            raise ValueError("the needle token window changed during derivation")

        derived_request = copy.deepcopy(request)
        derived_request["messages"] = new_messages
        result = {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(),
            "target_prompt_tokens": len(new_full_ids),
            "depth_requested": args.depth,
            "secret": secret,
            "needle_prompt_token_index": new_needle_index,
            "needle_prompt_depth": new_needle_index / len(new_full_ids),
            "prompt_token_ids_sha256": _token_ids_sha256(new_full_ids),
            "request_sha256": _request_sha256(derived_request),
            "source_request_path": str(source_path),
            "source_prompt_token_ids_sha256": source_hash,
            "splice": {
                "deleted_tokens_total": delete_total,
                "deleted_before_needle": delete_before,
                "deleted_after_needle": delete_after,
            },
            "request": derived_request,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "output": str(output_path),
                "target_prompt_tokens": result["target_prompt_tokens"],
                "needle_prompt_token_index": result["needle_prompt_token_index"],
                "needle_prompt_depth": result["needle_prompt_depth"],
                "prompt_token_ids_sha256": result["prompt_token_ids_sha256"],
                "request_sha256": result["request_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
