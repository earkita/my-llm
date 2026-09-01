from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import ConfigurationError, ROOT, load_model, load_runtime
from .service import managed_state


LITERAL_EXPECTED = "serwer działa"
LITERAL_SYSTEM = (
    "Zwróć wyłącznie tekst podany przez użytkownika. "
    "Nie dodawaj żadnych słów, znaków ani formatowania."
)


def get_json(url: str, timeout: float = 30) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError) as exc:
        raise ConfigurationError(f"API request failed: {url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigurationError(f"API returned a non-object: {url}")
    return payload


def post_json(url: str, body: dict[str, Any], timeout: float = 600) -> tuple[dict, float]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise ConfigurationError(
            f"API HTTP {exc.code}: {exc.read().decode(errors='replace')}"
        ) from exc
    except (OSError, urllib.error.URLError) as exc:
        raise ConfigurationError(f"API request failed: {url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigurationError("API returned a non-object")
    return payload, time.perf_counter() - started


def _first_choice(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ConfigurationError("API response contains no choice")
    return choices[0]


def served_context_length(served: dict[str, Any]) -> int | None:
    """Return the advertised OpenAI context length across supported backends."""
    for key in ("max_model_len", "max_context_length", "context_length"):
        direct = served.get(key)
        if isinstance(direct, int):
            return direct
    meta = served.get("meta")
    nested = meta.get("n_ctx") if isinstance(meta, dict) else None
    return nested if isinstance(nested, int) else None


def literal_oracle(
    payload: dict[str, Any], expected: str = LITERAL_EXPECTED
) -> tuple[bool, str]:
    choice = _first_choice(payload)
    message = choice.get("message")
    actual = message.get("content") if isinstance(message, dict) else None
    finish = choice.get("finish_reason")
    passed = actual == expected and finish == "stop"
    return passed, f"content={actual!r} finish_reason={finish!r}"


def test_api(
    *,
    url: str,
    model_name: str,
    runtime_name: str,
    output: Path | None = None,
    timeout: float = 600,
) -> dict[str, Any]:
    url = url.rstrip("/")
    model = load_model(model_name)
    runtime = load_runtime(runtime_name)
    state = managed_state()
    managed_ok = (
        state.get("url") == url
        and state.get("model") == model["name"]
        and state.get("runtime") == runtime["name"]
        and state.get("runtime_profile_sha256") == runtime["_sha256"]
    )
    started = datetime.now().astimezone()
    health_ok = False
    try:
        with urllib.request.urlopen(url + "/health", timeout=10) as response:
            health_ok = response.status == 200
    except (OSError, urllib.error.URLError):
        health_ok = False
    models = get_json(url + "/v1/models", timeout=30)
    rows = models.get("data")
    served = next(
        (row for row in rows or [] if isinstance(row, dict) and row.get("id") == model["served_name"]),
        None,
    )
    model_ok = isinstance(served, dict)
    context_ok = (
        model_ok
        and served_context_length(served) == runtime["limits"]["max_model_len"]
    )

    if model["family"] in {"deepseek_v4", "deepseek_v4_flash", "glm5next"}:
        body = {
            "model": model["served_name"],
            "messages": [
                {"role": "system", "content": LITERAL_SYSTEM},
                {"role": "user", "content": LITERAL_EXPECTED},
            ],
            "chat_template_kwargs": {"thinking": False},
            "temperature": 0,
            "max_tokens": 32,
        }
        if model["family"] == "glm5next":
            body["reasoning_effort"] = "low"
        response, elapsed = post_json(url + "/v1/chat/completions", body, timeout)
        generation_ok, detail = literal_oracle(response)
        check_name = "literal_chat"
    else:
        body = {
            "model": model["served_name"],
            "prompt": "Reply with one short word.",
            "temperature": 0,
            "max_tokens": 8,
        }
        response, elapsed = post_json(url + "/v1/completions", body, timeout)
        choice = _first_choice(response)
        generation_ok = isinstance(choice.get("text"), str) and bool(choice["text"])
        detail = f"text={choice.get('text')!r}"
        check_name = "generic_completion"
    usage = response.get("usage")
    usage_ok = (
        isinstance(usage, dict)
        and isinstance(usage.get("prompt_tokens"), int)
        and usage["prompt_tokens"] > 0
        and isinstance(usage.get("completion_tokens"), int)
        and usage["completion_tokens"] > 0
        and usage.get("total_tokens") == usage["prompt_tokens"] + usage["completion_tokens"]
    )
    checks = {
        "managed_identity": managed_ok,
        "health": health_ok,
        "served_model": model_ok,
        "max_model_len": context_ok,
        check_name: generation_ok,
        "exact_usage": usage_ok,
    }
    payload = {
        "schema_version": 1,
        "started_at": started.isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "url": url,
        "model": model["name"],
        "runtime": runtime["name"],
        "checks": checks,
        "passed": all(checks.values()),
        "elapsed_seconds": elapsed,
        "detail": detail,
        "usage": usage,
        "response": response,
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(output)
    print(json.dumps({"passed": payload["passed"], "checks": checks, "detail": detail}, indent=2))
    if not payload["passed"]:
        raise ConfigurationError("API gate failed")
    return payload
