from __future__ import annotations

from typing import Any


GLM53_MODEL_NAMES = frozenset(
    {
        "glm-5.3-flash-high",
        "glm-5.3-flash-quark-mxfp4",
    }
)
QWEN38_MODEL_PREFIXES = ("qwen3.8-flash-next", "qwen3.8-27b-worker")
MIMO_MODEL_FRAGMENT = "mimo-v2.6-flash"
MIMO_MAX_OUTPUT_TOKENS = 32000


def local_anthropic_count_tokens_endpoint(api_base: str) -> str:
    return api_base.rstrip("/") + "/v1/messages/count_tokens"


def _is_glm53_model(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.removeprefix("anthropic/") in GLM53_MODEL_NAMES


def _is_qwen38_model(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    name = value
    for provider in ("anthropic/", "hosted_vllm/", "openai/"):
        name = name.removeprefix(provider)
    return name.startswith(QWEN38_MODEL_PREFIXES)


def _is_mimo_model(value: Any) -> bool:
    return isinstance(value, str) and MIMO_MODEL_FRAGMENT in value.lower()


def limit_mimo_output_tokens(data: dict[str, Any]) -> dict[str, Any]:
    """Bound unusually long MiMo generations without changing normal calls."""
    if not _is_mimo_model(data.get("model")):
        return data

    updates: dict[str, int] = {}
    for key in ("max_tokens", "max_completion_tokens"):
        value = data.get(key)
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value > MIMO_MAX_OUTPUT_TOKENS
        ):
            updates[key] = MIMO_MAX_OUTPUT_TOKENS
    if not updates:
        return data
    updated = dict(data)
    updated.update(updates)
    return updated


def normalize_qwen38_reasoning_effort(data: dict[str, Any]) -> dict[str, Any]:
    """Translate Claude Code's high/max effort to Qwen3.8's xhigh level."""
    if not _is_qwen38_model(data.get("model")):
        return data
    if data.get("reasoning_effort") not in {"high", "max"}:
        return data

    updated = dict(data)
    updated["reasoning_effort"] = "xhigh"
    return updated


def _strict_tool(tool: Any) -> Any:
    if not isinstance(tool, dict):
        return tool

    # Anthropic custom tools put the schema directly under input_schema.
    if isinstance(tool.get("name"), str) and isinstance(
        tool.get("input_schema"), dict
    ):
        updated = dict(tool)
        updated["strict"] = True
        return updated

    # OpenAI-compatible clients wrap the same schema in a function object.
    function = tool.get("function")
    if tool.get("type") == "function" and isinstance(function, dict):
        updated_function = dict(function)
        updated_function["strict"] = True
        updated = dict(tool)
        updated["function"] = updated_function
        return updated

    # Do not add unsupported fields to provider/server tools such as web search.
    return tool


def enforce_glm53_strict_tools(data: dict[str, Any]) -> dict[str, Any]:
    """Force constrained tool calling for GLM 5.3 aliases and backend name."""
    if not _is_glm53_model(data.get("model")):
        return data
    tools = data.get("tools")
    if not isinstance(tools, list):
        return data

    updated = dict(data)
    updated["tools"] = [_strict_tool(tool) for tool in tools]
    return updated
