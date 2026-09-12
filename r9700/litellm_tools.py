from __future__ import annotations

from typing import Any


GLM53_MODEL_NAMES = frozenset(
    {
        "glm-5.3-flash-high",
        "glm-5.3-flash-quark-mxfp4",
    }
)
QWEN38_MODEL_PREFIXES = ("qwen3.8-flash-next", "qwen3.8-27b-worker")


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
