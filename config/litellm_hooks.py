from __future__ import annotations

import os
from typing import Any

from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.anthropic.count_tokens.transformation import (
    AnthropicCountTokensConfig,
)

from r9700.litellm_tools import (
    enforce_glm53_strict_tools,
    local_anthropic_count_tokens_endpoint,
)


def _local_count_tokens_endpoint(_config: AnthropicCountTokensConfig) -> str:
    return local_anthropic_count_tokens_endpoint(
        os.environ["HOSTED_INFERENCE_ANTHROPIC_BASE"]
    )


# LiteLLM 1.96.2 ignores a deployment's api_base in its Anthropic count-token
# helper and otherwise contacts api.anthropic.com. Keep the entire local model
# route, including Claude Code's context meter, on the selected vLLM backend.
AnthropicCountTokensConfig.get_anthropic_count_tokens_endpoint = (  # type: ignore[method-assign]
    _local_count_tokens_endpoint
)


class GLM53StrictToolsHook(CustomLogger):
    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict[str, Any],
        call_type: str,
    ) -> dict[str, Any]:
        return enforce_glm53_strict_tools(data)


proxy_handler_instance = GLM53StrictToolsHook()
