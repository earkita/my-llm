---
name: start-litellm-proxy
description: Start the repository LiteLLM proxy persistently as a separate systemd user service, using the checked-in DeepSeek YAML configuration. Use when the user asks to run, launch, or start LiteLLM, the API gateway, or the OpenAI-compatible proxy for the local R9700 vLLM backend.
---

# Start LiteLLM Proxy

1. Work from the repository containing this skill.
2. Confirm `./run service status` reports the vLLM backend ready. Do not start or replace vLLM implicitly.
3. Confirm LiteLLM is installed. If `.runtime/litellm/venv/bin/litellm` is absent, report that `./run proxy install` is required; do not download packages without approval.
4. Check `./run proxy status` and `systemctl --user is-active r9700-litellm-proxy.service`. Do not replace an active proxy.
5. Start the persistent proxy and wait for readiness:

   ```bash
   skills/start-litellm-proxy/scripts/start-proxy.sh
   ```

6. Run `./run proxy test`, then report the proxy URL, model alias, unit name, and authentication result.

Use `config/litellm.yaml` as the model/provider source. Do not pass a model or provider on the command line. The systemd user manager keeps the proxy outside the Codex execution cgroup. Do not change the master key, enable lingering, install dependencies, or start the backend unless explicitly requested.
