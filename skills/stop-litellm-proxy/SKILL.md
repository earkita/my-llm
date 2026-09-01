---
name: stop-litellm-proxy
description: Gracefully stop the repository LiteLLM proxy and its separate persistent systemd user unit without stopping the model backend. Use when the user asks to stop, shut down, terminate, or turn off LiteLLM, the API gateway, or the OpenAI-compatible proxy.
---

# Stop LiteLLM Proxy

1. Work from the repository containing this skill.
2. Inspect `./run proxy status` and `systemctl --user is-active r9700-litellm-proxy.service`.
3. Stop the proxy gracefully and clean up its keeper unit:

   ```bash
   skills/stop-litellm-proxy/scripts/stop-proxy.sh
   ```

4. Confirm `./run proxy status` prints `stopped` and the unit is inactive.
5. Report completion and explicitly state that the vLLM backend was left unchanged.

Never stop vLLM, guess a PID, send `SIGKILL`, delete state files, or force-reset the process. If identity verification or graceful shutdown fails, report the error and preserve state for diagnosis.
