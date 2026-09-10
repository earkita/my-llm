---
name: stop-r9700-runtime
description: Gracefully stop the complete managed production stack in dependency order—LiteLLM first, then the R9700 inference runtime—without SIGKILL. Use for ordinary stop, shutdown, terminate, or turn-off requests; component-only stops are diagnostic operations.
---

# Stop R9700 Production Stack

1. Work from the repository containing this skill.
2. Inspect `./run launcher status`, `r9700-runtime.service`, and
   `r9700-litellm-proxy.service`.
3. Preview when requested:

   ```bash
   ./run launcher stop --dry-run
   ```

4. Stop the complete stack in reverse dependency order:

   ```bash
   ./run launcher stop
   ```

   Use timeout overrides only when the user requests different graceful
   shutdown limits.
5. Confirm that both components report `stopped` and both systemd units are
   inactive. Report whether proxy and runtime cleanup completed.

The stack manager stops LiteLLM before sending the repository-managed SIGINT
to the inference runtime, allowing the API, engine, and GPU workers to shut
down. Never use `--runtime-only` for an ordinary production stop, `kill -9`,
guess a PID, delete managed state, or force-reset the process. If identity
verification or graceful shutdown fails, report the error and leave state
intact for diagnosis.
