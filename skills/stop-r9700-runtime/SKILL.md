---
name: stop-r9700-runtime
description: Gracefully stop the managed R9700 vLLM process and its persistent systemd user unit, waiting for engine and GPU workers to exit without SIGKILL. Use when the user asks to stop, shut down, terminate, or turn off the running baseline or other R9700 model service.
---

# Stop R9700 Runtime

1. Work from the repository containing this skill.
2. Inspect `./run service status` and `systemctl --user is-active r9700-runtime.service`.
3. Preview when requested:

   ```bash
   skills/stop-r9700-runtime/scripts/stop-runtime.sh --dry-run
   ```

4. Stop the managed process and persistent unit:

   ```bash
   skills/stop-r9700-runtime/scripts/stop-runtime.sh
   ```

   Use `--timeout SECONDS` only when the user requests a different graceful-shutdown limit.
5. Confirm that `./run service status` prints `stopped` and the systemd unit is inactive. Report the stopped PID when available and whether cleanup completed.

The script sends the repository-managed SIGINT first, allowing the API, engine, and GPU workers to shut down. It then stops the keeper unit. Never use `kill -9`, guess a PID, delete `.runtime/service.json`, or force-reset the process. If identity verification or graceful shutdown fails, report the error and leave state intact for diagnosis.
