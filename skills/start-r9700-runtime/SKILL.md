---
name: start-r9700-runtime
description: Start one self-contained my-llm production profile persistently through the r9700-runtime.service systemd user unit.
---

# Start R9700 Runtime

1. Work from the repository containing this skill.
2. Select one complete profile: `deepseek-v4-flash`, `glm53-flash`, or
   `qwen38-flash`. Use `glm53-flash` when none is supplied.
4. Check current state with `./run service status`. If a service is already running, report it and do not replace it unless the user explicitly asks to stop it first.
5. Require the configured maximum PPT0 power cap before launching. The start
   script defaults to at most 270 W on every visible GPU and fails closed if any
   card exceeds it, including after a GPU reset. A deliberately lower cap is
   accepted. It only reads `amd-smi`; it never invokes or
   bypasses `sudo`. On mismatch, stop and ask the user to run the exact
   `sudo amd-smi set` command printed by the script. Pass
   `--required-power-cap-w WATTS` only when the user explicitly requests a
   different limit.
6. Preview both the power check and resolved launch command when the request is ambiguous:

   ```bash
   skills/start-r9700-runtime/scripts/start-runtime.sh --dry-run
   ```

7. Start the requested service through the transient `r9700-runtime.service` user unit and wait for readiness:

   ```bash
   skills/start-r9700-runtime/scripts/start-runtime.sh \
     --profile deepseek-v4-flash
   ```

8. Confirm both `systemctl --user is-active r9700-runtime.service` and `./run service status`. Report the unit, backend, selected model, runtime, URL, and readiness result. On failure, inspect `journalctl --user-unit r9700-runtime.service` and the runtime log; do not silently fall back to another profile.

The systemd user manager keeps the workload outside the Codex execution cgroup. It survives the skill command and sandbox ending, but with user lingering disabled it does not promise survival across logout or reboot.

Forward `--host`, `--port`, and `--ready-timeout` only when requested. Do not enable lingering, run installation, download a model, stop a running service, or alter profiles without explicit user approval.
