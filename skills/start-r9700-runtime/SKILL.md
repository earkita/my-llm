---
name: start-r9700-runtime
description: "Start one self-contained my-llm production profile as a complete persistent stack: its R9700 inference runtime followed by the profile-bound LiteLLM proxy. Use for ordinary production model launch requests; component-only runtime starts are diagnostic operations."
---

# Start R9700 Production Stack

1. Work from the repository containing this skill.
2. Select one complete profile from `./run profiles list`. Use `glm53-flash`
   when none is supplied.
3. Check current state with `./run launcher status`. If another profile is
   already running, report it and do not replace it unless the user explicitly
   asks to switch or stop it first.
5. Require the configured maximum PPT0 power cap before launching. The start
   script defaults to at most 285 W on every visible GPU and fails closed if any
   card exceeds it, including after a GPU reset. A deliberately lower cap is
   accepted. It only reads `amd-smi`; it never invokes or
   bypasses `sudo`. On mismatch, stop and ask the user to run the exact
   `sudo amd-smi set` command printed by the script. Pass
   `--required-power-cap-w WATTS` only when the user explicitly requests a
   different limit.
6. Preview the complete runtime and proxy lifecycle when the request is
   ambiguous:

   ```bash
   ./run launcher start glm53-flash --dry-run
   ```

7. Start the requested production profile as one stack and wait for both
   components:

   ```bash
   ./run launcher start deepseek-v4-flash
   ```

8. Confirm both `r9700-runtime.service` and
   `r9700-litellm-proxy.service` are active and `./run launcher status` reports
   both components ready. Report the selected profile, runtime URL, LiteLLM
   URL, and proxy test result. On failure, inspect the failing component's
   journal and managed log; do not silently fall back to another profile.

The stack manager starts the model first, then LiteLLM, and rolls back only
components started by that invocation if startup fails. The systemd user
manager keeps both workloads outside the Codex execution cgroup. They survive
the skill command and sandbox ending, but with user lingering disabled they do
not promise survival across logout or reboot.

Do not use `--runtime-only` for an ordinary production launch. Do not enable
lingering, run installation, download a model, stop a running service, or alter
profiles without explicit user approval.
