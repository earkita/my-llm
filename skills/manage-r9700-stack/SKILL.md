---
name: manage-r9700-stack
description: Start or stop the complete local R9700 inference stack in dependency order—the managed model service selected by its runtime profile and the managed LiteLLM proxy. Use when the user wants both services launched together, the whole stack shut down, or a dry-run preview of either lifecycle operation. Do not use when only one service should change.
---

# Manage R9700 Stack

Run from the repository containing this skill.

Start the default `glm53-flash` preset. It binds the model, runtime, LiteLLM
routing and Claude Code settings as one tested selection:

```bash
./run stack start
```

Select another complete preset instead of choosing its components separately:

```bash
./run stack presets
./run stack start --preset qwen38-flash
```

The default GLM preset selects the verified vLLM Quark/MXFP4 target-only stack
with a 32K target context. Pass `--runtime-mode dflash2` to opt into the
qualified experimental DFlash2 K=7 mode. Qwen selects the production-ready
vLLM 0.28 cache-safe MTP K=2 alternative.

The script checks that LiteLLM is installed before changing service state. It
starts the selected model backend first, waits for readiness, then starts and
tests LiteLLM. It reuses
healthy services instead of replacing them. If startup fails, it rolls back
only components started by that invocation.

Stop the complete stack in reverse dependency order:

```bash
./run stack stop
```

Preview either operation without changing service state:

```bash
skills/manage-r9700-stack/scripts/manage-stack.sh start --preset glm53-flash --dry-run
skills/manage-r9700-stack/scripts/manage-stack.sh stop --dry-run
```

After `start`, report the selected runtime and model, the inference URL
`http://127.0.0.1:8000`, the LiteLLM URL `http://127.0.0.1:4000`, and the proxy
test result. After `stop`, confirm that both services are inactive.

Do not install LiteLLM, change keys or profiles, replace a healthy service, use
SIGKILL, or enable user lingering implicitly. Use the individual start/stop
skills when the request concerns only the model backend or only LiteLLM.
