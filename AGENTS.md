# Repository Guidelines

## Project Structure & Module Organization

`r9700/` contains the Python 3.12 control plane. Use `./run` as the public
entry point. Exactly three self-contained deployment files live in
`profiles/production/`; each embeds its model, runtime and stack preset.
Profile inheritance and `extends` are forbidden. Immutable software pins and
patch hashes belong in `manifest/`, `constraints/` and `patches/`. Generated
state stays under ignored `.runtime/` and `logs/`.

## Build, Test, and Development Commands

- `make unit` or `./run test unit`: run offline unit tests.
- `./run install --profile PROFILE --dry-run`: validate one complete install
  plan without building dependencies.
- `make check`: run both unit tests and the install dry run.
- `./run install --profile PROFILE`: create an isolated recipe under
  `.runtime/`, fetch pinned sources, apply patches, and build the runtime.
- `./run doctor --profile PROFILE`: check host tools, devices, and permissions.
- Use `skills/start-r9700-runtime/scripts/start-runtime.sh --profile PROFILE`
  for persistent launch through `r9700-runtime.service`.

Never reboot or reset GPUs. Never use SIGKILL. Do not replace a running model
implicitly.

## Coding Style & Naming Conventions

Follow existing Python conventions: four-space indentation, type annotations,
`snake_case` functions and modules, `PascalCase` classes, and uppercase
constants. Keep the control plane standard-library-first. Never use system
`python3` or bare `pip`; use `uv` and `.venv/bin/python`. Preserve numeric
prefixes in patch filenames because manifest order is significant.

## Recipe and Patch Naming

Recipe names use `backend_target_version`, for example
`vllm_deepseekv4flash_v0.28`, `vllm_qwen38flash_pr53896`, and
`vllm_glm53flash_v0.28`. Use lowercase ASCII letters, digits and
underscores; a dot is allowed only in a version such as `v0.28`.

The manifest filename must be `manifest/<recipe-name>.json`, and generated
artifacts must live under `.runtime/recipes/<recipe-name>/`. Store every patch
for a recipe, including patches for helper sources such as AITER, under
`patches/<recipe-name>/`. Keep the ordered numeric prefix on every patch file.
Do not embed absolute checkout paths in scripts or committed configuration;
derive paths from the repository root or the script's own directory.

## Testing Guidelines

Tests use `unittest`. Run `make check` before submitting. Every profile change
must preserve the single-file/no-`extends` invariant and pass hash validation.
API, GPU and lifecycle tests are explicit, potentially disruptive operations;
never run them as an implicit side effect of unit tests. Never commit
credentials, model weights or transient process state.

## Commit & Pull Request Guidelines

History currently uses Conventional Commit-style subjects, for example `feat: bootstrap reproducible R9700 vLLM runtime`. Use an imperative `type: summary` subject (`fix:`, `docs:`, `test:`) and keep each commit scoped. PRs should explain the motivation, affected profiles or patches, commands run, and hardware used. Link relevant issues and attach benchmark or validation artifacts when behavior, capacity, or performance changes; include screenshots only for documentation changes where rendering matters.
