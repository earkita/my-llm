---
name: clean-r9700-repo
description: Audit and clean R9700 profile boundaries so every production profile contains one approved runtime without experimental modes while unfinished candidates and experimental variants live under profiles/dev. Use when the user asks to clean profile sprawl, move runtime experiments out of production, or promote a runtime. Do not use for general source-code cleanup.
---

# Clean R9700 Repository Profiles

Work from the repository root and read its `AGENTS.md` first.

## Boundary

Each JSON file in `profiles/production/` is an approved, self-contained
profile containing exactly one default runtime:

- the profile and runtime both have `status: production-ready`;
- `runtime.experimental_modes` is absent;
- the file contains the complete model, runtime, and stack configuration and
  never uses `extends`;
- only an explicit user decision can select or replace the production runtime.

Keep other approved model profiles in production under the same rule. Put
unfinished, unapproved, diagnostic, benchmark, and alternative runtime work
in `profiles/dev/`. Development profiles remain self-contained; they may
contain `experimental_modes`. Use `status: development` for the profile and
`status: diagnostic-only` for its default runtime unless the user has
specified a more accurate non-production status.

Do not delete manifests, constraints, patches, or installed recipes merely
because production no longer selects them. They may still support development
profiles or reproducibility.

## Workflow

1. Inspect `git status`, the production and development profile inventories,
   and the current service state. Do not start, stop, install, build, or load a
   model as part of repository cleanup.
2. Run the boundary auditor:

   ```bash
   .venv/bin/python skills/clean-r9700-repo/scripts/audit-profile-boundaries.py
   ```

3. When the request only moves existing experimental modes out of production,
   preserve every current default runtime and do not ask the user to select it
   again. If cleanup would promote or replace a production runtime and the
   approved choice is not explicit, stop and ask which exact runtime should
   become the default. A commit, benchmark, existing `production-ready` label,
   or recent service history is evidence, not approval.
4. Preserve unrelated work. Never reset or discard a dirty tree. Before a
   broad restructuring, use a named, recoverable stash only when the user has
   asked for a clean tree or has approved stashing.
5. Materialize the selected runtime directly in its canonical production
   profile. Merge any selected development-mode overrides into the default
   runtime, remove `experimental_modes`, and update its name, recipe, required
   patches, environment, limits, multimodal settings, stack context, and
   verification notes consistently.
6. Move every non-selected runtime variant or useful candidate to
   `profiles/dev/`; do not move unrelated approved production profiles.
   Adjust development profile names and non-production statuses. Keep them
   self-contained; do not introduce inheritance. Remove obsolete variants
   only when the user explicitly asks to discard them.
7. Update direct consumers such as LiteLLM limits, Claude templates, profile
   inventories, tests, and `provenance.json`. Recompute profile hashes after
   the final profile edit.
8. Run the auditor again, followed by `git diff --check` and `make check`.
   Review the final diff for unrelated changes and confirm the service state is
   unchanged.
9. Commit only when requested. If the user asked for a clean working tree, use
   one scoped Conventional Commit and report any recovery stash that remains.

Report the canonical production runtime, development profiles retained, audit
and test results, service state, and whether the working tree is clean.
