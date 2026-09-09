---
name: validate-r9700-stack
description: Run an end-to-end health validation of the complete R9700 inference stack—host and device checks, live API correctness, client-observed throughput, and log diagnosis—and report one pass/fail verdict. Use when the user asks to validate the stack, run a full check, check stack health, verify everything works, or qualify a profile after install or a runtime change. Do not use for a single isolated task such as only benchmarking or only reading a log; run the specific skill instead.
---

# Validate R9700 Stack

Run from the repository containing this skill. Chains the single-purpose
skills in dependency order and folds their results into one verdict.

1. Resolve the profile to validate. Default to `glm53-flash`. If the user
   names a profile, use it consistently for every step.
2. Check whether a managed service is already running:

   ```bash
   ./run service status
   ```

   - If a service is running, record its identity (recipe, runtime, log
     path). Do not start, restart, or stop anything implicitly. Validate the
     running stack as-is, and treat a profile mismatch between the running
     service and the requested profile as a finding to report, not an error
     to fix.
   - If no service is running, ask the user before starting one. If the
     user declines, validate host prerequisites only and say the API steps
     were skipped.
3. Host and device prerequisites:

   ```bash
   ./run doctor --profile PROFILE
   ```

   Report failures here as blocking findings.
4. API correctness and throughput, using the measure skill's script:
   ```bash
   skills/measure-r9700-model/scripts/test-and-benchmark.sh --profile PROFILE
   ```
   Pass `--runtime-mode` when the running service's runtime embeds one.
   Report observed prefill, decode mean/min, TTFT mean/p95 and end-to-end
   p95 exactly as `measure-r9700-model` defines them.
5. Log diagnosis, using the diagnose skill's script:
   ```bash
   skills/diagnose-r9700-log/scripts/diagnose-last-log.sh
   ```
   Classify the newest log with the same conservatism as
   `diagnose-r9700-log`: report only what the evidence proves, and treat
   ordinary warnings as nonfatal.
6. Compose the verdict. Pass means: doctor clean, API gate correct,
   benchmark numbers reported, and log classification not showing an
   application error, OOM, signal, or unexplained termination. Anything
   else is fail with the failing step named. Separate observed facts from
   inference. Never restart, stop, or repair anything as part of
   validation—report findings and ask before any lifecycle action.

Comparison rules follow `measure-r9700-model`: compare benchmark numbers
only when model/profile identity, token lengths, concurrency, cache policy,
warmup, and repetitions match. Do not infer long-context capability or
hardware stability from this validation.
