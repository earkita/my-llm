---
name: measure-r9700-model
description: Validate a running R9700 vLLM model through its API and benchmark client-observed prefill and decode throughput with exact token counts. Use when the user asks to test a model, measure prefill or decode speed, run a performance check, or save benchmark evidence for baseline or another runtime profile.
---

# Measure R9700 Model

1. Work from the repository containing this skill.
2. Check `./run service status`. Require an already-running managed service; do not start, restart, or stop it implicitly.
3. Resolve one complete production profile from the running service. Do not mix
   model and runtime components from different profiles.
4. Run the API correctness gate before measuring performance:

   ```bash
   skills/measure-r9700-model/scripts/test-and-benchmark.sh
   ```

5. Pass requested workload dimensions explicitly. For example:

   ```bash
   skills/measure-r9700-model/scripts/test-and-benchmark.sh \
     --profile deepseek-v4-flash --prompt-tokens 256 --output-tokens 256 \
     --concurrency 2 --repetitions 5 --warmup 1
   ```

Use `--dry-run` to preview both commands without contacting the API.

For an exact input-plus-output context-boundary test, pass `--full-context`
instead of calculating the prompt length manually. The script reads
`max_model_len` from the selected runtime and subtracts `--output-tokens`.

After success, read the benchmark JSON and report:

- `observed_prefill_mean_tokens_per_second` as client-observed prefill throughput;
- `decode_mean_tokens_per_second` and `decode_min_tokens_per_second`;
- TTFT mean/p95, end-to-end p95, concurrency, repetitions, and artifact paths.

Do not describe observed prefill as kernel-only throughput: it includes HTTP and scheduling time. Compare runs only when model/profile identity, token lengths, concurrency, cache policy, warmup, and repetitions match. Do not infer long-context capability or hardware stability from this benchmark.
