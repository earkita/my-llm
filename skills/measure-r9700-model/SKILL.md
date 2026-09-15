---
name: measure-r9700-model
description: Validate a running R9700 vLLM model and measure client-observed prefill, decode, and speculative acceptance with the repository's official vLLM Bench client. Use when the user asks to test a model, measure performance, compare runtime profiles, or save benchmark evidence.
---

# Measure R9700 Model

1. Work from the repository containing this skill.
2. Check `./run service status`. Require an already-running managed service; do not start, restart, or stop it implicitly.
3. Resolve the complete deployment profile from the running service. Accept a
   bare production name, a directory-qualified name such as `dev/example`, or
   a JSON path. Do not mix model and runtime components from different profiles.
4. Run the API correctness gate before measuring performance. The helper then
   delegates measurement to `scripts/bench`, which wraps the packaged
   runtime's official `vllm bench serve` client. The helper skips vLLM Bench's
   redundant initial prompt probe after the gate so automatic prefix caching
   cannot contaminate the first measured request:

   ```bash
   skills/measure-r9700-model/scripts/test-and-benchmark.sh
   ```

5. Pass requested workload dimensions explicitly. For example:

   ```bash
   skills/measure-r9700-model/scripts/test-and-benchmark.sh \
     --profile dev/glm53-flash-new-dflash \
     --prompt-tokens 8192 --output-tokens 256 \
     --concurrency 1 --repetitions 5 --warmup 1 \
     --seed-base 20260914
   ```

   For an explicit embedded runtime mode, pass the same identity to both
   checks with `--runtime-mode`, for example
   `--runtime-mode mxfp4-gemv-dflash2-k7-256k`.

Use `--dry-run` to preview both commands without contacting the API. For a
direct one-off measurement without the API gate, use:

```bash
scripts/bench dev/glm53-flash-new-dflash decode --raw \
  --input-tokens 128 --output-tokens 512 --requests 5 \
  --concurrency 1 --warmups 1 --seed-base 20260914
```

Raw throughput measurements must use `scripts/bench ... --raw`: it sends
random-token prompts of exact length through `/v1/completions`, forces greedy
generation with `ignore_eos`, and records server-side speculative acceptance.
Do not use `./run benchmark` to estimate speculative acceptance or
representative decode throughput; its short tiled phrase is a deterministic
infrastructure probe and can make a draft model appear unrealistically accurate.

For an exact input-plus-output context-boundary test, pass `--full-context`
instead of calculating the prompt length manually. The script reads
`max_model_len` from the selected runtime and subtracts `--output-tokens`.

After success, read `summary.json` and report:

- `effective_prefill_tokens_per_second` as client-observed prefill throughput;
- `decode_tokens_per_second`, `tpot_mean_ms`, and output throughput;
- `spec_decode_acceptance_percent` when speculative decoding is active;
- TTFT mean/p95, end-to-end latency, concurrency, completed requests, seed,
  and artifact paths.

Do not describe effective prefill as kernel-only throughput: it includes HTTP
and scheduling time. Compare runs only when model/profile identity, access
mode, token lengths, concurrency, cache policy, warmups, request count, and
seed match. Random-token results measure engine throughput and draft behavior
under a reproducible synthetic load; add a real coding workload before claiming
agent-task performance. Do not infer long-context capability or hardware
stability from one benchmark.
