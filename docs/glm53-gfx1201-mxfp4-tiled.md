# GLM-5.3 output-tiled MXFP4 GEMV on gfx1201

This production lane isolates one decode optimization from the prior v0.29
`glm53-flash` runtime. It keeps the same software pins and first 25 vLLM
patches, then adds only patch `0026`: an output-tiled Triton MXFP4 GEMV with an
in-recipe scalar rollback. The qualified profile is
`profiles/production/glm53-flash.json`; the prior deployment remains available
as `profiles/production/glm53-flash-v029-rollback.json`.

## Current state on 2026-09-09

- The isolated `vllm_glm53flashrocm10_v0.31` recipe is installed.
- The benchmark-selected candidate is `VLLM_ROCM_MXFP4_GEMV_BLOCK_N=8`; embedded mode
  `scalar-gemv-rollback` selects `block_n=0` in the exact same recipe.
- The operator benchmark covers `block_n=0/1/2/4/8` and the two TP8/DFlash K4
  MoE shapes: up projection `5 x 4096 -> 40 x 512` and down projection
  `40 x 256 -> 40 x 4096`, each across eight local experts.
- The 2026-09-09 single-R9700 gate passed 16/16 focused tests. In the exact
  500-repetition microbenchmark, BN8 reached `1.039x` scalar on the up
  projection and `2.261x` on the down projection.
- The first full-model `32768 x 128`, C1 A/B used the earlier 1M
  language-only topology. Across nine requests per variant, BN8 improved mean
  decode from `48.835` to `50.435 tok/s` (`+3.28%`) and minimum decode from
  `48.449` to `49.403 tok/s` (`+1.97%`). Mean TTFT regressed `0.63%`, mean E2E
  regressed `0.32%`, and observed prefill regressed `0.63%`.
- The final deployment-exact nine-request A/B measured 49.820 mean and 48.790
  minimum decode tok/s for BN8 versus 49.065 and 48.213 for scalar: +1.54%
  mean and +1.20% minimum. Mean TTFT changed by +0.09%; mean E2E improved by
  0.05%.
- API and deterministic Vision smoke passed. NIAH 256K passed all four 5%,
  35%, 65% and 95% placements.
- The exact `786368 + 64 = 786432` boundary returned the expected 95% needle
  in 746.09 seconds. Across 746 one-second telemetry samples, all GPUs averaged
  at least 99.84% activity, peak power was 238 W, host memory available stayed
  above 302.93 GB, and the minimum reported VRAM margin was 20 MiB.
- No new MCE, watchdog, OOM, GPU reset or runtime traceback appeared during
  the qualification window. v0.31 was promoted; v0.29 remains the explicit
  topology-exact rollback.

v0.31 is simpler to qualify than v0.30: it changes only a bounded vLLM decode
kernel and provides a scalar control under the same build. v0.30 changes both
AITER and vLLM in sparse attention, so it also depends on top-k, KV dtype, page
geometry, prefill continuations, and long-context behavior.

## Work that is safe while production stays active

Validate the immutable plan and preview the GPU commands:

```bash
./run install \
  --profile glm53-flash \
  --dry-run

.venv/bin/python scripts/qualify-mxfp4-tiled-gemv.py --dry-run
```

The qualification runner checks the installed source identity and prints both
commands. Even with `--run-gpu-gates`, it refuses to allocate on a GPU whenever
`r9700-runtime.service` or the control-plane state is occupied. It never stops,
starts, or replaces a service.

## Qualification order after an explicitly authorized stop

Run the single-GPU correctness and microbenchmark gate first:

```bash
.venv/bin/python scripts/qualify-mxfp4-tiled-gemv.py \
  --run-gpu-gates --gpu 0
```

The completed gate selected BN8 because every tiled choice matched the scalar
result and BN8 won both exact GLM shapes. Re-run this gate whenever the kernel,
Triton pin, or GPU software stack changes.

Then perform two separate managed-runtime trials in the 768K Vision topology,
gracefully stopping between them: first `scalar-gemv-rollback`, then the
default tiled candidate. For each trial, run API validation before a matched
benchmark:

```bash
skills/measure-r9700-model/scripts/test-and-benchmark.sh \
  --profile glm53-flash \
  --runtime-mode scalar-gemv-rollback \
  --prompt-tokens 32768 --output-tokens 128 \
  --concurrency 1 --repetitions 9 --warmup 1
```

Repeat without `--runtime-mode` for the tiled candidate. Compare only runs with
the same profile identity, prompt/output sizes, concurrency, warmup, repetition
count, cache state, power cap, and DFlash K4 configuration. Promote only after:

1. Focused operator tests and exact-shape microbenchmark pass.
2. Short API smoke passes for scalar and tiled modes.
3. Nine-run `32768 x 128`, C1 A/B improves mean decode throughput without a
   worse minimum or material TTFT/E2E regression.
4. 256K NIAH passes with the established needle placements.
5. A final 768K boundary run passes with telemetry and no new GPU/system
   errors.

The reusable NIAH replay command validates every fixture hash before sending a
request and writes one result per needle depth:

```bash
.venv/bin/python scripts/replay-niah-requests.py \
  --requests-dir logs/results/niah-ab-bf16-vs-fp8-256k-20260906/requests-v2 \
  --output-dir logs/qualification/v031-tiled-bn8-niah-256k-RUN \
  --label v031-tiled-bn8-768k-vision
```

Prepare the 768K boundary case through the tokenizer-only endpoints while the
production service remains active. The resulting prompt reserves exactly 64
tokens for the answer and preserves the needle at 95%:

```bash
.venv/bin/python scripts/prepare-niah-boundary.py \
  --source logs/results/niah-mxfp4-gemv-fp8-1m-20260907/requests/request-depth-095.json \
  --output logs/results/niah-boundary-768k/requests/request-depth-095.json \
  --target-prompt-tokens 786368
```

## Follow-on upstream lanes

Keep these out of v0.31 so its result stays attributable:

- vLLM PR [#45559](https://github.com/vllm-project/vllm/pull/45559): GFX12
  skinny-GEMM work reports R9700 decode gains and may fit five-row DFlash K4
  linears, but dispatch must be confirmed by profiling.
- vLLM PR [#55106](https://github.com/vllm-project/vllm/pull/55106): fused ROCm
  SiLU-and-multiply-with-clamp may help GLM's dense/shared path; profile first.
- vLLM PR [#55736](https://github.com/vllm-project/vllm/pull/55736): further GLM
  decode hot-path cleanup; the router deduplication is already in local patch
  `0023`, while an AMD KDA port remains separate work.
- vLLM PR [#55917](https://github.com/vllm-project/vllm/pull/55917): RDNA4
  FlyDSL all-reduce is a future TP lane and should not be mixed into the GEMV
  comparison.

Native FP8/MXFP4 ports aimed at another weight format or GPU architecture are
not drop-in v0.31 candidates.
