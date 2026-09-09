# GLM-5.3 output-tiled MXFP4 GEMV on gfx1201

This development lane isolates one decode optimization from the production
`glm53-flash` runtime. It keeps the v0.29 software pins and first 25 vLLM
patches, then adds only patch `0026`: an output-tiled Triton MXFP4 GEMV with an
in-recipe scalar rollback. The explicit profile is
`profiles/dev/glm53-flash-rocm10-mxfp4-tiled.json`.

The lane is deliberately unqualified. The active production service must not
be stopped or replaced merely to inspect, build, or preview these gates.

## Current state on 2026-09-09

- The isolated `vllm_glm53flashrocm10_v0.31` recipe is installed.
- The default candidate is `VLLM_ROCM_MXFP4_GEMV_BLOCK_N=4`; embedded mode
  `scalar-gemv-rollback` selects `block_n=0` in the exact same recipe.
- The operator benchmark covers `block_n=0/1/2/4/8` and the two TP8/DFlash K4
  MoE shapes: up projection `5 x 4096 -> 40 x 512` and down projection
  `40 x 256 -> 40 x 4096`, each across eight local experts.
- No v0.31 GPU or API performance result exists yet. v0.29 remains the
  production and rollback target.

v0.31 is simpler to qualify than v0.30: it changes only a bounded vLLM decode
kernel and provides a scalar control under the same build. v0.30 changes both
AITER and vLLM in sparse attention, so it also depends on top-k, KV dtype, page
geometry, prefill continuations, and long-context behavior.

## Work that is safe while production stays active

Validate the immutable plan and preview the GPU commands:

```bash
./run install \
  --profile profiles/dev/glm53-flash-rocm10-mxfp4-tiled.json \
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

Keep v0.31 only if every tiled choice matches the scalar result and at least
one choice wins both exact GLM shapes. Select the block size from the measured
artifact, not from the current `block_n=4` hypothesis.

Then perform two separate managed-runtime trials, gracefully stopping between
them: first `scalar-gemv-rollback`, then the default tiled candidate. For each
trial, run API validation before a matched benchmark:

```bash
skills/measure-r9700-model/scripts/test-and-benchmark.sh \
  --profile profiles/dev/glm53-flash-rocm10-mxfp4-tiled.json \
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
5. A final 1M boundary run passes with telemetry and no new GPU/system errors.

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
