# Production profiles and GLM-5.3 ROCm 10 qualification summary

Date: 2026-09-07 (Europe/Warsaw)

## Outcome

The repository contains flat, self-contained production profiles.
`glm53-flash` is the repository's default profile and stack preset. It now
selects the ROCm 10 deployment directly; the superseded ROCm 7.14 GLM profile
and recipe have been removed.

The default runtime uses MRV2, TP8/EP8, packed RDNA4 MXFP4 decode GEMV,
DFlash2 K7, FP8 KV, one sequence and a 1,048,576-token context.

## Pinned GLM stack

- hardware: 8 x AMD Radeon AI PRO R9700 32 GB (`gfx1201`); NVIDIA remains the
  display adapter and is excluded from compute selection;
- ROCm SDK: 10.0.0 from the recipe-local wheel environment;
- target: `amd/GLM-5.3-Flash-Quark-MXFP4` revision
  `b5688f25491202978c19c4d036eef579f61bbe07`;
- drafter: `incoai/GLM-5.3-Flash-DFlash2` revision
  `bf582e4eacc1810f76656d1811693ff6c6737d2a`;
- vLLM: `7fbd44cbe0a90b9c8fd3a94a0f0401ac4b1bc719`;
- AITER: v0.1.21, `7ff5155f3ba772e534b6cf8dddc0099932327b9b`;
- runtime: MRV2, TP8/EP8, packed MXFP4 GEMV, DFlash2 K7, FP8 KV, 1M, no CPU
  offload and no prefix cache.

The repository-native recipe is isolated under `.runtime/recipes/`; it does
not use Docker or depend on the system ROCm user-space stack.

## Qualification result

The exact `1,048,560 + 16 = 1,048,576` boundary request completed at about
607 client-observed prefill tok/s and 28.31 decode tok/s. Full-context NIAH
passed 4/4 placements at 5%, 35%, 65% and 95% depth. No ECC, AER or runtime
OOM was observed in that qualification.

## Runtime commands

```bash
# default production runtime: ROCm 10, packed GEMV, DFlash2 K7, FP8 KV, 1M
./run launcher start glm53-flash

# explicit BF16 KV, 256K fallback
./run launcher start glm53-flash \
  --runtime-mode mxfp4-gemv-dflash2-k7-256k

# full configured boundary of an already-running default runtime
skills/measure-r9700-model/scripts/test-and-benchmark.sh \
  --profile glm53-flash --output-tokens 128 --full-context \
  --concurrency 1 --repetitions 1 --warmup 0
```

Stop the service before changing modes. Concurrency above one and a long
thermal soak remain outside the production qualification.

Detailed run artifacts remain under ignored `.runtime/` and `logs/` state.
