# Production profiles and GLM-5.3 ROCm 10 qualification summary

Date: 2026-09-06 (Europe/Warsaw)

## Outcome

The repository contains flat, self-contained production profiles.
`glm53-flash-rocm` is the repository's default profile and stack preset, while
`glm53-flash` remains an independent ROCm 7.14 production deployment.

The default runtime is the qualified ROCm 10 configuration: MRV2, TP8/EP8,
DFlash2 K7, BF16 KV, one sequence and a 262,144-token context. The existing
ROCm 7.14 profile remains production-ready under its original
`glm53-flash` name.

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
- runtime: MRV2, TP8/EP8, DFlash2 K7, BF16 KV, 256K, no CPU offload and no
  prefix cache.

The repository-native recipe is isolated under `.runtime/recipes/`; it does
not use Docker or depend on the system ROCm user-space stack.

## Qualification result

All target-only, native MTP K1, DFlash2 K1 and DFlash2 K7 32K gates passed API
correctness and repeated 4096-input/128-output measurements. DFlash2 K7 was
the fastest qualified mode at 26.213 mean decode tok/s with 465/504 accepted
draft tokens.

The exact `262016 + 128 = 262144` boundary request completed at 606.78
client-observed prefill tok/s and 26.99 decode tok/s, with coherent output and
111/111 accepted draft tokens. All GPU ECC counters remained zero and the
inspected runtime and kernel logs contained no OOM, GPU reset, illegal memory
access, HSA/amdgpu fault, AER or MCE.

## Runtime commands

```bash
# default production runtime: ROCm 10, DFlash2 K7, BF16 KV, 256K
./run launcher start glm53-flash-rocm

# qualified 32K controls
./run launcher start glm53-flash-rocm --runtime-mode target-only-32k
./run launcher start glm53-flash-rocm --runtime-mode native-mtp-k1
./run launcher start glm53-flash-rocm --runtime-mode dflash2-k1
./run launcher start glm53-flash-rocm --runtime-mode dflash2-k7

# full configured boundary of an already-running default runtime
skills/measure-r9700-model/scripts/test-and-benchmark.sh \
  --profile glm53-flash-rocm --output-tokens 128 --full-context \
  --concurrency 1 --repetitions 1 --warmup 0
```

Stop the service before changing modes. Contexts above 256K, FP8 KV,
concurrency above one and a long thermal soak remain outside the production
qualification.

Full evidence is under `profiles/dev/glm53-flash-rocm10/results/`.
