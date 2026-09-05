# GLM-5.3 Flash on ROCm 10

This is an isolated development profile for GLM-5.3-Flash Quark MXFP4 on
eight R9700 (`gfx1201`) GPUs. It follows the repository's existing native
recipe architecture: a pinned wheel SDK, source checkouts, ordered patches,
and a recipe-local virtual environment under `.runtime/recipes/`. It does not
add or use Docker.

The qualified default is deliberately conservative: target-only MRV2,
TP8/EP8, BF16 KV, one sequence, and a 32K context. Speculative decoding is
enabled only through an explicit runtime mode. Target-only, native MTP K1,
DFlash2 K1, and DFlash2 K7 passed the development qualification on
2026-09-05; the profile remains outside production pending explicit approval.

## Identity

- Profile: `profiles/dev/glm53-flash-rocm10/glm53-flash-rocm10.json`
- Recipe: `vllm_glm53flashrocm10_v0.29`
- ROCm SDK wheels: 10.0.0 from `https://stable.repo.amd.com/rocm/whl-next/`
- PyTorch: 2.13.0+rocm10.0.0
- vLLM: `7fbd44cbe0a90b9c8fd3a94a0f0401ac4b1bc719`
- AITER: v0.1.21, `7ff5155f3ba772e534b6cf8dddc0099932327b9b`
- AMD SMI Python: 27.0.0+6b0e43f3 from the pinned ROCm SDK wheel

The profile is not discoverable by production name. Always pass its explicit
path while it remains under `profiles/dev`.

## Build and host checks

```bash
PROFILE=profiles/dev/glm53-flash-rocm10/glm53-flash-rocm10.json
./run install --profile "$PROFILE" --dry-run
./run doctor --profile "$PROFILE"
./run install --profile "$PROFILE"
```

The install creates a separate ROCm 10 recipe. It does not modify system ROCm,
the production GLM recipe, or the NVIDIA display driver.

## Qualification order

Start the target-only baseline:

```bash
PROFILE=profiles/dev/glm53-flash-rocm10/glm53-flash-rocm10.json
skills/start-r9700-runtime/scripts/start-runtime.sh --profile "$PROFILE"
```

Stop it gracefully before changing modes:

```bash
skills/stop-r9700-runtime/scripts/stop-runtime.sh
```

Then qualify each explicit mode in order:

```bash
skills/start-r9700-runtime/scripts/start-runtime.sh \
  --profile "$PROFILE" --runtime-mode native-mtp-k1

skills/start-r9700-runtime/scripts/start-runtime.sh \
  --profile "$PROFILE" --runtime-mode dflash2-k1

skills/start-r9700-runtime/scripts/start-runtime.sh \
  --profile "$PROFILE" --runtime-mode dflash2-k7
```

Do not switch modes over a running service. Stop each mode gracefully and
inspect its log before starting the next one.

## Basic API and throughput gate

With one mode running:

```bash
curl -fsS http://127.0.0.1:8000/health
./run benchmark \
  --profile "$PROFILE" \
  --prompt-tokens 4096 \
  --output-tokens 128 \
  --concurrency 1 \
  --repetitions 3 \
  --output profiles/dev/glm53-flash-rocm10/results/benchmark-4k.json
```

Record exact versions, source identities, GPU visibility, API correctness,
prefill/decode rates, speculative acceptance, and relevant kernel/AER events
in `results/`. Promotion to `profiles/production` requires separate explicit
approval.

## Best qualified mode

DFlash2 K7 was the fastest qualified decode mode. Start it with:

```bash
skills/start-r9700-runtime/scripts/start-runtime.sh \
  --profile profiles/dev/glm53-flash-rocm10/glm53-flash-rocm10.json \
  --runtime-mode dflash2-k7
```

It measured 26.21 client-observed decode tokens/s, 868.34 prefill tokens/s,
and 92.26% draft-token acceptance in the qualification run. See
`results/qualification-20260905.md` for the complete evidence and caveats.
