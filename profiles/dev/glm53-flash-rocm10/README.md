# GLM-5.3 Flash on ROCm 10

This directory preserves the qualification evidence for the production
`glm53-flash-rocm` profile on eight R9700 (`gfx1201`) GPUs. The deployment
follows the repository's native recipe architecture: a pinned wheel SDK,
source checkouts, ordered patches, and a recipe-local virtual environment
under `.runtime/recipes/`. It does not add or use Docker.

Target-only, native MTP K1, DFlash2 K1, and DFlash2 K7 passed 32K development
qualification on 2026-09-05. DFlash2 K7 subsequently passed exact 32K and
256K boundaries, and is now the production default with BF16 KV and one
sequence. The 32K variants remain explicit fallback or diagnostic modes.

## Identity

- Profile: `profiles/production/glm53-flash-rocm.json`
- Recipe: `vllm_glm53flashrocm10_v0.29`
- ROCm SDK wheels: 10.0.0 from `https://stable.repo.amd.com/rocm/whl-next/`
- PyTorch: 2.13.0+rocm10.0.0
- vLLM: `7fbd44cbe0a90b9c8fd3a94a0f0401ac4b1bc719`
- AITER: v0.1.21, `7ff5155f3ba772e534b6cf8dddc0099932327b9b`
- AMD SMI Python: 27.0.0+6b0e43f3 from the pinned ROCm SDK wheel

The production profile is discoverable as `glm53-flash-rocm` and is also the
repository default.

## Build and host checks

```bash
PROFILE=glm53-flash-rocm
./run install --profile "$PROFILE" --dry-run
./run doctor --profile "$PROFILE"
./run install --profile "$PROFILE"
```

The install creates a separate ROCm 10 recipe. It does not modify system ROCm,
the production GLM recipe, or the NVIDIA display driver.

## Qualification order

Start the target-only baseline:

```bash
PROFILE=glm53-flash-rocm
skills/start-r9700-runtime/scripts/start-runtime.sh \
  --profile "$PROFILE" --runtime-mode target-only-32k
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
in `results/`.

## Best qualified mode

DFlash2 K7 was the fastest qualified decode mode and is now the default. Start
it with:

```bash
skills/start-r9700-runtime/scripts/start-runtime.sh \
  --profile glm53-flash-rocm
```

It measured 26.21 client-observed decode tokens/s, 868.34 prefill tokens/s,
and 92.26% draft-token acceptance in the qualification run. See
`results/qualification-20260905.md` for the complete evidence and caveats.

The default uses the qualified long-context limits:
`max_model_len=262144`, scheduler chunk 512, and
`gpu_memory_utilization=0.97`. See
`results/qualification-256k-20260906.md` for the exact boundary result.
