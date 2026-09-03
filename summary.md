# Production profiles and GLM-5.3 qualification summary

Date: 2026-09-03 (Europe/Warsaw)

## Outcome

The repository contains exactly three flat production profiles. GLM-5.3 was
migrated from the historical llama.cpp/GGUF deployment to the requested AMD
Quark/MXFP4 checkpoint and a repository-local vLLM `glm-release` recipe.

The qualified GLM production mode is target-only. It loads on all eight R9700
GPUs, completes prefill and decode, exposes a correct OpenAI-compatible chat
API, and returns coherent text. DFlash2 is available only through explicit
diagnostic runtime modes because current MRV2 multi-token verification and
rollback are not lossless.

## Test host and pinned GLM stack

- 8 x AMD Radeon AI PRO R9700 32 GB (`gfx1201`), ROCm 7.14
- target: `amd/GLM-5.3-Flash-Quark-MXFP4` revision
  `b5688f25491202978c19c4d036eef579f61bbe07`
- 62 safetensors shards, 185,066,521,464 tensor bytes
- vLLM: `ZJY0516/vllm` `glm-release` commit
  `4500c80c080328dfe62435d083f4063e00d987df` (PR #53906)
- MRV2 forced with `VLLM_USE_V2_MODEL_RUNNER=1`
- TP8 + EP8, BF16 KV, 32,768 context, no CPU offload, prefix cache disabled
- Quark emulation backends on RDNA4; native FP4 BMM is intentionally disabled
- DFlash2 artifact revision `bf582e4eacc1810f76656d1811693ff6c6737d2a`,
  SHA-256 `b038e1d9d1e7833fa3880c2c0135ba9b673013f03da1b29fb831931584759dac`

## Production profile checks

| Profile | Backend and layout | Context | Current checkpoint/API evidence |
|---|---|---:|---|
| `glm53-flash` | vLLM glm-release, TP8/EP8, target-only | 32,768 | 62 shards verified; 8 GPUs active; API 6/6; coherent chat |
| `qwen38-flash` | vLLM 0.28, TP8/EP8, MTP K2 | 262,144 | verified on 2026-09-02 |
| `deepseek-v4-flash` | vLLM 0.28, TP1/PP6, DSpark K5 | 1,048,576 | verified on 2026-09-02 |

## Fast benchmark

Workload: 256 input tokens, 64 output tokens, concurrency 1, two measured
repetitions after one warm-up. These are smoke/regression measurements, not a
capacity ranking.

| Profile | Date | Mean TTFT (s) | Mean E2E (s) | Observed prefill (tok/s) | Mean decode (tok/s) | Aggregate output (tok/s) |
|---|---|---:|---:|---:|---:|---:|
| `glm53-flash` Quark target | 2026-09-03 | 0.574 | 17.399 | 445.74 | 3.74 | 3.68 |
| `qwen38-flash` | 2026-09-02 | 0.347 | 2.715 | 738.41 | 26.76 | 23.68 |
| `deepseek-v4-flash` | 2026-09-02 | 0.585 | 2.569 | 437.49 | 31.76 | 24.92 |

The GLM artifact is stored locally under `.runtime/results/glm-target/`;
DFlash captures are under `.runtime/diagnostics/glm-dflash/`. Generated
evidence is intentionally not committed.

## DFlash2 diagnosis

The checkpoint itself is the correct GLM-5.3 DFlash2 model: architecture,
vocabulary, hidden size, non-causal setting, mask token and five
`target_layer_ids` match the target. The vLLM capture points resolve to target
layers 6, 15, 25, 34 and 43, and the GLM mHC output is contracted as in the
independently validated SGLang implementation.

The decisive tests used one frozen 20-token prompt and greedy generation:

| Runtime | Initial output IDs | Accepted drafts | Interpretation |
|---|---|---:|---|
| target-only | `3555, 374, 279, 12935, 914, 1939, ...` | n/a | baseline |
| minimal DFlash K=1 | `3555, 374, 279, 12935, 914, 1939, ...` | 1/62 | initial target state is correct; diverges after rollback |
| minimal DFlash K=7 | `3555, 374, 279, 4226, 30, 7943, 7943, ...` | 2/427 | first two drafts accepted; multi-token verify then degrades |

An earlier 16-patch image made K=1 diverge at token zero. Ablation isolated
that additional regression to the unrelated PR #54163 prefix-cache scheduler
patch. It was removed from the final recipe: prefix caching is disabled, and
the patch changes KDA/Mamba cache-boundary handling outside this
qualification. Diagnostic-only error wrapping was also removed. The bounded
trace patch remains opt-in and was proven not to affect the clean A/B.

A separate A/B port of the BF16-to-FP32 causal-convolution change from PR
#52905 reduced rather than improved acceptance and was reverted. No checkpoint
conversion or weight modification was used.

The remaining behavior matches independent upstream reports: DFlash can fail
under MRV2 even on NVIDIA, greedy output can diverge at K=1, and GLM target
verification with more than one query token can degrade on a different
accelerator stack. The likely remaining defect is therefore the MRV2 GLM
multi-token target verify/recurrent-state path, not the R9700 Quark loader or
the DFlash checkpoint.

## Final runtime modes

```bash
# qualified production path
./run launcher start glm53-flash

# explicit diagnostic controls only
./run launcher start glm53-flash --runtime-mode extract-hidden-states-k1
./run launcher start glm53-flash --runtime-mode dflash2-k1
./run launcher start glm53-flash --runtime-mode dflash2
```

LiteLLM is not required for inference. Add `--with-litellm` to the qualified
target-only start when Claude Code or a unified proxy endpoint is needed.

## Commands exercised

```bash
./run install --profile glm53-flash --rebuild
./run test runtime --profile glm53-flash
./run model verify glm53-flash
./run doctor --profile glm53-flash
make check
skills/start-r9700-runtime/scripts/start-runtime.sh --profile glm53-flash
./run test api --profile glm53-flash --timeout 600 --output OUTPUT
./run benchmark --profile glm53-flash --prompt-tokens 256 --output-tokens 64 \
  --concurrency 1 --repetitions 2 --warmup 1 --timeout 600 --output OUTPUT
skills/stop-r9700-runtime/scripts/stop-runtime.sh
```

## Relevant upstream status

- vLLM PR #53906 (`glm-release`) supplies GLM-5.3 target support and is not yet
  merged at the tested commit.
- vLLM issue #49559 reports near-zero DFlash acceptance under MRV2 on H800,
  while MRV1 works.
- vLLM issue #54928 reports greedy divergence with DFlash at K=1.
- vLLM-Ascend issue #15329 reports healthy GLM prefill/single-token decode but
  degraded logits in multi-token target verification.
- SGLang PR #36755 validates contracted mHC hidden states with the official
  DFlash2 checkpoint, supporting the auxiliary-state mapping used here.
