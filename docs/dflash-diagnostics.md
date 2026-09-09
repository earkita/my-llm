# GLM DFlash MRV2 diagnostics

The diagnostic path has two independent parts:

- `scripts/dflash_diagnostics.py` freezes one tokenized prompt and captures
  exact request/response token IDs, top logprobs and speculative-decoding
  metrics from the OpenAI-compatible API.
- vLLM patch `0016-dflash-mrv2-bounded-diagnostics.patch` can record the first
  few MRV2 DFlash steps as JSONL. It is disabled unless explicitly enabled.

The client never starts, stops or switches a model. Runtime selection remains
an explicit operator action.

## Deterministic API A/B

Create a diagnostic directory under generated state and tokenize the prompt
once while either GLM mode is running:

```bash
mkdir -p .runtime/diagnostics/glm-dflash
.venv/bin/python scripts/dflash_diagnostics.py prepare \
  --model glm-5.3-flash-quark-mxfp4 \
  --output .runtime/diagnostics/glm-dflash/case.json
```

Start the explicit BF16-KV 256K fallback, then capture its baseline:

```bash
./run launcher start glm53-flash-v029-rollback \
  --runtime-mode mxfp4-gemv-dflash2-k7-256k
.venv/bin/python scripts/dflash_diagnostics.py capture \
  --case .runtime/diagnostics/glm-dflash/case.json \
  --label bf16 \
  --max-tokens 64 \
  --logprobs 5 \
  --output .runtime/diagnostics/glm-dflash/bf16.json
```

After an explicit stop, start the default DFlash2 K4 runtime and run the same
command with a different label and output path:

```bash
./run launcher stop
./run launcher start glm53-flash
.venv/bin/python scripts/dflash_diagnostics.py capture \
  --case .runtime/diagnostics/glm-dflash/case.json \
  --label fp8 \
  --max-tokens 64 \
  --logprobs 5 \
  --output .runtime/diagnostics/glm-dflash/fp8.json
```

Compare the exact output token sequences:

```bash
.venv/bin/python scripts/dflash_diagnostics.py compare \
  .runtime/diagnostics/glm-dflash/bf16.json \
  .runtime/diagnostics/glm-dflash/fp8.json \
  --output .runtime/diagnostics/glm-dflash/comparison.json \
  --summary-output .runtime/diagnostics/glm-dflash/summary.md
```

The first capture is always the baseline. Comparison fails if prompt token IDs
or deterministic generation settings differ. `greedy_equivalent=true` means
the returned output token IDs are exactly equal. Cross-restart equality is a
diagnostic signal rather than the sole gate. Per-request speculative metrics
are preferred when exposed by vLLM; Prometheus counter deltas are also saved,
but are process-wide and require an otherwise idle server.

The default capture is deliberately bounded to 64 generated tokens and five
logprobs per position. Hard limits are 4,096 input tokens, 256 output tokens and
20 logprobs. An API key may be supplied through `VLLM_API_KEY`; its value is
used only as an HTTP header and is never written to an artifact.

## Qualification evidence on R9700

The production default `glm53-flash` uses ROCm 10 v0.31, MRV2, TP8 without
expert parallelism, output-tiled BN8 RDNA4 MXFP4 decode GEMV, DFlash2 K4, FP8
KV, Vision and a 786,432-token context. Prefix caching is disabled. The
explicit `mxfp4-gemv-dflash2-k7-256k` mode remains in the
`glm53-flash-v029-rollback` profile and uses scalar GEMV with BF16 KV at
262,144 tokens.

Before K4 promotion, the exact K7 full-context request used 1,048,560 prompt
plus 16 output tokens,
measured about 607 tok/s observed prefill and 28.31 tok/s decode. Full-context
NIAH then passed 4/4 placements at 5%, 35%, 65% and 95% depth without an ECC,
AER or runtime OOM report.

The default ROCm 10 v0.31 recipe carries 26 ordered vLLM patches. Its cache-page geometry,
slot guards, bounded indexer workspaces, sharded DFlash projection, staged
OCP-MX dequantization and packed RDNA4 MXFP4 GEMV are covered by focused
repository tests.
