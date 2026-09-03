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

Start the target-only runtime, then capture its baseline:

```bash
./run launcher start glm53-flash
.venv/bin/python scripts/dflash_diagnostics.py capture \
  --case .runtime/diagnostics/glm-dflash/case.json \
  --label target \
  --max-tokens 64 \
  --logprobs 5 \
  --output .runtime/diagnostics/glm-dflash/target.json
```

After an explicit stop and start in DFlash mode, run the same command with a
different label and output path:

```bash
./run launcher start glm53-flash --experimental-dflash2
.venv/bin/python scripts/dflash_diagnostics.py capture \
  --case .runtime/diagnostics/glm-dflash/case.json \
  --label dflash-k7 \
  --max-tokens 64 \
  --logprobs 5 \
  --output .runtime/diagnostics/glm-dflash/dflash-k7.json
```

Compare the exact output token sequences:

```bash
.venv/bin/python scripts/dflash_diagnostics.py compare \
  .runtime/diagnostics/glm-dflash/target.json \
  .runtime/diagnostics/glm-dflash/dflash-k7.json \
  --output .runtime/diagnostics/glm-dflash/comparison.json \
  --summary-output .runtime/diagnostics/glm-dflash/summary.md
```

The first capture is always the baseline. Comparison fails if prompt token IDs
or deterministic generation settings differ. `greedy_equivalent=true` means
the returned output token IDs are exactly equal. On this gfx1201 EP/emulation
stack, target-only itself has produced different token hashes across identical
seeded restarts, so cross-restart equality is a diagnostic signal rather than
the sole gate. Per-request speculative metrics are preferred when exposed by
vLLM; Prometheus counter deltas are also saved, but are process-wide and
require an otherwise idle server.

The default capture is deliberately bounded to 64 generated tokens and five
logprobs per position. Hard limits are 4,096 input tokens, 256 output tokens and
20 logprobs. An API key may be supplied through `VLLM_API_KEY`; its value is
used only as an HTTP header and is never written to an artifact.

## Bounded internal MRV2 trace

The trace is synchronous and intended only for a short, single-request run.
Before starting the persistent user service, publish these temporary variables
to the user service manager:

```bash
systemctl --user set-environment \
  VLLM_DFLASH_TRACE_FILE=.runtime/diagnostics/glm-dflash/dflash-k7-trace.jsonl \
  VLLM_DFLASH_TRACE_STEPS=4 \
  VLLM_DFLASH_TRACE_VALUES=16 \
  VLLM_DFLASH_TRACE_TOPK=5
```

Then explicitly start the DFlash runtime and run exactly one capture. Remove
the variables after stopping it:

```bash
./run launcher start glm53-flash --runtime-mode dflash2-k1
```

After the capture, stop the runtime and remove the variables:

```bash
./run launcher stop
systemctl --user unset-environment \
  VLLM_DFLASH_TRACE_FILE \
  VLLM_DFLASH_TRACE_STEPS \
  VLLM_DFLASH_TRACE_VALUES \
  VLLM_DFLASH_TRACE_TOPK
```

Only global rank zero writes. Each event type is capped independently (default
four, maximum 32). The trace contains no prompt text or full tensor data. It
records:

- each auxiliary hidden state, their concatenation and combined drafter input;
- context/query positions, sample indices and target/draft slot mappings;
- proposed token IDs;
- target verify top-k logits, sampled IDs, accepted/rejected counts;
- request, KV-slot and recurrent-state metadata before and after rollback.

Tensor records contain shape, dtype, finite check, mean, standard deviation,
L2 norm, maximum magnitude and at most 64 deterministically selected scalar
values. Enabling the trace introduces GPU synchronization and is therefore not
a performance benchmark mode.

## Qualified result on R9700

The production target-only mode is correct. DFlash2 K7 is now a validated but
explicit experimental mode:

- two K1 captures accepted 52/74 draft tokens (61.5-80.0% per capture) and
  returned coherent text;
- three K7 captures accepted 139/385 draft tokens (33.1-38.7% per capture),
  with accepted tokens at every one of the seven draft positions;
- the final fresh K7 start passed all six API checks and produced 46/119
  accepted drafts, mean acceptance length 3.71;
- a 256-input/128-output benchmark measured 23.63 tok/s mean decode and
  21.65 tok/s minimum decode at concurrency one.

The correction is the combination of aligned DFlash/MLA cache pages (`0010`),
the PR #55239 Triton multi-token verify path (`0012`), the focused PR #55219
kpool rollback ring (`0017`) and PR #55201 invalid-pool rejection (`0018`). The
old overlay patches, #54163 and the experimental #52905 causal-convolution
change are intentionally excluded.

K7 is not the default until the upstream fixes merge and full-context,
concurrent and deterministic-losslessness tests are complete.
