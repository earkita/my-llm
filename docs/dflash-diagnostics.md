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
the returned output token IDs are exactly equal, which is the losslessness
gate. Per-request speculative metrics are preferred when exposed by vLLM;
Prometheus counter deltas are also saved, but are process-wide and require an
otherwise idle server.

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

The production target-only mode is correct. DFlash remains diagnostic-only:

- K=1 without the unrelated prefix-cache patch #54163 matches the first six
  target tokens, then diverges after rollback; 1 of 62 drafts was accepted.
- K=7 matches the first three target tokens and accepts the first two drafts,
  then multi-token target verification degrades and output becomes repetitive;
  2 of 427 draft tokens were accepted.
- Adding #54163 caused an additional local regression: K=1 diverged at token
  zero. It is intentionally excluded because prefix caching is disabled and
  the patch is not required for this qualification.

These results distinguish a valid DFlash checkpoint and valid initial hidden
states from the remaining MRV2 multi-token verification/rollback defect.
