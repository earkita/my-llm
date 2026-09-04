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

Start the explicit target-only fallback, then capture its baseline:

```bash
./run launcher start glm53-flash --runtime-mode target-only-32k
.venv/bin/python scripts/dflash_diagnostics.py capture \
  --case .runtime/diagnostics/glm-dflash/case.json \
  --label target \
  --max-tokens 64 \
  --logprobs 5 \
  --output .runtime/diagnostics/glm-dflash/target.json
```

After an explicit stop, start the default DFlash2 K7 runtime and run the same
command with a different label and output path:

```bash
./run launcher stop
./run launcher start glm53-flash
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

The production default is MRV2, TP8/EP8, DFlash2 K7, BF16 KV and a 262,144
token context. Prefix caching is disabled. The explicit `target-only-32k` mode
is the fallback:

- two K1 captures accepted 52/74 draft tokens (61.5-80.0% per capture) and
  returned coherent text;
- three K7 captures accepted 139/385 draft tokens (33.1-38.7% per capture),
  with accepted tokens at every one of the seven draft positions;
- the final fresh K7 start passed all six API checks and produced 46/119
  accepted drafts, mean acceptance length 3.71;
- a 256-input/128-output benchmark measured 23.63 tok/s mean decode and
  21.65 tok/s minimum decode at concurrency one.
- the exact full-context run used 262,016 prompt plus 128 output tokens,
  measured 599.39 tok/s observed prefill and 23.84 tok/s decode, returned
  coherent output and accepted 111/111 drafts.

The correction is the combination of aligned DFlash/MLA cache pages (`0010`),
the PR #55239 Triton multi-token verify path (`0012`), the focused PR #55219
kpool rollback ring (`0017`), PR #55201 invalid-pool rejection (`0018`), the
bounded indexer workspace (`0019`), ROCm kpool alignment (`0020`), the correct
128/256-token kernel-page advertisement (`0021`) and guarded block-table loads
from PR #54296 (`0022`). The old overlay patches, #54163 and the experimental
#52905 causal-convolution change are intentionally excluded.

The `0021` page geometry is essential: a 768-token storage block contains
three 256-token kernel pages. Before the block table represented that split,
the effective table ended around 87,552 tokens for the 256K profile and
136,704 for 400K. Those thresholds explain why a 134K control passed under the
400K configuration while the earlier full-boundary requests faulted. `0022`
also masks out-of-range loads, but does not replace the corrected geometry.

Only the ring behavior from PR #55219 commit `de63c847` is backported. The
entire draft PR also carries a broader generic packed-layout refactor without
GLM/MTP ROCm end-to-end evidence; importing it would enlarge the patch surface
without addressing a failing local test.

K7 is now the default after the full 256K pass. Concurrency above one and 400K
remain unqualified. The 400K replay at a verified 285 W cap kept the host alive
but ended with GPU `illegal memory access`; it predates `0021`/`0022` and was
not repeated on the final image. The dedicated GPU telemetry stream observed
maxima of 255 W, 82°C edge, 106°C hotspot and 88°C memory. Prefix cache remains
off because #54163 caused a K1 token-zero regression in local A/B testing.
