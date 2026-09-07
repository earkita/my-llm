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
./run launcher start glm53-flash-rocm --runtime-mode target-only-32k
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
./run launcher start glm53-flash-rocm
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
./run launcher start glm53-flash-rocm --runtime-mode dflash2-k1
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

## Qualification evidence on R9700

The production default `glm53-flash-rocm` uses ROCm 10, MRV2, TP8/EP8,
DFlash2 K7, BF16 KV and a 262,144-token context. Prefix caching is disabled.
The explicit `target-only-32k`, `native-mtp-k1`, `dflash2-k1` and
`dflash2-k7` modes preserve the qualified 32K control configurations.

At 32K configured context, every mode passed API correctness followed by three
4096-input/128-output measurements. DFlash2 K7 measured 868.34 observed
prefill tok/s, 26.213 mean decode tok/s and accepted 465/504 draft tokens
(92.26%). The exact full-context run used 262,016 prompt plus 128 output
tokens, measured 606.78 tok/s observed prefill and 26.99 tok/s decode, returned
coherent output and accepted 111/111 drafts.

The 256K startup allocated about 5.8 GiB KV cache per GPU and exposed 480,827
KV tokens. During the boundary request the maximum sampled hotspot was 91°C;
all ECC counters remained zero and the inspected application and kernel logs
contained no OOM, illegal memory access, GPU reset, HSA/amdgpu error, AER or
machine-check event.

The ROCm 10 recipe carries 18 ordered vLLM patches. Its cache-page geometry,
slot guards, bounded indexer workspaces, sharded DFlash projection and staged
OCP-MX dequantization are covered by focused repository tests and by the 32K
and 256K runtime gates. Detailed identities, measurements and artifact hashes
are recorded in
`profiles/dev/glm53-flash-rocm10/results/qualification-20260905.md` and
`profiles/dev/glm53-flash-rocm10/results/qualification-256k-20260906.md`.
