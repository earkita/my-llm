# Production profiles and GLM-5.3 qualification summary

Date: 2026-09-04 (Europe/Warsaw)

## Outcome

The repository still contains exactly three flat production profiles. The
GLM-5.3 recipe now builds from an immutable snapshot of the official vLLM
`main` after PR #53906 was merged, rather than from the contributor's
`glm-release` branch.

The target-only Quark/MXFP4 path remains the default production mode. DFlash2
K7 is now functional on 8 x R9700: it loads, passes every OpenAI API gate,
returns coherent text and accepts draft tokens. It remains an explicit
experimental mode because three required upstream fixes are not merged and
full-context, concurrent and deterministic-losslessness qualification is not
complete.

## Pinned GLM stack

- hardware: 8 x AMD Radeon AI PRO R9700 32 GB (`gfx1201`), ROCm 7.14;
- target: `amd/GLM-5.3-Flash-Quark-MXFP4` revision
  `b5688f25491202978c19c4d036eef579f61bbe07`;
- target data: 62 safetensors shards and 185,066,521,464 tensor bytes;
- vLLM: official `vllm-project/vllm` main snapshot
  `c7e6e36fa93a5b8cb95b74fa96e4abdf2f0be51d`, containing the squash merge
  `98ed0856f31fa3aaf5e27464e2b4ef5a8ee6b2f5` of PR #53906;
- runner/layout: MRV2 forced by `VLLM_USE_V2_MODEL_RUNNER=1`, TP8 + EP8,
  BF16 KV, 32,768 context, no CPU offload and prefix cache disabled;
- target attention: `ROCM_AITER_MLA_SPARSE`; DFlash attention:
  `TRITON_ATTN` for the required non-causal path;
- Quark uses the correctness-first emulation path on RDNA4; native AITER FP4
  BMM and ASM GEMM remain disabled;
- DFlash2 artifact: `incoai/GLM-5.3-Flash-DFlash2` revision
  `bf582e4eacc1810f76656d1811693ff6c6737d2a`, SHA-256
  `b038e1d9d1e7833fa3880c2c0135ba9b673013f03da1b29fb831931584759dac`.

During the final audit, upstream main advanced to
`bc2ee480738d7dcc558262a0c6d81956b515b050` by two performance commits that do
not address GLM/DFlash correctness. The tested `c7e6e36` snapshot is retained
so the committed recipe exactly matches the built and qualified image.

## DFlash2 root cause and fixes

The DFlash checkpoint, its five auxiliary layer IDs and the GLM mHC
`residual=None` contract were correct. The principal local regression was an
unsafe cache overlay: the earlier custom patches let draft tensors and live
target MLA pages use incompatible physical strides. They also suppressed an
upstream layout assertion. Those patches were removed rather than carried
forward.

The current image applies four focused corrections:

1. `0010` gives DFlash pages the target MLA physical block stride and aliases
   only matching global block IDs. Draft and target pages no longer overlap
   across different block IDs.
2. `0012` is the focused fix from vLLM PR #55239, routing rope-free BF16
   multi-token verification to ragged Triton instead of an unsupported AITER
   sparse-MLA path.
3. `0017` ports the ring semantics of PR #55219 commit `de63c847` without its
   broad generic-layout refactor. K7 uses a 12-slot tail ring, preserving
   committed pool state across rejected speculative rows and redo.
4. `0018` is PR #55201: unwritten top-k pool IDs start at `-1`, and both the
   fallback and fused AMD/NVIDIA expansion paths reject IDs outside the number
   of completed pools.

The full draft PR #55219 is intentionally not included. Its generic packed
layout refactor is larger than required and has no upstream ROCm/MTP end-to-end
validation. PR #54163 and the experimental causal-convolution change from
#52905 also remain excluded because controlled A/B tests reduced correctness or
acceptance on this stack.

The ring slot-mapping suite passed 14/14 CPU tests. Selected real gfx1201
kernel tests passed 10/10; the deterministic K7 regression proves that 4- and
8-slot tails corrupt the redo while the 12-slot ring matches the reference.
The focused GPU regression added by PR #55201 was also run after freeing the
serving GPUs and passed 1/1.

## Functional evidence

All captures below used the same frozen 20-token prompt, greedy sampling and
64 requested output tokens on an otherwise idle server.

| Runtime | API gate | Accepted / drafted | Mean acceptance evidence | Result |
|---|---:|---:|---:|---|
| target-only | 6/6 | n/a | n/a | coherent response; prefill and decode work |
| DFlash2 K1, two captures | 6/6 | 52/74 (70.3%) | 1.62-1.80 tokens/target step | coherent responses |
| DFlash2 K7, three captures | 6/6 | 139/385 (36.1%) | 3.32-3.71 tokens/target step | coherent responses; all seven draft positions accepted |

The final fresh K7 restart produced 46/119 accepted drafts (38.7%), a mean
acceptance length of 3.71 and a coherent prime-number answer. The active log
contains no runtime failure; all eight workers initialized, and `/health` and
OpenAI endpoints return HTTP 200.

Exact token hashes can differ between identical seeded target-only restarts on
this EP/emulation stack, including at the first token. Therefore cross-restart
token equality is retained as a diagnostic signal but is not treated as the
sole DFlash gate. API correctness, coherent output, isolated Prometheus deltas
and explicit acceptance counters are the current qualification evidence.

## Benchmarks

The new DFlash measurement used 256 input tokens, 128 output tokens,
concurrency 1, three measured repetitions and one warm-up:

| Runtime | Mean TTFT | p95 TTFT | Mean E2E | p95 E2E | Observed prefill | Mean decode | Minimum decode | Aggregate output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GLM DFlash2 K7 | 0.573 s | 0.576 s | 5.967 s | 6.373 s | 446.60 tok/s | 23.63 tok/s | 21.65 tok/s | 21.52 tok/s |

The previous fast smoke results for the other production profiles are retained
for operational comparison. Their workload was 256 input + 64 output tokens,
concurrency 1, two repetitions after one warm-up, so they are not a strict
apples-to-apples comparison with the DFlash row.

| Profile/runtime | Date | Mean TTFT | Mean E2E | Observed prefill | Mean decode |
|---|---|---:|---:|---:|---:|
| GLM Quark target-only | 2026-09-03 | 0.574 s | 17.399 s | 445.74 tok/s | 3.74 tok/s |
| Qwen3.8 Flash-Next MTP K2 | 2026-09-02 | 0.347 s | 2.715 s | 738.41 tok/s | 26.76 tok/s |
| DeepSeek V4 Flash DSpark K5 | 2026-09-02 | 0.585 s | 2.569 s | 437.49 tok/s | 31.76 tok/s |

Client-observed prefill includes HTTP and scheduling time; these short tests
are regression checks, not capacity rankings.

## Runtime commands

```bash
# default production path
./run launcher start glm53-flash

# validated but explicit DFlash2 K7 path
./run launcher start glm53-flash --runtime-mode dflash2

# diagnostic controls
./run launcher start glm53-flash --runtime-mode dflash2-k1
./run launcher start glm53-flash --runtime-mode extract-hidden-states-k1

# validate and benchmark the already-running K7 identity
skills/measure-r9700-model/scripts/test-and-benchmark.sh \
  --profile glm53-flash --runtime-mode dflash2 \
  --prompt-tokens 256 --output-tokens 128 \
  --concurrency 1 --repetitions 3 --warmup 1
```

LiteLLM is optional. Add `--with-litellm` only when a proxy endpoint or Claude
Code integration is wanted.

## Evidence and remaining limits

- API artifacts are under `logs/validation/`; the benchmark is
  `logs/benchmarks/glm53-flash-dflash2-k7-c1-256x128-main-c7e6.json`.
- Bounded DFlash captures are under `.runtime/diagnostics/glm-dflash/`.
  Generated state and logs are intentionally ignored by git.
- GLM was qualified at a configured 32K limit, but a full 32K request was not
  run in DFlash mode.
- DFlash concurrency above one and prefix caching were not qualified.
- The full nearby GPU kernel fuzz file passed 29/30 cases; seed 9 retains a
  one-byte difference in the quantized KV cache. The focused ring and invalid
  pool-ID regressions pass, and no corresponding E2E quality failure was seen,
  but the byte-level discrepancy remains tracked rather than hidden.
- PR #55239 and PR #55201 remain open; PR #55219 remains a draft. Until their
  final upstream forms are known, K7 stays opt-in.

## Upstream status at final audit

- PR #53906 is merged into official vLLM main.
- PR #54373, which fixes the DFlash RoPE configuration, is already in main.
- PR #55239 (`d608ced8445a21ec3bffa01b73a889e914aff2fd`) is open.
- PR #55219 (`d004ff11c7e4f2cb7dc040efb76f62291c5d3fb6`) is a draft; only the focused
  ring behavior from `de63c84731cd719a6ab5c9e8d73289c9c96a7587` is ported.
- PR #55201 (`40bf4af864fb4e0fd3f84c5650ca9ba465c31ac8`) is open.
- Issues #49559, #54928, #54451 and #53323 remain open. The current R9700
  result demonstrates a working local path but does not make those broader
  upstream reports resolved.
