# Production profiles and GLM-5.3 qualification summary

Date: 2026-09-04 (Europe/Warsaw)

## Outcome

The repository still contains exactly three flat production profiles. The
GLM-5.3 recipe now builds from an immutable snapshot of the official vLLM
`main` after PR #53906 was merged, rather than from the contributor's
`glm-release` branch.

The default GLM path is now MRV2, TP8/EP8, DFlash2 K7, BF16 KV and a 262,144
token context on 8 x R9700. It loads, passes every OpenAI API gate, returns
coherent text and accepts drafts. An exact `262016 + 128` request qualified the
full configured boundary at concurrency one. Target-only 32K remains an
explicit fallback; 400K and higher-capacity modes remain diagnostic.

An opt-in target-only FP8 mode now allocates the complete 1,048,576-token
context without CPU offload. The engine completed model load, cache allocation
and warm-up with a measured capacity of 1,187,115 tokens. This is a capacity
result, not yet an inference qualification: the run was stopped before API
readiness after new correctable PCIe `BadTLP` events appeared.

## Pinned GLM stack

- hardware: 8 x AMD Radeon AI PRO R9700 32 GB (`gfx1201`), ROCm 7.14;
- target: `amd/GLM-5.3-Flash-Quark-MXFP4` revision
  `b5688f25491202978c19c4d036eef579f61bbe07`;
- target data: 62 safetensors shards and 185,066,521,464 tensor bytes;
- vLLM: official `vllm-project/vllm` main snapshot
  `c7e6e36fa93a5b8cb95b74fa96e4abdf2f0be51d`, containing the squash merge
  `98ed0856f31fa3aaf5e27464e2b4ef5a8ee6b2f5` of PR #53906;
- runner/layout: MRV2 forced by `VLLM_USE_V2_MODEL_RUNNER=1`, TP8 + EP8,
  PP1, draft TP8, DFlash2 K7, BF16 KV, 262,144 context, no CPU offload and
  prefix cache disabled;
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

The current image applies the following focused corrections:

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
5. `0019` replaces the fixed `max_model_len * 40` indexer prefill workspace
   with the exact scheduler bound `max_model_len * min(40, max_num_seqs)`,
   which makes single-request long-context startup practical.
6. `0020`, following issue #55280, aligns ROCm GLM kpool storage to
   `index_kpool * 64`. With `index_kpool=4`, the hybrid allocator selects a
   768-token attention block containing 192 compressed states.
7. `0021` advertises the actual 128/256-token GLM kernel pages corresponding
   to 32/64 compressed states. A 768-token storage block is consequently
   represented by three 256-token virtual pages in the block table.
8. `0022`, from PR #54296, guards block-table loads and returns `PAD_ID` for an
   out-of-range virtual page. It is defense in depth, not a substitute for the
   corrected geometry in `0021`.

The old failure boundaries corroborated the geometry error. With a 768-token
storage block treated as one table entry but addressed in 256-token kernel
pages, the effective table ended at about 87,552 tokens for a 256K profile and
136,704 for a 400K profile. The 134,000-token control stayed below the latter;
the previous 262,016- and 409,000-token requests crossed their respective
thresholds and faulted.

The full draft PR #55219 is intentionally not included. Its generic packed
layout refactor is larger than required, changes code unrelated to the observed
failure and has no upstream ROCm/MTP end-to-end validation. The focused
`de63c847` ring behavior passes the local regression suite; carrying the rest
would add unqualified patch surface without evidence that it fixes another
local failure. PR #54163 and the experimental causal-convolution change from
#52905 also remain excluded because controlled A/B tests reduced correctness or
acceptance on this stack.

The ring slot-mapping suite passed 14/14 CPU tests. Selected real gfx1201
kernel tests passed 10/10; the deterministic K7 regression proves that 4- and
8-slot tails corrupt the redo while the 12-slot ring matches the reference.
The focused GPU regression added by PR #55201 was also run after freeing the
serving GPUs and passed 1/1.

## FP8 MLA 1M capacity probe

The `long-context-1m-fp8` mode is MRV2, TP8, target-only, has no CPU offload or
prefix caching, and requests `--kv-cache-dtype fp8 --max-model-len 1048576`.
It uses the existing vLLM FP8 cache writer and scale tensors. On gfx1201, where
the pinned AITER build has no compatible sparse-MLA code object, patch `0025`
routes the rope-free reader through the existing Triton sparse path and applies
the vLLM per-tensor `k_scale` before BF16 attention arithmetic. It does not
introduce another FP8 representation.

The checkpoint does not contain calibrated MLA KV scales. The runtime therefore
used explicit finite FP32 `q_scale=1` and `k_scale=1`, and logged that fact. The
representation and kernel path are valid, but the unit scales make a BF16
quality comparison mandatory before this mode can be promoted.

Measured startup evidence:

| Target-only runtime | Model load / GPU | Peak activation / GPU | Allocated cache / GPU | Reported token capacity | Configured boundary |
|---|---:|---:|---:|---:|---:|
| BF16 KV baseline | 21.24 GiB | 1.82 GiB | 5.88 GiB | 538,269 | 524,288 |
| FP8 KV probe | 21.23 GiB | 1.76 GiB | 6.67 GiB | 1,187,115 | 1,048,576 |

The rows use different requested GPU utilization (`0.97` and `0.99`), so the
allocated GiB values are not themselves the compression ratio. At exactly 1M
tokens, the 11-layer 512-element MLA latent cache is 11.00 GiB/GPU in BF16 and
5.50 GiB/GPU in FP8. The already-quantized four-token indexer occupies another
0.354 GiB/GPU in either mode. This reduces those context-proportional cache
elements by 48.4%, from 11.354 GiB to 5.854 GiB before the small tail and page
rounding. The measured usable token capacity increased by 2.21x after the FP8
conversion and indexer workspace corrections.

Runtime allocation audit confirmed:

- target MLA latent: `torch.uint8` backing, `cache_dtype=fp8`,
  `quant_mode=1`; 6.27 GiB logical allocation at the reported 1.19M-token
  capacity;
- indexer/kpool: existing packed E4M3 values plus FP32 UE8M0 scale,
  `torch.uint8`; 0.40 GiB at the allocated capacity and one state per four
  tokens;
- kpool tail: BF16, 0.02 GiB. It holds the small incomplete four-token pooling
  tail consumed by the existing writer, so converting it would save little and
  would require changing that contract;
- recurrent/Mamba state views: INT8, not BF16. Their logged logical views share
  the hybrid backing allocation and must not be added to the physical 6.67 GiB;
- DFlash draft cache: absent from this target-only probe. DFlash modes retain an
  explicit BF16 draft cache instead of accidentally inheriting target FP8.

All 62 target shards loaded on all eight workers, the physical cache allocation
succeeded, and full engine initialization/warm-up completed. During that same
startup the kernel recorded 12 correctable PCIe Data-Link `BadTLP` events: 11
on the switch path to logical AMD GPU 5 (`c1:00.0` -> `c3:00.0`) and one on the
path to logical AMD GPU 7 (`e1:00.0` -> `e3:00.0`). There were no uncorrectable
AER events, GPU faults, AMDGPU resets, MCE/EDAC reports or OOM kills. Both links
returned to Gen5 x16, but the runtime was deliberately stopped with SIGTERM at
the safety gate. Consequently no OpenAI request, cache read/write end-to-end
comparison, NaN/Inf check or long-context retrieval test has yet passed.

A staged 32K retry at 21:07 used the same 1M FP8 runtime and was prepared for
exactly 32,640 input plus 128 output tokens. Before the request could be sent,
startup added 11 correctable `BadTLP` events: ten on the switch path to logical
AMD GPU 5 and one on a newly affected path to logical AMD GPU 6
(`c4:00.0` -> `c6:00.0`). The model again loaded all shards, allocated the
6.67 GiB/GPU cache and completed warm-up, but the safety monitor caused an
immediate graceful stop. There were again zero uncorrectable AER events, GPU
faults, resets, MCE/EDAC errors or OOMs, and no AER events appeared after the
service became inactive. Therefore the 32K inference gate is recorded as not
executed, not as passed or failed.

After the PCIe path is stable, qualification must proceed on this same mode at
32K, 128K, 256K, 512K, 768K and 1M, with identical deterministic BF16/FP8
prompts, output-quality and finite-value checks, a needle retrieval at each long
boundary, and an AER check after every stage. DFlash2 remains out of this gate
until target-only FP8 passes.

## Functional evidence

All captures below used the same frozen 20-token prompt, greedy sampling and
64 requested output tokens on an otherwise idle server.

| Runtime | API gate | Accepted / drafted | Mean acceptance evidence | Result |
|---|---:|---:|---:|---|
| target-only | 6/6 | n/a | n/a | coherent response; prefill and decode work |
| DFlash2 K1, two captures | 6/6 | 52/74 (70.3%) | 1.62-1.80 tokens/target step | coherent responses |
| DFlash2 K7, three captures | 6/6 | 139/385 (36.1%) | 3.32-3.71 tokens/target step | coherent responses; all seven draft positions accepted |

The short K7 qualification restart produced 46/119 accepted drafts (38.7%), a
mean acceptance length of 3.71 and a coherent prime-number answer. After the
page-geometry fixes, a fresh 256K runtime initialized all eight workers and
reported 421,649 tokens of GPU KV capacity. Its OpenAI API gate passed 6/6.

The capacity run then completed exactly 262,016 prompt tokens plus 128 output
tokens. It returned coherent output, reported exact total usage of 262,144 and
accepted 111/111 drafts. This is the production context qualification; it does
not imply concurrency-above-one or 400K stability.

The final 256K telemetry stream contains 1,832 per-GPU samples. Its P95 values
were 234 W socket power, 85°C edge, 103°C hotspot and 94°C memory; maxima were
370 W, 87°C, 109°C and 106°C. Static PPT0 still reported 285 W on all eight
cards before and after the run, although eight instantaneous power samples
exceeded that value. The hotspot maximum was only 1°C below its slowdown
threshold and memory was 2°C below its threshold, so this is a capacity and
correctness qualification rather than a thermal soak.

Exact token hashes can differ between identical seeded target-only restarts on
this EP/emulation stack, including at the first token. Therefore cross-restart
token equality is retained as a diagnostic signal but is not treated as the
sole DFlash gate. API correctness, coherent output, isolated Prometheus deltas
and explicit acceptance counters are the current qualification evidence.

## Benchmarks

The full-context qualification used concurrency 1, one measured repetition and
no warm-up:

| Runtime | Workload | TTFT | E2E | Observed prefill | Decode | Draft acceptance |
|---|---:|---:|---:|---:|---:|---:|
| GLM DFlash2 K7 256K | 262,016 + 128 | 437.137 s | 442.463 s | 599.39 tok/s | 23.84 tok/s | 111/111 |

The earlier short DFlash measurement used 256 input tokens, 128 output tokens,
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
# default production path: MRV2, TP8/EP8, DFlash2 K7, BF16 KV, 256K
./run launcher start glm53-flash

# compatibility alias for the same DFlash2 K7 runtime
./run launcher start glm53-flash --runtime-mode dflash2

# diagnostic controls
./run launcher start glm53-flash --runtime-mode target-only-32k
./run launcher start glm53-flash --runtime-mode dflash2-k1
./run launcher start glm53-flash --runtime-mode extract-hidden-states-k1

# diagnostic 1M target-only FP8 capacity/correctness mode
./run launcher start glm53-flash --runtime-mode long-context-1m-fp8

# validate the configured 256K boundary of an already-running default runtime
skills/measure-r9700-model/scripts/test-and-benchmark.sh \
  --profile glm53-flash --output-tokens 128 --full-context \
  --concurrency 1 --repetitions 1 --warmup 0
```

LiteLLM is optional. Add `--with-litellm` only when a proxy endpoint or Claude
Code integration is wanted.

## Evidence and remaining limits

- The full-boundary artifacts are
  `logs/validation/api-glm53-flash-long-context-256k-dflash2-20260904T122236.json`
  and
  `logs/benchmarks/glm53-flash-long-context-256k-dflash2-c1-262016x128-20260904T122236.json`.
  Generated logs are intentionally ignored by git.
- Bounded DFlash captures are under `.runtime/diagnostics/glm-dflash/`.
  Generated state and logs are intentionally ignored by git.
- GLM DFlash2 was qualified at its configured 256K boundary with concurrency
  one. DFlash concurrency above one, 400K and prefix caching are not qualified.
- Prefix caching remains disabled. PR #54163 caused a K1 token-zero regression
  in the local A/B despite prefix reuse not being needed for the test.
- The first 400K boundary attempt coincided with a fatal CPU/Data-Fabric MCE
  and host reset. Its repeat at a verified 285 W cap kept the host alive but
  ended in GPU `illegal memory access`. Both runs predate `0021`/`0022`; 400K
  was not repeated on the final image. The dedicated GPU telemetry stream
  peaked at 255 W, 82°C edge, 106°C hotspot and 88°C memory: margins of 30 W,
  28°C, 4°C and 20°C to the configured cap/reported critical thresholds. This
  is not evidence of a thermal trip, and the 400K mode remains diagnostic-only.
- The official `c7e6e36` GPU kernel test file passes 29/30 invocations: 19/20
  randomized seeds and all 10 deterministic variants. The locally extended
  file passes 35/36; both fail only seed 9. `max diff 1` is the magnitude of
  the raw FP8-code difference, not its count: exactly two of 270,336 KV-cache
  bytes differ, both by one adjacent FP8 code, while the scale and tail cache
  are exact. A 1,000-seed diagnostic found five such bytes across three seeds,
  with no scale or tail failures; the production prefill and decode writers
  agreed exactly for all 751 written vectors. The same seed-9 failure was
  reproduced against the otherwise clean official commit, so it is not
  introduced by the local ring or bounds patches and is retained as a strict
  cross-implementation FP8-oracle discrepancy rather than hidden.
- PR #55239 and PR #55201 remain open; PR #55219 remains a draft. Their focused
  fixes are pinned in the recipe even though K7 is now the default.

## Upstream status at final audit

- PR #53906 is merged into official vLLM main.
- PR #54373, which fixes the DFlash RoPE configuration, is already in main.
- PR #55239 is open; the recipe's focused backport comes from
  `d608ced8445a21ec3bffa01b73a889e914aff2fd`, while its audited head is
  `0f1d78db4495fed10384a2b048e70367e61d3dd1`.
- PR #55219 is a draft at audited head
  `6712aa109b856d2fefd75a28084db358eb7f9e1b`; only the focused ring behavior
  from `de63c84731cd719a6ab5c9e8d73289c9c96a7587` is ported.
- PR #55201 (`40bf4af864fb4e0fd3f84c5650ca9ba465c31ac8`) is open.
- Issues #49559, #54928, #54451 and #53323 remain open. The current R9700
  result demonstrates a working local path but does not make those broader
  upstream reports resolved.
