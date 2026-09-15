# GLM-5.3-Flash W4A16 on 8x R9700

Results below were measured on 15 September 2026 with the development profile
`dev/glm53-flash-new-dflash`, vLLM recipe `vllm_glm53flashrocm10_v0.31`, TP8,
FP8 KV cache, and concurrency 1. Decode comparisons use the raw completion
endpoint, greedy sampling, exact output lengths, and the official vLLM Bench
client. Results with a different endpoint or cache state are not compared.

The selected 256K/4096 configuration is promoted without experimental modes as
the production profile `glm53-flash-new`. Its dedicated LiteLLM and Claude Code
alias is `glm-5.3-flash-new-high`.

## Runtime selection

| experiment | result | decision |
| --- | ---: | --- |
| DFlash2 K4 eager, 128 -> 512 | 50.8 decode tok/s | DFlash beats matched MTP for short coding decode |
| MTP N3 eager, 128 -> 512 | 38.9 decode tok/s | rejected for the short coding workload |
| DFlash `FULL_DECODE_ONLY`, 128 -> 512 | 43.7 decode tok/s | rejected |
| DFlash `FULL_AND_PIECEWISE`, 128 -> 512 | 90.1 decode tok/s | selected |
| same graphs with PyNccl/RCCL wrapper | 84.3 decode tok/s | rejected; 6.4% slower |
| request custom/AITER all-reduce | vLLM resolved it to disabled; backend list `[]` | unavailable on `gfx1201`; not forced |

The selected graph mode requires `VLLM_USE_BREAKABLE_CUDAGRAPH=1` for this GLM
class. Without it, vLLM rejects piecewise graphs because the model is not
torch-compiled.

## Chunk size and 256K context

| context/workload | batch tokens | TTFT | prefill | decode | DFlash acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32,768 + 128, three cold requests | 4,096 | 27.668 s | 1,184.3 tok/s | 61.9 tok/s | 76.8% |
| 32,768 + 128, three cold requests | 8,192 | 27.947 s | 1,172.5 tok/s | 67.7 tok/s | 86.5% |
| 262,016 + 128, full boundary | 4,096 | 214.085 s | 1,223.9 tok/s | 35.7 tok/s | 22.9% |
| 262,016 + 128, full boundary | 8,192 | 216.438 s | 1,210.6 tok/s | 35.9 tok/s | 22.2% |

At the full 256K boundary, 8,192 was 1.1% slower for prefill and only 0.6%
faster for decode. The selected mode therefore keeps 4,096. Its 2.2 GB fixed
FP8 KV reservation provides 317,682 tokens of measured GPU KV capacity, or
1.21 full 262,144-token requests.

The near-boundary prefix test used two requests of 262,080 input tokens and 32
output tokens. Cold TTFT was 214.441 s and warm TTFT was 2.729 s, a 78.6x
speedup. This validates automatic prefix caching at 256K for one retained
prefix.

Artifacts:

- `logs/benchmarks/glm53-flash-new-dflash-dflash-full-piecewise-256k/primary/20260915T013221-llm-bench/summary.json`
- `logs/benchmarks/glm53-flash-new-dflash-dflash-full-piecewise-256k-batch8192/primary/20260915T014009-llm-bench/summary.json`
- `logs/benchmarks/glm53-flash-new-dflash/primary/20260915T014812+0200/summary.json`

## Async scheduling and concurrent sequences

The concurrency comparison used the raw completion endpoint, 8,192 input
tokens, 256 forced output tokens, greedy sampling, `--seed-base 424242`
(resolved vLLM Bench seed `427543`), and three complete waves for every
concurrency. The C1 runs therefore contain 3 requests, C2 contains 6, and C4
contains 12. This avoids averaging a partial final batch.

| scheduler / max sequences | requests | TTFT mean / p95 | decode per request | aggregate output | E2E mean / p95 | DFlash acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sync / C1 | 3 | 4.802 / 6.429 s | 88.3 tok/s | 33.3 tok/s | 7.689 / 9.416 s | 92.7% |
| async / C1 | 3 | 4.833 / 6.471 s | 87.0 tok/s | 33.0 tok/s | 7.763 / 9.467 s | 91.7% |
| async / C2 | 6 | 6.023 / 7.682 s | 22.8 tok/s | 29.6 tok/s | 17.199 / 20.156 s | 88.9% |
| async / C4 | 12 | 12.276 / 18.297 s | 13.9 tok/s | 31.9 tok/s | 30.559 / 39.170 s | 71.3% |

Async C1 was 1.5% slower in decode, 0.9% lower in aggregate output throughput,
and 1.0% slower end to end than sync C1. Increasing `max_num_seqs` did not
recover throughput: async C2 and C4 were respectively 11.0% and 4.3% below the
sync C1 aggregate while materially increasing per-request latency. No runtime
preemption was observed. Greedy generated-output arrays were not byte-identical
between the sync and async C1 restarts, so async also requires a semantic
correctness qualification before any future promotion; this observation alone
does not distinguish scheduling behavior from nondeterministic ROCm kernels.

Decision: retain synchronous scheduling and `max_num_seqs=1` in production.
Keep the async C1/C2/C4 modes diagnostic-only in
`profiles/dev/glm53-flash-new-dflash.json`.

Final comparison artifacts:

- `logs/benchmarks/glm53-flash-new-concurrency/final-seed424242/sync-c1/llm-bench-20260915T095701/summary.json`
- `logs/benchmarks/glm53-flash-new-concurrency/final-seed424242/async-c1/llm-bench-20260915T095228/summary.json`
- `logs/benchmarks/glm53-flash-new-concurrency/final-seed424242/async-c2/llm-bench-20260915T094136/summary.json`
- `logs/benchmarks/glm53-flash-new-concurrency/final-seed424242/async-c4/llm-bench-20260915T094639/summary.json`

## Remaining decode opportunities

The active 256K log proves that sparse MLA uses a native Triton HIP `gfx1201`
kernel named `_rdna4_fp8_paged_mqa_logits_kernel`; its generated assembly uses
`v_wmma_f32_16x16x16_fp8_fp8`. W4A16 MoE selects
`CompressedTensorsWNA16MoEMethod` and `TritonWNA16Experts`. However, vLLM
reports that the exact R9700 tuning file
`E=288,N=256,device_name=AMD_Radeon_R9700,dtype=int4_w4a16.json` is missing and
uses its default MoE configuration.

The next controlled experiments should be:

1. DFlash K1/K2 versus K4 and target-only at 256K, because K4 acceptance falls
   to about 23% at the boundary.
2. Tune and validate the exact R9700 W4A16 MoE configuration, then retain it
   only after end-to-end and correctness A/B.
3. Prewarm the observed first-request Triton JIT shapes to remove cold latency.
4. Capture a `rocprofv3` trace before changing attention or GEMM backends.

The profile's AITER FP8/FP4 BMM flags and `ROCBLAS_USE_HIPBLASLT=0` are not
evidence that hardware is unused. They guard alternate implementations whose
correctness and speed on this checkpoint are not established. The
`VLLM_ROCM_USE_AITER_FP4_ASM_GEMM` setting is reported as unknown by this vLLM
build and can be removed in a separate configuration cleanup.
