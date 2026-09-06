# ROCm 10 development qualification — 2026-09-05

Result: all four staged gates passed at 32K configured context on eight R9700
GPUs. DFlash2 K7 is the best qualified mode. The service was stopped cleanly
after testing. The qualified recipe was later promoted as `glm53-flash-rocm`.

## Tested identity

- Host: Ubuntu 26.04.1 LTS, kernel `7.0.0-31-generic`, ROCk module
  `7.1.3.31500000`
- GPUs: 8 x AMD Radeon AI PRO R9700 (`gfx1201`); NVIDIA remained the display
  adapter and was not reset or modified
- Python: 3.12.3
- ROCm SDK: 10.0.0; runtime log: `ROCm 10.0.0.0-9999-6b0e43f3`,
  `HIP 7.15.26333`, `RCCL 2.30.4-HEAD:6b0e43f`
- PyTorch: 2.13.0+rocm10.0.0; torchvision: 0.28.0+rocm10.0.0
- Triton: 3.8.0+git4cff872c.rocm10.0.0
- vLLM: `0.1.dev1+g7fbd44cbe.d20260905.rocm100`, source
  `7fbd44cbe0a90b9c8fd3a94a0f0401ac4b1bc719`
- AITER: `0.1.1.dev1+g7ff5155f3`, source
  `7ff5155f3ba772e534b6cf8dddc0099932327b9b`
- AMD SMI Python: 27.0.0+6b0e43f3
- Target checkpoint: `amd/GLM-5.3-Flash-Quark-MXFP4` revision
  `b5688f25491202978c19c4d036eef579f61bbe07`
- DFlash2 checkpoint: `incoai/GLM-5.3-Flash-DFlash2` revision
  `bf582e4eacc1810f76656d1811693ff6c6737d2a`; weight SHA-256
  `b038e1d9d1e7833fa3880c2c0135ba9b673013f03da1b29fb831931584759dac`
- Final install manifest SHA-256:
  `f7474450c398ba641fb4d63d4ebf5bf94a2ada5846b07a1c1889318164ba2b74`
- Patched vLLM full-index diff SHA-256:
  `b5235df9bca775a33f5486123fb9584e63874b6034a3d2fc1c91553336ca96cc`

The isolated recipe uses ROCm 10 wheel libraries from the recipe-local virtual
environment. It does not use Docker or the system ROCm user-space stack.

## Common gate

Every mode used TP8/EP8, BF16 KV, MRV2, eager execution, one sequence, 32K
configured context, `max_num_batched_tokens=2048`, and three measured requests
after one warmup. Each measured request had exactly 4096 input and 128 output
tokens at concurrency one. API checks covered managed identity, health, served
model, context limit, deterministic literal output, and exact usage; all
passed.

Reported prefill is client-observed prompt tokens divided by TTFT. Reported
decode excludes TTFT and is not a standalone kernel microbenchmark.

## Performance

| Mode | Prefill tok/s | Decode mean/min tok/s | TTFT mean/p95 s | E2E mean/p95 s | Draft acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| target-only | 931.86 | 3.754 / 3.704 | 4.395 / 4.397 | 38.231 / 38.633 | n/a |
| native MTP K1 | 921.18 | 6.927 / 6.897 | 4.446 / 4.449 | 22.781 / 22.861 | 258/258, 100.00% |
| DFlash2 K1 | 867.81 | 6.952 / 6.906 | 4.720 / 4.728 | 22.990 / 23.101 | 261/262, 99.62% |
| DFlash2 K7 | 868.34 | 26.213 / 26.104 | 4.717 / 4.722 | 9.562 / 9.585 | 465/504, 92.26% |

DFlash2 K7 delivered 6.98x the target-only client-observed decode rate. Its
72 draft rounds accepted a mean 6.46 draft tokens; accepted tokens by draft
position were `71, 69, 69, 64, 64, 64, 64`.

## Capacity and device observations

| Mode | Weights + non-Torch / GPU | Peak activation / GPU | KV cache / GPU | KV tokens | Temperature evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| target-only | 23.31 GiB | 1.33 GiB | 4.35 GiB | 356,352 | maximum observed hotspot 87 C during load/test |
| native MTP K1 | 24.32 GiB | 1.33 GiB | 3.34 GiB | 240,517 | post-run hotspots at or below 62 C |
| DFlash2 K1 | 24.12 GiB | 1.41 GiB | 3.46 GiB | 230,486 | no separate snapshot retained; no thermal or driver fault |
| DFlash2 K7 | 24.10 GiB | 1.41 GiB | 3.48 GiB | 165,024 | post-run hotspot maximum 65 C |

ECC counters were zero after the speculative runs. No application OOM, GPU
reset, HSA fault, AER event, or amdgpu fault appeared in the qualification
logs or the checked kernel journal. One MTP-era kernel warning reported
`svm_range_deferred_list_work` holding a workqueue for more than 10 ms; the
run completed and no reset or memory fault followed it.

## Evidence files

| Stage | Committed artifacts | Runtime log and final SHA-256 |
| --- | --- | --- |
| target-only | `stage1-target/api-20260905T144344.json` (`5be23820...`), `stage1-target/benchmark-20260905T144344.json` (`9ae352ee...`) | `logs/runtime/vllm-glm53-flash-quark-mxfp4-rocm10-mrv2-8xr9700-tp8-target-bf16kv-32k-20260905T143846.log`, `1fecfc3dced187ca9d8024d7250aabc035e44d5bdb8dd491a8cee34bbe448f8f` |
| native MTP K1 | `stage2-mtp-k1/api-20260905T150702.json` (`fb244992...`), `stage2-mtp-k1/benchmark-20260905T150702.json` (`fe385894...`) | `logs/runtime/vllm-glm53-flash-quark-mxfp4-rocm10-mrv2-8xr9700-tp8-mtp-k1-bf16kv-32k-20260905T150254.log`, `2c9982aa492ae0216b9b8d5956a67f89086c426ed98786e22cd35b4eabee53c8` |
| DFlash2 K1 | `stage3a-dflash2-k1/api-20260905T151143.json` (`c1fadd98...`), `stage3a-dflash2-k1/benchmark-20260905T151143.json` (`d416b3b1...`) | `logs/runtime/vllm-glm53-flash-quark-mxfp4-rocm10-mrv2-8xr9700-tp8-dflash2-k1-bf16kv-32k-20260905T150924.log`, `5e3a22664fb4540e1cd7b7d96acee072c4d2c6a8a90b4c69f8e2eb5cd1999ae0` |
| DFlash2 K7 | `stage3b-dflash2-k7/api-20260905T151608.json` (`f7199fc6...`), `stage3b-dflash2-k7/benchmark-20260905T151608.json` (`3b78ac42...`) | `logs/runtime/vllm-glm53-flash-quark-mxfp4-rocm10-mrv2-8xr9700-tp8-dflash2-k7-bf16kv-32k-20260905T151349.log`, `9fe5b0f34873649b2072919f1e517cf6221795adeb85e2fcdfea8befe67bf7cc` |

The target-only run predates patch 0017 and has install manifest
`976a01d8...`; patch 0017 touches only Quark block-FP8 MoE used by native MTP.
The final recipe was rebuilt from the same source with all 17 patches before
MTP and DFlash qualification.

## Patch-specific test notes

- All repository unit tests passed.
- The complete recipe built successfully and its lock/import checks passed.
- Focused sparse/kpool tests passed except one ROCm 10 seed-9 expectation that
  differs by one raw FP8 cache code in two bytes; tail and scale values match,
  and the runtime writer invariant plus all runtime gates passed. It was not
  patched merely to hide a backend rounding difference.
- Patch 0017 passes `ruff check`, `ruff format --check`, applies after the
  first 16 patches, and passed the stronger full checkpoint MTP load/decode
  gate. The upstream Quark test module itself was not collected in the recipe
  environment because optional `lm_eval` is not installed.

## Start the best mode

```bash
skills/start-r9700-runtime/scripts/start-runtime.sh \
  --profile glm53-flash-rocm \
  --runtime-mode dflash2-k7
```

Stop it gracefully before switching modes:

```bash
skills/stop-r9700-runtime/scripts/stop-runtime.sh
```
