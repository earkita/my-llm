# ROCm 10 DFlash2 K7 boundary qualification — 2026-09-06

Result: the development ROCm 10 recipe started DFlash2 K7 with a 256K BF16
KV context and completed exact 32K and 256K input-plus-output boundary
requests. This qualified configuration was later promoted as the
`glm53-flash-rocm` production default.

## Configuration

- Runtime mode: `dflash2-k7-256k`
- Runtime: `glm53-flash-quark-mxfp4-rocm10-mrv2-8xr9700-tp8-dflash2-k7-bf16kv-256k`
- Runtime profile SHA-256:
  `6bd8abbb29dcd91bb1ea8dcfcce2b2c3f30ad34c09b34307b98a8f45a5b49b87`
- Install manifest SHA-256:
  `f7474450c398ba641fb4d63d4ebf5bf94a2ada5846b07a1c1889318164ba2b74`
- `max_model_len=262144`, BF16 KV, scheduler chunk 512,
  `gpu_memory_utilization=0.97`, TP8/EP8, DFlash2 K7
- Concurrency 1, one measured request, no separate benchmark warmup, 128
  output tokens

The startup profiler allocated about 5.8 GiB KV cache per GPU, exposing
480,827 KV tokens and 1.83x maximum concurrency at 262,144 tokens. Reported
weights plus non-Torch memory was 23.89–23.91 GiB per GPU and peak activation
was 1.2 GiB.

## Results

| Boundary | Prompt + output | TTFT | Observed prefill | Decode | E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32K | 32,640 + 128 | 54.432 s | 599.65 tok/s | 23.24 tok/s | 59.896 s |
| 256K | 262,016 + 128 | 431.810 s | 606.78 tok/s | 26.99 tok/s | 436.515 s |

The production ROCm 7.14 reference recorded 599.39 observed prefill tok/s and
23.84 decode tok/s for the same 262,016 + 128 boundary. The single ROCm 10
run is 1.2% faster in observed prefill and 13.2% faster in decode, but one
sample is not a statistical performance claim.

The 256K request proposed and accepted 111/111 draft tokens. API identity,
health, model, context limit, literal output, and exact usage checks passed
before both measurements.

## Device and error checks

- GPU power cap: 225 W on all eight cards.
- During prefill, sampled socket power was 193–202 W, clocks were about
  3.07–3.09 GHz, and the maximum sampled hotspot was 91 C; the driver reported
  throttling.
- Immediately after completion, hotspot temperatures were 68–77 C and memory
  temperatures were 72–81 C.
- Correctable, uncorrectable, deferred, and cache ECC counts were all zero on
  every GPU.
- No application OOM, illegal memory access, GPU reset, HSA/amdgpu error, AER,
  or machine-check event appeared in the inspected runtime log or kernel
  journal.

## Evidence

- 32K API: `stage4-dflash2-k7-256k/32k/api-20260906T025424.json`, SHA-256
  `885dbc6a0509fbc685dc584787377414583b69e4127f81dbad8fd8ad7c4a095d`
- 32K benchmark: `stage4-dflash2-k7-256k/32k/benchmark-20260906T025424.json`,
  SHA-256 `b5735fa49843fe21b937f12042c62d6c93b58d25b944e772f0ad5c9bd6835a65`
- 256K API: `stage4-dflash2-k7-256k/256k/api-20260906T025534.json`, SHA-256
  `1431de25e43b51703db27e2998526cb8018ae2cf60052c33ae2a701ec4328689`
- 256K benchmark:
  `stage4-dflash2-k7-256k/256k/benchmark-20260906T025534.json`, SHA-256
  `21678ce11467cc1c8012ed6274a7a5c2bb25e94b4e0586e96cdbf30c713c6809`
- Live runtime log:
  `logs/runtime/vllm-glm53-flash-quark-mxfp4-rocm10-mrv2-8xr9700-tp8-dflash2-k7-bf16kv-256k-20260906T025146.log`

The runtime log has no final digest in this historical capture because the
service was still active when the report was written. It was subsequently
stopped cleanly.
