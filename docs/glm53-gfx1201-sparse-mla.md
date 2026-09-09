# GLM-5.3 sparse MLA on gfx1201

This development lane isolates the sparse-MLA experiment from the production
`glm53-flash` deployment. The production service remains on
`vllm_glm53flashrocm10_v0.29`; the experiment is selected only by the explicit
path `profiles/dev/glm53-flash-rocm10-gluon.json`.

## Upstream state on 2026-09-09

- AITER classifies Radeon AI PRO R9700 (`gfx1201`) as experimental. Triton,
  FlyDSL and selected HIP kernels are available, while most CK and assembly
  kernels remain CDNA-only: <https://github.com/ROCm/aiter>.
- AITER PR #4919 adds the GLM-5.3 rope-free sparse-MLA API, but its production
  kernel is gfx950-only and uses CDNA4 MFMA, wave64 and CDNA-specific memory
  operations: <https://github.com/ROCm/aiter/pull/4919>.
- AITER PR #4255 explicitly keeps gfx1201 paged-MQA on Triton because the
  existing AITER Gluon implementation does not support gfx1201:
  <https://github.com/ROCm/aiter/pull/4255>.
- The ROCm 10 Triton pin used by this repository does expose the RDNA4 Gluon
  `wmma` primitive. Patch `1001` therefore supplies a separate synchronous
  gather, wave32/WMMA prototype instead of pretending that the gfx950 kernel is
  portable.
- AITER PR #4188 demonstrates that native gfx1201 attention work is currently
  progressing most actively in FlyDSL. It covers dense diffusion attention,
  not the 512-wide sparse MLA layout required by GLM-5.3:
  <https://github.com/ROCm/aiter/pull/4188>.

The most visible R9700 vLLM enablement is around Qwen: AITER has a gfx1201
Qwen3.6-35B-A3B MoE stack in progress
(<https://github.com/ROCm/aiter/pull/5059>), and vLLM's opt-in RDNA4 FlyDSL
all-reduce was exercised with Qwen3.8-27B-FP8
(<https://github.com/vllm-project/vllm/pull/55917>). This is useful
infrastructure evidence, but it is not proof that GLM-5.3 sparse MLA is ready.

## Recipe boundary

`vllm_glm53flashrocm10_v0.30` contains the complete v0.29 patch image plus:

- AITER pinned to PR #4919 head `83075b6701d554cf096e35fae71777ffe3ef3aac`;
- patch `1001`, the gfx1201 rope-free BF16/FP8-KV Gluon prototype;
- patch `0026`, the strict vLLM opt-in dispatch;
- `VLLM_ROCM_USE_GLUON_SPARSE_MLA=1` only in the development profile.

The supported prototype geometry is deliberately narrow: `D=512`, no RoPE
tail, BF16 query and BF16 WMMA math, with BF16 KV or scalar-scaled FP8 KV. Any
other geometry fails before kernel launch.

## Qualification order

First validate the immutable install plan without building or touching the
running service:

```bash
./run install --profile profiles/dev/glm53-flash-rocm10-gluon.json --dry-run
```

Build the isolated recipe with the production service still running only when
host CPU/RAM headroom is acceptable:

```bash
./run install --profile profiles/dev/glm53-flash-rocm10-gluon.json --jobs 4
```

GPU qualification must not share the nearly full R9700 devices with the 1M
production runtime. After a separately authorized graceful stop, run:

```bash
.venv/bin/python scripts/qualify-gluon-sparse-mla.py --operator
```

Promotion gates, in order:

1. BF16 and FP8-KV operator correctness at top-k 17, 256 and 2048.
2. Repeated operator timing against the existing vLLM Triton kernel; retain the
   new route only if it wins without numerical drift.
3. Short API smoke and a controlled 32K prefill/decode A/B.
4. 256K NIAH with the same needle placements used by the production baseline.
5. Repeated 1M boundary qualification with power, temperature and system-error
   monitoring.

Until all gates pass, the profile stays under `profiles/dev`, and v0.29 remains
the default and rollback target.
