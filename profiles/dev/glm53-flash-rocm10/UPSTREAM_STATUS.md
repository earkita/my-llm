# Upstream and local patch status

Selected bases:

- vLLM main `7fbd44cbe0a90b9c8fd3a94a0f0401ac4b1bc719` (2026-09-05)
- AITER v0.1.21 `7ff5155f3ba772e534b6cf8dddc0099932327b9b`
- ROCm SDK wheels 10.0.0 and PyTorch 2.13.0+rocm10.0.0

The ROCm 10 recipe carries 17 vLLM patches, compared with 24 in the ROCm
7.14 production recipe. No AITER patch is needed: v0.1.21 already contains
the ROCm 10 sampling and hipCUB compatibility changes missing from the old
AITER pin. AMD SMI Python 27.0.0+6b0e43f3 comes from the pinned SDK because
PyPI's 7.0.2 bindings require a symbol absent from ROCm 10's
`libamd_smi.so.27`.

## Comparison with the ROCm 7.14 recipe

| Old patch | ROCm 10 disposition | Reason |
| --- | --- | --- |
| 0001 | omitted | Quark per-block linear `weight_scale` support merged in vLLM PR #54770. The separate MoE gap found during MTP qualification is handled narrowly by new 0017. |
| 0002 | new 0002 | GLM's checkpoint-specific loader still does not accept both Quark and native FP8 scale names. |
| 0003, 0009, 0010, 0013 | replaced by new 0001 | Current PR #55423 supplies one coherent Eagle3/mHC and DFlash KV-layout implementation on selected main. |
| 0005 | new 0003 | Still required on this mixed NVIDIA-display/AMD-compute host. |
| 0007 | refreshed as new 0004 | Still required around the current `MixtureOfExperts` and `SupportsEagle3` declarations. |
| 0008 | new 0005 | Current main still excludes the RDNA4 AITER kpool path. |
| 0012 | exact current PR as new 0006 | PR #55239 remains open. |
| 0015 | omitted | Its scale handling is upstream; its FP4 BMM guard is unreachable because AITER FP4 BMM is disabled. |
| 0016 | omitted | Bounded diagnostics are not runtime correctness; metrics and logs provide the evidence. |
| 0017-0023 | new 0007-0013 | Speculative rollback, bounds, page geometry, and bounded workspace fixes remain absent from selected main. |
| 0024, 0025 | omitted | They are FP8 KV-only; this profile qualifies BF16 KV. |
| 0026-0028 | new 0014-0016 | Still needed to fit and run GLM/DFlash on 32 GB R9700 cards. |
| none | new 0017 | Runtime qualification exposed missing Quark 128x128 block-FP8 MoE dispatch for the checkpoint's MTP layer. |

This removes obsolete, diagnostic-only, disabled-path, and FP8-KV patches
instead of copying the old patch stack wholesale.

## Already merged upstream

The selected vLLM base already contains:

- PR #53906, GLM sparse MLA support;
- PR #54826, MRV2 draft-attention handling;
- PR #54770, Quark scale support for the existing linear path;
- commit `5093e4844a75f279d1921d7e34ccd23d68529f27`, DFlash
  sliding-window AOT handling.

The selected AITER v0.1.21 base already contains the relevant ROCm 10
sampling and hipCUB changes, so its source remains unpatched.

## Applied upstream PR patches

| ID | Upstream source and purpose | Verification / removal condition |
| --- | --- | --- |
| 0001 | Architecture-neutral port of PR #55423 head `374e367e`; enables GLM-5.3 DFlash2, contracted mHC auxiliary states, and safe draft cache-page overlay. | DFlash K1/K7 API and throughput gates passed. Remove when equivalent support reaches the pinned base. |
| 0006 | Exact PR #55239 head `0f1d78db`; routes speculative verification through ragged sparse MLA. | MTP and DFlash gates passed. Remove when the PR lands. |
| 0007 | Architecture-neutral backport of PR #55219 commit `de63c847`; preserves kpool state across speculative rollback. | Kpool rollback tests and rejected-draft runtime path. Remove when equivalent semantics land. |
| 0008 | Backport of PR #55201 head `40bf4af8`; rejects invalid kpool indices. | Invalid-index unit gate and runtime decode. Remove when the bounds fix lands. |
| 0012 | Backport of PR #54296 head `191f82d7`; guards block-table loads used to build slot mappings. | CPU/GPU guard tests and 4096-token prompt gates. Remove when equivalent bounds checks land. |
| 0013 | Backport of PR #55222 head `8eda002d`; sizes GLM indexer prefill storage in compressed states. | Allocation inspection and 32K runtime load. Remove when the PR lands. |

## Locally authored compatibility patches

| ID | Purpose | Verification / removal condition |
| --- | --- | --- |
| 0002 | Accept Quark `weight_scale` and native `weight_scale_inv` names in the GLM projection loader. | Target checkpoint load and deterministic API output. Remove when the GLM loader accepts both upstream. |
| 0003 | Make `VLLM_TARGET_DEVICE=rocm` authoritative and select Triton's AMD backend on a mixed NVIDIA-display/AMD-compute host. | Import probe, doctor, and eight-R9700 visibility. Remove when upstream explicit selection is equivalent. |
| 0004 | Map fused GLM gate/up modules to Quark's separate overrides. | Checkpoint load and quantization selection. Remove when upstream maps them. |
| 0005 | Permit AITER kpool indexing on RDNA4. | Target and speculative prefill/decode on gfx1201. Remove after upstream RDNA4 qualification. |
| 0009 | Cap sparse-indexer prefill workspace by scheduler capacity rather than context alone. | Workspace-size check and 32K load. Remove when upstream bounds it equivalently. |
| 0010 | Align ROCm GLM kpool storage with supported paged-MQA geometry for issue #55280. | Layout checks and all API gates. Remove when upstream resolves the contract. |
| 0011 | Advertise actual compressed GLM kpool kernel page sizes for issue #54359. | Layout/unit and prompt-boundary gates. Remove when the kernel/backend contract is upstream. |
| 0014 | Size decode logits/workspace in compressed kpool states. | Shape checks and sustained decode. Remove when upstream uses compressed sizing. |
| 0015 | Replace the replicated DFlash auxiliary projection with a reduced row-parallel projection. | TP8 DFlash K1/K7 load and output gates. Remove when upstream shards it without changing checkpoint semantics. |
| 0016 | Dequantize OCP MX expert weights one GEMM at a time and release temporaries between stages. | MoE checks and target/DFlash loading within 32 GB. Remove when upstream bounds the peak or gfx1201 gains a qualified native route. |
| 0017 | Add Quark 128x128 block-FP8 MoE dispatch, scale shapes, and kernel configuration for the checkpoint's native MTP layer. | Focused unit source, formatting, full MTP load, API output, and sustained K1 decode. Remove when Quark block-FP8 MoE support lands upstream. |

Patch 0017 is deliberately narrower than the broader GLM-5.3 Quark naming
discussion in issue #54547: it only fills the missing MoE scheme already
supported by vLLM's generic block-FP8 fused-MoE path.

## Investigated but not applied

- PR #55340 head `c83572a3` constrains speculative draft backend selection by
  existing KV layouts. DFlash explicitly pins `TRITON_ATTN`, and all runtime
  gates passed without it.
- PR #51915 is unnecessary because the affected AITER FP4 BMM route is
  explicitly disabled.
- PR #54163 remains excluded because the earlier local A/B run regressed token
  zero with DFlash K1; prefix caching also remains disabled.
- Issue #48568 describes an older GLM-5.2 MTP/RCCL failure. Native MTP
  completed real speculative decoding here, so no speculative all-gather
  patch was imported without a reproduced fault.
- Old FP8 KV patches remain excluded because this profile qualifies BF16 KV.
- Old bounded-diagnostic patches remain excluded because they do not affect
  runtime correctness.

## Remaining blockers

There is no known blocker for the qualified 32K development scope. Promotion
is intentionally blocked on explicit user approval, not on a runtime failure.
The 17-patch upstream delta remains maintenance debt, and contexts beyond 32K,
FP8 KV, higher concurrency, and longer thermal soak tests are not qualified by
this result.
