# Qualification results

This directory stores concise, committed evidence for the ROCm 10 development
profile. Large raw runtime logs remain under the repository's ignored `logs/`
directory; reports here should identify the corresponding log path and digest.

Qualification order:

| Stage | Runtime mode | State |
| --- | --- | --- |
| 1 | target-only default | passed |
| 2 | `native-mtp-k1` | passed |
| 3a | `dflash2-k1` | passed |
| 3b | `dflash2-k7` | passed; best qualified decode throughput |

Each result must record exact profile/manifest hashes, Python, ROCm, PyTorch,
vLLM and AITER identities, visible GPU count/architecture, request token counts,
prefill/decode throughput, speculative acceptance when applicable, GPU memory,
temperatures/power, and kernel/AER events observed during the run.

The consolidated report is `qualification-20260905.md`. Machine-readable API
and benchmark artifacts remain in the per-stage subdirectories.

The later exact 32K/256K DFlash2 K7 boundary qualification is recorded in
`qualification-256k-20260906.md` and `stage4-dflash2-k7-256k/`.
