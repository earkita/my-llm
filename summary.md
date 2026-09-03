# Production profile validation and smoke benchmark

Date: 2026-09-02 (Europe/Warsaw)

## Outcome

All three self-contained profiles in `profiles/production/` were validated,
started sequentially, tested through their OpenAI-compatible API, benchmarked,
and stopped gracefully. The managed runtime and its persistent systemd user
unit were inactive after the run.

The repository is a Python 3.12 control plane for pinned, isolated inference
recipes. `./run` is its public entry point; each production profile embeds its
model, runtime, and stack configuration. Recipe builds and generated state live
outside version-controlled configuration, under `.runtime/` and `logs/`.

## Test host

- CPU: AMD Ryzen Threadripper PRO 7955WX, 16 cores / 32 threads
- RAM: 345 GiB
- GPUs: 8 x AMD Radeon AI PRO R9700 (`gfx1201`)
- GPU power cap verified before every start: 270 W on all eight R9700 cards
- Kernel: Linux 7.0.0-29-generic x86_64
- An NVIDIA RTX 3090 is also installed, but is intentionally excluded from the
  production profiles.

## Profile validation

| Profile | Backend and layout | Context limit | Checkpoint verification | API smoke test | Final lifecycle |
|---|---|---:|---|---|---|
| `glm53-flash` | llama.cpp, 8 GPUs | 1,048,576 | PASS: 6 GGUF shards, 199,707,321,347 bytes, plus verified DFlash artifact | PASS, 6/6 checks | start/ready/stop PASS |
| `qwen38-flash` | vLLM, TP8 + EP | 262,144 | PASS: 131 safetensors shards, 185,502,232,570 tensor bytes | PASS, 6/6 checks | start/ready/stop PASS |
| `deepseek-v4-flash` | vLLM, TP1 x PP6 | 1,048,576 | PASS: 48 safetensors shards, 166,886,535,336 file bytes | PASS, 6/6 checks | start/ready/stop PASS |

Runtime recipe manifests and installed artifacts were hash-verified. Runtime
import tests passed for all three recipes, host doctor checks passed for each
profile, and all installation dry runs completed successfully.

## Fast benchmark

The same low-cost smoke workload was used for every profile:

```text
prompt tokens: 256
output tokens: 64
concurrency: 1
measured repetitions: 2
warm-up repetitions: 1
timeout: 600 seconds
```

| Profile | Mean TTFT (s) | Mean E2E (s) | Observed input (tok/s) | Mean decode (tok/s) | Min decode (tok/s) | Aggregate output (tok/s) |
|---|---:|---:|---:|---:|---:|---:|
| `glm53-flash` | 0.992 | 2.247 | 257.98 | 50.22 | 49.91 | 28.48 |
| `qwen38-flash` | 0.347 | 2.715 | 738.41 | 26.76 | 24.75 | 23.68 |
| `deepseek-v4-flash` | 0.585 | 2.569 | 437.49 | 31.76 | 31.72 | 24.92 |

Raw benchmark artifacts:

- [GLM benchmark](logs/smoke/glm53-flash/benchmark-20260902T145040.json)
- [Qwen benchmark](logs/smoke/qwen38-flash/benchmark-20260902T150459.json)
- [DeepSeek benchmark](logs/smoke/deepseek-v4-flash/benchmark-20260902T151237.json)
- [GLM API test](logs/smoke/glm53-flash/api-20260902T145040.json)

The observed input rate is derived from client-observed time to first token.
These figures are a startup and regression smoke test, not a capacity or
long-context benchmark. The small sample size should not be used to rank the
models for production workloads.

## GLM prefix-cache regression check

On 2026-09-03, the GLM llama.cpp recipe was rebuilt with a HIP-only multi-GPU
device-context fix derived from upstream llama.cpp PR #21170. Prompt caching
was then enabled with `--cache-prompt --cache-reuse 0 --cache-ram 0` and tested
against the production 8-GPU process.

An identical 14,416-token request was issued repeatedly. The cold request
evaluated all 14,416 prompt tokens in 69.53 seconds (207.34 tokens/s). Each
measured warm request reported 14,412 cached tokens and evaluated only four new
prompt tokens:

| Warm request | Client E2E (s) | Cached tokens | Evaluated prompt tokens | Prompt eval (ms) |
|---:|---:|---:|---:|---:|
| 1 | 0.302 | 14,412 | 4 | 210.014 |
| 2 | 0.276 | 14,412 | 4 | 195.828 |
| 3 | 0.274 | 14,412 | 4 | 195.459 |

The server survived every repeat, remained ready afterward, and the LiteLLM
chat-completion probe passed through alias `glm-5.3-flash-high`. This directly
covers the former second-request ROCm illegal-memory-access failure.

## Changes required to run the profiles

- Corrected code paths that invoked the Bash `run` entry point through Python.
- Added an optional `R9700_RECIPE_ROOT` so exact hash-matched recipe builds can
  be reused from shared storage without duplicating roughly 25 GiB locally.
- Made model verification inspect the current checkpoint instead of trusting a
  previously written source record. Safetensors validation now compares actual
  tensor payload bytes and excludes container headers.
- Updated Qwen and DeepSeek GPU BDF ordering for the host's current PCIe
  placement, keeping the slower relocated links at the end of the layout.
- Added a narrowly scoped ROCm Triton bootstrap for the vLLM parent and every
  spawned worker. This is required on the mixed AMD/NVIDIA host because both
  Triton drivers otherwise report active and vLLM disables Triton. Two Qwen
  startup attempts exposed the parent/worker distinction; the final run passed
  after the worker bootstrap was added.
- Improved persistent keeper shutdown handling so a normal managed stop is not
  misreported as an unexpected runtime exit.
- Added regression tests for shared recipe roots, live checkpoint revalidation,
  safetensors byte semantics, vLLM entry-point construction, and worker Triton
  bootstrapping.
- Added a HIP-scoped llama.cpp multi-GPU device-context patch and enabled GLM
  common-prefix prompt caching after a repeated-request regression test.
- Made the LiteLLM health test validate all aliases exposed by the active flat
  production profile and probe the profile's configured Claude model alias.

## Commands exercised

```bash
./run test unit
make dry-run
./run test runtime --profile PROFILE
./run model verify --profile PROFILE
./run doctor --profile PROFILE
skills/start-r9700-runtime/scripts/start-runtime.sh --profile PROFILE
./run test api --profile PROFILE --timeout 600 --output OUTPUT
./run benchmark --profile PROFILE --prompt-tokens 256 --output-tokens 64 \
  --concurrency 1 --repetitions 2 --warmup 1 --timeout 600 --output OUTPUT
skills/stop-r9700-runtime/scripts/stop-runtime.sh
make check
```

## Remaining observations

- DeepSeek's first start took about seven minutes because its six stages
  compiled and warmed custom kernels. This was active compilation, not a hang.
- The final Qwen and DeepSeek logs contain no OOM or fatal runtime error. vLLM
  warns that some R9700-specific FP8/MoE tuning files are absent and falls back
  to default kernels; targeted tuning could improve performance.
- Qwen also reports that its speculative-token scheduling may be suboptimal.
  That setting was left unchanged because this run establishes correctness and
  a smoke baseline rather than retuning the production profile.
- No long-context, concurrent-load, soak, or power-efficiency benchmark was run.
