#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
recipe=${QWEN38_RECIPE:-vllm_qwen38flash_pr53896}
recipe_root="$repo_root/.runtime/recipes/$recipe"
python="$recipe_root/venv/bin/python"
rocm_sdk="$recipe_root/venv/bin/rocm-sdk"

[[ -x $python ]] || { printf 'missing recipe Python: %s\n' "$python" >&2; exit 1; }
[[ -x $rocm_sdk ]] || { printf 'missing recipe ROCm SDK helper: %s\n' "$rocm_sdk" >&2; exit 1; }
rocm_root=$($rocm_sdk path --root)

export ROCM_HOME="$rocm_root"
export ROCM_PATH="$rocm_root"
export LD_LIBRARY_PATH="$rocm_root/share/amd_smi/amdsmi:$rocm_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TRITON_DEFAULT_BACKEND=amd
export PYTORCH_ROCM_ARCH=gfx1201
export GPU_ARCHS=gfx1201

exec "$python" "$script_dir/validate-safetensors-checkpoint.py" "$@"
