#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(dirname -- "$script_dir")
env_file="$repo_dir/.env"

litellm_key=${LITELLM_MASTER_KEY:-}
if [[ -z "$litellm_key" && -r "$env_file" ]]; then
  litellm_key=$(sed -n 's/^LITELLM_MASTER_KEY=//p' "$env_file" | tail -n 1)
fi
[[ -n $litellm_key ]] || {
  printf '%s\n' 'LITELLM_MASTER_KEY is missing from the environment and .env' >&2
  exit 1
}

printf '%s' "$litellm_key"
