#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(dirname -- "$script_dir")
settings="$repo_root/.claude/settings.local.json"

[[ -f $settings ]] || {
  printf 'Claude Code settings are absent: %s\n' "$settings" >&2
  printf '%s\n' 'Start a stack preset first, for example: ./run stack start --preset qwen38-flash' >&2
  exit 1
}

export MY_LLM_REPO_ROOT="$repo_root"
exec claude --settings "$settings" "$@"
