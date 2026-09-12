#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(dirname -- "$script_dir")
settings="$repo_root/.claude/settings.local.json"
agents="$repo_root/.claude/agents.local.json"

[[ -f $settings ]] || {
  printf 'Claude Code settings are absent: %s\n' "$settings" >&2
  printf '%s\n' 'Start a stack preset first, for example: ./run stack start --preset qwen38-flash' >&2
  exit 1
}

export MY_LLM_REPO_ROOT="$repo_root"
claude_args=(--settings "$settings")
if [[ -f $agents ]]; then
  agents_json=$(<"$agents")
  [[ -n $agents_json ]] || {
    printf 'Claude Code agents file is empty: %s\n' "$agents" >&2
    exit 1
  }
  claude_args+=(--agents "$agents_json")
fi
exec claude "${claude_args[@]}" "$@"
