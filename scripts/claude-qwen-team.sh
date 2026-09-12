#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(dirname -- "$script_dir")
agents="$repo_root/.claude/agents.local.json"

[[ -f $agents ]] || {
  printf 'Qwen team agents are absent: %s\n' "$agents" >&2
  printf '%s\n' 'Activate the combined stack first: ./run launcher start qwen-multi' >&2
  exit 1
}

exec "$script_dir/claude-local.sh" --agent qwen-team-lead "$@"
