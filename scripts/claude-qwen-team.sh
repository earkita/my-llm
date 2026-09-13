#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
default_repo_root=$(dirname -- "$script_dir")

# Prefer the repository from which Claude is being launched.
project_root=$(git -C "$PWD" rev-parse --show-toplevel 2>/dev/null || printf '%s' "$PWD")

project_agents="$project_root/.claude/agents.local.json"
default_agents="$default_repo_root/.claude/agents.local.json"

if [[ -f "$project_agents" ]]; then
    agents="$project_agents"
elif [[ -f "$default_agents" ]]; then
    agents="$default_agents"
else
    printf 'Qwen team agents are absent.\n' >&2
    printf 'Checked:\n' >&2
    printf '  %s\n' "$project_agents" >&2
    printf '  %s\n' "$default_agents" >&2
    printf '%s\n' 'Activate the combined stack first: ./run launcher start qwen-multi' >&2
    exit 1
fi

printf '[qwen-team] project: %s\n' "$project_root"
printf '[qwen-team] agents:  %s\n' "$agents"

exec "$script_dir/claude-local.sh" \
    --agents "$(cat "$agents")" \
    --agent qwen-team-lead \
    "$@"