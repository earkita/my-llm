#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(dirname -- "$script_dir")
python="$repo_root/.venv/bin/python"

[[ -x $python ]] || {
  printf 'Repository Python is unavailable: %s\n' "$python" >&2
  exit 1
}

exec "$python" "$script_dir/watch-claude-throughput.py" \
  --target all \
  --gpu \
  --dashboard \
  "$@"
