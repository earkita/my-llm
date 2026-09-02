#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=${R9700_REPO_ROOT:-$(cd -- "$script_dir/../../.." && pwd)}
if [[ ! -x "$repo_root/run" && -x "$PWD/run" ]]; then
  repo_root=$(pwd -P)
fi
python="$repo_root/.venv/bin/python"
[[ -x "$python" ]] || {
  printf 'runtime environment is missing: %s\n' "$python" >&2
  exit 1
}

stopping=0
stop_managed_service() {
  ((stopping == 0)) || return
  stopping=1
  "$repo_root/run" service stop --timeout 180 || true
  exit 0
}
trap stop_managed_service INT TERM

cd -- "$repo_root"
"$repo_root/run" service start "$@" --wait

while "$repo_root/run" service status >/dev/null 2>&1; do
  sleep 5 &
  wait $!
done

for ((attempt = 0; attempt < 10; attempt++)); do
  if [[ ! -e "$repo_root/.runtime/service.json" ]]; then
    printf '%s\n' 'managed runtime stopped cleanly; persistent keeper is stopping'
    exit 0
  fi
  sleep 0.2
done

printf '%s\n' 'managed runtime exited; persistent keeper is stopping' >&2
exit 1
