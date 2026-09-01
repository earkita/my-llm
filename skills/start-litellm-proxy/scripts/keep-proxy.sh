#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
stopping=0

stop_proxy() {
  ((stopping == 0)) || return
  stopping=1
  "$repo_root/run" proxy stop --timeout 30 || true
  exit 0
}
trap stop_proxy INT TERM

cd -- "$repo_root"
"$repo_root/run" proxy start --wait

while "$repo_root/run" proxy status >/dev/null 2>&1; do
  sleep 5 &
  wait $!
done

printf '%s\n' 'managed LiteLLM proxy exited; keeper is stopping' >&2
exit 1
