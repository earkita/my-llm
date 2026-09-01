#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
log_path=
lines=120

usage() {
  printf 'Usage: %s [--log PATH] [--lines N]\n' "$0"
}

while (($#)); do
  case "$1" in
    --log|--lines)
      (($# >= 2)) || { printf 'missing value for %s\n' "$1" >&2; exit 2; }
      if [[ $1 == --log ]]; then log_path=$2; else lines=$2; fi
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ $lines =~ ^[1-9][0-9]*$ ]] || { printf '%s\n' '--lines must be a positive integer' >&2; exit 2; }
cd -- "$repo_root"

if [[ -z $log_path ]]; then
  log_path=$(find logs/runtime -maxdepth 1 -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)
fi
[[ -n $log_path && -f $log_path ]] || { printf 'runtime log not found: %s\n' "${log_path:-<none>}" >&2; exit 1; }

failure_pattern='traceback|fatal|segmentation fault|segfault|out of memory|hipErrorOutOfMemory|memoryerror|oom-kill|killed process|received signal|terminated by signal|sigsegv|sigkill|sigabrt|sigterm|aborted|exception|(^|[[:space:]])error([:[:space:]])'
shutdown_pattern='shutting down|shutdown complete|finished server process|graceful shutdown'
ready_pattern='application startup complete|GET /health HTTP/[^ ]+" 200'

printf 'LOG=%s\n' "$log_path"
printf '%s\n' 'SERVICE_STATUS:'
./run service status || true
printf '%s\n' 'RECORDED_STATE:'
if [[ -f .runtime/service.json ]]; then
  sed -n '1,160p' .runtime/service.json
else
  printf '%s\n' '<absent>'
fi

if rg -n -i "$shutdown_pattern" "$log_path" >/dev/null; then
  classification='graceful shutdown markers found'
elif rg -n -i "$failure_pattern" "$log_path" >/dev/null; then
  classification='failure markers found in application log'
elif rg -n -i "$ready_pattern" "$log_path" >/dev/null; then
  classification='ready without in-log failure; external termination is possible if PID is absent'
else
  classification='inconclusive; no recognized readiness, shutdown, or failure marker'
fi
printf 'CLASSIFICATION=%s\n' "$classification"

printf '%s\n' 'MATCHED_EVIDENCE:'
rg -n -i "$failure_pattern|$shutdown_pattern|$ready_pattern" "$log_path" | tail -n 80 || printf '%s\n' '<none>'
printf 'FINAL_%s_LINES:\n' "$lines"
tail -n "$lines" "$log_path"
