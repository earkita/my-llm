#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
unit=r9700-litellm-proxy.service
ready_timeout=120
dry_run=0

while (($#)); do
  case "$1" in
    --ready-timeout)
      (($# >= 2)) || { printf 'missing value for %s\n' "$1" >&2; exit 2; }
      ready_timeout=$2
      shift 2
      ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help)
      printf 'Usage: %s [--ready-timeout SECONDS] [--dry-run]\n' "$0"
      exit 0
      ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

[[ $ready_timeout =~ ^[1-9][0-9]*$ ]] || {
  printf '%s\n' '--ready-timeout must be a positive integer' >&2
  exit 2
}

keeper=("$script_dir/keep-proxy.sh")
command=(systemd-run --user --unit "$unit" --collect --service-type=exec --property=KillMode=mixed --property=TimeoutStopSec=60 "${keeper[@]}")

if ((dry_run)); then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

cd -- "$repo_root"
[[ -x .runtime/litellm/venv/bin/litellm ]] || {
  printf '%s\n' 'LiteLLM is not installed; run ./run proxy install first' >&2
  exit 1
}
./run service status >/dev/null 2>&1 || {
  printf '%s\n' 'inference backend is not ready; start it before LiteLLM' >&2
  exit 1
}
if systemctl --user is-active --quiet "$unit"; then
  printf 'persistent unit is already active: %s\n' "$unit" >&2
  exit 1
fi
"${command[@]}"

deadline=$((SECONDS + ready_timeout))
while ((SECONDS < deadline)); do
  status=$(./run proxy status 2>&1) && {
    printf '%s unit=%s\n' "$status" "$unit"
    exit 0
  }
  if systemctl --user is-failed --quiet "$unit" || ! systemctl --user is-active --quiet "$unit"; then
    printf 'persistent unit failed before readiness: %s\n' "$unit" >&2
    journalctl --user-unit "$unit" -n 80 --no-pager >&2 || true
    exit 1
  fi
  sleep 1
done

printf 'proxy readiness timed out after %ss; unit remains active: %s\n' "$ready_timeout" "$unit" >&2
exit 1
