#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=${MY_LLM_REPO_ROOT:-$(cd -- "$script_dir/../../.." && pwd)}
if [[ ! -x "$repo_root/run" && -x "$PWD/run" ]]; then
  repo_root=$(pwd -P)
fi
python="$repo_root/.venv/bin/python"
[[ -x $python ]] || {
  printf 'control environment is missing: %s\n' "$python" >&2
  exit 1
}

profile=glm53-flash
ready_timeout=900
required_power_cap_w=270
dry_run=0
extra=()
unit=r9700-runtime.service

usage() {
  printf 'Usage: %s [--profile NAME|PATH] [--runtime-mode NAME] [--model-directory PATH] [--host HOST] [--port PORT] [--ready-timeout SECONDS] [--required-power-cap-w WATTS] [--dry-run]\n' "$0"
}

while (($#)); do
  case "$1" in
    --profile|--runtime-mode|--model-directory|--host|--port|--ready-timeout|--required-power-cap-w)
      (($# >= 2)) || { printf 'missing value for %s\n' "$1" >&2; exit 2; }
      case "$1" in
        --profile) profile=$2 ;;
        --runtime-mode) extra+=("$1" "$2") ;;
        --ready-timeout) ready_timeout=$2 ;;
        --required-power-cap-w) required_power_cap_w=$2 ;;
        *) extra+=("$1" "$2") ;;
      esac
      shift 2
      ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ $ready_timeout =~ ^[1-9][0-9]*$ ]] || {
  printf '%s\n' '--ready-timeout must be a positive integer' >&2
  exit 2
}
[[ $required_power_cap_w =~ ^[1-9][0-9]*$ ]] || {
  printf '%s\n' '--required-power-cap-w must be a positive integer' >&2
  exit 2
}

keeper=(
  "$script_dir/keep-runtime.sh"
  --profile "$profile"
  "${extra[@]}"
  --ready-timeout "$ready_timeout"
)
command=(
  systemd-run --user --unit "$unit" --collect --service-type=exec
  --property=KillMode=control-group
  --property=KillSignal=SIGINT
  --property=SendSIGKILL=no
  --property=TimeoutStopSec=240
  --working-directory="$repo_root"
  "${keeper[@]}"
)
power_check=("$python" "$script_dir/check-power-cap.py" --watts "$required_power_cap_w")
host_check=("$repo_root/run" doctor --profile "$profile")

if ((dry_run)); then
  printf '%q ' "${host_check[@]}"
  printf '\n'
  printf '%q ' "${power_check[@]}"
  printf '\n'
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

cd -- "$repo_root"
if systemctl --user is-active --quiet "$unit"; then
  printf 'persistent unit is already active: %s\n' "$unit" >&2
  exit 1
fi
"${host_check[@]}"
"${power_check[@]}"
"${command[@]}"
deadline=$((SECONDS + ready_timeout))
while ((SECONDS < deadline)); do
  status=$(./run service status 2>&1) && {
    printf '%s unit=%s\n' "$status" "$unit"
    exit 0
  }
  if systemctl --user is-failed --quiet "$unit" \
      || ! systemctl --user is-active --quiet "$unit"; then
    printf 'persistent unit failed before readiness: %s\n' "$unit" >&2
    journalctl --user-unit "$unit" -n 80 --no-pager >&2 || true
    exit 1
  fi
  sleep 1
done

printf 'service readiness timed out after %ss; unit remains active: %s\n' \
  "$ready_timeout" "$unit" >&2
exit 1
