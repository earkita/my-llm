#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
unit=r9700-runtime.service
timeout=180
dry_run=0

unit_has_processes() {
  local control_group cgroup_procs first_pid
  control_group=$(
    systemctl --user show "$unit" --property=ControlGroup --value 2>/dev/null
  ) || return 1
  [[ $control_group == */$unit ]] || return 1
  cgroup_procs="/sys/fs/cgroup${control_group}/cgroup.procs"
  [[ -r $cgroup_procs ]] || return 1
  IFS= read -r first_pid < "$cgroup_procs" || return 1
  [[ -n $first_pid ]]
}

wait_for_empty_cgroup() {
  local remaining=$1
  while unit_has_processes && ((remaining > 0)); do
    sleep 1
    remaining=$((remaining - 1))
  done
  ! unit_has_processes
}

usage() {
  printf 'Usage: %s [--timeout SECONDS] [--dry-run]\n' "$0"
}

while (($#)); do
  case "$1" in
    --timeout)
      (($# >= 2)) || { printf 'missing value for %s\n' "$1" >&2; exit 2; }
      timeout=$2
      shift 2
      ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ $timeout =~ ^[1-9][0-9]*$ ]] || {
  printf '%s\n' '--timeout must be a positive integer' >&2
  exit 2
}

stop_command=("$repo_root/run" service stop --timeout "$timeout")
unit_command=(systemctl --user stop "$unit")
if ((dry_run)); then
  printf '%q ' "${stop_command[@]}"
  printf '\n'
  printf '%q ' "${unit_command[@]}"
  printf '\n'
  exit 0
fi

cd -- "$repo_root"
set +e
status=$(./run service status 2>&1)
status_rc=$?
set -e
unit_active=0
systemctl --user is-active --quiet "$unit" && unit_active=1

if ((status_rc == 3 && unit_active == 0)) && ! unit_has_processes; then
  printf '%s\n' 'already stopped; persistent unit is inactive'
  exit 0
fi

if ((status_rc != 3)); then
  printf 'before: %s\n' "$status"
  set +e
  "${stop_command[@]}"
  stop_rc=$?
  set -e
  if ((stop_rc != 0)); then
    printf '%s\n' 'managed SIGINT timed out; continuing with systemd cgroup SIGINT' >&2
  fi
fi

if ((unit_active)); then
  # The unit uses KillMode=control-group, KillSignal=SIGINT and
  # SendSIGKILL=no. Even if the direct managed stop timed out, this signals
  # every remaining worker without PID guessing and can never escalate to
  # SIGKILL. The transient unit can disappear while SIGINT is completing.
  systemctl --user is-active --quiet "$unit" && "${unit_command[@]}"
fi

if unit_has_processes; then
  printf '%s\n' 'signaling remaining service cgroup with SIGINT' >&2
  systemctl --user kill --kill-who=all --signal=SIGINT "$unit"
  sigint_timeout=$((timeout / 3))
  ((sigint_timeout > 0)) || sigint_timeout=1
  wait_for_empty_cgroup "$sigint_timeout" || true
fi

if unit_has_processes; then
  printf '%s\n' 'service cgroup ignored SIGINT; escalating to cgroup SIGTERM' >&2
  systemctl --user kill --kill-who=all --signal=SIGTERM "$unit"
  sigterm_timeout=$((timeout / 3))
  ((sigterm_timeout > 0)) || sigterm_timeout=1
  wait_for_empty_cgroup "$sigterm_timeout" || true
fi

if unit_has_processes; then
  printf '%s\n' 'service cgroup ignored SIGTERM; using final cgroup SIGHUP' >&2
  systemctl --user kill --kill-who=all --signal=SIGHUP "$unit"
  sighup_timeout=$((timeout - sigint_timeout - sigterm_timeout))
  ((sighup_timeout > 0)) || sighup_timeout=1
  wait_for_empty_cgroup "$sighup_timeout" || true
fi

set +e
after=$(./run service status 2>&1)
after_rc=$?
set -e
if ((after_rc != 3)) || systemctl --user is-active --quiet "$unit" || unit_has_processes; then
  printf 'shutdown verification failed: service=%s unit=%s\n' "$after" "$unit" >&2
  exit 1
fi

printf 'stopped; persistent unit inactive: %s\n' "$unit"
