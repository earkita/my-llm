#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
unit=r9700-litellm-proxy.service
timeout=30
dry_run=0

while (($#)); do
  case "$1" in
    --timeout)
      (($# >= 2)) || { printf 'missing value for %s\n' "$1" >&2; exit 2; }
      timeout=$2
      shift 2
      ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help)
      printf 'Usage: %s [--timeout SECONDS] [--dry-run]\n' "$0"
      exit 0
      ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

[[ $timeout =~ ^[1-9][0-9]*$ ]] || {
  printf '%s\n' '--timeout must be a positive integer' >&2
  exit 2
}

if ((dry_run)); then
  printf '%q ' "$repo_root/run" proxy stop --timeout "$timeout"
  printf '\n'
  printf '%q ' systemctl --user stop "$unit"
  printf '\n'
  exit 0
fi

cd -- "$repo_root"
set +e
./run proxy status >/dev/null 2>&1
proxy_rc=$?
set -e
unit_active=0
systemctl --user is-active --quiet "$unit" && unit_active=1

if ((proxy_rc == 3 && unit_active == 0)); then
  printf '%s\n' 'already stopped; persistent unit is inactive'
  exit 0
fi
if ((proxy_rc != 3)); then
  ./run proxy stop --timeout "$timeout"
fi
if ((unit_active)); then
  if ! systemctl --user stop "$unit"; then
    # A transient --collect unit can disappear immediately after the managed
    # proxy exits. Treat that race as success only when the unit is no longer
    # active; preserve genuine systemd stop failures.
    systemctl --user is-active --quiet "$unit" && exit 1
  fi
fi

set +e
./run proxy status >/dev/null 2>&1
after_rc=$?
set -e
if ((after_rc != 3)) || systemctl --user is-active --quiet "$unit"; then
  printf '%s\n' 'LiteLLM shutdown verification failed' >&2
  exit 1
fi
printf 'stopped; persistent unit inactive: %s\n' "$unit"
