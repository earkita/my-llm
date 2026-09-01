#!/usr/bin/env bash
set -Eeuo pipefail

packages=(
  build-essential ca-certificates cmake curl git git-lfs jq libdrm-dev
  libelf-dev libnuma-dev libpciaccess-dev libssl-dev ninja-build numactl
  pciutils pkg-config python3-dev python3-venv ripgrep
)

if [[ ${1:-} == --dry-run ]]; then
  printf 'sudo apt-get update\n'
  printf 'sudo apt-get install -y --no-install-recommends'
  printf ' %q' "${packages[@]}"
  printf '\n'
  exit 0
fi

[[ $EUID == 0 ]] || {
  printf 'run this command through sudo: sudo scripts/bootstrap-host.sh\n' >&2
  exit 1
}
apt-get update
apt-get install -y --no-install-recommends "${packages[@]}"
