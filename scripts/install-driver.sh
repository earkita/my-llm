#!/usr/bin/env bash
set -Eeuo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
readarray -t driver < <(python3 - "$root/manifest/runtime.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))["kernel_driver"]
print(value["filename"])
print(value["url"])
print(value["sha256"])
PY
)
filename=${driver[0]}
url=${driver[1]}
sha256=${driver[2]}
download_dir="$root/.runtime/downloads"
work_dir="$root/.runtime/driver-work"
installer="$download_dir/$filename"

if [[ ${1:-} == --dry-run ]]; then
  printf 'download %s\nverify sha256=%s\n' "$url" "$sha256"
  printf 'default action: validate only; --install performs privileged driver installation\n'
  exit 0
fi

mkdir -p "$download_dir" "$work_dir"
curl --fail --location --continue-at - --output "$installer" "$url"
printf '%s  %s\n' "$sha256" "$installer" | sha256sum --check --strict
(
  cd "$work_dir"
  bash "$installer" version
  bash "$installer" buildinfo
)

if [[ ${1:-} != --install ]]; then
  printf 'Driver bundle validated. Re-run with sudo %q --install to install it.\n' "$0"
  exit 0
fi
[[ $EUID == 0 ]] || {
  printf '%s\n' '--install requires root; use sudo scripts/install-driver.sh --install' >&2
  exit 1
}
printf '%s\n' 'Installing the pinned AMDGPU driver. Reboot is required afterward.'
(
  cd "$work_dir"
  bash "$installer" deps=install amdgpu assumeyes
)
