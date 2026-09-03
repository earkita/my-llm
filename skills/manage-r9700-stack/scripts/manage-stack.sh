#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
python="$repo_root/.venv/bin/python"
[[ -x $python ]] || {
  printf 'control environment is missing: %s\n' "$python" >&2
  exit 1
}

start_runtime="$repo_root/skills/start-r9700-runtime/scripts/start-runtime.sh"
stop_runtime="$repo_root/skills/stop-r9700-runtime/scripts/stop-runtime.sh"
start_proxy="$repo_root/skills/start-litellm-proxy/scripts/start-proxy.sh"
stop_proxy="$repo_root/skills/stop-litellm-proxy/scripts/stop-proxy.sh"
run="$repo_root/run"
profiles_dir="$repo_root/profiles/production"
claude_settings="$repo_root/.claude/settings.local.json"
runtime_state="$repo_root/.runtime/service.json"

preset=glm53-flash
runtime_ready_timeout=900
proxy_ready_timeout=120
runtime_stop_timeout=180
proxy_stop_timeout=30
required_power_cap_w=270
runtime_mode=
dry_run=0

usage() {
  cat <<EOF
Usage:
  $0 start [--preset NAME] [--runtime-ready-timeout SECONDS]
     [--proxy-ready-timeout SECONDS] [--runtime-mode NAME]
     [--required-power-cap-w WATTS] [--dry-run]
  $0 stop [--runtime-timeout SECONDS] [--proxy-timeout SECONDS] [--dry-run]
  $0 status
  $0 presets
EOF
}

action=${1:-}
case "$action" in
  start|stop|status|presets) shift ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

while (($#)); do
  case "$1" in
    --preset|--runtime-mode|--runtime-ready-timeout|--proxy-ready-timeout|--runtime-timeout|--proxy-timeout|--required-power-cap-w)
      (($# >= 2)) || { printf 'missing value for %s\n' "$1" >&2; exit 2; }
      case "$1" in
        --preset) preset=$2 ;;
        --runtime-mode) runtime_mode=$2 ;;
        --runtime-ready-timeout) runtime_ready_timeout=$2 ;;
        --proxy-ready-timeout) proxy_ready_timeout=$2 ;;
        --runtime-timeout) runtime_stop_timeout=$2 ;;
        --proxy-timeout) proxy_stop_timeout=$2 ;;
        --required-power-cap-w) required_power_cap_w=$2 ;;
      esac
      shift 2
      ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ $action == presets ]]; then
  "$python" - "$profiles_dir" <<'PY'
import json
import pathlib
import sys

for path in sorted(pathlib.Path(sys.argv[1]).glob("*.json")):
    profile = json.loads(path.read_text(encoding="utf-8"))
    marker = " (default)" if profile["name"] == "glm53-flash" else ""
    print(
        f"{profile['name']}{marker}\t{profile['status']}\t"
        f"{profile['description']}"
    )
PY
  exit
fi

if [[ $action == status ]]; then
  set +e
  "$run" service status
  runtime_rc=$?
  "$run" proxy status
  proxy_rc=$?
  set -e
  ((runtime_rc == 0 && proxy_rc == 0))
  exit
fi

profile_path="$profiles_dir/$preset.json"
[[ -f $profile_path ]] || {
  printf 'unknown production preset: %s\n' "$preset" >&2
  exit 2
}

for value in "$runtime_ready_timeout" "$proxy_ready_timeout" \
  "$runtime_stop_timeout" "$proxy_stop_timeout" "$required_power_cap_w"; do
  [[ $value =~ ^[1-9][0-9]*$ ]] || {
    printf '%s\n' 'timeouts and power cap must be positive integers' >&2
    exit 2
  }
done

runtime_start_command=(
  "$start_runtime"
  --profile "$preset"
  --ready-timeout "$runtime_ready_timeout"
  --required-power-cap-w "$required_power_cap_w"
)
if [[ -n $runtime_mode ]]; then
  runtime_start_command+=(--runtime-mode "$runtime_mode")
fi
proxy_start_command=("$start_proxy" --ready-timeout "$proxy_ready_timeout")
proxy_stop_command=("$stop_proxy" --timeout "$proxy_stop_timeout")
runtime_stop_command=("$stop_runtime" --timeout "$runtime_stop_timeout")

if ((dry_run)); then
  if [[ $action == start ]]; then
    printf '# preset %s\n' "$preset"
    printf 'materialize embedded Claude settings -> %q\n' "$claude_settings"
    "${runtime_start_command[@]}" --dry-run
    "${proxy_start_command[@]}" --dry-run
  else
    "${proxy_stop_command[@]}" --dry-run
    "${runtime_stop_command[@]}" --dry-run
  fi
  exit 0
fi

cd -- "$repo_root"

if [[ $action == stop ]]; then
  set +e
  "${proxy_stop_command[@]}"
  proxy_rc=$?
  "${runtime_stop_command[@]}"
  runtime_rc=$?
  set -e
  if ((proxy_rc != 0 || runtime_rc != 0)); then
    printf 'stack shutdown incomplete: proxy_rc=%d runtime_rc=%d\n' \
      "$proxy_rc" "$runtime_rc" >&2
    exit 1
  fi
  printf '%s\n' 'stack stopped: LiteLLM inactive; inference runtime inactive'
  exit 0
fi

[[ -x .runtime/litellm/venv/bin/litellm ]] || {
  printf '%s\n' 'LiteLLM is not installed; run ./run proxy install first' >&2
  exit 1
}

activate_claude_settings() {
  "$python" - "$profile_path" "$claude_settings" <<'PY'
import json
import os
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
profile = json.loads(source.read_text(encoding="utf-8"))
content = json.dumps(profile["stack"]["claude_settings"], indent=2) + "\n"
target.parent.mkdir(parents=True, exist_ok=True)
temporary = target.with_suffix(target.suffix + ".tmp")
temporary.write_text(content, encoding="utf-8")
os.replace(temporary, target)
print(f"Claude Code settings activated: {target}")
PY
}

assert_runtime_identity() {
  "$python" - "$runtime_state" "$profile_path" "$runtime_mode" <<'PY'
import json
import sys

state = json.load(open(sys.argv[1], encoding="utf-8"))
profile = json.load(open(sys.argv[2], encoding="utf-8"))
mode = sys.argv[3] or None
runtime_name = profile["runtime"]["name"]
if mode:
    try:
        runtime_name = profile["runtime"]["experimental_modes"][mode]["runtime_name"]
    except KeyError as exc:
        raise SystemExit(f"unknown experimental runtime mode: {mode}") from exc
expected = (
    profile["name"],
    profile["model"]["name"],
    runtime_name,
    mode,
)
actual = (
    state.get("profile"),
    state.get("model"),
    state.get("runtime"),
    state.get("runtime_mode"),
)
present = [index for index, value in enumerate(actual[:3]) if value is not None]
identity_matches = bool(present) and all(
    actual[index] == expected[index] for index in present
)
if mode is None:
    mode_matches = actual[3] in (None, "")
else:
    # Experimental modes cannot be inferred safely from a legacy state.
    mode_matches = actual[2] == expected[2] and actual[3] == mode
if not identity_matches or not mode_matches:
    raise SystemExit(
        "running inference identity differs from the production preset: "
        f"profile={actual[0]} model={actual[1]} runtime={actual[2]} "
        f"mode={actual[3]}; "
        f"expected profile={expected[0]} model={expected[1]} "
        f"runtime={expected[2]} mode={expected[3]}"
    )
PY
}

runtime_started=0
proxy_attempted=0
rollback_start() {
  set +e
  ((proxy_attempted == 0)) || "${proxy_stop_command[@]}" >&2
  ((runtime_started == 0)) || "${runtime_stop_command[@]}" >&2
  set -e
}

if runtime_status=$("$run" service status 2>&1); then
  assert_runtime_identity
  printf 'inference runtime already ready: %s\n' "$runtime_status"
else
  activate_claude_settings
  runtime_started=1
  "${runtime_start_command[@]}" || {
    failure_rc=$?
    rollback_start
    exit "$failure_rc"
  }
fi
((runtime_started != 0)) || activate_claude_settings

set +e
proxy_status=$("$run" proxy status 2>&1)
proxy_status_rc=$?
set -e
if ((proxy_status_rc == 0)); then
  printf 'LiteLLM already ready: %s\n' "$proxy_status"
else
  if ((proxy_status_rc == 2)) && [[ $proxy_status == stale-config* ]]; then
    "${proxy_stop_command[@]}"
  fi
  proxy_attempted=1
  "${proxy_start_command[@]}" || {
    failure_rc=$?
    rollback_start
    exit "$failure_rc"
  }
fi

"$run" proxy test || {
  failure_rc=$?
  rollback_start
  exit "$failure_rc"
}

printf 'stack ready: preset=%s inference=http://127.0.0.1:8000 litellm=http://127.0.0.1:4000\n' "$preset"
"$run" service status
"$run" proxy status
