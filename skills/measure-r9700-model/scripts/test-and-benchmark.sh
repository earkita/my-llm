#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)

profile=glm53-flash
runtime_mode=
url=http://127.0.0.1:8000
prompt_tokens=256
full_context=0
output_tokens=256
concurrency=1
repetitions=5
warmup=1
timeout=600
output_dir=
dry_run=0
pin_data_parallel=0
data_parallel_rank=
prompt_variant_offset=0
telemetry=0
telemetry_interval=0.25

usage() {
  printf 'Usage: %s [--profile NAME|PATH] [--runtime-mode NAME] [--url URL] [--prompt-tokens N | --full-context] [--output-tokens N] [--concurrency N] [--repetitions N] [--warmup N] [--prompt-variant-offset N] [--timeout SECONDS] [--output-dir DIR] [--pin-data-parallel | --data-parallel-rank N] [--telemetry] [--telemetry-interval SECONDS] [--dry-run]\n' "$0"
}

while (($#)); do
  case "$1" in
    --profile|--runtime-mode|--url|--prompt-tokens|--output-tokens|--concurrency|--repetitions|--warmup|--prompt-variant-offset|--timeout|--output-dir|--data-parallel-rank|--telemetry-interval)
      (($# >= 2)) || { printf 'missing value for %s\n' "$1" >&2; exit 2; }
      case "$1" in
        --profile) profile=$2 ;;
        --runtime-mode) runtime_mode=$2 ;;
        --url) url=$2 ;;
        --prompt-tokens) prompt_tokens=$2 ;;
        --output-tokens) output_tokens=$2 ;;
        --concurrency) concurrency=$2 ;;
        --repetitions) repetitions=$2 ;;
        --warmup) warmup=$2 ;;
        --prompt-variant-offset) prompt_variant_offset=$2 ;;
        --timeout) timeout=$2 ;;
        --output-dir) output_dir=$2 ;;
        --data-parallel-rank) data_parallel_rank=$2 ;;
        --telemetry-interval) telemetry_interval=$2 ;;
      esac
      shift 2
      ;;
    --dry-run) dry_run=1; shift ;;
    --full-context) full_context=1; shift ;;
    --pin-data-parallel) pin_data_parallel=1; shift ;;
    --telemetry) telemetry=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if ((full_context)); then
  max_model_len=$(
    "$repo_root/.venv/bin/python" -c \
      'import sys; from r9700.config import load_runtime; print(load_runtime(sys.argv[1], sys.argv[2] or None)["limits"]["max_model_len"])' \
      "$profile" "$runtime_mode"
  )
  [[ $max_model_len =~ ^[1-9][0-9]*$ ]] || {
    printf 'runtime max_model_len is not a positive integer\n' >&2
    exit 2
  }
  prompt_tokens=$((max_model_len - output_tokens))
  ((prompt_tokens > 0)) || {
    printf 'output tokens leave no room for a full-context prompt\n' >&2
    exit 2
  }
fi

stamp=$(date +%Y%m%dT%H%M%S)
profile_label=${profile##*/}
profile_label=${profile_label%.json}
if [[ -n $runtime_mode ]]; then
  profile_label=$profile_label-$runtime_mode
fi
if [[ -n $output_dir ]]; then
  validation_output=$output_dir/api-$stamp.json
  benchmark_output=$output_dir/benchmark-$stamp.json
  telemetry_output=$output_dir/telemetry-$stamp.json
else
  validation_output=$repo_root/logs/validation/api-$profile_label-$stamp.json
  benchmark_output=$repo_root/logs/benchmarks/$profile_label-c$concurrency-${prompt_tokens}x${output_tokens}-$stamp.json
  telemetry_output=$repo_root/logs/telemetry/$profile_label-c$concurrency-${prompt_tokens}x${output_tokens}-$stamp.json
fi

api_command=("$repo_root/run" test api --profile "$profile" --url "$url" --timeout "$timeout" --output "$validation_output")
benchmark_command=("$repo_root/run" benchmark --profile "$profile" --url "$url" --prompt-tokens "$prompt_tokens" --output-tokens "$output_tokens" --concurrency "$concurrency" --repetitions "$repetitions" --warmup "$warmup" --timeout "$timeout" --output "$benchmark_output")
if [[ -n $runtime_mode ]]; then
  api_command+=(--runtime-mode "$runtime_mode")
  benchmark_command+=(--runtime-mode "$runtime_mode")
fi
benchmark_command+=(--prompt-variant-offset "$prompt_variant_offset")
if ((pin_data_parallel)); then
  benchmark_command+=(--pin-data-parallel)
fi
if [[ -n $data_parallel_rank ]]; then
  benchmark_command+=(--data-parallel-rank "$data_parallel_rank")
fi
telemetry_command=(
  "$repo_root/.venv/bin/python" "$script_dir/measure-runtime-telemetry.py"
  --output "$telemetry_output"
  --benchmark-output "$benchmark_output"
  --interval "$telemetry_interval"
  --require-cpu-io
  -- "${benchmark_command[@]}"
)

if ((dry_run)); then
  printf '%q ' "${api_command[@]}"
  printf '\n'
  if ((telemetry)); then
    printf '%q ' "${telemetry_command[@]}"
  else
    printf '%q ' "${benchmark_command[@]}"
  fi
  printf '\n'
  exit 0
fi

cd -- "$repo_root"
[[ $(./run service status) != stopped ]] || {
  printf 'managed service is stopped; start the requested model before testing\n' >&2
  exit 1
}

"${api_command[@]}"
if ((telemetry)); then
  "${telemetry_command[@]}"
  printf 'validation=%s\nbenchmark=%s\ntelemetry=%s\n' "$validation_output" "$benchmark_output" "$telemetry_output"
else
  "${benchmark_command[@]}"
  printf 'validation=%s\nbenchmark=%s\n' "$validation_output" "$benchmark_output"
fi
