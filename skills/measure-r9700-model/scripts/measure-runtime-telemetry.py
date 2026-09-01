#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from r9700.telemetry import run_with_telemetry


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a benchmark command with phase-labelled R9700 telemetry"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--unit", default="r9700-runtime.service")
    parser.add_argument("--require-cpu-io", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return run_with_telemetry(
        command=command,
        output=args.output,
        benchmark_output=args.benchmark_output,
        interval=args.interval,
        unit=args.unit,
        require_cpu_io=args.require_cpu_io,
    )


if __name__ == "__main__":
    raise SystemExit(main())
