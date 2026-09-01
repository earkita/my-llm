#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from typing import Any


def positive_watts(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("watts must be an integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("watts must be positive")
    return value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Fail closed unless every visible GPU has the required PPT0 cap."
    )
    result.add_argument("--watts", type=positive_watts, required=True)
    return result


def read_caps(payload: dict[str, Any]) -> list[tuple[int, int]]:
    gpu_data = payload.get("gpu_data")
    if not isinstance(gpu_data, list) or not gpu_data:
        raise ValueError("amd-smi JSON contains no gpu_data")
    caps: list[tuple[int, int]] = []
    for item in gpu_data:
        try:
            gpu = int(item["gpu"])
            power_limit = item["limit"]["ppt0"]["socket_power_limit"]
            watts = int(power_limit["value"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("amd-smi JSON has an invalid PPT0 power limit") from exc
        caps.append((gpu, watts))
    return sorted(caps)


def main() -> int:
    args = parser().parse_args()
    amd_smi = shutil.which("amd-smi")
    if amd_smi is None:
        print("power-cap check failed: amd-smi is not installed", file=sys.stderr)
        return 1
    result = subprocess.run(
        [amd_smi, "static", "--limit", "--gpu", "all", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        print(f"power-cap check failed: amd-smi: {detail}", file=sys.stderr)
        return 1
    try:
        caps = read_caps(json.loads(result.stdout))
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"power-cap check failed: {exc}", file=sys.stderr)
        return 1
    mismatches = [(gpu, watts) for gpu, watts in caps if watts > args.watts]
    if mismatches:
        actual = ", ".join(f"GPU {gpu}={watts} W" for gpu, watts in mismatches)
        print(
            f"power-cap check failed: expected at most {args.watts} W on every GPU; "
            f"{actual}",
            file=sys.stderr,
        )
        print(
            f"set it before launch: sudo amd-smi set -o ppt0 {args.watts} -g all",
            file=sys.stderr,
        )
        return 1
    reported = sorted({watts for _gpu, watts in caps})
    print(
        f"GPU power cap verified: {len(caps)} GPU(s), "
        f"maximum {args.watts} W, reported {reported} W"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
