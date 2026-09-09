#!/usr/bin/env python3
"""Run the isolated v0.31 MXFP4 GEMV qualification gates.

The command never stops, starts, or replaces the managed runtime. GPU work is
refused while that runtime is active; ``--dry-run`` remains available so the
complete command sequence can be inspected without allocating on a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "profiles/dev/glm53-flash-rocm10-mxfp4-tiled.json"
EXPECTED_RECIPE = "vllm_glm53flashrocm10_v0.31"
SERVICE_UNIT = "r9700-runtime.service"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _managed_runtime() -> tuple[bool, str]:
    status = subprocess.run(
        [str(ROOT / "run"), "service", "status"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    detail = (status.stdout or status.stderr).strip() or f"exit={status.returncode}"
    unit = subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", SERVICE_UNIT],
        check=False,
    )
    # Exit 3 plus the literal `stopped` is the control plane's cleanly stopped
    # state. Any unknown/error state is treated conservatively as occupied.
    stopped = status.returncode == 3 and detail == "stopped"
    return unit.returncode == 0 or not stopped, detail


def _resolve_recipe(
    profile_path: Path,
) -> tuple[dict[str, Any], Path, Path, Path]:
    profile = _load_json(profile_path)
    runtime = profile.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError(f"profile has no runtime object: {profile_path}")
    recipe = runtime.get("recipe")
    if recipe != EXPECTED_RECIPE:
        raise ValueError(
            f"qualification requires {EXPECTED_RECIPE}, profile selects {recipe!r}"
        )

    recipe_root = ROOT / ".runtime/recipes" / EXPECTED_RECIPE
    python = recipe_root / "venv/bin/python"
    vllm_source = recipe_root / "src/vllm"
    aiter_source = recipe_root / "src/aiter"
    install_path = recipe_root / "install.json"
    required = (python, vllm_source, aiter_source, install_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("isolated recipe is incomplete: " + ", ".join(missing))

    install = _load_json(install_path)
    if install.get("recipe") != EXPECTED_RECIPE:
        raise ValueError(f"installed recipe identity differs in {install_path}")
    manifest = _load_json(ROOT / "manifest" / f"{EXPECTED_RECIPE}.json")
    expected_vllm = manifest["sources"]["vllm"]
    installed_vllm = install.get("sources", {}).get("vllm", {})
    if (
        installed_vllm.get("head") != expected_vllm.get("commit")
        or installed_vllm.get("tracked_diff_sha256")
        != expected_vllm.get("expected_diff_sha256")
    ):
        raise ValueError("installed vLLM source identity differs from the manifest")
    return runtime, python, vllm_source, aiter_source


def _commands(
    python: Path,
    vllm_source: Path,
    output_dir: Path,
) -> tuple[list[str], list[str]]:
    test = vllm_source / "tests/kernels/quantization/test_rocm_mxfp4_gemv.py"
    benchmark = vllm_source / "benchmarks/kernels/benchmark_rocm_mxfp4_gemv.py"
    missing = [str(path) for path in (test, benchmark) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "v0.31 qualification files are absent: " + ", ".join(missing)
        )
    pytest_command = [
        str(python),
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-q",
        str(test),
    ]
    benchmark_command = [
        str(python),
        str(benchmark),
        "--output",
        str(output_dir / "benchmark.json"),
    ]
    return pytest_command, benchmark_command


def _environment(
    runtime: dict[str, Any],
    python: Path,
    vllm_source: Path,
    aiter_source: Path,
    gpu: int,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {str(key): str(value) for key, value in runtime.get("environment", {}).items()}
    )
    environment["HIP_VISIBLE_DEVICES"] = str(gpu)
    environment["ROCR_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (
            str(vllm_source),
            str(aiter_source),
            environment.get("PYTHONPATH", ""),
        )
        if part
    )

    rocm_sdk = python.parent / "rocm-sdk"
    rocm_root = Path(
        subprocess.run(
            [str(rocm_sdk), "path", "--root"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    environment["ROCM_HOME"] = str(rocm_root)
    environment["ROCM_PATH"] = str(rocm_root)
    environment["PYTORCH_ROCM_ARCH"] = "gfx1201"
    environment["GPU_ARCHS"] = "gfx1201"
    environment["TRITON_DEFAULT_BACKEND"] = "amd"
    environment["PATH"] = os.pathsep.join(
        (str(rocm_root / "bin"), str(python.parent), environment.get("PATH", ""))
    )
    environment["CPLUS_INCLUDE_PATH"] = os.pathsep.join(
        part
        for part in (
            str(rocm_root / "include"),
            environment.get("CPLUS_INCLUDE_PATH", ""),
        )
        if part
    )
    library_dirs = (rocm_root / "share/amd_smi/amdsmi", rocm_root / "lib")
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        part
        for part in (
            *(str(path) for path in library_dirs if path.is_dir()),
            environment.get("LD_LIBRARY_PATH", ""),
        )
        if part
    )
    return environment


def _run_and_record(
    command: list[str], *, cwd: Path, environment: dict[str, str], output: Path
) -> int:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output.write_text(result.stdout, encoding="utf-8")
    print(result.stdout, end="")
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-gpu-gates", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("--gpu must be a non-negative ROCm device index")

    try:
        profile_path = args.profile.expanduser().resolve()
        runtime, python, vllm_source, aiter_source = _resolve_recipe(profile_path)
        stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir
            else ROOT / "logs/qualification" / f"v031-mxfp4-gemv-{stamp}"
        )
        pytest_command, benchmark_command = _commands(
            python, vllm_source, output_dir
        )
    except (FileNotFoundError, KeyError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2

    occupied, service_status = _managed_runtime()
    print(f"profile={profile_path}")
    print(f"recipe={EXPECTED_RECIPE}")
    print(f"managed_runtime_occupied={str(occupied).lower()}")
    print(f"managed_runtime_status={service_status}")
    print("recipe_installed=true")
    print(f"gpu={args.gpu}")

    if args.dry_run:
        environment_prefix = [
            "env",
            f"HIP_VISIBLE_DEVICES={args.gpu}",
            f"ROCR_VISIBLE_DEVICES={args.gpu}",
        ]
        print("pytest_command=" + shlex.join(environment_prefix + pytest_command))
        print("benchmark_command=" + shlex.join(environment_prefix + benchmark_command))
        if occupied:
            print("gpu_gates_would_refuse=true")
        return 0
    if not args.run_gpu_gates:
        return 0
    if occupied:
        print(
            "refusing GPU qualification while the managed runtime is occupied; "
            "the script did not stop or replace it",
            file=sys.stderr,
        )
        return 2

    try:
        environment = _environment(
            runtime, python, vllm_source, aiter_source, args.gpu
        )
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"cannot resolve the isolated ROCm environment: {error}", file=sys.stderr)
        return 2

    if output_dir.exists():
        print(f"refusing to overwrite an existing artifact directory: {output_dir}")
        return 2
    output_dir.mkdir(parents=True)
    metadata = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "profile": str(profile_path),
        "recipe": EXPECTED_RECIPE,
        "gpu": args.gpu,
        "service_status_before": service_status,
        "pytest_command": pytest_command,
        "benchmark_command": benchmark_command,
        "status": "running",
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    pytest_rc = _run_and_record(
        pytest_command,
        cwd=vllm_source,
        environment=environment,
        output=output_dir / "pytest.log",
    )
    if pytest_rc:
        metadata["status"] = "pytest-failed"
        metadata["pytest_exit_code"] = pytest_rc
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        return pytest_rc

    occupied, service_status_after_tests = _managed_runtime()
    if occupied:
        metadata["status"] = "service-became-occupied"
        metadata["service_status_after_tests"] = service_status_after_tests
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(
            "refusing the benchmark because the managed runtime became occupied",
            file=sys.stderr,
        )
        return 2

    benchmark_rc = _run_and_record(
        benchmark_command,
        cwd=vllm_source,
        environment=environment,
        output=output_dir / "benchmark.stdout.log",
    )
    metadata["status"] = "passed" if benchmark_rc == 0 else "benchmark-failed"
    metadata["pytest_exit_code"] = pytest_rc
    metadata["benchmark_exit_code"] = benchmark_rc
    metadata["finished_at"] = datetime.now().astimezone().isoformat()
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"artifacts={output_dir}")
    return benchmark_rc


if __name__ == "__main__":
    raise SystemExit(main())
