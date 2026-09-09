#!/usr/bin/env python3
"""Run the isolated gfx1201 sparse-MLA operator gate.

The command refuses to allocate on the GPUs while the managed production model
is active. It never stops or replaces that service itself.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "profiles/dev/glm53-flash-rocm10-gluon.json"


def _service_active() -> bool:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", "r9700-runtime.service"],
        check=False,
    )
    return result.returncode == 0


def _load_paths(profile_path: Path) -> tuple[str, Path, Path]:
    profile = json.loads(profile_path.read_text())
    recipe = profile["runtime"]["recipe"]
    recipe_root = ROOT / ".runtime/recipes" / recipe
    return recipe, recipe_root / "venv/bin/python", recipe_root / "src/aiter"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--operator", action="store_true")
    parser.add_argument(
        "--allow-shared-gpu",
        action="store_true",
        help="override the active-service guard; unsafe with a nearly full KV pool",
    )
    args = parser.parse_args()

    profile_path = args.profile.resolve()
    recipe, python, aiter_source = _load_paths(profile_path)
    active = _service_active()
    print(f"profile={profile_path}")
    print(f"recipe={recipe}")
    print(f"managed_runtime_active={str(active).lower()}")
    print(f"recipe_installed={str(python.is_file() and aiter_source.is_dir()).lower()}")

    if not args.operator:
        return 0
    if active and not args.allow_shared_gpu:
        print(
            "refusing GPU operator tests while r9700-runtime.service is active; "
            "the script did not stop it",
            file=sys.stderr,
        )
        return 2
    if not python.is_file() or not aiter_source.is_dir():
        print(
            "isolated recipe is not installed; run ./run install --profile "
            f"{profile_path}",
            file=sys.stderr,
        )
        return 2

    environment = os.environ.copy()
    rocm_sdk = python.parent / "rocm-sdk"
    rocm_root = Path(
        subprocess.run(
            [str(rocm_sdk), "path", "--root"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    library_dirs = [
        rocm_root / "share/amd_smi/amdsmi",
        rocm_root / "lib",
    ]
    environment["LD_LIBRARY_PATH"] = ":".join(
        part
        for part in (
            *(str(path) for path in library_dirs if path.is_dir()),
            environment.get("LD_LIBRARY_PATH", ""),
        )
        if part
    )
    environment["PYTHONPATH"] = ":".join(
        part
        for part in (str(aiter_source), environment.get("PYTHONPATH", ""))
        if part
    )
    environment["AITER_TRITON_ONLY"] = "1"
    test = aiter_source / "op_tests/triton_tests/attention/test_sparse_mla_rdna4.py"
    return subprocess.run(
        [str(python), "-m", "pytest", "-q", str(test)],
        cwd=aiter_source,
        env=environment,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
