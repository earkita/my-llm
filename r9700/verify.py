from __future__ import annotations

import importlib
import importlib.metadata
import json
import re
import subprocess
import sys
from pathlib import Path

from .backends import build_environment, runtime_backend, verify_backend_install
from .config import ConfigurationError, load_model, load_runtime, validate_compatibility
from .manifest import (
    recipe_constraints_path,
    recipe_source_root,
    recipe_venv,
    runtime_manifest,
    verify_install,
)

PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^;\s]+)")


def _locked_packages(path: Path) -> dict[str, str]:
    packages: dict[str, str] = {}
    for line in path.read_text().splitlines():
        match = PIN.match(line.strip())
        if match:
            packages[match.group(1).lower().replace("_", "-")] = match.group(2)
    return packages


def _source_package_overrides(manifest: dict) -> dict[str, tuple[str, str]]:
    overrides = {}
    for source_name, source in manifest["sources"].items():
        distribution = source.get("python_distribution")
        import_name = source.get("python_import")
        if distribution and import_name:
            normalized = distribution.lower().replace("_", "-")
            overrides[normalized] = (source_name, import_name)
    return overrides


def _import_resolves_to_source(
    recipe_name: str, source_name: str, import_name: str
) -> bool:
    module = importlib.import_module(import_name)
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False
    imported = Path(module_file).resolve()
    source = (recipe_source_root(recipe_name) / source_name).resolve()
    return imported.is_relative_to(source)


def verify_python_environment(recipe_name: str) -> dict:
    manifest = runtime_manifest(recipe_name)
    verify_install(recipe_name)
    expected_python = manifest["platform"]["python"]
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != expected_python:
        raise ConfigurationError(
            f"Python mismatch: expected {expected_python}, found {actual_python}"
        )
    constraints = recipe_constraints_path(recipe_name)
    source_overrides = _source_package_overrides(manifest)
    mismatches = []
    for name, expected in _locked_packages(constraints).items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            mismatches.append(f"missing {name}=={expected}")
            continue
        if actual != expected:
            override = source_overrides.get(name)
            if override and _import_resolves_to_source(recipe_name, *override):
                continue
            mismatches.append(f"{name}: {actual} != {expected}")
    if mismatches:
        raise ConfigurationError("Python lock mismatch:\n" + "\n".join(mismatches))
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise ConfigurationError("pip check failed:\n" + result.stdout + result.stderr)
    payload = {
        "python": actual_python,
        "locked_packages": len(_locked_packages(constraints)),
        "source_package_overrides": len(source_overrides),
        "pip_check": "pass",
    }
    print(json.dumps(payload, indent=2))
    return payload


def verify_runtime(
    model_name: str, runtime_name: str, runtime_mode: str | None = None
) -> dict:
    model = load_model(model_name)
    runtime = load_runtime(runtime_name, runtime_mode)
    validate_compatibility(model, runtime)
    backend = runtime_backend(runtime)
    installed = verify_backend_install(runtime)
    if backend == "llama-cpp":
        binary = Path(installed["binary"])
        probe = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            env=build_environment(runtime),
            check=False,
        )
        if probe.returncode:
            raise ConfigurationError(
                "llama.cpp runtime probe failed:\n" + probe.stdout + probe.stderr
            )
        payload = {
            "backend": backend,
            "model": model["name"],
            "runtime": runtime["name"],
            "install": installed,
            "version": probe.stdout.strip() or probe.stderr.strip(),
        }
        print(json.dumps(payload, indent=2))
        return payload
    venv_python = recipe_venv(runtime["recipe"]) / "bin" / "python"
    probe = subprocess.run(
        [
            venv_python,
            "-c",
            (
                "import torch,vllm,aiter; print(torch.__version__); "
                "print(vllm.__version__); print(aiter.__file__)"
            ),
        ],
        capture_output=True,
        text=True,
        env=build_environment(runtime),
        check=False,
    )
    if probe.returncode:
        raise ConfigurationError(
            "runtime imports failed:\n" + probe.stdout + probe.stderr
        )
    payload = {
        "backend": backend,
        "recipe": runtime["recipe"],
        "model": model["name"],
        "runtime": runtime["name"],
        "install": installed,
        "imports": probe.stdout.strip().splitlines(),
    }
    print(json.dumps(payload, indent=2))
    return payload
