from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

from .config import ConfigurationError, ROOT, load_runtime
from .backends.common import rocm_root


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    required: bool = True


def _command(command: list[str], timeout: float = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, capture_output=True, text=True, timeout=timeout, check=False
    )


def _rocm_root(runtime: dict) -> Path | None:
    try:
        return rocm_root(runtime)
    except (ConfigurationError, OSError, subprocess.SubprocessError):
        pass
    for candidate in (Path("/opt/rocm"), Path("/usr")):
        if (candidate / "bin" / "rocminfo").is_file():
            return candidate
    return None


def _tool(name: str) -> Check:
    path = shutil.which(name)
    return Check(f"tool:{name}", path is not None, path or "not found")


def _device_access(path: Path, mode: int, name: str) -> Check:
    exists = path.exists()
    allowed = exists and os.access(path, mode)
    return Check(name, allowed, f"path={path} exists={exists} access={allowed}")


def _render_nodes(required: int) -> Check:
    nodes = sorted(Path("/dev/dri").glob("renderD*")) if Path("/dev/dri").is_dir() else []
    usable = [str(path) for path in nodes if os.access(path, os.R_OK | os.W_OK)]
    return Check(
        "render-nodes",
        len(usable) >= required,
        f"required={required} writable={len(usable)} nodes={','.join(usable) or '-'}",
    )


def _amdgpu_runtime_pm_errors(
    devices_root: Path = Path("/sys/bus/pci/devices"),
) -> list[str]:
    errors: list[str] = []
    if not devices_root.is_dir():
        return errors
    for device in sorted(devices_root.iterdir()):
        try:
            vendor = (device / "vendor").read_text().strip().lower()
            device_class = (device / "class").read_text().strip().lower()
            runtime_status = (device / "power" / "runtime_status").read_text().strip()
        except OSError:
            continue
        if vendor == "0x1002" and device_class.startswith("0x03"):
            if runtime_status == "error":
                errors.append(device.name)
    return errors


def _amdgpu_runtime_pm() -> Check:
    errors = _amdgpu_runtime_pm_errors()
    return Check(
        "amdgpu-runtime-pm",
        not errors,
        "error_devices=" + (",".join(errors) if errors else "-"),
    )


def _compact_command_error(result: subprocess.CompletedProcess[str]) -> str:
    error = result.stderr.strip() or result.stdout.strip()
    error = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", error)
    return " ".join(error.split())[:600] or "-"


def _gpu_arch(required: int, runtime: dict) -> Check:
    root = _rocm_root(runtime)
    executable = (root / "bin" / "rocminfo") if root else None
    if not executable or not executable.is_file():
        return Check("gfx1201-devices", False, "rocminfo unavailable")
    result = _command([str(executable)], timeout=60)
    count = result.stdout.count("Name:                    gfx1201")
    if count == 0:
        # Formatting differs slightly across ROCm releases.
        count = sum(1 for line in result.stdout.splitlines() if "gfx1201" in line)
    return Check(
        "gfx1201-devices",
        result.returncode == 0 and count >= required,
        f"required={required} detected={count} rocminfo={executable} "
        f"returncode={result.returncode} error={_compact_command_error(result)}",
    )


def _gpu_arch_after_runtime_pm(
    runtime_pm: Check, required: int, runtime: dict
) -> Check:
    if not runtime_pm.passed:
        return Check(
            "gfx1201-devices",
            False,
            "rocminfo skipped because amdgpu runtime-PM is already in error",
        )
    return _gpu_arch(required, runtime)


def _iommu() -> Check:
    cmdline = Path("/proc/cmdline").read_text().strip() if Path("/proc/cmdline").is_file() else ""
    enabled = "iommu=pt" in cmdline or "amd_iommu=on" in cmdline
    return Check("iommu", enabled, cmdline or "unavailable", required=False)


def doctor(runtime_name: str, *, output: Path | None = None) -> dict:
    runtime = load_runtime(runtime_name)
    world_size = runtime["parallel"]["tensor"] * runtime["parallel"]["pipeline"]
    checks = [_tool(name) for name in ("git", "gcc", "g++", "cmake", "ninja")]
    runtime_pm = _amdgpu_runtime_pm()
    checks.extend(
        [
            Check(
                "operating-system",
                platform.system() == "Linux",
                f"{platform.system()} {platform.release()}",
            ),
            _device_access(Path("/dev/kfd"), os.R_OK | os.W_OK, "kfd-access"),
            _render_nodes(world_size),
            runtime_pm,
            _gpu_arch_after_runtime_pm(runtime_pm, world_size, runtime),
            _iommu(),
        ]
    )
    required_pass = all(check.passed for check in checks if check.required)
    payload = {
        "schema_version": 1,
        "runtime": runtime["name"],
        "world_size": world_size,
        "checks": [asdict(check) for check in checks],
        "passed": required_pass,
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, output)
    for check in checks:
        status = "PASS" if check.passed else ("WARN" if not check.required else "FAIL")
        print(f"{status:4} {check.name}: {check.detail}")
    if not required_pass:
        raise ConfigurationError("host preflight failed")
    return payload
