from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from ..config import ConfigurationError, read_dotenv
from ..manifest import (
    default_recipe_name,
    recipe_record,
    recipe_venv,
    runtime_manifest,
)


def rocm_root(
    runtime: dict[str, Any] | None = None, *, recipe_name: str | None = None
) -> Path:
    selected = recipe_name or (
        str(runtime["recipe"])
        if runtime is not None
        else default_recipe_name(backend="vllm")
    )
    record = recipe_record(selected)
    if record["backend"] != "vllm":
        selected = str(record.get("foundation_recipe") or "")
        if not selected:
            raise ConfigurationError(
                f"{record['backend']} recipe does not select a ROCm foundation recipe"
            )
    helper = recipe_venv(selected) / "bin" / "rocm-sdk"
    if helper.is_file():
        result = subprocess.run(
            [helper, "path", "--root"], check=True, capture_output=True, text=True
        )
        return Path(result.stdout.strip()).resolve()

    configured = runtime_manifest(selected)["environment"].get("rocm_root")
    if isinstance(configured, str) and configured:
        root = Path(configured).resolve()
        if root.is_dir():
            return root
        raise ConfigurationError(
            f"configured ROCm root is absent for recipe {selected}: {root}"
        )
    raise ConfigurationError(
        f"ROCm SDK helper is absent; install recipe {selected} first"
    )


def _property_value(path: Path, key: str) -> int | None:
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] == key:
            return int(fields[1], 0)
    return None


def resolve_gpu_bdfs(
    bdfs: list[str],
    topology_root: Path = Path("/sys/class/kfd/kfd/topology/nodes"),
) -> list[str]:
    """Resolve stable PCI BDFs to the current HIP enumeration order."""
    discovered: dict[str, str] = {}
    gpu_ordinal = 0
    try:
        nodes = sorted(topology_root.iterdir(), key=lambda path: int(path.name))
        for node in nodes:
            properties = node / "properties"
            if not properties.is_file():
                continue
            simd_count = _property_value(properties, "simd_count")
            if not simd_count:
                continue
            location = _property_value(properties, "location_id")
            domain = _property_value(properties, "domain")
            if location is None or domain is None:
                continue
            bus = (location >> 8) & 0xFF
            device = (location >> 3) & 0x1F
            function = location & 0x7
            bdf = f"{domain:04x}:{bus:02x}:{device:02x}.{function}"
            discovered[bdf] = str(gpu_ordinal)
            gpu_ordinal += 1
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"cannot read KFD GPU topology: {exc}") from exc
    normalized = [item.strip().lower() for item in bdfs]
    missing = [item for item in normalized if item not in discovered]
    if missing:
        raise ConfigurationError(
            f"profile GPU BDFs are absent from KFD topology: {','.join(missing)}"
        )
    return [discovered[item] for item in normalized]


def visible_devices(runtime: dict[str, Any]) -> list[str]:
    gpu_order = os.environ.get("GPU_ORDER")
    if gpu_order:
        devices = [item.strip() for item in gpu_order.split(",") if item.strip()]
    elif runtime.get("gpu_bdfs"):
        devices = resolve_gpu_bdfs(runtime["gpu_bdfs"])
        gpu_order = ",".join(devices)
    else:
        gpu_order = ",".join(str(item) for item in runtime["gpu_order"])
        devices = [item.strip() for item in gpu_order.split(",") if item.strip()]
    parallel = runtime["parallel"]
    world_size = (
        parallel["tensor"] * parallel["pipeline"] * parallel.get("data", 1)
    )
    if len(devices) < world_size or len(set(devices)) != len(devices):
        raise ConfigurationError(
            f"GPU_ORDER={gpu_order!r} is incompatible with world size {world_size}"
        )
    return devices[:world_size]


def base_environment(runtime: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    dotenv = read_dotenv()
    if "HF_TOKEN" not in env and "HF_TOKEN" in dotenv:
        env["HF_TOKEN"] = dotenv["HF_TOKEN"]
    rocm_home = rocm_root(runtime)
    devices = visible_devices(runtime)
    env.update(
        {
            "PATH": f"{rocm_home / 'bin'}:{env.get('PATH', '')}",
            "LD_LIBRARY_PATH": f"{rocm_home / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}",
            "ROCM_HOME": str(rocm_home),
            "ROCM_PATH": str(rocm_home),
            "PYTORCH_ROCM_ARCH": "gfx1201",
            "GPU_ARCHS": "gfx1201",
            "HIP_VISIBLE_DEVICES": ",".join(devices),
        }
    )
    return env
