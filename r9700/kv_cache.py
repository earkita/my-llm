from __future__ import annotations

import errno
import json
import os
import shutil
import stat
from heapq import heapify, heappop
from pathlib import Path
from typing import Any, Collection

from .config import ConfigurationError, load_profile, load_runtime


CACHE_MARKER = ".r9700-kv-cache.json"
BOUNDED_FS_TIER = "BoundedFileSystemTierManager"
BOUNDED_FS_MODULE = "r9700.vllm_bootstrap.bounded_fs_tier"
MANAGED_CACHE_ROOT = Path("/mnt/ai/r9700-kv-cache")


def _regular_files(root: Path, pattern: str = "*"):
    if not root.is_dir():
        return
    for path in root.rglob(pattern):
        try:
            mode = path.stat(follow_symlinks=False).st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISREG(mode):
            yield path


def cache_usage(root: Path) -> tuple[int, int]:
    total_bytes = 0
    file_count = 0
    for path in _regular_files(root):
        try:
            total_bytes += path.stat(follow_symlinks=False).st_size
        except FileNotFoundError:
            continue
        file_count += 1
    return total_bytes, file_count


def block_cache_usage(root: Path) -> tuple[int, int]:
    total_bytes = 0
    file_count = 0
    for path in _regular_files(root, "*.bin"):
        try:
            total_bytes += path.stat(follow_symlinks=False).st_size
        except FileNotFoundError:
            continue
        file_count += 1
    return total_bytes, file_count


def reclaim_for_write(
    root: Path,
    *,
    current_bytes: int,
    incoming_bytes: int,
    max_bytes: int,
    min_free_bytes: int,
    protected_paths: Collection[str] = (),
) -> tuple[int, int]:
    """Evict oldest complete KV blocks before a filesystem-tier write.

    The caller serializes this function with its write. Only ``*.bin`` files
    are eligible: vLLM temporary files and the namespace configuration are
    never removed while a transfer may be using them.
    """
    if incoming_bytes < 0 or current_bytes < 0:
        raise ValueError("cache byte counts cannot be negative")
    if max_bytes <= 0 or min_free_bytes < 0:
        raise ValueError("cache limits are invalid")
    if incoming_bytes > max_bytes:
        raise OSError(
            errno.ENOSPC,
            f"one KV write ({incoming_bytes} bytes) exceeds cache limit "
            f"({max_bytes} bytes)",
        )

    free_bytes = shutil.disk_usage(root).free
    required_reclaim = max(
        0,
        current_bytes + incoming_bytes - max_bytes,
        min_free_bytes + incoming_bytes - free_bytes,
    )
    if required_reclaim == 0:
        return current_bytes, 0

    protected = {os.path.abspath(path) for path in protected_paths}
    candidates: list[tuple[int, str, int]] = []
    for path in _regular_files(root, "*.bin"):
        absolute = os.path.abspath(path)
        if absolute in protected:
            continue
        try:
            file_stat = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        candidates.append((file_stat.st_mtime_ns, absolute, file_stat.st_size))
    heapify(candidates)

    reclaimed = 0
    while candidates and reclaimed < required_reclaim:
        _, candidate, expected_size = heappop(candidates)
        try:
            os.unlink(candidate)
        except FileNotFoundError:
            continue
        reclaimed += expected_size

    if reclaimed < required_reclaim:
        raise OSError(
            errno.ENOSPC,
            "KV cache cannot preserve its capacity/free-space limits because "
            "too many blocks are protected by active transfers",
        )
    return max(0, current_bytes - reclaimed), reclaimed


def _storage_tiers(runtime: dict[str, Any]) -> list[dict[str, Any]]:
    transfer = runtime.get("kv_transfer_config")
    if not isinstance(transfer, dict):
        return []
    extra = transfer.get("kv_connector_extra_config")
    if not isinstance(extra, dict):
        return []
    tiers = extra.get("secondary_tiers", [])
    return [tier for tier in tiers if isinstance(tier, dict)]


def configured_cache_roots(runtime: dict[str, Any]) -> list[Path]:
    return [
        Path(str(tier["root_dir"])).expanduser().resolve()
        for tier in _storage_tiers(runtime)
        if tier.get("type") == BOUNDED_FS_TIER and tier.get("root_dir")
    ]


def _selected_cache(
    profile_name: str, runtime_mode: str
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    profile = load_profile(profile_name)
    runtime = load_runtime(profile_name, runtime_mode)
    tiers = [
        tier
        for tier in _storage_tiers(runtime)
        if tier.get("type") == BOUNDED_FS_TIER
        and tier.get("module_path") == BOUNDED_FS_MODULE
    ]
    if len(tiers) != 1:
        raise ConfigurationError(
            f"runtime mode {runtime_mode!r} must configure exactly one managed "
            "bounded filesystem KV tier"
        )
    root = Path(str(tiers[0]["root_dir"])).expanduser().resolve()
    if root == MANAGED_CACHE_ROOT or not root.is_relative_to(MANAGED_CACHE_ROOT):
        raise ConfigurationError(f"unsafe KV cache root: {root}")
    return profile, tiers[0], root


def prepare(profile_name: str, runtime_mode: str) -> dict[str, Any]:
    profile, tier, root = _selected_cache(profile_name, runtime_mode)
    root.mkdir(parents=True, exist_ok=True)
    marker_path = root / CACHE_MARKER
    existing_children = list(root.iterdir())
    if existing_children and not marker_path.is_file():
        raise ConfigurationError(
            f"refusing to adopt non-empty unowned KV cache directory: {root}"
        )
    if marker_path.is_file():
        try:
            existing_marker = json.loads(marker_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"invalid KV cache marker: {marker_path}") from exc
        expected_identity = (profile["name"], runtime_mode, str(root))
        actual_identity = (
            existing_marker.get("profile"),
            existing_marker.get("runtime_mode"),
            existing_marker.get("root"),
        )
        if actual_identity != expected_identity:
            raise ConfigurationError(
                f"refusing to replace foreign KV cache marker: {marker_path}"
            )
    marker = {
        "schema_version": 1,
        "profile": profile["name"],
        "runtime_mode": runtime_mode,
        "root": str(root),
        "max_bytes": tier["max_bytes"],
        "min_free_bytes": tier["min_free_bytes"],
    }
    temporary = marker_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, marker_path)
    return status(profile_name, runtime_mode)


def status(profile_name: str, runtime_mode: str) -> dict[str, Any]:
    _, tier, root = _selected_cache(profile_name, runtime_mode)
    used_bytes, file_count = cache_usage(root)
    block_bytes, block_count = block_cache_usage(root)
    existing = root
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    filesystem = shutil.disk_usage(existing)
    return {
        "profile": profile_name,
        "runtime_mode": runtime_mode,
        "root": str(root),
        "prepared": (root / CACHE_MARKER).is_file(),
        "used_bytes": used_bytes,
        "file_count": file_count,
        "block_bytes": block_bytes,
        "block_count": block_count,
        "max_bytes": tier["max_bytes"],
        "min_free_bytes": tier["min_free_bytes"],
        "filesystem_free_bytes": filesystem.free,
    }


def _assert_not_active(root: Path) -> None:
    from .service import managed_state

    try:
        active = managed_state()
    except ConfigurationError:
        return
    active_mode = active.get("runtime_mode")
    if not active_mode:
        return
    try:
        runtime = load_runtime(str(active["profile"]), str(active_mode))
    except (ConfigurationError, KeyError):
        raise ConfigurationError(
            "refusing to clear persistent KV cache because the active runtime "
            "mode cannot be resolved safely"
        )
    if root in configured_cache_roots(runtime):
        raise ConfigurationError(
            "refusing to clear persistent KV cache while its runtime mode is "
            "active; stop that runtime gracefully first"
        )


def clear(profile_name: str, runtime_mode: str) -> dict[str, Any]:
    profile, _, root = _selected_cache(profile_name, runtime_mode)
    marker_path = root / CACHE_MARKER
    if not marker_path.is_file():
        raise ConfigurationError(
            f"refusing to clear unprepared KV cache directory: {root}"
        )
    try:
        marker = json.loads(marker_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"invalid KV cache marker: {marker_path}") from exc
    if (
        marker.get("root") != str(root)
        or marker.get("profile") != profile["name"]
        or marker.get("runtime_mode") != runtime_mode
    ):
        raise ConfigurationError(f"KV cache marker does not own directory: {root}")
    _assert_not_active(root)

    removed_bytes, removed_files = cache_usage(root)
    marker_bytes = marker_path.stat(follow_symlinks=False).st_size
    for child in root.iterdir():
        if child == marker_path:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)
    return {
        "profile": profile_name,
        "runtime_mode": runtime_mode,
        "root": str(root),
        "removed_bytes": max(0, removed_bytes - marker_bytes),
        "removed_files": max(0, removed_files - 1),
        "note": "persistent filesystem tier cleared; live GPU/CPU KV is unchanged",
    }
