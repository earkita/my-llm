from __future__ import annotations

from pathlib import Path

from .config import ConfigurationError


def _parse_counter_text(path: Path, text: str) -> dict[str, int]:
    stripped = text.strip()
    if not stripped:
        raise ConfigurationError(f"RAS counter unreadable: {path}: empty value")

    try:
        return {str(path): int(stripped)}
    except ValueError:
        pass

    counters: dict[str, int] = {}
    for line in stripped.splitlines():
        fields = line.split()
        if len(fields) != 2:
            raise ConfigurationError(
                f"RAS counter unreadable: {path}: invalid row {line!r}"
            )
        label, raw_value = fields
        key = f"{path}::{label}"
        if key in counters:
            raise ConfigurationError(
                f"RAS counter unreadable: {path}: duplicate label {label!r}"
            )
        try:
            counters[key] = int(raw_value)
        except ValueError as exc:
            raise ConfigurationError(
                f"RAS counter unreadable: {path}: invalid value {raw_value!r}"
            ) from exc
    return counters


def snapshot() -> dict:
    counters: dict[str, int] = {}
    patterns = (
        "/sys/devices/system/edac/mc/mc*/ce_count",
        "/sys/devices/system/edac/mc/mc*/ue_count",
        "/sys/devices/system/edac/mc/mc*/dimm*/dimm_ce_count",
        "/sys/devices/system/edac/mc/mc*/dimm*/dimm_ue_count",
        "/sys/bus/pci/devices/*/aer_dev_correctable",
        "/sys/bus/pci/devices/*/aer_dev_nonfatal",
        "/sys/bus/pci/devices/*/aer_dev_fatal",
    )
    for pattern in patterns:
        for path in sorted(Path("/").glob(pattern.lstrip("/"))):
            try:
                parsed = _parse_counter_text(path, path.read_text())
            except OSError as exc:
                raise ConfigurationError(f"RAS counter unreadable: {path}: {exc}") from exc
            overlap = counters.keys() & parsed.keys()
            if overlap:
                raise ConfigurationError(
                    f"duplicate RAS counter identity: {sorted(overlap)[0]}"
                )
            counters.update(parsed)
    boot_path = Path("/proc/sys/kernel/random/boot_id")
    if not boot_path.is_file() or not counters:
        raise ConfigurationError("complete boot/RAS telemetry is unavailable")
    return {"boot_id": boot_path.read_text().strip(), "counters": counters}


def compare(before: dict, after: dict) -> dict:
    if before["boot_id"] != after["boot_id"]:
        raise ConfigurationError("host rebooted during the gate")
    if set(before["counters"]) != set(after["counters"]):
        raise ConfigurationError("RAS counter device set changed during the gate")
    delta = {}
    for path, initial in before["counters"].items():
        change = after["counters"][path] - initial
        if change < 0:
            raise ConfigurationError(f"RAS counter reset: {path}")
        if change:
            delta[path] = change
    return {"passed": not delta, "delta": delta}
