from __future__ import annotations

import json
import math
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


_VALUE_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")
_IO_LINKS = ("P0", "P1", "P2", "P3", "P4", "G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7")


def _number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and (match := _VALUE_RE.search(value)):
        return float(match.group())
    return None


def _read_meminfo() -> dict[str, int]:
    result: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, raw = line.split(":", 1)
        value = raw.strip().split()[0]
        result[key] = int(value) * 1024
    return result


def _service_memory_path(unit: str) -> Path:
    result = subprocess.run(
        ["systemctl", "--user", "show", unit, "--property=ControlGroup", "--value"],
        check=True,
        capture_output=True,
        text=True,
    )
    control_group = result.stdout.strip()
    if not control_group:
        raise RuntimeError(f"systemd unit {unit!r} has no control group")
    return Path("/sys/fs/cgroup") / control_group.lstrip("/") / "memory.current"


class AmdSmiSampler:
    """Low-overhead R9700 telemetry using the AMD SMI Python API."""

    def __init__(self, *, unit: str, require_cpu_io: bool = False):
        try:
            import amdsmi
        except ImportError as exc:  # pragma: no cover - host dependency
            raise RuntimeError("AMD SMI Python bindings are unavailable") from exc

        self.amdsmi = amdsmi
        self.amdsmi.amdsmi_init(amdsmi.AmdSmiInitFlags.INIT_ALL_PROCESSORS)
        self._closed = False
        self.memory_path = _service_memory_path(unit)
        self.gpu_handles = self.amdsmi.amdsmi_get_processor_handles()
        sockets = self.amdsmi.amdsmi_get_cpusocket_handles()
        self.cpu_handle = sockets[0] if sockets else None
        self.gpu_bdfs = [
            self.amdsmi.amdsmi_get_gpu_device_bdf(handle)
            for handle in self.gpu_handles
        ]

        self.valid_io_links: list[str] = []
        self.unsupported_io_links: list[str] = []
        if self.cpu_handle is not None:
            for link in _IO_LINKS:
                try:
                    value = self.amdsmi.amdsmi_get_cpu_current_io_bandwidth(
                        self.cpu_handle, 1, link
                    )
                except Exception:
                    self.unsupported_io_links.append(link)
                    continue
                if _number(value) is None:
                    self.unsupported_io_links.append(link)
                else:
                    self.valid_io_links.append(link)
        if require_cpu_io and not self.valid_io_links:
            self.close()
            raise RuntimeError("AMD SMI exposes no measurable CPU I/O links")

    def close(self) -> None:
        if not self._closed:
            self.amdsmi.amdsmi_shut_down()
            self._closed = True

    def sample(self) -> dict[str, Any]:
        sampled_perf = time.perf_counter()
        meminfo = _read_meminfo()
        service_memory = int(self.memory_path.read_text().strip())

        io_links: dict[str, float] = {}
        ddr: dict[str, Any] | None = None
        if self.cpu_handle is not None:
            for link in self.valid_io_links:
                raw = self.amdsmi.amdsmi_get_cpu_current_io_bandwidth(
                    self.cpu_handle, 1, link
                )
                if (value := _number(raw)) is not None:
                    io_links[link] = value
            try:
                ddr = self.amdsmi.amdsmi_get_cpu_ddr_bw(self.cpu_handle)
            except Exception:
                ddr = None

        gpus = []
        for index, (handle, bdf) in enumerate(zip(self.gpu_handles, self.gpu_bdfs)):
            vram = self.amdsmi.amdsmi_get_gpu_vram_usage(handle)
            activity = self.amdsmi.amdsmi_get_gpu_activity(handle)
            power = self.amdsmi.amdsmi_get_power_info(handle)
            pcie = self.amdsmi.amdsmi_get_pcie_info(handle).get("pcie_metric", {})
            gpus.append(
                {
                    "gpu": index,
                    "bdf": bdf,
                    "vram_used_mb": vram.get("vram_used"),
                    "vram_total_mb": vram.get("vram_total"),
                    "gfx_percent": activity.get("gfx_activity"),
                    "memory_percent": activity.get("umc_activity"),
                    "socket_power_w": power.get("average_socket_power"),
                    "pcie_width": pcie.get("pcie_width"),
                    "pcie_speed_mt_s": pcie.get("pcie_speed"),
                    "pcie_bandwidth": pcie.get("pcie_bandwidth"),
                }
            )

        return {
            "perf_counter": sampled_perf,
            "timestamp": datetime.now().astimezone().isoformat(),
            "phase": "outside_measured_requests",
            "service_memory_bytes": service_memory,
            "host_memory_total_bytes": meminfo.get("MemTotal"),
            "host_memory_available_bytes": meminfo.get("MemAvailable"),
            "cpu_ddr_bandwidth": ddr,
            "cpu_io_aggregate_mbps": io_links,
            "gpus": gpus,
        }


def label_samples(samples: list[dict[str, Any]], rows: Sequence[dict[str, Any]]) -> None:
    for sample in samples:
        point = float(sample["perf_counter"])
        phases: set[str] = set()
        for row in rows:
            start = float(row["started_perf"])
            end = float(row["ended_perf"])
            first = start + float(row["ttft_seconds"])
            if start <= point <= end:
                phases.add("prefill" if point < first else "decode")
        if len(phases) == 1:
            sample["phase"] = next(iter(phases))
        elif phases:
            sample["phase"] = "mixed"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_samples(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    phases = sorted({str(sample["phase"]) for sample in samples})
    summary: dict[str, Any] = {}
    for phase in phases:
        selected = [sample for sample in samples if sample["phase"] == phase]
        service_memory = [float(row["service_memory_bytes"]) for row in selected]
        host_available = [
            float(row["host_memory_available_bytes"])
            for row in selected
            if row.get("host_memory_available_bytes") is not None
        ]
        io_links = sorted(
            {
                link
                for row in selected
                for link in row.get("cpu_io_aggregate_mbps", {})
            }
        )
        gpu_indexes = sorted(
            {int(gpu["gpu"]) for row in selected for gpu in row.get("gpus", [])}
        )
        gpu_summary = []
        for gpu_index in gpu_indexes:
            records = [
                gpu
                for row in selected
                for gpu in row.get("gpus", [])
                if gpu["gpu"] == gpu_index
            ]
            used = [float(row["vram_used_mb"]) for row in records]
            total = [float(row["vram_total_mb"]) for row in records]
            gfx = [float(row["gfx_percent"]) for row in records]
            power = [float(row["socket_power_w"]) for row in records]
            gpu_summary.append(
                {
                    "gpu": gpu_index,
                    "bdf": records[0]["bdf"],
                    "vram_used_peak_mb": max(used),
                    "vram_free_min_mb": min(t - u for t, u in zip(total, used)),
                    "gfx_mean_percent": _mean(gfx),
                    "gfx_peak_percent": max(gfx),
                    "power_mean_w": _mean(power),
                    "power_peak_w": max(power),
                }
            )
        summary[phase] = {
            "samples": len(selected),
            "service_memory_mean_bytes": _mean(service_memory),
            "service_memory_peak_bytes": max(service_memory),
            "host_memory_available_min_bytes": min(host_available),
            "cpu_io_aggregate_mbps": {
                link: {
                    "mean": _mean(
                        [
                            float(row["cpu_io_aggregate_mbps"][link])
                            for row in selected
                            if link in row.get("cpu_io_aggregate_mbps", {})
                        ]
                    ),
                    "peak": max(
                        float(row["cpu_io_aggregate_mbps"][link])
                        for row in selected
                        if link in row.get("cpu_io_aggregate_mbps", {})
                    ),
                }
                for link in io_links
            },
            "gpus": gpu_summary,
        }
    return summary


def run_with_telemetry(
    *,
    command: Sequence[str],
    output: Path,
    benchmark_output: Path,
    interval: float,
    unit: str = "r9700-runtime.service",
    require_cpu_io: bool = False,
) -> int:
    if not command:
        raise ValueError("telemetry wrapper requires a child command")
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("telemetry interval must be positive")

    sampler = AmdSmiSampler(unit=unit, require_cpu_io=require_cpu_io)
    samples: list[dict[str, Any]] = []
    started = datetime.now().astimezone().isoformat()
    process = subprocess.Popen(list(command))
    try:
        deadline = time.perf_counter()
        while process.poll() is None:
            samples.append(sampler.sample())
            deadline += interval
            delay = deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        samples.append(sampler.sample())
        returncode = process.wait()
    except KeyboardInterrupt:
        process.send_signal(2)
        returncode = process.wait()
    finally:
        sampler.close()

    rows: list[dict[str, Any]] = []
    benchmark_error = None
    if benchmark_output.exists():
        try:
            rows = json.loads(benchmark_output.read_text()).get("rows", [])
        except (OSError, json.JSONDecodeError) as exc:
            benchmark_error = str(exc)
    else:
        benchmark_error = "benchmark output was not created"
    label_samples(samples, rows)

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "started_at": started,
        "command": list(command),
        "child_returncode": returncode,
        "benchmark_output": str(benchmark_output),
        "benchmark_error": benchmark_error,
        "measurement": {
            "interval_seconds": interval,
            "systemd_unit": unit,
            "cpu_io_semantics": "AMD SMI aggregate current CPU I/O bandwidth per link",
            "cpu_io_unit": "Mbps",
            "cpu_io_links": sampler.valid_io_links,
            "unsupported_cpu_io_links": sampler.unsupported_io_links,
            "per_gpu_pcie_bandwidth_note": (
                "gfx1201 reports N/A; CPU I/O link counters are the measured "
                "PCIe traffic source"
            ),
        },
        "samples": samples,
        "phases": summarize_samples(samples),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps(payload["phases"], indent=2))
    return returncode
