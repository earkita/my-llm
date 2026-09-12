from __future__ import annotations

import signal
import subprocess
import time
from pathlib import Path
from threading import Event
from typing import Any

from . import doctor as doctor_module
from . import service as runtime_service
from .config import ConfigurationError, ROOT, load_profile


STATE_PATH = runtime_service.WORKER_POOL_STATE_PATH
UNIT = "r9700-worker-pool.service"
DEFAULT_PROFILE = "qwen38-4x27b"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8100
POWER_CHECK = (
    ROOT
    / "skills"
    / "start-r9700-runtime"
    / "scripts"
    / "check-power-cap.py"
)


def _profile(name: str) -> dict[str, Any]:
    profile = load_profile(name)
    runtime = profile["runtime"]
    parallel = runtime["parallel"]
    if runtime.get("role") != "worker-pool":
        raise ConfigurationError(
            f"profile {name} is not marked as a worker-pool deployment"
        )
    if (
        parallel["tensor"] != 1
        or parallel["pipeline"] != 1
        or parallel.get("data", 1) < 2
    ):
        raise ConfigurationError(
            "worker-pool requires TP1, PP1 and at least two local DP replicas"
        )
    return profile


def _unit_state() -> str:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", UNIT],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "inactive"


def _assert_disjoint_from_primary(profile: dict[str, Any]) -> None:
    try:
        state = runtime_service.managed_state()
    except ConfigurationError:
        return
    primary_name = state.get("profile")
    if not isinstance(primary_name, str) or not primary_name:
        raise ConfigurationError(
            "the active primary runtime has no profile identity; refusing to "
            "guess its GPU allocation"
        )
    primary = load_profile(primary_name)
    primary_bdfs = primary["runtime"].get("gpu_bdfs")
    worker_bdfs = profile["runtime"].get("gpu_bdfs")
    if not isinstance(primary_bdfs, list) or not isinstance(worker_bdfs, list):
        raise ConfigurationError(
            "both primary and worker-pool profiles must use stable GPU BDFs"
        )
    overlap = sorted(
        {str(value).lower() for value in primary_bdfs}
        & {str(value).lower() for value in worker_bdfs}
    )
    if overlap:
        raise ConfigurationError(
            "worker-pool GPU allocation overlaps the active primary runtime: "
            + ",".join(overlap)
        )


def _start_command(
    profile: str,
    *,
    host: str,
    port: int,
    ready_timeout: int,
) -> list[str]:
    supervisor = [
        str(ROOT / "run"),
        "worker-pool",
        "supervise",
        "--profile",
        profile,
        "--host",
        host,
        "--port",
        str(port),
        "--ready-timeout",
        str(ready_timeout),
    ]
    return [
        "systemd-run",
        "--user",
        "--unit",
        UNIT,
        "--collect",
        "--service-type=exec",
        "--property=KillMode=control-group",
        "--property=KillSignal=SIGINT",
        "--property=SendSIGKILL=no",
        "--property=TimeoutStopSec=240",
        f"--working-directory={ROOT}",
        *supervisor,
    ]


def start(
    profile_name: str = DEFAULT_PROFILE,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    ready_timeout: int = 1200,
    required_power_cap_w: int = 285,
    dry_run: bool = False,
) -> None:
    if not 1 <= port <= 65535:
        raise ConfigurationError("worker-pool port must be between 1 and 65535")
    if ready_timeout < 1 or required_power_cap_w < 1:
        raise ConfigurationError("timeouts and power cap must be positive")
    profile = _profile(profile_name)
    _assert_disjoint_from_primary(profile)
    command = _start_command(
        profile_name,
        host=host,
        port=port,
        ready_timeout=ready_timeout,
    )
    if dry_run:
        print(" ".join(command))
        return
    if _unit_state() == "active":
        raise ConfigurationError(f"persistent unit is already active: {UNIT}")

    doctor_module.doctor(profile_name)
    subprocess.run(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            str(POWER_CHECK),
            "--watts",
            str(required_power_cap_w),
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(command, cwd=ROOT, check=True)

    deadline = time.monotonic() + ready_timeout
    while time.monotonic() < deadline:
        try:
            state = runtime_service.managed_state(state_path=STATE_PATH)
        except ConfigurationError:
            state = None
        if state is not None and runtime_service._ready(state):
            print(
                f"worker-pool ready PID={state['pid']} URL={state['url']} "
                f"unit={UNIT}"
            )
            return
        unit_state = _unit_state()
        if unit_state not in ("active", "activating"):
            raise ConfigurationError(
                f"worker-pool unit failed before readiness: state={unit_state}"
            )
        time.sleep(1)
    raise ConfigurationError(
        f"worker-pool readiness timed out after {ready_timeout}s; unit remains active"
    )


def supervise(
    profile_name: str,
    *,
    host: str,
    port: int,
    ready_timeout: int,
) -> None:
    _profile(profile_name)
    stopping = Event()

    def request_stop(signum: int, frame: object) -> None:
        del signum, frame
        stopping.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        runtime_service.start(
            profile_name,
            profile_name,
            host=host,
            port=port,
            wait_ready=True,
            ready_timeout=ready_timeout,
            state_path=STATE_PATH,
        )
        while not stopping.wait(5):
            try:
                runtime_service.managed_state(state_path=STATE_PATH)
            except ConfigurationError as exc:
                raise ConfigurationError("worker-pool runtime exited") from exc
    except ConfigurationError:
        if not stopping.is_set():
            raise
    finally:
        if stopping.is_set():
            try:
                runtime_service.stop(timeout=180, state_path=STATE_PATH)
            except ConfigurationError:
                STATE_PATH.unlink(missing_ok=True)


def status() -> int:
    unit_state = _unit_state()
    try:
        state = runtime_service.managed_state(state_path=STATE_PATH)
    except ConfigurationError:
        print(f"stopped unit={UNIT} state={unit_state}")
        return 3 if unit_state in ("inactive", "failed") else 2
    label = "ready" if runtime_service._ready(state) else "starting"
    print(
        f"{label} PID={state['pid']} URL={state['url']} "
        f"profile={state['profile']} unit={UNIT} state={unit_state}"
    )
    return 0 if label == "ready" and unit_state == "active" else 2


def stop(*, timeout: int = 240, dry_run: bool = False) -> None:
    if timeout < 1:
        raise ConfigurationError("worker-pool stop timeout must be positive")
    command = ["systemctl", "--user", "stop", UNIT]
    if dry_run:
        print(" ".join(command))
        return
    if _unit_state() in ("inactive", "failed"):
        try:
            runtime_service.managed_state(state_path=STATE_PATH)
        except ConfigurationError:
            print("worker-pool already stopped")
            return
    subprocess.run(command, cwd=ROOT, timeout=timeout, check=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _unit_state() in ("inactive", "failed"):
            try:
                runtime_service.managed_state(state_path=STATE_PATH)
            except ConfigurationError:
                STATE_PATH.unlink(missing_ok=True)
                print(f"worker-pool stopped unit={UNIT}")
                return
        time.sleep(0.5)
    raise ConfigurationError(
        "worker-pool did not stop before timeout; no force signal was sent"
    )


def logs(*, follow: bool = False, lines: int = 100) -> None:
    runtime_service.logs(
        follow=follow,
        lines=lines,
        state_path=STATE_PATH,
    )
