from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from .backends import (
    backend_manifest_sha256,
    build_command,
    build_environment,
    runtime_backend,
    verify_backend_install,
)
from .config import (
    ConfigurationError,
    ROOT,
    load_profile,
    read_dotenv,
    resolve_model_directory,
    validate_compatibility,
)
from .models import verify_model


STATE_PATH = ROOT / ".runtime" / "service.json"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def _start_ticks(pid: int) -> int:
    fields = Path(f"/proc/{pid}/stat").read_text().split()
    return int(fields[21])


def _state() -> dict[str, Any] | None:
    if not STATE_PATH.is_file():
        return None
    try:
        value = json.loads(STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def managed_state() -> dict[str, Any]:
    state = _state()
    if not state or not _identity_alive(state):
        raise ConfigurationError("managed service is not running")
    return state


def _identity_alive(state: dict[str, Any]) -> bool:
    try:
        pid = int(state["pid"])
        process_status = Path(f"/proc/{pid}/status").read_text()
        process_state = next(
            line.split()[1] for line in process_status.splitlines() if line.startswith("State:")
        )
        return (
            process_state != "Z"
            and state.get("boot_id") == _boot_id()
            and _start_ticks(pid) == int(state["start_ticks"])
            and os.getpgid(pid) == int(state["pgid"])
        )
    except (KeyError, ValueError, OSError, ProcessLookupError, StopIteration):
        return False


def _health(url: str, timeout: float = 2) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _vllm_engine_alive(pid: int) -> bool:
    try:
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
    except OSError:
        return False
    for child in children:
        try:
            child_pid = int(child)
            process_name = Path(f"/proc/{child_pid}/comm").read_text().strip()
            process_status = Path(f"/proc/{child_pid}/status").read_text()
            process_state = next(
                line.split()[1]
                for line in process_status.splitlines()
                if line.startswith("State:")
            )
        except (OSError, ValueError, StopIteration):
            continue
        if process_name.startswith("VLLM::Engine") and process_state != "Z":
            return True
    return False


def _ready(state: dict[str, Any]) -> bool:
    if not _health(state["url"]):
        return False
    if state.get("backend", "vllm") == "vllm":
        return _vllm_engine_alive(int(state["pid"]))
    return True


def _port_available(host: str, port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.3)
        return probe.connect_ex((host, port)) != 0


def _native_environment(runtime: dict[str, Any]) -> dict[str, str]:
    return build_environment(runtime)


def _command(
    model: dict[str, Any],
    runtime: dict[str, Any],
    model_directory: Path,
    host: str,
    port: int,
) -> list[str]:
    return build_command(model, runtime, model_directory, host, port)


def start(
    model_name: str,
    runtime_name: str,
    *,
    model_directory: str | None = None,
    host: str | None = None,
    port: int | None = None,
    backend: str | None = None,
    wait_ready: bool = False,
    ready_timeout: float = 900,
) -> dict[str, Any]:
    deployment = load_profile(model_name)
    runtime_deployment = load_profile(runtime_name)
    if deployment["_path"] != runtime_deployment["_path"]:
        raise ConfigurationError(
            "model and runtime must come from the same production profile"
        )
    model = deployment["model"]
    runtime = runtime_deployment["runtime"]
    selected_backend = runtime_backend(runtime)
    selected_manifest_sha256 = backend_manifest_sha256(runtime)
    if backend is not None and backend != selected_backend:
        raise ConfigurationError(
            f"runtime {runtime['name']} selects {selected_backend}, not {backend}"
        )
    verify_backend_install(runtime)
    validate_compatibility(model, runtime)
    directory = resolve_model_directory(model, model_directory)
    verify_model(model_name, str(directory))
    existing = _state()
    if existing and _identity_alive(existing):
        raise ConfigurationError(
            f"service is already running: PID={existing['pid']} URL={existing['url']}"
        )
    if existing:
        STATE_PATH.unlink(missing_ok=True)
    dotenv = read_dotenv()
    host = host or os.environ.get("TARGET_HOST") or dotenv.get("TARGET_HOST", "127.0.0.1")
    port = port or int(os.environ.get("TARGET_PORT") or dotenv.get("TARGET_PORT", "8000"))
    if not _port_available(host, port):
        raise ConfigurationError(f"listener already owns {host}:{port}")
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    log_path = (
        ROOT
        / "logs"
        / "runtime"
        / f"{selected_backend}-{runtime['name']}-{timestamp}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = _command(model, runtime, directory, host, port)
    environment = _native_environment(runtime)
    header = {
        "event": "launch",
        "timestamp": datetime.now().astimezone().isoformat(),
        "backend": selected_backend,
        "profile": deployment["name"],
        "profile_sha256": deployment["_sha256"],
        "recipe": runtime["recipe"],
        "model": model["name"],
        "model_profile_sha256": model["_sha256"],
        "runtime": runtime["name"],
        "runtime_profile_sha256": runtime["_sha256"],
        "backend_manifest_sha256": selected_manifest_sha256,
        "runtime_manifest_sha256": selected_manifest_sha256,
        "command": command,
    }
    with log_path.open("w") as log:
        log.write(json.dumps(header, sort_keys=True) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    state = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "pid": process.pid,
        "pgid": os.getpgid(process.pid),
        "start_ticks": _start_ticks(process.pid),
        "boot_id": _boot_id(),
        "url": f"http://{host}:{port}",
        "log": str(log_path),
        "model": model["name"],
        "profile": deployment["name"],
        "profile_sha256": deployment["_sha256"],
        "backend": selected_backend,
        "recipe": runtime["recipe"],
        "runtime": runtime["name"],
        "runtime_profile_sha256": runtime["_sha256"],
        "backend_manifest_sha256": selected_manifest_sha256,
        "runtime_manifest_sha256": selected_manifest_sha256,
    }
    _atomic_json(STATE_PATH, state)
    print(f"started PID={process.pid} URL={state['url']} log={log_path}")
    if wait_ready:
        wait(timeout=ready_timeout)
    return state


def wait(*, timeout: float = 900) -> dict[str, Any]:
    state = _state()
    if not state or not _identity_alive(state):
        raise ConfigurationError("managed service is not running")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _identity_alive(state):
            raise ConfigurationError(f"service exited before readiness; log={state['log']}")
        if _ready(state):
            print(f"ready PID={state['pid']} URL={state['url']}")
            return state
        time.sleep(1)
    raise ConfigurationError(f"service readiness timed out after {timeout}s")


def status() -> int:
    state = _state()
    if not state or not _identity_alive(state):
        print("stopped")
        return 3
    label = "ready" if _ready(state) else "starting"
    print(
        f"{label} PID={state['pid']} PGID={state['pgid']} "
        f"URL={state['url']} backend={state.get('backend', 'vllm')} "
        f"recipe={state.get('recipe', '-')} runtime={state['runtime']} log={state['log']}"
    )
    return 0 if label == "ready" else 2


def stop(*, timeout: float = 180) -> None:
    state = _state()
    if not state:
        print("already stopped")
        return
    if not _identity_alive(state):
        raise ConfigurationError(
            "service identity is stale; refusing to signal an unverified PID"
        )
    pid = int(state["pid"])
    os.kill(pid, signal.SIGINT)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _identity_alive(state):
            STATE_PATH.unlink(missing_ok=True)
            print(f"stopped PID={pid}")
            return
        time.sleep(0.5)
    raise ConfigurationError(
        f"service did not stop in {timeout}s; no force signal was sent"
    )


def logs(*, follow: bool = False, lines: int = 100) -> None:
    state = _state()
    if not state:
        raise ConfigurationError("no managed service state")
    command = ["tail", "-n", str(lines)]
    if follow:
        command.append("-f")
    command.append(state["log"])
    raise SystemExit(subprocess.run(command, check=False).returncode)
