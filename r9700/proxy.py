from __future__ import annotations

import json
import hashlib
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import ConfigurationError, ROOT, load_model, read_dotenv
from .service import managed_state


PROXY_ROOT = ROOT / ".runtime" / "litellm"
VENV = PROXY_ROOT / "venv"
STATE_PATH = PROXY_ROOT / "service.json"
CONFIG_PATH = ROOT / "config" / "litellm.yaml"
REQUIREMENTS_PATH = ROOT / "constraints" / "litellm-py312.txt"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 4000
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


def _config_sha256() -> str:
    return hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()


def _config_matches(state: dict[str, Any]) -> bool:
    return state.get("config_sha256") == _config_sha256()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def _start_ticks(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text().split()[21])


def _process_stat(pid: int) -> list[str]:
    return Path(f"/proc/{pid}/stat").read_text().split()


def _state() -> dict[str, Any] | None:
    try:
        value = json.loads(STATE_PATH.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _identity_alive(state: dict[str, Any]) -> bool:
    try:
        pid = int(state["pid"])
        fields = _process_stat(pid)
        return (
            fields[2] != "Z"
            and state.get("boot_id") == _boot_id()
            and int(fields[21]) == int(state["start_ticks"])
            and os.getpgid(pid) == int(state["pgid"])
        )
    except (KeyError, ValueError, OSError, ProcessLookupError):
        return False


def _get(url: str, *, key: str | None = None, timeout: float = 2) -> Any:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _post(url: str, payload: dict[str, Any], *, key: str, timeout: float) -> Any:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _healthy(url: str, timeout: float = 2) -> bool:
    try:
        _get(url.rstrip("/") + "/health/liveliness", timeout=timeout)
        return True
    except (OSError, ValueError, urllib.error.URLError):
        return False


def _backend_healthy(url: str, timeout: float = 2) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _port_available(host: str, port: int) -> bool:
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    with socket.socket() as probe:
        probe.settimeout(0.3)
        return probe.connect_ex((probe_host, port)) != 0


def _master_key(dotenv: dict[str, str]) -> str:
    key = os.environ.get("LITELLM_MASTER_KEY") or dotenv.get(
        "LITELLM_MASTER_KEY"
    )
    if not key:
        raise ConfigurationError(
            "LITELLM_MASTER_KEY must be set in the environment or .env"
        )
    return key


def install() -> None:
    if sys.version_info[:2] != (3, 12):
        raise ConfigurationError("LiteLLM installer requires Python 3.12")
    uv = shutil.which("uv")
    if uv is None:
        raise ConfigurationError("uv is required to install LiteLLM")
    PROXY_ROOT.mkdir(parents=True, exist_ok=True)
    if not (VENV / "bin" / "python").is_file():
        subprocess.run(
            [uv, "venv", "--python", sys.executable, str(VENV)],
            check=True,
        )
    python = VENV / "bin" / "python"
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--requirement",
            str(REQUIREMENTS_PATH),
        ],
        check=True,
    )
    subprocess.run(
        [uv, "pip", "check", "--python", str(python)],
        check=True,
    )
    version = subprocess.run(
        [python, "-c", "from importlib.metadata import version; print(version('litellm'))"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"LiteLLM installed: version={version} venv={VENV}")


def start(
    *,
    host: str | None = None,
    port: int | None = None,
    backend_url: str | None = None,
    wait_ready: bool = False,
    ready_timeout: float = 120,
) -> dict[str, Any]:
    executable = VENV / "bin" / "litellm"
    if not executable.is_file():
        raise ConfigurationError("LiteLLM is not installed; run ./run proxy install")
    dotenv = read_dotenv()
    host = host or os.environ.get("LITELLM_HOST") or dotenv.get("LITELLM_HOST", DEFAULT_HOST)
    port = port or int(os.environ.get("LITELLM_PORT") or dotenv.get("LITELLM_PORT", DEFAULT_PORT))
    backend_url = (
        backend_url
        or os.environ.get("LITELLM_BACKEND_URL")
        or dotenv.get("LITELLM_BACKEND_URL", DEFAULT_BACKEND_URL)
    )
    master_key = _master_key(dotenv)
    existing = _state()
    if existing and _identity_alive(existing):
        raise ConfigurationError(
            f"LiteLLM proxy is already running: PID={existing['pid']} URL={existing['url']}"
        )
    if existing:
        STATE_PATH.unlink(missing_ok=True)
    if not _backend_healthy(backend_url):
        raise ConfigurationError(f"inference backend is not healthy: {backend_url}")
    if not _port_available(host, port):
        raise ConfigurationError(f"listener already owns {host}:{port}")
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    log_path = ROOT / "logs" / "litellm" / f"proxy-{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(executable),
        "--config",
        str(CONFIG_PATH),
    ]
    environment = os.environ.copy()
    # LiteLLM/Click treats DEBUG as a boolean CLI environment variable. Host
    # values such as DEBUG=release must not leak into the isolated proxy.
    environment.pop("DEBUG", None)
    environment.update(
        {
            "HOST": host,
            "PORT": str(port),
            "LITELLM_MASTER_KEY": master_key,
            "HOSTED_INFERENCE_API_BASE": backend_url.rstrip("/") + "/v1",
        }
    )
    with log_path.open("w") as log:
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
        "probe_url": f"http://127.0.0.1:{port}" if host == "0.0.0.0" else f"http://{host}:{port}",
        "backend_url": backend_url,
        "config_sha256": _config_sha256(),
        "log": str(log_path),
    }
    _atomic_json(STATE_PATH, state)
    print(f"LiteLLM started PID={process.pid} URL={state['url']} log={log_path}")
    if wait_ready:
        wait(timeout=ready_timeout)
    return state


def wait(*, timeout: float = 120) -> dict[str, Any]:
    state = _state()
    if not state or not _identity_alive(state):
        raise ConfigurationError("LiteLLM proxy is not running")
    if not _config_matches(state):
        raise ConfigurationError("LiteLLM proxy is running with a stale config")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _identity_alive(state):
            raise ConfigurationError(f"LiteLLM exited before readiness; log={state['log']}")
        if _healthy(state["probe_url"]):
            print(f"LiteLLM ready PID={state['pid']} URL={state['url']}")
            return state
        time.sleep(0.5)
    raise ConfigurationError(f"LiteLLM readiness timed out after {timeout}s")


def status() -> int:
    state = _state()
    if not state or not _identity_alive(state):
        print("stopped")
        return 3
    if not _config_matches(state):
        print(
            f"stale-config PID={state['pid']} URL={state['url']} "
            f"backend={state['backend_url']}"
        )
        return 2
    label = "ready" if _healthy(state["probe_url"]) else "starting"
    print(f"{label} PID={state['pid']} URL={state['url']} backend={state['backend_url']}")
    return 0 if label == "ready" else 2


def stop(*, timeout: float = 30) -> None:
    state = _state()
    if not state:
        print("already stopped")
        return
    if not _identity_alive(state):
        raise ConfigurationError("LiteLLM identity is stale; refusing to signal an unverified PID")
    pid = int(state["pid"])
    os.kill(pid, signal.SIGINT)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _identity_alive(state):
            STATE_PATH.unlink(missing_ok=True)
            print(f"LiteLLM stopped PID={pid}")
            return
        time.sleep(0.25)
    raise ConfigurationError(f"LiteLLM did not stop in {timeout}s; no force signal was sent")


def logs(*, follow: bool = False, lines: int = 100) -> None:
    state = _state()
    if not state:
        raise ConfigurationError("no LiteLLM service state")
    command = ["tail", "-n", str(lines)]
    if follow:
        command.append("-f")
    command.append(state["log"])
    raise SystemExit(subprocess.run(command, check=False).returncode)


def test(*, timeout: float = 120) -> None:
    state = _state()
    if not state or not _identity_alive(state):
        raise ConfigurationError("LiteLLM proxy is not running")
    if not _config_matches(state):
        raise ConfigurationError("LiteLLM proxy is running with a stale config")
    dotenv = read_dotenv()
    key = _master_key(dotenv)
    runtime_state = managed_state()
    active_profile = runtime_state.get("profile")
    if not isinstance(active_profile, str) or not active_profile:
        raise ConfigurationError(
            "managed runtime state predates flat production profiles; restart it"
        )
    active_model = load_model(active_profile)["served_name"]
    try:
        models = _get(state["probe_url"] + "/v1/models", key=key, timeout=timeout)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise ConfigurationError(f"LiteLLM model-list request failed: {exc}") from exc
    identifiers = {row.get("id") for row in models.get("data", []) if isinstance(row, dict)}
    if active_model not in identifiers:
        raise ConfigurationError(
            f"LiteLLM model list does not expose active model {active_model}"
        )
    try:
        completion = _post(
            state["probe_url"] + "/v1/chat/completions",
            {
                "model": active_model,
                "messages": [{"role": "user", "content": "Reply with OK"}],
                "max_tokens": 8,
            },
            key=key,
            timeout=timeout,
        )
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise ConfigurationError(f"LiteLLM chat-completion request failed: {exc}") from exc
    choices = completion.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ConfigurationError("LiteLLM chat completion returned no choices")
    print(f"LiteLLM proxy test: PASS model={active_model} chat_completion=true")
