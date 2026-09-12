from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from . import proxy, service as runtime_service, worker_pool
from .backends import runtime_backend
from .config import (
    ConfigurationError,
    ROOT,
    activate_runtime_mode,
    list_runtime_profiles,
    load_profile,
)
from .service import managed_state, status as service_status
from .service import wait as service_wait


START_SCRIPT = (
    ROOT / "skills" / "start-r9700-runtime" / "scripts" / "start-runtime.sh"
)
STOP_SCRIPT = (
    ROOT / "skills" / "stop-r9700-runtime" / "scripts" / "stop-runtime.sh"
)
STACK_SCRIPT = (
    ROOT / "skills" / "manage-r9700-stack" / "scripts" / "manage-stack.sh"
)
RUNTIME_UNIT = "r9700-runtime.service"
PROXY_UNIT = "r9700-litellm-proxy.service"
MULTI_STATE_PATH = ROOT / ".runtime" / "qwen-multi.json"


def _running_state() -> dict[str, Any] | None:
    try:
        return managed_state()
    except ConfigurationError:
        return None


def _proxy_running_state() -> dict[str, Any] | None:
    try:
        return proxy.managed_state()
    except ConfigurationError:
        return None


def _target(profile_name: str) -> tuple[str, dict[str, Any]]:
    profile = load_profile(profile_name)
    if profile["runtime"].get("role") == "worker-pool":
        raise ConfigurationError(
            f"{profile_name} is a secondary worker pool; use "
            f"'./run worker-pool start {profile_name}'"
        )
    return str(profile["name"]), profile


def _state_matches_target(
    state: dict[str, Any],
    *,
    profile_name: str,
    model_name: str,
    runtime_name: str,
    runtime_mode: str | None,
    compatible_profiles: set[str] | None = None,
) -> bool:
    """Match both current and pre-runtime-mode managed state records."""
    expected = {
        "profile": profile_name,
        "model": model_name,
        "runtime": runtime_name,
    }
    present = [key for key in expected if state.get(key) is not None]
    accepted_profiles = {profile_name, *(compatible_profiles or set())}
    if (
        not present
        or (
            state.get("profile") is not None
            and state.get("profile") not in accepted_profiles
        )
        or any(
            state.get(key) != expected[key]
            for key in ("model", "runtime")
            if state.get(key) is not None
        )
    ):
        return False
    active_mode = state.get("runtime_mode")
    if runtime_mode is None:
        return active_mode in (None, "")
    # An experimental runtime must carry its explicit mode and runtime name;
    # an older ambiguous state must never be upgraded implicitly.
    return active_mode == runtime_mode and state.get("runtime") == runtime_name


def _compatible_primary_profiles(profile: dict[str, Any]) -> set[str]:
    components = profile.get("components")
    if not isinstance(components, dict):
        return set()
    primary = components.get("primary")
    if not isinstance(primary, dict):
        return set()
    compatible = primary.get("compatible_profile")
    return {compatible} if isinstance(compatible, str) and compatible else set()


def _ensure_worker_component(
    profile: dict[str, Any],
    *,
    dry_run: bool,
) -> dict[str, Any] | None:
    components = profile.get("components")
    if not isinstance(components, dict):
        return None
    component = components["worker_pool"]
    worker_name = str(component["profile"])
    expected = load_profile(worker_name)
    try:
        state = runtime_service.managed_state(
            state_path=runtime_service.WORKER_POOL_STATE_PATH
        )
    except ConfigurationError:
        state = None
    if state is not None:
        checks = {
            "profile": state.get("profile") == worker_name,
            "model": state.get("model") == expected["model"]["name"],
            "runtime": state.get("runtime") == expected["runtime"]["name"],
            "runtime_profile_sha256": (
                state.get("runtime_profile_sha256")
                == expected["runtime"]["_sha256"]
            ),
        }
        if not all(checks.values()):
            raise ConfigurationError(
                "active worker pool differs from the multi-profile component: "
                + str(checks)
            )
        print(
            f"worker-pool already ready: profile={worker_name} "
            f"PID={state['pid']} URL={state['url']}"
        )
        return state
    worker_pool.start(worker_name, dry_run=dry_run)
    if dry_run:
        return None
    return runtime_service.managed_state(
        state_path=runtime_service.WORKER_POOL_STATE_PATH
    )


def _record_multi_state(
    profile: dict[str, Any], worker_state: dict[str, Any] | None
) -> None:
    primary_state = managed_state()
    try:
        proxy_state = proxy.managed_state()
    except ConfigurationError:
        proxy_state = {}
    runtime_service._atomic_json(
        MULTI_STATE_PATH,
        {
            "schema_version": 1,
            "profile": profile["name"],
            "profile_sha256": profile["_sha256"],
            "primary": {
                "profile": primary_state.get("profile"),
                "pid": primary_state.get("pid"),
                "url": primary_state.get("url"),
                "runtime_profile_sha256": primary_state.get(
                    "runtime_profile_sha256"
                ),
            },
            "worker_pool": {
                "profile": (worker_state or {}).get("profile"),
                "pid": (worker_state or {}).get("pid"),
                "url": (worker_state or {}).get("url"),
                "runtime_profile_sha256": (worker_state or {}).get(
                    "runtime_profile_sha256"
                ),
            },
            "proxy": {
                "pid": proxy_state.get("pid"),
                "url": proxy_state.get("url"),
                "config_sha256": proxy_state.get("config_sha256"),
            },
        },
    )


def _desired_runtime(
    profile: dict[str, Any],
    runtime_mode: str | None,
) -> tuple[str | None, dict[str, Any]]:
    runtime = profile["runtime"]
    if runtime_mode:
        runtime = activate_runtime_mode(profile["model"], runtime, runtime_mode)
    return runtime_mode, runtime


def _run(command: list[str]) -> None:
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode:
        raise ConfigurationError(
            f"launcher command failed with exit code {result.returncode}"
        )


def _start_command(
    profile_name: str,
    *,
    host: str | None = None,
    port: int | None = None,
    ready_timeout: int = 900,
    runtime_mode: str | None = None,
    dry_run: bool = False,
) -> list[str]:
    if ready_timeout < 1:
        raise ConfigurationError("ready timeout must be a positive integer")
    if port is not None and not 1 <= port <= 65535:
        raise ConfigurationError("port must be between 1 and 65535")
    command = [str(START_SCRIPT), "--profile", profile_name]
    if runtime_mode:
        command.extend(("--runtime-mode", runtime_mode))
    if host:
        command.extend(("--host", host))
    if port is not None:
        command.extend(("--port", str(port)))
    command.extend(("--ready-timeout", str(ready_timeout)))
    if dry_run:
        command.append("--dry-run")
    return command


def _stack_start_command(
    profile_name: str,
    *,
    ready_timeout: int = 900,
    proxy_ready_timeout: int = 120,
    runtime_mode: str | None = None,
    dry_run: bool = False,
) -> list[str]:
    if ready_timeout < 1 or proxy_ready_timeout < 1:
        raise ConfigurationError("ready timeouts must be positive integers")
    command = [
        str(STACK_SCRIPT),
        "start",
        "--preset",
        profile_name,
        "--runtime-ready-timeout",
        str(ready_timeout),
        "--proxy-ready-timeout",
        str(proxy_ready_timeout),
    ]
    if runtime_mode:
        command.extend(("--runtime-mode", runtime_mode))
    if dry_run:
        command.append("--dry-run")
    return command


def _stack_stop_command(
    *,
    runtime_timeout: int | None = None,
    proxy_timeout: int | None = None,
    dry_run: bool = False,
) -> list[str]:
    if runtime_timeout is not None and runtime_timeout < 1:
        raise ConfigurationError("runtime stop timeout must be a positive integer")
    if proxy_timeout is not None and proxy_timeout < 1:
        raise ConfigurationError("proxy stop timeout must be a positive integer")
    command = [str(STACK_SCRIPT), "stop"]
    if runtime_timeout is not None:
        command.extend(("--runtime-timeout", str(runtime_timeout)))
    if proxy_timeout is not None:
        command.extend(("--proxy-timeout", str(proxy_timeout)))
    if dry_run:
        command.append("--dry-run")
    return command


def start(
    profile_name: str,
    *,
    host: str | None = None,
    port: int | None = None,
    ready_timeout: int = 900,
    proxy_ready_timeout: int = 120,
    with_litellm: bool = True,
    runtime_mode: str | None = None,
    dry_run: bool = False,
) -> None:
    name, profile = _target(profile_name)
    runtime_mode, desired_runtime = _desired_runtime(profile, runtime_mode)
    state = _running_state()
    proxy_state = _proxy_running_state()
    matches_running = bool(
        state
        and _state_matches_target(
            state,
            profile_name=name,
            model_name=profile["model"]["name"],
            runtime_name=desired_runtime["name"],
            runtime_mode=runtime_mode,
            compatible_profiles=_compatible_primary_profiles(profile),
        )
    )
    if state and not matches_running and not dry_run:
        raise ConfigurationError(
            f"{state.get('profile', state.get('model', 'unknown'))} "
            f"({state.get('runtime', 'legacy state')}) is already running at "
            f"{state['url']}; "
            f"use './run launcher switch {name}' to replace it explicitly"
        )
    if state and not with_litellm and not dry_run:
        raise ConfigurationError(
            f"{name} is already running at {state['url']}"
        )
    if proxy_state and not with_litellm and not dry_run:
        raise ConfigurationError(
            "LiteLLM is already running; omit --runtime-only or stop the full stack"
        )
    if with_litellm:
        if host or port is not None:
            raise ConfigurationError(
                "--host and --port are available only with --runtime-only; "
                "configure TARGET_* and LITELLM_* in .env for the full stack"
            )
        if state and not dry_run:
            service_wait(timeout=ready_timeout)
        _run(
            _stack_start_command(
                name,
                ready_timeout=ready_timeout,
                proxy_ready_timeout=proxy_ready_timeout,
                runtime_mode=runtime_mode,
                dry_run=dry_run,
            )
        )
        worker_state = _ensure_worker_component(profile, dry_run=dry_run)
        if not dry_run and profile.get("components"):
            _record_multi_state(profile, worker_state)
            print(
                f"multi stack ready: profile={name} primary=:8000 "
                "workers=:8100 litellm=:4000"
            )
        return
    _run(
        _start_command(
            name,
            host=host,
            port=port,
            ready_timeout=ready_timeout,
            runtime_mode=runtime_mode,
            dry_run=dry_run,
        )
    )
    worker_state = _ensure_worker_component(profile, dry_run=dry_run)
    if not dry_run and profile.get("components"):
        _record_multi_state(profile, worker_state)
        print(f"multi runtime ready: profile={name} primary=:8000 workers=:8100")


def stop(
    *,
    timeout: int | None = None,
    proxy_timeout: int | None = None,
    with_litellm: bool = True,
    dry_run: bool = False,
    profile_name: str | None = None,
) -> None:
    multi_profile = load_profile(profile_name) if profile_name else None
    if (
        with_litellm
        and multi_profile is not None
        and isinstance(multi_profile.get("components"), dict)
    ):
        proxy_stop = [
            str(ROOT / "skills" / "stop-litellm-proxy" / "scripts" / "stop-proxy.sh")
        ]
        runtime_stop = [str(STOP_SCRIPT)]
        worker_stop = [str(ROOT / "run"), "worker-pool", "stop"]
        if proxy_timeout is not None:
            proxy_stop.extend(("--timeout", str(proxy_timeout)))
        if timeout is not None:
            runtime_stop.extend(("--timeout", str(timeout)))
            worker_stop.extend(("--timeout", str(timeout)))
        if dry_run:
            proxy_stop.append("--dry-run")
            runtime_stop.append("--dry-run")
            worker_stop.append("--dry-run")
        _run(proxy_stop)
        _run(worker_stop)
        _run(runtime_stop)
        if not dry_run:
            MULTI_STATE_PATH.unlink(missing_ok=True)
        return
    if with_litellm:
        _run(
            _stack_stop_command(
                runtime_timeout=timeout,
                proxy_timeout=proxy_timeout,
                dry_run=dry_run,
            )
        )
        return
    if not dry_run and _proxy_running_state():
        raise ConfigurationError(
            "LiteLLM is running; omit --runtime-only to stop the proxy "
            "before the model"
        )
    if timeout is not None and timeout < 1:
        raise ConfigurationError("stop timeout must be a positive integer")
    command = [str(STOP_SCRIPT)]
    if timeout is not None:
        command.extend(("--timeout", str(timeout)))
    if dry_run:
        command.append("--dry-run")
    _run(command)


def switch(
    profile_name: str,
    *,
    host: str | None = None,
    port: int | None = None,
    ready_timeout: int = 900,
    proxy_ready_timeout: int = 120,
    stop_timeout: int | None = None,
    proxy_stop_timeout: int | None = None,
    with_litellm: bool = True,
    runtime_mode: str | None = None,
    dry_run: bool = False,
) -> None:
    name, profile = _target(profile_name)
    runtime_mode, desired_runtime = _desired_runtime(profile, runtime_mode)
    if with_litellm:
        if host or port is not None:
            raise ConfigurationError(
                "--host and --port are available only with --runtime-only; "
                "configure TARGET_* and LITELLM_* in .env for the full stack"
            )
        _stack_start_command(
            name,
            ready_timeout=ready_timeout,
            proxy_ready_timeout=proxy_ready_timeout,
            runtime_mode=runtime_mode,
            dry_run=dry_run,
        )
        _stack_stop_command(
            runtime_timeout=stop_timeout,
            proxy_timeout=proxy_stop_timeout,
            dry_run=dry_run,
        )
    else:
        _start_command(
            name,
            host=host,
            port=port,
            ready_timeout=ready_timeout,
            runtime_mode=runtime_mode,
            dry_run=dry_run,
        )
        if stop_timeout is not None and stop_timeout < 1:
            raise ConfigurationError("stop timeout must be a positive integer")
    state = _running_state()
    matches_running = bool(
        state
        and _state_matches_target(
            state,
            profile_name=name,
            model_name=profile["model"]["name"],
            runtime_name=desired_runtime["name"],
            runtime_mode=runtime_mode,
            compatible_profiles=_compatible_primary_profiles(profile),
        )
    )
    if state and matches_running and with_litellm:
        start(
            name,
            ready_timeout=ready_timeout,
            proxy_ready_timeout=proxy_ready_timeout,
            with_litellm=True,
            runtime_mode=runtime_mode,
            dry_run=dry_run,
        )
        return
    if state and matches_running and not dry_run:
        print(
            f"already running profile={name} PID={state['pid']} URL={state['url']}"
        )
        return

    if state:
        active = state.get("profile", state.get("model", "unknown"))
        print(f"switching {active} -> {name}", flush=True)
        stop(
            timeout=stop_timeout,
            proxy_timeout=proxy_stop_timeout,
            with_litellm=with_litellm,
            dry_run=dry_run,
        )
    elif dry_run:
        print(
            f"no managed runtime is active; switch will start {name}",
            flush=True,
        )

    start(
        name,
        host=host,
        port=port,
        ready_timeout=ready_timeout,
        proxy_ready_timeout=proxy_ready_timeout,
        with_litellm=with_litellm,
        runtime_mode=runtime_mode,
        dry_run=dry_run,
    )


def _unit_state(unit: str) -> str:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", unit],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "unavailable"


def _print_component_status(
    label: str,
    status_function: Any,
    unit: str,
) -> None:
    print(f"{label}:")
    status_rc = status_function()
    print(f"unit={unit} state={_unit_state(unit)}")
    if status_rc not in (0, 2, 3):
        raise ConfigurationError(
            f"unexpected {label.lower()} status code: {status_rc}"
        )


def _layout(profile: dict[str, Any]) -> tuple[str, int, str]:
    runtime = profile["runtime"]
    parallel = runtime["parallel"]
    tensor = int(parallel["tensor"])
    pipeline = int(parallel["pipeline"])
    data = int(parallel.get("data", 1))
    gpu_count = tensor * pipeline * data
    backend = runtime_backend(runtime)
    components = profile.get("components")
    if isinstance(components, dict):
        worker = load_profile(str(components["worker_pool"]["profile"]))
        worker_parallel = worker["runtime"]["parallel"]
        worker_count = (
            int(worker_parallel["tensor"])
            * int(worker_parallel["pipeline"])
            * int(worker_parallel.get("data", 1))
        )
        primary_layout = f"TP{tensor}/PP{pipeline}"
        if parallel.get("enable_expert_parallel"):
            primary_layout += "/EP"
        return backend, gpu_count + worker_count, primary_layout + "+DP4/TP1"
    if backend == "llama-cpp":
        return backend, gpu_count, "layer-split"

    parts = [f"TP{tensor}", f"PP{pipeline}"]
    if data > 1:
        parts.append(f"DP{data}")
    if parallel.get("enable_expert_parallel"):
        parts.append("EP")
    return backend, gpu_count, "/".join(parts)


def list_profiles() -> None:
    rows: list[tuple[str, str, str, str, str]] = []
    for record in list_runtime_profiles():
        profile = load_profile(record["name"])
        backend, gpu_count, layout = _layout(profile)
        context = f"{int(profile['runtime']['limits']['max_model_len']):,}"
        rows.append(
            (
                str(profile["name"]),
                backend,
                str(gpu_count),
                layout,
                context,
            )
        )

    headers = ("PROFILE", "BACKEND", "GPUS", "LAYOUT", "MAX CONTEXT")
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print(
        "  ".join(
            value.ljust(widths[index]) for index, value in enumerate(headers)
        )
    )
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(
            "  ".join(
                value.ljust(widths[index]) for index, value in enumerate(row)
            )
        )


def status() -> None:
    _print_component_status("Inference runtime", service_status, RUNTIME_UNIT)
    _print_component_status("Qwen worker pool", worker_pool.status, worker_pool.UNIT)
    _print_component_status("LiteLLM proxy", proxy.status, PROXY_UNIT)


def logs(
    *,
    component: str = "runtime",
    follow: bool = False,
    lines: int = 100,
) -> None:
    if lines < 1:
        raise ConfigurationError("log line count must be a positive integer")
    if component == "litellm":
        command = [str(ROOT / "run"), "proxy", "logs", "--lines", str(lines)]
        if follow:
            command.append("--follow")
        _run(command)
        return
    if component != "runtime":
        raise ConfigurationError(f"unknown log component: {component}")
    state = _running_state()
    if not state:
        raise ConfigurationError("no managed runtime is active")
    command = ["tail", "-n", str(lines)]
    if follow:
        command.append("-f")
    command.append(str(state["log"]))
    _run(command)


def _confirm_switch(current: str, target: str) -> bool:
    answer = input(f"Stop {current} and start {target}? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def interactive() -> None:
    if not sys.stdin.isatty():
        raise ConfigurationError(
            "interactive launcher requires a terminal; use "
            "'./run launcher --help' for scriptable commands"
        )

    profiles = list_runtime_profiles()
    while True:
        state = _running_state()
        print("\nR9700 model launcher")
        if state:
            print(
                f"Current: {state['profile']} ({state['backend']}) "
                f"PID={state['pid']} URL={state['url']}"
            )
        else:
            print("Current: stopped")
        print()
        for index, record in enumerate(profiles, start=1):
            print(f"  {index}) {record['name']:<22} {record['description']}")
        print("  s) status    l) runtime logs    p) proxy logs")
        print("  x) stop full stack               q) quit")

        try:
            choice = input("\nSelect: ").strip().lower()
            if choice in {"q", "quit"}:
                return
            if choice in {"s", "status"}:
                status()
                continue
            if choice in {"l", "logs"}:
                logs(lines=60)
                continue
            if choice in {"p", "proxy"}:
                logs(component="litellm", lines=60)
                continue
            if choice in {"x", "stop"}:
                stop(with_litellm=True)
                continue
            if not choice.isdigit() or not 1 <= int(choice) <= len(profiles):
                print("Unknown selection.", file=sys.stderr)
                continue

            selected = str(profiles[int(choice) - 1]["name"])
            if state and not _confirm_switch(str(state["profile"]), selected):
                print("Switch cancelled.")
                continue
            if state:
                switch(selected)
            else:
                start(selected)
        except ConfigurationError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
        except (EOFError, KeyboardInterrupt):
            print()
            return
