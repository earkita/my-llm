from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from .backends import runtime_backend
from .config import ConfigurationError, ROOT, list_runtime_profiles, load_profile
from .service import managed_state, status as service_status


START_SCRIPT = (
    ROOT / "skills" / "start-r9700-runtime" / "scripts" / "start-runtime.sh"
)
STOP_SCRIPT = (
    ROOT / "skills" / "stop-r9700-runtime" / "scripts" / "stop-runtime.sh"
)
UNIT = "r9700-runtime.service"


def _running_state() -> dict[str, Any] | None:
    try:
        return managed_state()
    except ConfigurationError:
        return None


def _target(profile_name: str) -> tuple[str, dict[str, Any]]:
    profile = load_profile(profile_name)
    return str(profile["name"]), profile


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
    dry_run: bool = False,
) -> list[str]:
    if ready_timeout < 1:
        raise ConfigurationError("ready timeout must be a positive integer")
    if port is not None and not 1 <= port <= 65535:
        raise ConfigurationError("port must be between 1 and 65535")
    command = [str(START_SCRIPT), "--profile", profile_name]
    if host:
        command.extend(("--host", host))
    if port is not None:
        command.extend(("--port", str(port)))
    command.extend(("--ready-timeout", str(ready_timeout)))
    if dry_run:
        command.append("--dry-run")
    return command


def start(
    profile_name: str,
    *,
    host: str | None = None,
    port: int | None = None,
    ready_timeout: int = 900,
    dry_run: bool = False,
) -> None:
    name, _ = _target(profile_name)
    state = _running_state()
    if state and not dry_run:
        raise ConfigurationError(
            f"{state['profile']} is already running at {state['url']}; "
            f"use './run launcher switch {name}' to replace it explicitly"
        )
    _run(
        _start_command(
            name,
            host=host,
            port=port,
            ready_timeout=ready_timeout,
            dry_run=dry_run,
        )
    )


def stop(*, timeout: int | None = None, dry_run: bool = False) -> None:
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
    stop_timeout: int | None = None,
    dry_run: bool = False,
) -> None:
    name, _ = _target(profile_name)
    state = _running_state()
    if state and state.get("profile") == name and not dry_run:
        print(
            f"already running profile={name} PID={state['pid']} URL={state['url']}"
        )
        return

    if state:
        print(f"switching {state['profile']} -> {name}", flush=True)
        stop(timeout=stop_timeout, dry_run=dry_run)
    elif dry_run:
        print(
            f"no managed runtime is active; switch will start {name}",
            flush=True,
        )

    _run(
        _start_command(
            name,
            host=host,
            port=port,
            ready_timeout=ready_timeout,
            dry_run=dry_run,
        )
    )


def _layout(profile: dict[str, Any]) -> tuple[str, int, str]:
    runtime = profile["runtime"]
    parallel = runtime["parallel"]
    tensor = int(parallel["tensor"])
    pipeline = int(parallel["pipeline"])
    data = int(parallel.get("data", 1))
    gpu_count = tensor * pipeline * data
    backend = runtime_backend(runtime)
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
    service_rc = service_status()
    result = subprocess.run(
        ["systemctl", "--user", "is-active", UNIT],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    unit_state = result.stdout.strip() or "unavailable"
    print(f"unit={UNIT} state={unit_state}")
    if service_rc not in (0, 2, 3):
        raise ConfigurationError(f"unexpected service status code: {service_rc}")


def logs(*, follow: bool = False, lines: int = 100) -> None:
    if lines < 1:
        raise ConfigurationError("log line count must be a positive integer")
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
        print("  s) status    l) logs    x) stop    q) quit")

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
            if choice in {"x", "stop"}:
                if state:
                    stop()
                else:
                    print("already stopped")
                continue
            if not choice.isdigit() or not 1 <= int(choice) <= len(profiles):
                print("Unknown selection.", file=sys.stderr)
                continue

            selected = str(profiles[int(choice) - 1]["name"])
            if state and state.get("profile") == selected:
                print(f"{selected} is already running.")
                continue
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
