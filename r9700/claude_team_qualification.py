from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from . import proxy
from .config import ConfigurationError, ROOT, load_profile
from .service import WORKER_POOL_STATE_PATH, managed_state


LEAD = "qwen-team-lead"
WORKER_MARKERS = {
    "qwen-worker-explorer": "EXPLORER_OK",
    "qwen-worker-implementer-a": "IMPLEMENTER_A_OK",
    "qwen-worker-implementer-b": "IMPLEMENTER_B_OK",
    "qwen-worker-verifier": "VERIFIER_OK",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_status() -> str:
    return subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _stream_events(stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(
                f"Claude Code emitted invalid stream JSON: {line[:200]!r}"
            ) from exc
        if isinstance(event, dict):
            events.append(event)
    if not events:
        raise ConfigurationError("Claude Code emitted no stream events")
    return events


def _agent_calls(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for event in events:
        message = event.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            if block.get("name") not in {"Agent", "Task"}:
                continue
            value = block.get("input")
            if isinstance(value, dict):
                calls.append(value)
    return calls


def _resolved_workers(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    workers: dict[str, dict[str, Any]] = {}
    for event in events:
        value = event.get("tool_use_result")
        if not isinstance(value, dict):
            continue
        agent_type = value.get("agentType")
        if agent_type not in WORKER_MARKERS:
            continue
        content = value.get("content")
        texts = [
            str(block.get("text", "")).strip()
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ] if isinstance(content, list) else []
        workers[str(agent_type)] = {
            "agent_id": value.get("agentId"),
            "resolved_model": value.get("resolvedModel"),
            "text": "\n".join(texts).strip(),
            "duration_ms": value.get("totalDurationMs"),
            "tokens": value.get("totalTokens"),
            "tool_uses": value.get("totalToolUseCount"),
        }
    return workers


def _task_event_indexes(events: list[dict[str, Any]], subtype: str) -> list[int]:
    return [
        index
        for index, event in enumerate(events)
        if event.get("type") == "system" and event.get("subtype") == subtype
    ]


def _role_probe(
    agent_type: str,
    marker: str,
    *,
    timeout: float,
) -> dict[str, Any]:
    command = [
        str(ROOT / "scripts" / "claude-local.sh"),
        "--agent",
        agent_type,
        "--print",
        "--verbose",
        "--output-format",
        "stream-json",
        "--no-session-persistence",
        "--permission-mode",
        "dontAsk",
        "--tools=",
        f"Call no tools and respond with exactly {marker}.",
    ]
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    finished = time.monotonic()
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ConfigurationError(
            f"Claude worker probe {agent_type} exited with {completed.returncode}: "
            f"{detail[-1000:]}"
        )
    events = _stream_events(completed.stdout)
    result_events = [event for event in events if event.get("type") == "result"]
    if not result_events:
        raise ConfigurationError(f"Claude worker probe {agent_type} has no result")
    result = result_events[-1]
    model = next(
        (
            event.get("model")
            for event in events
            if event.get("type") == "system" and event.get("subtype") == "init"
        ),
        None,
    )
    return {
        "model": model,
        "result": str(result.get("result", "")).strip(),
        "success": result.get("subtype") == "success" and not result.get("is_error"),
        "permission_denials": result.get("permission_denials") or [],
        "started_monotonic": started,
        "finished_monotonic": finished,
        "duration_seconds": finished - started,
    }


def qualify(
    *,
    profile_name: str,
    output: Path,
    timeout: float = 300,
) -> dict[str, Any]:
    profile = load_profile(profile_name)
    agents = profile["stack"].get("claude_agents")
    expected_agents = {LEAD, *WORKER_MARKERS}
    if not isinstance(agents, dict) or set(agents) != expected_agents:
        raise ConfigurationError(
            "Claude team qualification requires the embedded Qwen lead and four workers"
        )

    settings_path = ROOT / ".claude" / "settings.local.json"
    agents_path = ROOT / ".claude" / "agents.local.json"
    if not settings_path.is_file() or not agents_path.is_file():
        raise ConfigurationError(
            "activate qwen-multi before qualification: ./run launcher start qwen-multi"
        )
    settings = json.loads(settings_path.read_text())
    materialized_agents = json.loads(agents_path.read_text())

    primary_state = managed_state()
    worker_state = managed_state(state_path=WORKER_POOL_STATE_PATH)
    proxy_state = proxy.managed_state()
    worker_profile = load_profile(profile["components"]["worker_pool"]["profile"])
    compatible_primary = profile["components"]["primary"]["compatible_profile"]
    identity_checks = {
        "primary_profile": primary_state.get("profile")
        in {profile["name"], compatible_primary},
        "primary_model": primary_state.get("model") == profile["model"]["name"],
        "primary_runtime": primary_state.get("runtime")
        == profile["runtime"]["name"],
        "worker_profile": worker_state.get("profile") == worker_profile["name"],
        "worker_model": worker_state.get("model") == worker_profile["model"]["name"],
        "worker_runtime": worker_state.get("runtime")
        == worker_profile["runtime"]["name"],
        "worker_runtime_sha256": worker_state.get("runtime_profile_sha256")
        == worker_profile["runtime"]["_sha256"],
        "proxy_config_sha256": proxy._config_matches(proxy_state),
        "settings_materialized": settings == profile["stack"]["claude_settings"],
        "agents_materialized": materialized_agents == agents,
    }
    if not all(identity_checks.values()):
        raise ConfigurationError(
            "Claude team qualification requires the exact active qwen-multi stack: "
            + json.dumps(identity_checks, sort_keys=True)
        )

    executable = shutil.which("claude")
    if executable is None:
        raise ConfigurationError("Claude Code executable is not installed")
    version = subprocess.run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    prompt = (
        "This is a five-agent background-launch qualification; do not edit, create, "
        "delete, read, search, or execute anything in the workspace. As "
        "qwen-team-lead, launch exactly four Agent tasks, one for each configured "
        "worker type. Set run_in_background=true on every Agent call and give them "
        "the unique names explorer, implementer-a, implementer-b, and verifier. "
        "Issue all four launches before waiting for any result. Tell every worker to "
        "call no tools and return its configured marker. Immediately after all four "
        "background launches are accepted, respond with exactly BACKGROUND_LAUNCHED. "
        "Do not wait for or summarize worker results in this headless launch phase."
    )
    command = [
        str(ROOT / "scripts" / "claude-qwen-team.sh"),
        "--print",
        "--verbose",
        "--output-format",
        "stream-json",
        "--no-session-persistence",
        "--permission-mode",
        "dontAsk",
        "--allowedTools=Agent",
        prompt,
    ]
    environment = os.environ.copy()
    environment["MY_LLM_REPO_ROOT"] = str(ROOT)
    status_before = _git_status()
    started_at = datetime.now().astimezone()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ConfigurationError(
            f"Claude team qualification timed out after {timeout}s"
        ) from exc
    status_after = _git_status()
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ConfigurationError(
            f"Claude Code exited with {completed.returncode}: {detail[-2000:]}"
        )

    events = _stream_events(completed.stdout)
    calls = _agent_calls(events)
    resolved = _resolved_workers(events)
    starts = _task_event_indexes(events, "task_started")
    finishes = _task_event_indexes(events, "task_notification")
    result_events = [event for event in events if event.get("type") == "result"]
    if not result_events:
        raise ConfigurationError("Claude team stream contains no final result")
    result = result_events[-1]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            name: executor.submit(
                _role_probe,
                name,
                marker,
                timeout=min(timeout, 180),
            )
            for name, marker in WORKER_MARKERS.items()
        }
        role_probes = {name: future.result() for name, future in futures.items()}
    status_after_probes = _git_status()

    expected_models = {
        name: agents[name]["model"] for name in WORKER_MARKERS
    }
    call_types = [call.get("subagent_type") for call in calls]
    call_names = [call.get("name") for call in calls]
    checks = {
        **identity_checks,
        "claude_success": (
            result.get("subtype") == "success" and not result.get("is_error")
        ),
        "lead_model": next(
            (
                event.get("model")
                for event in events
                if event.get("type") == "system"
                and event.get("subtype") == "init"
            ),
            None,
        )
        == agents[LEAD]["model"],
        "four_agent_calls": len(calls) == 4,
        "exact_worker_types": set(call_types) == set(WORKER_MARKERS),
        "unique_worker_names": len(call_names) == 4 and len(set(call_names)) == 4,
        "background_launches": len(calls) == 4
        and all(call.get("run_in_background") is True for call in calls),
        "all_started_before_first_finish": len(starts) == 4
        and (not finishes or max(starts) < min(finishes)),
        "role_probes_overlap": max(
            probe["started_monotonic"] for probe in role_probes.values()
        )
        < min(probe["finished_monotonic"] for probe in role_probes.values()),
        "worker_models": all(
            role_probes[name]["model"] == expected_models[name]
            for name in WORKER_MARKERS
        ),
        "worker_markers": all(
            role_probes[name]["result"] == WORKER_MARKERS[name]
            for name in WORKER_MARKERS
        ),
        "worker_success": all(
            role_probes[name]["success"]
            and not role_probes[name]["permission_denials"]
            for name in WORKER_MARKERS
        ),
        "lead_returned_after_launches": bool(
            str(result.get("result", "")).strip()
        ),
        "no_permission_denials": not result.get("permission_denials"),
        "workspace_unchanged": status_before == status_after == status_after_probes,
    }
    payload = {
        "schema_version": 1,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "profile": profile["name"],
        "claude_code_version": version,
        "settings_sha256": _sha256(settings_path),
        "agents_sha256": _sha256(agents_path),
        "primary": {
            key: primary_state.get(key)
            for key in ("pid", "profile", "model", "runtime", "url")
        },
        "worker_pool": {
            key: worker_state.get(key)
            for key in ("pid", "profile", "model", "runtime", "url")
        },
        "proxy": {
            key: proxy_state.get(key)
            for key in ("pid", "profile", "url", "config_sha256")
        },
        "lead": {
            "agent_type": LEAD,
            "model": agents[LEAD]["model"],
            "result": str(result.get("result", "")).strip(),
            "turns": result.get("num_turns"),
            "duration_ms": result.get("duration_ms"),
        },
        "workers_completed_during_launch": resolved,
        "worker_role_probes": role_probes,
        "launches": calls,
        "checks": checks,
        "passed": all(checks.values()),
        "stderr": completed.stderr.strip(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "passed": payload["passed"],
                "profile": profile["name"],
                "checks": checks,
                "output": str(output),
            },
            indent=2,
        )
    )
    if not payload["passed"]:
        raise ConfigurationError("Claude team qualification failed")
    return payload
