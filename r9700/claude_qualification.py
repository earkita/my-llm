from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

from . import proxy
from .config import ConfigurationError, ROOT, load_profile
from .service import managed_state as runtime_managed_state, state_path_for_runtime


LITERAL_EXPECTED = "DZIAŁA"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _template_path(profile_name: str) -> Path:
    matches = sorted(
        (ROOT / "templates" / ".claude").glob(
            f"*/{profile_name}.settings.local.json"
        )
    )
    if len(matches) != 1:
        raise ConfigurationError(
            f"expected one Claude Code template for {profile_name}, found {len(matches)}"
        )
    return matches[0]


def _health(url: str, *, timeout: float = 5) -> bool:
    try:
        with urllib.request.urlopen(
            url.rstrip("/") + "/health/liveliness", timeout=timeout
        ) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _parse_stream(stdout: str) -> tuple[dict[str, Any], list[str], dict[str, int]]:
    result: dict[str, Any] | None = None
    tool_ids: set[str] = set()
    tools: list[str] = []
    thinking_fingerprints: set[str] = set()
    thinking_characters = 0
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(
                f"Claude Code emitted invalid stream JSON: {line[:200]!r}"
            ) from exc
        if not isinstance(event, dict):
            continue
        if event.get("type") == "result":
            result = event
        message = event.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in {"thinking", "redacted_thinking", "reasoning"}:
                fingerprint = json.dumps(block, sort_keys=True)
                if fingerprint not in thinking_fingerprints:
                    thinking_fingerprints.add(fingerprint)
                    for key in ("thinking", "text", "content"):
                        value = block.get(key)
                        if isinstance(value, str):
                            thinking_characters += len(value)
                            break
            if block.get("type") != "tool_use":
                continue
            tool_id = str(block.get("id", ""))
            if tool_id and tool_id in tool_ids:
                continue
            if tool_id:
                tool_ids.add(tool_id)
            name = block.get("name")
            if isinstance(name, str):
                tools.append(name)
    if result is None:
        raise ConfigurationError("Claude Code stream contains no final result")
    return result, tools, {
        "blocks": len(thinking_fingerprints),
        "characters": thinking_characters,
    }


def _run_claude(
    *,
    executable: str,
    settings: Path,
    model: str,
    cwd: Path,
    prompt: str,
    tools: str,
    timeout: float,
) -> dict[str, Any]:
    command = [
        executable,
        "--bare",
        "--settings",
        str(settings),
        "--model",
        model,
        "--print",
        "--verbose",
        "--output-format",
        "stream-json",
        "--no-session-persistence",
        "--permission-mode",
        "dontAsk",
        f"--tools={tools}",
    ]
    if tools:
        command.append(f"--allowedTools={tools}")
    command.append(prompt)
    environment = os.environ.copy()
    environment["MY_LLM_REPO_ROOT"] = str(ROOT)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ConfigurationError(
            f"Claude Code qualification timed out after {timeout}s"
        ) from exc
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ConfigurationError(
            f"Claude Code exited with {completed.returncode}: {detail[-2000:]}"
        )
    result, tool_names, thinking = _parse_stream(completed.stdout)
    return {
        "elapsed_seconds": elapsed,
        "result": result,
        "tool_names": tool_names,
        "thinking": thinking,
        "stderr": completed.stderr.strip(),
    }


def qualify(
    *,
    profile_name: str,
    output: Path,
    timeout: float = 600,
) -> dict[str, Any]:
    profile = load_profile(profile_name)
    model = profile["model"]
    runtime = profile["runtime"]
    settings_path = _template_path(profile_name)
    settings = json.loads(settings_path.read_text())
    settings_env = settings.get("env")
    if not isinstance(settings_env, dict):
        raise ConfigurationError("Claude Code template has no environment")
    alias = settings_env.get("ANTHROPIC_MODEL")
    if not isinstance(alias, str) or alias not in profile["stack"]["litellm_aliases"]:
        raise ConfigurationError("Claude Code template model alias differs from profile")
    fast_alias = settings_env.get("ANTHROPIC_SMALL_FAST_MODEL")
    if not isinstance(fast_alias, str) or fast_alias not in profile["stack"][
        "litellm_aliases"
    ]:
        raise ConfigurationError("Claude Code fast model alias differs from profile")

    runtime_state = runtime_managed_state(
        state_path=state_path_for_runtime(runtime)
    )
    proxy_state = proxy.managed_state()
    aliases = profile["stack"]["litellm_aliases"]
    if len(aliases) != 2:
        raise ConfigurationError(
            "Claude Code qualification requires thinking and fast aliases"
        )
    thinking_alias, expected_fast_alias = aliases
    worker_pool = runtime.get("role") == "worker-pool"
    identity_checks = {
        "active_profile": runtime_state.get("profile") == profile["name"],
        "active_model": runtime_state.get("model") == model["name"],
        "active_runtime": runtime_state.get("runtime") == runtime["name"],
        "runtime_profile_sha256": (
            runtime_state.get("runtime_profile_sha256") == runtime["_sha256"]
        ),
        "proxy_profile": worker_pool or proxy_state.get("profile") == profile["name"],
        "proxy_config_sha256": proxy._config_matches(proxy_state),
        "proxy_health": _health(str(settings_env["ANTHROPIC_BASE_URL"])),
        "thinking_enabled": (
            settings.get("effortLevel") == "high"
            and alias == thinking_alias
            and settings_env.get("ANTHROPIC_DEFAULT_SONNET_MODEL") == alias
            and settings_env.get("ANTHROPIC_DEFAULT_OPUS_MODEL") == alias
            and settings_env.get("ANTHROPIC_SMALL_FAST_MODEL")
            == expected_fast_alias
            and settings_env.get("ANTHROPIC_DEFAULT_HAIKU_MODEL")
            == expected_fast_alias
            and "CLAUDE_CODE_DISABLE_THINKING" not in settings_env
            and "MAX_THINKING_TOKENS" not in settings_env
        ),
    }
    if not all(identity_checks.values()):
        raise ConfigurationError(
            "Claude Code qualification requires the exact managed profile and proxy: "
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
    started = datetime.now().astimezone()
    literal = _run_claude(
        executable=executable,
        settings=settings_path,
        model=alias,
        cwd=ROOT,
        prompt=f"Odpowiedz dokładnie jednym słowem: {LITERAL_EXPECTED}",
        tools="",
        timeout=min(timeout, 180),
    )
    literal_result = literal["result"]
    fast = _run_claude(
        executable=executable,
        settings=settings_path,
        model=fast_alias,
        cwd=ROOT,
        prompt="Odpowiedz dokładnie: FAST_OK",
        tools="",
        timeout=min(timeout, 180),
    )
    fast_result = fast["result"]
    reasoning = _run_claude(
        executable=executable,
        settings=settings_path,
        model=alias,
        cwd=ROOT,
        prompt=(
            "Rozwiąż równanie 17x + 29 = 199. Przeprowadź rozumowanie, ale w "
            "odpowiedzi końcowej wypisz dokładnie: X=10"
        ),
        tools="",
        timeout=min(timeout, 180),
    )
    reasoning_result = reasoning["result"]

    with tempfile.TemporaryDirectory(prefix="qwen-claude-code-") as temporary:
        workspace = Path(temporary)
        implementation = workspace / "calculator.py"
        tests = workspace / "test_calculator.py"
        implementation.write_text(
            "def add(left: int, right: int) -> int:\n"
            "    \"\"\"Return the sum of two integers.\"\"\"\n"
            "    return left - right\n"
        )
        tests.write_text(
            "import unittest\n\n"
            "from calculator import add\n\n\n"
            "class CalculatorTests(unittest.TestCase):\n"
            "    def test_positive(self) -> None:\n"
            "        self.assertEqual(add(17, 19), 36)\n\n"
            "    def test_negative(self) -> None:\n"
            "        self.assertEqual(add(-4, 9), 5)\n\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        )
        test_hash_before = _sha256(tests)
        code = _run_claude(
            executable=executable,
            settings=settings_path,
            model=alias,
            cwd=workspace,
            prompt=(
                "Inspect calculator.py and test_calculator.py. Fix the bug by editing "
                "calculator.py only, then run "
                f"{ROOT / '.venv/bin/python'} -m unittest -v. "
                "Use the available tools; finish with KWALIFIKACJA_OK only after tests pass."
            ),
            tools="Read,Edit,Bash",
            timeout=timeout,
        )
        verification = subprocess.run(
            [str(ROOT / ".venv/bin/python"), "-m", "unittest", "-v"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
        )
        code_checks = {
            "fixture_tests_pass": verification.returncode == 0,
            "implementation_changed": "return left + right" in implementation.read_text(),
            "tests_unchanged": _sha256(tests) == test_hash_before,
            "tool_loop": (
                int(code["result"].get("num_turns", 0)) >= 2
                and bool(code["tool_names"])
            ),
            "no_permission_denials": not code["result"].get("permission_denials"),
        }

    checks = {
        **identity_checks,
        "literal_response": str(literal_result.get("result", "")).strip()
        == LITERAL_EXPECTED,
        "literal_success": (
            literal_result.get("subtype") == "success"
            and not literal_result.get("is_error")
        ),
        "fast_response": str(fast_result.get("result", "")).strip() == "FAST_OK",
        "fast_success": (
            fast_result.get("subtype") == "success"
            and not fast_result.get("is_error")
        ),
        "fast_without_thinking": fast["thinking"]["blocks"] == 0,
        "reasoning_response": str(reasoning_result.get("result", ""))
        .strip()
        .endswith("X=10"),
        "reasoning_success": (
            reasoning_result.get("subtype") == "success"
            and not reasoning_result.get("is_error")
        ),
        "reasoning_block": sum(
            case["thinking"]["blocks"] for case in (literal, reasoning, code)
        )
        > 0,
        **code_checks,
    }
    warnings = []
    if literal_result.get("result") != LITERAL_EXPECTED:
        warnings.append("literal response required surrounding-whitespace normalization")
    if "</antml>" in str(code["result"].get("result", "")):
        warnings.append("final response contains a stray </antml> protocol tag")
    payload = {
        "schema_version": 2,
        "started_at": started.isoformat(),
        "finished_at": datetime.now().astimezone().isoformat(),
        "profile": profile["name"],
        "model": model["name"],
        "runtime": runtime["name"],
        "runtime_state": {
            key: runtime_state.get(key)
            for key in (
                "pid",
                "url",
                "profile",
                "model",
                "runtime",
                "runtime_profile_sha256",
            )
        },
        "proxy_state": {
            key: proxy_state.get(key)
            for key in ("pid", "url", "profile", "config_sha256")
        },
        "claude_code_version": version,
        "template": str(settings_path.relative_to(ROOT)),
        "template_sha256": _sha256(settings_path),
        "aliases": {"thinking": alias, "fast": fast_alias},
        "thinking_mode": "enabled-high-normalized-to-xhigh",
        "checks": checks,
        "passed": all(checks.values()),
        "warnings": warnings,
        "cases": {
            "literal": literal,
            "fast": fast,
            "reasoning": reasoning,
            "code_edit_and_test": {
                **code,
                "host_verification_stdout": verification.stdout,
                "host_verification_stderr": verification.stderr,
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    temporary_output.write_text(json.dumps(payload, indent=2) + "\n")
    temporary_output.replace(output)
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
        raise ConfigurationError("Claude Code qualification failed")
    return payload
