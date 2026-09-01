from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from .api import test_api
from .config import ConfigurationError
from .ras import compare, snapshot
from .service import managed_state, start, stop


FORBIDDEN = (
    "Traceback (most recent call last)",
    "force killing remaining processes",
    "resource_tracker: There appear to be",
    "OutOfMemoryError",
    "RCCL error",
)


def _try_stop(timeout: float) -> None:
    try:
        managed_state()
    except ConfigurationError:
        return
    stop(timeout=timeout)


def gate(
    *,
    model_name: str,
    runtime_name: str,
    ready_timeout: float,
    stop_timeout: float,
    output: Path,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ConfigurationError(f"refusing to overwrite lifecycle artifact: {output}")
    try:
        managed_state()
    except ConfigurationError:
        pass
    else:
        raise ConfigurationError("lifecycle gate requires a stopped managed service")
    before = snapshot()
    legs = []
    logs = []
    error = None
    try:
        for leg in range(2):
            state = start(
                model_name,
                runtime_name,
                wait_ready=True,
                ready_timeout=ready_timeout,
            )
            logs.append(state["log"])
            api_result = test_api(
                url=state["url"],
                model_name=model_name,
                runtime_name=runtime_name,
                timeout=600,
            )
            stop(timeout=stop_timeout)
            legs.append({"leg": leg + 1, "api": api_result, "clean_stop": True})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            _try_stop(stop_timeout)
        except Exception as cleanup_exc:
            error = f"{error or ''}; cleanup={type(cleanup_exc).__name__}: {cleanup_exc}"
        after = snapshot()
        ras = compare(before, after)
        forbidden_hits = {}
        for name in logs:
            text = Path(name).read_text(errors="replace") if Path(name).is_file() else ""
            hits = [marker for marker in FORBIDDEN if marker in text]
            if hits:
                forbidden_hits[name] = hits
        payload = {
            "schema_version": 1,
            "generated_at": datetime.now().astimezone().isoformat(),
            "model": model_name,
            "runtime": runtime_name,
            "legs": legs,
            "logs": logs,
            "ras": ras,
            "forbidden_log_hits": forbidden_hits,
            "error": error,
            "passed": len(legs) == 2 and ras["passed"] and not forbidden_hits and not error,
        }
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, output)
    if not payload["passed"]:
        raise ConfigurationError("lifecycle gate failed")
    print(json.dumps({"passed": True, "output": str(output)}, indent=2))
    return payload
