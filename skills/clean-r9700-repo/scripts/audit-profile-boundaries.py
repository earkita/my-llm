#!/usr/bin/env python3
"""Audit the production/development boundary for self-contained profiles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _contains_extends(value: Any) -> bool:
    if isinstance(value, dict):
        return "extends" in value or any(
            _contains_extends(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_extends(item) for item in value)
    return False


def _load_profiles(
    directory: Path, errors: list[str]
) -> list[tuple[Path, dict[str, Any]]]:
    profiles: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: invalid JSON: {exc}")
            continue
        if not isinstance(value, dict):
            errors.append(f"{path}: profile root must be an object")
            continue
        if _contains_extends(value):
            errors.append(f"{path}: profile inheritance is forbidden")
        profiles.append((path, value))
    return profiles


def audit(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    production = _load_profiles(root / "profiles" / "production", errors)
    development = _load_profiles(root / "profiles" / "dev", errors)

    if not production:
        errors.append("profiles/production must contain at least one profile")

    production_summary: list[dict[str, Any]] = []
    development_summary: list[dict[str, Any]] = []

    for path, profile in production:
        model = profile.get("model")
        runtime = profile.get("runtime")
        if not isinstance(model, dict) or not isinstance(runtime, dict):
            errors.append(f"{path}: production profile needs object model and runtime")
            continue
        model_name = model.get("name")
        runtime_name = runtime.get("name")
        if not isinstance(model_name, str) or not model_name:
            errors.append(f"{path}: model.name must be a non-empty string")
            model_name = f"<invalid:{path.name}>"
        if not isinstance(runtime_name, str) or not runtime_name:
            errors.append(f"{path}: runtime.name must be a non-empty string")
        if profile.get("status") != "production-ready":
            errors.append(f"{path}: production profile status must be production-ready")
        if runtime.get("status") != "production-ready":
            errors.append(f"{path}: production runtime status must be production-ready")
        if "experimental_modes" in runtime:
            errors.append(f"{path}: runtime.experimental_modes belongs in profiles/dev")
        production_summary.append(
            {
                "path": str(path.relative_to(root)),
                "model": model_name,
                "runtime": runtime_name,
                "recipe": runtime.get("recipe"),
            }
        )

    for path, profile in development:
        runtime = profile.get("runtime")
        if not isinstance(runtime, dict):
            errors.append(f"{path}: development profile needs an object runtime")
            continue
        if profile.get("status") == "production-ready":
            errors.append(f"{path}: production-ready profile belongs in profiles/production")
        if runtime.get("status") == "production-ready":
            errors.append(f"{path}: production-ready runtime belongs in profiles/production")
        development_summary.append(
            {
                "path": str(path.relative_to(root)),
                "model": profile.get("model", {}).get("name"),
                "runtime": runtime.get("name"),
                "recipe": runtime.get("recipe"),
                "experimental_modes": sorted(runtime.get("experimental_modes", {})),
            }
        )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "production": production_summary,
        "development": development_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="repository root (defaults to the skill's repository)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()

    result = audit(args.root.resolve())
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for profile in result["production"]:
            print(
                "PRODUCTION "
                f"{profile['model']} -> {profile['runtime']} ({profile['recipe']})"
            )
        for profile in result["development"]:
            modes = ",".join(profile["experimental_modes"]) or "-"
            print(
                "DEV "
                f"{profile['model']} -> {profile['runtime']} "
                f"({profile['recipe']}); modes={modes}"
            )
        for warning in result["warnings"]:
            print(f"WARNING {warning}", file=sys.stderr)
        for error in result["errors"]:
            print(f"ERROR {error}", file=sys.stderr)
        print("PASS profile boundary audit" if result["ok"] else "FAIL profile boundary audit")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
