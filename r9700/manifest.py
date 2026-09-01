from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import ConfigurationError, ROOT, load_json


RECIPES_PATH = ROOT / "manifest" / "recipes.json"
DEFAULT_MANIFEST_PATH = ROOT / "manifest" / "runtime.json"
DEFAULT_INSTALL_PATH = ROOT / ".runtime" / "install.json"
# Compatibility names for callers that address the original DeepSeek recipe.
MANIFEST_PATH = DEFAULT_MANIFEST_PATH
INSTALL_PATH = DEFAULT_INSTALL_PATH


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recipe_registry() -> dict[str, Any]:
    registry = load_json(RECIPES_PATH)
    if registry.get("schema_version") != 1:
        raise ConfigurationError("unsupported runtime recipe registry schema")
    recipes = registry.get("recipes")
    if not isinstance(recipes, dict) or not recipes:
        raise ConfigurationError("runtime recipe registry is empty")
    default = registry.get("default_recipe")
    if not isinstance(default, str) or default not in recipes:
        raise ConfigurationError("runtime recipe registry has an invalid default")
    return registry


def recipe_names(*, backend: str | None = None) -> list[str]:
    registry = recipe_registry()
    return sorted(
        name
        for name, record in registry["recipes"].items()
        if backend is None or record.get("backend") == backend
    )


def default_recipe_name(*, backend: str = "vllm") -> str:
    registry = recipe_registry()
    preferred = registry["default_recipe"]
    if registry["recipes"][preferred].get("backend") == backend:
        return preferred
    matches = recipe_names(backend=backend)
    if not matches:
        raise ConfigurationError(f"no recipe is registered for backend {backend}")
    return matches[0]


def recipe_record(name: str) -> dict[str, Any]:
    records = recipe_registry()["recipes"]
    try:
        record = records[name]
    except KeyError as exc:
        raise ConfigurationError(f"unknown runtime recipe: {name}") from exc
    if not isinstance(record, dict):
        raise ConfigurationError(f"runtime recipe record is invalid: {name}")
    backend = record.get("backend")
    if backend not in ("vllm", "llama-cpp"):
        raise ConfigurationError(f"runtime recipe has an invalid backend: {name}")
    for field in ("path", "install_path"):
        if not isinstance(record.get(field), str) or not record[field]:
            raise ConfigurationError(f"runtime recipe {name} is missing {field}")
    return dict(record)


def recipe_manifest_path(name: str) -> Path:
    path = (ROOT / recipe_record(name)["path"]).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise ConfigurationError(f"runtime recipe leaves the repository: {name}") from exc
    return path


def recipe_install_path(name: str) -> Path:
    path = (ROOT / recipe_record(name)["install_path"]).resolve()
    try:
        path.relative_to(ROOT / ".runtime")
    except ValueError as exc:
        raise ConfigurationError(
            f"runtime recipe install path leaves .runtime: {name}"
        ) from exc
    return path


def recipe_manifest(name: str) -> dict[str, Any]:
    manifest = load_json(recipe_manifest_path(name))
    if manifest.get("schema_version") != 1:
        raise ConfigurationError(f"unsupported runtime recipe schema: {name}")
    return manifest


def runtime_manifest(recipe_name: str | None = None) -> dict[str, Any]:
    name = recipe_name or default_recipe_name(backend="vllm")
    if recipe_record(name)["backend"] != "vllm":
        raise ConfigurationError(f"runtime recipe is not a vLLM recipe: {name}")
    manifest = recipe_manifest(name)
    if not isinstance(manifest.get("environment"), dict) or not isinstance(
        manifest.get("sources"), dict
    ):
        raise ConfigurationError(f"vLLM runtime recipe is incomplete: {name}")
    return manifest


def manifest_sha256(recipe_name: str | None = None) -> str:
    name = recipe_name or default_recipe_name(backend="vllm")
    return sha256_file(recipe_manifest_path(name))


def recipe_venv(name: str) -> Path:
    manifest = runtime_manifest(name)
    path = (ROOT / manifest["environment"]["venv"]).resolve()
    try:
        path.relative_to(ROOT / ".runtime")
    except ValueError as exc:
        raise ConfigurationError(f"recipe venv leaves .runtime: {name}") from exc
    return path


def recipe_source_root(name: str) -> Path:
    manifest = runtime_manifest(name)
    path = (ROOT / manifest["environment"]["source_root"]).resolve()
    try:
        path.relative_to(ROOT / ".runtime")
    except ValueError as exc:
        raise ConfigurationError(f"recipe source root leaves .runtime: {name}") from exc
    return path


def _constraint_key(line: str) -> str | None:
    match = re.match(r"^\s*([A-Za-z0-9_.-]+)\s*(?:===|==|~=|!=|<=|>=|<|>)", line)
    if not match:
        return None
    return match.group(1).lower().replace("_", "-")


def effective_constraints_text(name: str) -> str:
    environment = runtime_manifest(name)["environment"]
    base = (ROOT / environment["constraints"]).read_text().splitlines()
    overlays = environment.get("constraint_overlays", [])
    if not overlays:
        return "\n".join(base) + "\n"
    overlay_lines: list[str] = []
    replaced: set[str] = set()
    for overlay in overlays:
        lines = (ROOT / overlay["path"]).read_text().splitlines()
        overlay_lines.extend(lines)
        replaced.update(
            key for line in lines if (key := _constraint_key(line)) is not None
        )
    retained = [line for line in base if _constraint_key(line) not in replaced]
    return "\n".join(
        [*retained, "", f"# Recipe overrides for {name}", *overlay_lines, ""]
    )


def recipe_constraints_path(name: str) -> Path:
    environment = runtime_manifest(name)["environment"]
    if not environment.get("constraint_overlays"):
        return ROOT / environment["constraints"]
    path = recipe_install_path(name).parent / "constraints.txt"
    content = effective_constraints_text(name)
    if not path.is_file() or path.read_text() != content:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(content)
        os.replace(temporary, path)
    return path


def recipe_patch_ids(name: str) -> set[str]:
    metadata = recipe_record(name)
    manifest = recipe_manifest(name)
    if metadata["backend"] != "vllm":
        patches = manifest.get("patches", [])
        identifiers: set[str] = set()
        for patch in patches:
            stem = Path(patch["path"]).name
            prefix = stem.split("-", 1)[0]
            if prefix.isdigit():
                identifiers.add(prefix)
        return identifiers
    identifiers: set[str] = set()
    for source in manifest["sources"].values():
        for patch in source.get("patches", []):
            stem = Path(patch["path"]).name
            prefix = stem.split("-", 1)[0]
            if prefix.isdigit():
                identifiers.add(prefix)
    return identifiers


def verify_assets(
    manifest: dict[str, Any] | None = None, *, recipe_name: str | None = None
) -> None:
    name = recipe_name or default_recipe_name(backend="vllm")
    manifest = manifest or runtime_manifest(name)
    constraints = manifest["environment"]
    entries = [(constraints["constraints"], constraints["constraints_sha256"])]
    entries.extend(
        (overlay["path"], overlay["sha256"])
        for overlay in constraints.get("constraint_overlays", [])
    )
    for source in manifest["sources"].values():
        entries.extend((row["path"], row["sha256"]) for row in source["patches"])
    problems = []
    for relative, expected in entries:
        path = ROOT / relative
        if not path.is_file():
            problems.append(f"missing: {relative}")
        else:
            actual = sha256_file(path)
            if actual != expected:
                problems.append(f"sha256 mismatch: {relative}: {actual} != {expected}")
    if problems:
        raise ConfigurationError(
            f"runtime recipe assets are invalid ({name}):\n" + "\n".join(problems)
        )


def git_output(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def tracked_diff_sha256(tree: Path, *, full_index: bool = False) -> str:
    command = ["git", "diff", "--binary"]
    if full_index:
        # Abbreviated object IDs depend on the checkout's object database.
        # A full index makes a patch-image hash stable across shallow and
        # object-rich clones of the same commit.
        command.append("--full-index")
    command.append("HEAD")
    result = subprocess.run(
        command, cwd=tree, check=True, capture_output=True
    )
    return hashlib.sha256(result.stdout).hexdigest()


def source_identity(
    name: str, manifest: dict[str, Any] | None = None, *, recipe_name: str | None = None
) -> dict[str, str]:
    selected = recipe_name or default_recipe_name(backend="vllm")
    manifest = manifest or runtime_manifest(selected)
    spec = manifest["sources"][name]
    tree = recipe_source_root(selected) / name
    if not (tree / ".git").is_dir():
        raise ConfigurationError(f"source tree is absent: {tree}")
    head = git_output("rev-parse", "HEAD", cwd=tree)
    diff = tracked_diff_sha256(
        tree, full_index=bool(spec.get("full_index_diff", False))
    )
    if head != spec["commit"]:
        raise ConfigurationError(f"{name} HEAD mismatch: {head} != {spec['commit']}")
    if diff != spec["expected_diff_sha256"]:
        raise ConfigurationError(
            f"{name} patch image mismatch: {diff} != {spec['expected_diff_sha256']}"
        )
    return {"head": head, "tracked_diff_sha256": diff}


def verify_install(recipe_name: str | None = None) -> dict[str, Any]:
    selected = recipe_name or default_recipe_name(backend="vllm")
    verify_assets(recipe_name=selected)
    install_path = recipe_install_path(selected)
    if not install_path.is_file():
        raise ConfigurationError(
            f"runtime recipe {selected} is not installed; "
            f"run ./run install --recipe {selected}"
        )
    installed = load_json(install_path)
    expected_manifest = manifest_sha256(selected)
    if installed.get("runtime_manifest_sha256") != expected_manifest:
        raise ConfigurationError("installed runtime belongs to a different recipe manifest")
    installed_recipe = installed.get("recipe")
    if installed_recipe is not None and installed_recipe != selected:
        raise ConfigurationError("installed runtime attests a different recipe")
    manifest = runtime_manifest(selected)
    actual_sources = {
        name: source_identity(name, manifest, recipe_name=selected)
        for name in sorted(manifest["sources"])
    }
    if installed.get("sources") != actual_sources:
        raise ConfigurationError("installed source identities are stale")
    return installed


def write_install_manifest(recipe_name: str | None = None) -> dict[str, Any]:
    selected = recipe_name or default_recipe_name(backend="vllm")
    manifest = runtime_manifest(selected)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "recipe": selected,
        "runtime_manifest_sha256": manifest_sha256(selected),
        "recipe_registry_sha256": sha256_file(RECIPES_PATH),
        "repository_head": repository_head(),
        "sources": {
            name: source_identity(name, manifest, recipe_name=selected)
            for name in sorted(manifest["sources"])
        },
    }
    install_path = recipe_install_path(selected)
    install_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = install_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, install_path)
    return payload


def repository_head() -> str | None:
    try:
        return git_output("rev-parse", "HEAD", cwd=ROOT)
    except subprocess.CalledProcessError:
        return None
