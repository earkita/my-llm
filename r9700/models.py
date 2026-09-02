from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from .config import (
    ConfigurationError,
    ROOT,
    load_model,
    load_profile,
    resolve_model_directory,
)
from .manifest import recipe_record, recipe_venv, verify_install


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_worker_python(profile_name: str) -> Path:
    deployment = load_profile(profile_name)
    recipe = str(deployment["runtime"]["recipe"])
    record = recipe_record(recipe)
    if record["backend"] != "vllm":
        recipe = str(record.get("foundation_recipe") or "")
        if not recipe:
            raise ConfigurationError(
                "non-vLLM model download requires a foundation recipe"
            )
    verify_install(recipe)
    return recipe_venv(recipe) / "bin" / "python"


def _verify_conversion(model: dict, destination: Path) -> None:
    expected = model.get("conversion")
    if expected is None:
        return
    if not isinstance(expected, dict):
        raise ConfigurationError("model conversion identity must be an object")
    expected_id = expected.get("id")
    expected_manifest_sha256 = expected.get("manifest_sha256")
    if not isinstance(expected_id, str) or not isinstance(
        expected_manifest_sha256, str
    ):
        raise ConfigurationError("model conversion identity is incomplete")

    manifest_path = destination / "conversion-manifest.json"
    success_path = destination / "_SUCCESS.json"
    if not manifest_path.is_file() or not success_path.is_file():
        raise ConfigurationError("converted checkpoint attestation is absent")
    actual_manifest_sha256 = _sha256_file(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ConfigurationError("conversion manifest SHA-256 differs from profile")

    manifest = json.loads(manifest_path.read_text())
    success = json.loads(success_path.read_text())
    if manifest.get("state") != "complete":
        raise ConfigurationError("conversion manifest is not complete")
    if manifest.get("conversion_id") != expected_id:
        raise ConfigurationError("conversion ID differs from profile")
    if success.get("conversion_id") != expected_id:
        raise ConfigurationError("conversion success marker has another ID")
    if success.get("manifest_sha256") != actual_manifest_sha256:
        raise ConfigurationError("conversion success marker has another manifest hash")

    result = manifest.get("result")
    if not isinstance(result, dict):
        raise ConfigurationError("conversion manifest has no result inventory")
    expected_weight_bytes = model.get("checkpoint_weight_bytes")
    if (
        expected_weight_bytes is not None
        and result.get("tensor_bytes") != expected_weight_bytes
    ):
        raise ConfigurationError("converted tensor payload differs from profile")
    for name, result_key in (
        ("config.json", "config_sha256"),
        (model.get("index_file", "model.safetensors.index.json"), "index_sha256"),
    ):
        path = destination / name
        if not path.is_file() or _sha256_file(path) != result.get(result_key):
            raise ConfigurationError(f"converted {name} differs from manifest")


def _verify_auxiliary_artifacts(model: dict) -> list[dict[str, object]]:
    artifacts = model.get("auxiliary_artifacts", [])
    if not isinstance(artifacts, list):
        raise ConfigurationError("auxiliary_artifacts must be a list")
    verified: list[dict[str, object]] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ConfigurationError("auxiliary artifact must be an object")
        name = artifact.get("name")
        raw_path = artifact.get("path")
        if not isinstance(name, str) or not name:
            raise ConfigurationError("auxiliary artifact has no name")
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise ConfigurationError(
                f"auxiliary artifact {name} path must be absolute"
            )
        path = Path(raw_path)
        if not path.is_file():
            raise ConfigurationError(f"auxiliary artifact is absent: {path}")
        expected_size = artifact.get("size_bytes")
        if not isinstance(expected_size, int) or path.stat().st_size != expected_size:
            raise ConfigurationError(f"auxiliary artifact size differs: {path}")
        expected_sha256 = artifact.get("sha256")
        if not isinstance(expected_sha256, str) or (
            _sha256_file(path).lower() != expected_sha256.lower()
        ):
            raise ConfigurationError(f"auxiliary artifact SHA-256 differs: {path}")
        verified.append(
            {
                "name": name,
                "path": str(path),
                "size_bytes": expected_size,
                "sha256": expected_sha256,
            }
        )
    return verified


def download_model(
    model_name: str,
    *,
    directory: str | None = None,
    dry_run: bool = False,
) -> None:
    model = load_model(model_name)
    destination = resolve_model_directory(model, directory)
    print(f"MODEL repository={model['repository']}")
    print(f"MODEL revision={model['revision']}")
    print(f"MODEL destination={destination}")
    print(f"MODEL expected_shards={model.get('expected_shards', 0)}")
    if dry_run:
        return
    python = _model_worker_python(model_name)
    subprocess.run(
        [
            python,
            "-m",
            "r9700.model_worker",
            "--profile",
            model["_path"],
            "--destination",
            destination,
        ],
        cwd=ROOT,
        env=os.environ.copy(),
        check=True,
    )


def adopt_model(model_name: str, *, directory: str | None = None) -> None:
    model = load_model(model_name)
    destination = resolve_model_directory(model, directory)
    print(f"MODEL repository={model['repository']}")
    print(f"MODEL revision={model['revision']}")
    print(f"MODEL destination={destination}")
    print(f"MODEL expected_shards={model.get('expected_shards', 0)}")
    python = _model_worker_python(model_name)
    subprocess.run(
        [
            python,
            "-m",
            "r9700.model_worker",
            "--profile",
            model["_path"],
            "--destination",
            destination,
            "--adopt-existing",
        ],
        cwd=ROOT,
        env=os.environ.copy(),
        check=True,
    )


def verify_model(model_name: str, directory: str | None = None) -> dict:
    model = load_model(model_name)
    destination = resolve_model_directory(model, directory)
    manifest_path = destination / ".model-source.json"
    if not manifest_path.is_file():
        raise ConfigurationError(f"model source manifest is absent: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(
            f"model source manifest is invalid: {manifest_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ConfigurationError(f"model source manifest is invalid: {manifest_path}")
    if (payload.get("repository"), payload.get("revision")) != (
        model["repository"],
        model["revision"],
    ):
        raise ConfigurationError("model source identity differs from its profile")
    accepted_profile_hashes = {model["_sha256"]}
    if model.get("source_profile_sha256"):
        accepted_profile_hashes.add(model["source_profile_sha256"])
    if payload.get("profile_sha256") not in accepted_profile_hashes:
        raise ConfigurationError("model source manifest belongs to another profile")
    from .model_worker import validate

    try:
        checkpoint = validate(model, destination)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"model checkpoint validation failed: {exc}") from exc
    if payload.get("checkpoint") != checkpoint:
        raise ConfigurationError("model checkpoint differs from its source manifest")
    _verify_conversion(model, destination)
    payload["auxiliary_artifacts"] = _verify_auxiliary_artifacts(model)
    return payload
