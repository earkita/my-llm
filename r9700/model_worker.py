from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from datetime import datetime
from pathlib import Path

from .config import canonical_sha256, load_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safetensors_payload_bytes(path: Path) -> int:
    """Read the safetensors header and return tensor data bytes, not file bytes."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise RuntimeError(f"invalid safetensors header: {path}")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size < 2 or header_size > size - 8:
            raise RuntimeError(f"invalid safetensors header size: {path}")
        try:
            header = json.loads(stream.read(header_size))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid safetensors header JSON: {path}") from exc
    if not isinstance(header, dict):
        raise RuntimeError(f"invalid safetensors header object: {path}")
    data_size = size - 8 - header_size
    total = 0
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        offsets = metadata.get("data_offsets") if isinstance(metadata, dict) else None
        if not (
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(isinstance(value, int) for value in offsets)
            and 0 <= offsets[0] <= offsets[1] <= data_size
        ):
            raise RuntimeError(f"invalid safetensors data offsets: {path}: {name}")
        total += offsets[1] - offsets[0]
    return total


def validate(model: dict, destination: Path) -> dict:
    weight_pattern = model.get("weight_pattern")
    index_path = destination / model.get("index_file", "model.safetensors.index.json")
    expected = int(model.get("expected_shards", 0))
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f"invalid model index: {index_path}")
        shards = sorted(set(weight_map.values()))
    elif weight_pattern:
        shards = sorted(
            str(path.relative_to(destination))
            for path in destination.glob(str(weight_pattern))
            if path.is_file()
        )
    else:
        shards = sorted(path.name for path in destination.glob("*.safetensors"))
    if expected and len(shards) != expected:
        raise RuntimeError(f"expected {expected} shards, found {len(shards)}")
    missing = [name for name in shards if not (destination / name).is_file()]
    if missing:
        raise RuntimeError(f"model shards are absent: {missing}")
    required = model.get("required_files", [])
    missing_required = [name for name in required if not (destination / name).is_file()]
    if missing_required:
        raise RuntimeError(f"required model files are absent: {missing_required}")
    expected_hashes = model.get("file_sha256", {})
    if not isinstance(expected_hashes, dict):
        raise RuntimeError("file_sha256 must be an object")
    for name, expected_hash in expected_hashes.items():
        path = destination / name
        if not path.is_file():
            raise RuntimeError(f"hashed model file is absent: {name}")
        actual_hash = sha256_file(path)
        if actual_hash.lower() != str(expected_hash).lower():
            raise RuntimeError(
                f"model file SHA-256 differs for {name}: "
                f"{actual_hash} != {expected_hash}"
            )
    sizes = {name: (destination / name).stat().st_size for name in shards}
    expected_weight_bytes = model.get("checkpoint_weight_bytes")
    if expected_weight_bytes is not None:
        if all(Path(name).suffix == ".safetensors" for name in shards):
            actual_weight_bytes = sum(
                safetensors_payload_bytes(destination / name) for name in shards
            )
        else:
            actual_weight_bytes = sum(sizes.values())
        if actual_weight_bytes != int(expected_weight_bytes):
            raise RuntimeError(
                "checkpoint tensor bytes differ: "
                f"{actual_weight_bytes} != {expected_weight_bytes}"
            )
    return {
        "index": index_path.name if index_path.is_file() else None,
        "index_sha256": sha256_file(index_path) if index_path.is_file() else None,
        "shard_count": len(shards),
        "shard_sizes": sizes,
        "weight_bytes": sum(sizes.values()),
    }


def download_root(model: dict, destination: Path) -> Path:
    source_subdirectory = model.get("source_subdirectory")
    if not source_subdirectory:
        return destination
    relative = Path(str(source_subdirectory))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("model source_subdirectory must be a safe relative path")
    root = destination
    for _ in relative.parts:
        root = root.parent
    if (root / relative).resolve() != destination.resolve():
        raise RuntimeError(
            "model destination must end with its source_subdirectory"
        )
    return root


def verify_auxiliary_artifacts(
    model: dict, *, download: bool, hf_hub_download
) -> None:
    artifacts = model.get("auxiliary_artifacts", [])
    if not isinstance(artifacts, list):
        raise RuntimeError("auxiliary_artifacts must be a list")
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise RuntimeError("auxiliary artifact must be an object")
        path = Path(str(artifact.get("path", "")))
        if not path.is_absolute():
            raise RuntimeError("auxiliary artifact path must be absolute")
        if download and not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            downloaded = Path(
                hf_hub_download(
                    repo_id=artifact["repository"],
                    revision=artifact["revision"],
                    filename=artifact["filename"],
                    local_dir=path.parent,
                )
            )
            if downloaded.resolve() != path.resolve():
                raise RuntimeError(
                    f"auxiliary artifact resolved to an unexpected path: {downloaded}"
                )
        if not path.is_file():
            raise RuntimeError(f"auxiliary artifact is absent: {path}")
        if path.stat().st_size != int(artifact["size_bytes"]):
            raise RuntimeError(f"auxiliary artifact size differs: {path}")
        if sha256_file(path).lower() != str(artifact["sha256"]).lower():
            raise RuntimeError(f"auxiliary artifact SHA-256 differs: {path}")


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--adopt-existing", action="store_true")
    args = parser.parse_args()
    document = load_json(Path(args.profile))
    model = document.get("model", document)
    if not isinstance(model, dict):
        raise SystemExit("deployment profile has no embedded model object")
    destination = Path(args.destination).expanduser().resolve()
    verify_auxiliary_artifacts(
        model,
        download=not args.adopt_existing,
        hf_hub_download=hf_hub_download,
    )
    destination.mkdir(parents=True, exist_ok=True)
    if not os.access(destination, os.W_OK | os.X_OK):
        raise SystemExit(f"model destination is not writable: {destination}")
    manifest_path = destination / ".model-source.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if (existing.get("repository"), existing.get("revision")) != (
            model["repository"], model["revision"]
        ):
            raise SystemExit("existing model manifest has a different identity")
        summary = validate(model, destination)
        if existing.get("checkpoint") != summary:
            raise SystemExit("existing checkpoint differs from its source manifest")
        print(f"model already complete: {manifest_path}")
        return

    resolved = HfApi().model_info(
        model["repository"], revision=model["revision"]
    ).sha
    if resolved.lower() != model["revision"].lower():
        raise SystemExit(f"Hub resolved an unexpected revision: {resolved}")
    if not args.adopt_existing:
        patterns = model.get("allow_patterns") or None
        snapshot_download(
            repo_id=model["repository"],
            revision=model["revision"],
            local_dir=download_root(model, destination),
            allow_patterns=patterns,
            max_workers=int(os.environ.get("MODEL_DOWNLOAD_WORKERS", "8")),
        )
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "repository": model["repository"],
        "revision": model["revision"],
        "resolved_revision": resolved,
        "profile_sha256": canonical_sha256(model),
        "checkpoint": validate(model, destination),
    }
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, manifest_path)
    print(f"model source manifest: {manifest_path}")


if __name__ == "__main__":
    main()
