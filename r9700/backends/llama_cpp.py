from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import ConfigurationError, ROOT, load_json
from ..manifest import (
    default_recipe_name,
    recipe_install_path,
    recipe_manifest,
    recipe_manifest_path,
    sha256_file,
    tracked_diff_sha256,
)
from .common import base_environment, rocm_root


EMPTY_DIFF_SHA256 = hashlib.sha256(b"").hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected_recipe(recipe_name: str | None = None) -> str:
    return recipe_name or default_recipe_name(backend="llama-cpp")


def manifest(recipe_name: str | None = None) -> dict[str, Any]:
    selected = _selected_recipe(recipe_name)
    value = recipe_manifest(selected)
    if value.get("schema_version") != 1 or value.get("backend") != "llama-cpp":
        raise ConfigurationError("unsupported llama.cpp manifest schema")
    return value


def manifest_sha256(recipe_name: str | None = None) -> str:
    return sha256_file(recipe_manifest_path(_selected_recipe(recipe_name)))


def _git_output(tree: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=tree, check=True, capture_output=True, text=True
    ).stdout.strip()


def _source_identity(tree: Path) -> dict[str, Any]:
    if not (tree / ".git").exists():
        raise ConfigurationError("llama.cpp Git source identity is absent")
    return {
        "commit": _git_output(tree, "rev-parse", "HEAD"),
        "tracked_diff_sha256": tracked_diff_sha256(tree),
    }


def _require_no_untracked_source(tree: Path) -> None:
    if _git_output(tree, "ls-files", "--others", "--exclude-standard"):
        raise ConfigurationError("llama.cpp source contains untracked files")


def _verify_patch_assets(spec: dict[str, Any]) -> None:
    for patch in spec.get("patches", []):
        path = ROOT / patch["path"]
        if not path.is_file():
            raise ConfigurationError(f"llama.cpp patch is absent: {patch['path']}")
        if sha256_file(path) != patch["sha256"]:
            raise ConfigurationError(
                f"llama.cpp patch identity is stale: {patch['path']}"
            )


def _prepare_source(spec: dict[str, Any]) -> Path:
    _verify_patch_assets(spec)
    source = ROOT / spec["source_directory"]
    source.mkdir(parents=True, exist_ok=True)
    if not (source / ".git").exists():
        if any(source.iterdir()):
            raise ConfigurationError("llama.cpp recipe source directory is not empty")
        subprocess.run(["git", "init", source], check=True)
        subprocess.run(
            ["git", "remote", "add", "origin", spec["repository"]],
            cwd=source,
            check=True,
        )
    elif _git_output(source, "remote", "get-url", "origin") != spec["repository"]:
        raise ConfigurationError("llama.cpp source origin differs from the recipe")
    try:
        head = _git_output(source, "rev-parse", "HEAD")
    except subprocess.CalledProcessError:
        head = ""
    if head != spec["commit"]:
        if head:
            if tracked_diff_sha256(source) != EMPTY_DIFF_SHA256:
                raise ConfigurationError(
                    "llama.cpp source has changes and cannot switch commits"
                )
            _require_no_untracked_source(source)
        subprocess.run(
            ["git", "fetch", "--depth", "1", "origin", spec["commit"]],
            cwd=source,
            check=True,
        )
        subprocess.run(
            ["git", "checkout", "--detach", spec["commit"]],
            cwd=source,
            check=True,
        )
    if _git_output(source, "rev-parse", "HEAD") != spec["commit"]:
        raise ConfigurationError("llama.cpp Git source identity is stale")
    expected_diff = spec.get("expected_diff_sha256", EMPTY_DIFF_SHA256)
    current_diff = tracked_diff_sha256(source)
    if current_diff != expected_diff:
        if current_diff != EMPTY_DIFF_SHA256:
            raise ConfigurationError(
                f"llama.cpp source has unsupported changes: {current_diff}"
            )
        for patch in spec.get("patches", []):
            patch_path = ROOT / patch["path"]
            subprocess.run(
                ["git", "apply", "--check", patch_path], cwd=source, check=True
            )
            subprocess.run(["git", "apply", patch_path], cwd=source, check=True)
        current_diff = tracked_diff_sha256(source)
    if current_diff != expected_diff:
        raise ConfigurationError(
            f"llama.cpp patch image mismatch: {current_diff} != {expected_diff}"
        )
    _require_no_untracked_source(source)
    return source


def install(
    *,
    dry_run: bool = False,
    jobs: int | None = None,
    recipe_name: str | None = None,
    rebuild: bool = False,
) -> None:
    selected = _selected_recipe(recipe_name)
    spec = manifest(selected)
    _verify_patch_assets(spec)
    print(
        f"INSTALL llama.cpp {spec['tag']} {spec['commit']} "
        f"HIP target={spec['cmake']['options']['AMDGPU_TARGETS']}"
    )
    if dry_run:
        for index, patch in enumerate(spec.get("patches", []), 1):
            print(
                f"  llama.cpp[{index:02d}] {patch['path']} {patch['sha256']}"
            )
        for key, value in spec["cmake"]["options"].items():
            print(f"  cmake {key}={value}")
        return
    if not rebuild:
        try:
            installed = verify_install(selected)
            print(
                f"llama.cpp recipe already installed: {selected} "
                f"({installed['manifest_sha256']})"
            )
            return
        except ConfigurationError:
            pass
    for executable in ("git", "cmake", "ninja"):
        if shutil.which(executable) is None:
            raise ConfigurationError(f"required build tool is absent: {executable}")
    source = _prepare_source(spec)
    build = ROOT / spec["build_directory"]
    binary = ROOT / spec["binary"]
    rocm_home = rocm_root(recipe_name=selected)
    env = os.environ.copy()
    env.update(
        {
            "ROCM_HOME": str(rocm_home),
            "ROCM_PATH": str(rocm_home),
            "HIP_PATH": str(rocm_home),
            "HIPCXX": str(rocm_home / "llvm" / "bin" / "clang"),
            "PATH": f"{rocm_home / 'bin'}:{env.get('PATH', '')}",
            "LD_LIBRARY_PATH": f"{rocm_home / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}",
        }
    )
    cmake = [
        "cmake",
        "-S",
        str(source),
        "-B",
        str(build),
        "-G",
        spec["cmake"]["generator"],
        f"-DCMAKE_BUILD_TYPE={spec['cmake']['build_type']}",
    ]
    cmake.extend(
        f"-D{key}={value}" for key, value in spec["cmake"]["options"].items()
    )
    subprocess.run(cmake, env=env, check=True)
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build),
            "--target",
            "llama-server",
            "-j",
            str(jobs or os.cpu_count() or 1),
        ],
        env=env,
        check=True,
    )
    if not os.access(binary, os.X_OK):
        raise ConfigurationError(f"llama.cpp server was not built: {binary}")
    version_probe = subprocess.run(
        [binary, "--version"], check=True, capture_output=True, text=True, env=env
    )
    version = "\n".join(
        line
        for line in (version_probe.stdout + version_probe.stderr).splitlines()
        if line.startswith(("version:", "built with"))
    )
    artifacts = [binary]
    artifacts.extend(
        path
        for path in sorted(binary.parent.glob("*.so*"))
        if path.is_file() and not path.is_symlink()
    )
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "recipe": selected,
        "manifest_sha256": manifest_sha256(selected),
        "source": _source_identity(source),
        "binary": str(binary),
        "artifact_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in artifacts
        },
        "version": version,
    }
    install_path = recipe_install_path(selected)
    install_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = install_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, install_path)
    print(f"llama.cpp backend installed: {install_path}")


def verify_install(recipe_name: str | None = None) -> dict[str, Any]:
    selected = _selected_recipe(recipe_name)
    spec = manifest(selected)
    binary = ROOT / spec["binary"]
    source = ROOT / spec["source_directory"]
    install_path = recipe_install_path(selected)
    _verify_patch_assets(spec)
    if not install_path.is_file():
        raise ConfigurationError(
            f"llama.cpp recipe is not installed; run ./run install --recipe {selected}"
        )
    installed = load_json(install_path)
    if installed.get("manifest_sha256") != manifest_sha256(selected):
        raise ConfigurationError("llama.cpp install belongs to another manifest")
    installed_recipe = installed.get("recipe")
    if installed_recipe is not None and installed_recipe != selected:
        raise ConfigurationError("llama.cpp install attests another recipe")
    if not os.access(binary, os.X_OK):
        raise ConfigurationError(f"llama.cpp server binary is absent: {binary}")
    identity = _source_identity(source)
    if identity != {
        "commit": spec["commit"],
        "tracked_diff_sha256": spec.get(
            "expected_diff_sha256", EMPTY_DIFF_SHA256
        ),
    }:
        raise ConfigurationError("llama.cpp source identity is stale")
    _require_no_untracked_source(source)
    if installed.get("source") != identity:
        raise ConfigurationError("llama.cpp install source attestation is stale")
    artifact_hashes = installed.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        raise ConfigurationError("llama.cpp artifact attestation is incomplete")
    for relative, expected in artifact_hashes.items():
        path = ROOT / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ConfigurationError(f"llama.cpp artifact identity is stale: {relative}")
    return installed


def environment(runtime: dict[str, Any]) -> dict[str, str]:
    env = base_environment(runtime)
    env.update(
        {key: str(value) for key, value in runtime.get("environment", {}).items()}
    )
    return env


def command(
    model: dict[str, Any],
    runtime: dict[str, Any],
    model_directory: Path,
    host: str,
    port: int,
) -> list[str]:
    settings = runtime.get("llama_cpp", {})
    model_settings = model.get("llama_cpp", {})
    model_file = model_settings.get("model_file")
    if not model_file:
        raise ConfigurationError("llama.cpp model profile does not define model_file")
    model_path = model_directory / model_file
    spec = manifest(runtime["recipe"])
    args = [
        str(ROOT / spec["binary"]),
        "--model",
        str(model_path),
        "--alias",
        model["served_name"],
        "--host",
        host,
        "--port",
        str(port),
        "--ctx-size",
        str(runtime["limits"]["max_model_len"]),
        "--parallel",
        str(runtime["limits"]["max_num_seqs"]),
        "--batch-size",
        str(settings.get("batch_size", runtime["limits"]["max_num_batched_tokens"])),
        "--ubatch-size",
        str(settings.get("ubatch_size", 512)),
        "--split-mode",
        str(settings.get("split_mode", "layer")),
        "--cache-type-k",
        str(settings.get("cache_type_k", "q8_0")),
        "--cache-type-v",
        str(settings.get("cache_type_v", "q8_0")),
        "--flash-attn",
        str(settings.get("flash_attention", "on")),
        "--fit",
        "on" if settings.get("fit", True) else "off",
        "--fit-target",
        str(settings.get("fit_target_mib", 2048)),
        "--load-mode",
        str(settings.get("load_mode", "mmap")),
        "--reasoning-format",
        str(model_settings.get("reasoning_format", "deepseek")),
        "--reasoning",
        str(model_settings.get("reasoning", "on")),
        "--jinja",
        (
            "--cache-prompt"
            if settings.get(
                "cache_prompt", runtime.get("cache", {}).get("prefix_cache", True)
            )
            else "--no-cache-prompt"
        ),
        "--cache-reuse",
        str(settings.get("cache_reuse", 0)),
        "--cache-ram",
        str(settings.get("cache_ram_mib", 8192)),
        "--metrics",
        "--slots",
        "--perf",
    ]
    n_gpu_layers = settings.get("n_gpu_layers", "all")
    if n_gpu_layers is not None:
        args += ["--n-gpu-layers", str(n_gpu_layers)]
    tensor_split = settings.get("tensor_split")
    if tensor_split:
        args += ["--tensor-split", ",".join(str(value) for value in tensor_split)]
    tensor_overrides = settings.get("tensor_overrides")
    if tensor_overrides:
        if isinstance(tensor_overrides, str):
            tensor_overrides = [tensor_overrides]
        args += ["--override-tensor", ",".join(str(value) for value in tensor_overrides)]
    speculative_type = settings.get("speculative_type")
    if speculative_type:
        args += ["--spec-type", str(speculative_type)]
    if settings.get("draft_model") is not None:
        args += ["--spec-draft-model", str(settings["draft_model"])]
    if settings.get("draft_n_max") is not None:
        args += ["--spec-draft-n-max", str(settings["draft_n_max"])]
    if settings.get("draft_n_min") is not None:
        args += ["--spec-draft-n-min", str(settings["draft_n_min"])]
    if settings.get("draft_p_min") is not None:
        args += ["--spec-draft-p-min", str(settings["draft_p_min"])]
    if settings.get("draft_ubatch_size") is not None:
        args += ["--spec-draft-ubatch-size", str(settings["draft_ubatch_size"])]
    if settings.get("draft_context_size") is not None:
        args += ["--spec-draft-ctx-size", str(settings["draft_context_size"])]
    if settings.get("draft_cache_type_k") is not None:
        args += ["--spec-draft-type-k", str(settings["draft_cache_type_k"])]
    if settings.get("draft_cache_type_v") is not None:
        args += ["--spec-draft-type-v", str(settings["draft_cache_type_v"])]
    if settings.get("draft_n_gpu_layers") is not None:
        args += ["--spec-draft-ngl", str(settings["draft_n_gpu_layers"])]
    if settings.get("ngram_simple_size_n") is not None:
        args += [
            "--spec-ngram-simple-size-n",
            str(settings["ngram_simple_size_n"]),
        ]
    if settings.get("ngram_simple_size_m") is not None:
        args += [
            "--spec-ngram-simple-size-m",
            str(settings["ngram_simple_size_m"]),
        ]
    if settings.get("ngram_simple_min_hits") is not None:
        args += [
            "--spec-ngram-simple-min-hits",
            str(settings["ngram_simple_min_hits"]),
        ]
    args.append(
        "--cont-batching"
        if settings.get("continuous_batching", True)
        else "--no-cont-batching"
    )
    return args
