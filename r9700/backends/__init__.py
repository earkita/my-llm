from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import ConfigurationError
from ..manifest import manifest_sha256, recipe_record, verify_install


SUPPORTED_BACKENDS = ("vllm", "llama-cpp")


def runtime_backend(runtime: dict[str, Any]) -> str:
    recipe = str(runtime.get("recipe", ""))
    if not recipe:
        raise ConfigurationError("runtime does not select a recipe")
    recipe_backend = str(recipe_record(recipe)["backend"])
    backend = str(runtime.get("backend", recipe_backend))
    if backend not in SUPPORTED_BACKENDS:
        raise ConfigurationError(f"unsupported inference backend: {backend}")
    if backend != recipe_backend:
        raise ConfigurationError(
            f"runtime backend {backend} differs from recipe backend {recipe_backend}"
        )
    return backend


def build_command(
    model: dict[str, Any],
    runtime: dict[str, Any],
    model_directory: Path,
    host: str,
    port: int,
) -> list[str]:
    backend = runtime_backend(runtime)
    if backend == "vllm":
        from .vllm import command
    else:
        from .llama_cpp import command
    return command(model, runtime, model_directory, host, port)


def build_environment(runtime: dict[str, Any]) -> dict[str, str]:
    backend = runtime_backend(runtime)
    if backend == "vllm":
        from .vllm import environment
    else:
        from .llama_cpp import environment
    return environment(runtime)


def verify_backend_install(runtime: dict[str, Any]) -> dict[str, Any]:
    backend = runtime_backend(runtime)
    recipe = str(runtime["recipe"])
    if backend == "vllm":
        return verify_install(recipe)
    if backend == "llama-cpp":
        from .llama_cpp import verify_install as verify_llama_install

        return verify_llama_install(recipe)
    raise ConfigurationError(f"unsupported inference backend: {backend}")


def backend_manifest_sha256(runtime: dict[str, Any]) -> str:
    backend = runtime_backend(runtime)
    recipe = str(runtime["recipe"])
    if backend == "vllm":
        return manifest_sha256(recipe)
    if backend == "llama-cpp":
        from .llama_cpp import manifest_sha256 as llama_manifest_sha256

        return llama_manifest_sha256(recipe)
    raise ConfigurationError(f"unsupported inference backend: {backend}")
