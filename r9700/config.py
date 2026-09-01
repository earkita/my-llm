from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_PROFILE_ROOT = ROOT / "profiles" / "production"
DEFAULT_PROFILE = "glm53-flash"
DEFAULT_MODEL_PROFILE = DEFAULT_PROFILE
DEFAULT_RUNTIME_PROFILE = DEFAULT_PROFILE
DEFAULT_STACK_PRESET = "glm53-flash"


class ConfigurationError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ConfigurationError(f"configuration file is absent: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"configuration root must be an object: {path}")
    return value


def _profile_path(name_or_path: str) -> Path:
    candidate = Path(name_or_path)
    if candidate.is_absolute() or candidate.parent != Path("."):
        return candidate.expanduser().resolve()
    suffix = "" if candidate.suffix else ".json"
    return PRODUCTION_PROFILE_ROOT / f"{name_or_path}{suffix}"


def _contains_extends(value: Any) -> bool:
    if isinstance(value, dict):
        return "extends" in value or any(
            _contains_extends(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_extends(item) for item in value)
    return False


def _validate_model(profile: dict[str, Any], path: Path) -> None:
    if profile.get("schema_version") != 1:
        raise ConfigurationError(f"unsupported model profile schema: {path}")
    required = ("name", "family", "repository", "revision", "served_name")
    missing = [key for key in required if key not in profile]
    if missing:
        raise ConfigurationError(f"model profile is missing {missing}: {path}")
    if not any(
        isinstance(profile.get(backend), dict)
        for backend in ("vllm", "llama_cpp")
    ):
        raise ConfigurationError(
            f"model profile must configure at least one backend: {path}"
        )
    revision = profile["revision"]
    if not isinstance(revision, str) or not (
        len(revision) == 40
        and all(character in "0123456789abcdefABCDEF" for character in revision)
    ):
        raise ConfigurationError(f"model revision must be a 40-character SHA: {path}")
    source_profile_sha256 = profile.get("source_profile_sha256")
    if source_profile_sha256 is not None and not (
        isinstance(source_profile_sha256, str)
        and len(source_profile_sha256) == 64
        and all(
            character in "0123456789abcdefABCDEF"
            for character in source_profile_sha256
        )
    ):
        raise ConfigurationError(
            f"source_profile_sha256 must be a 64-character SHA-256: {path}"
        )
    auxiliary_artifacts = profile.get("auxiliary_artifacts", [])
    if not isinstance(auxiliary_artifacts, list):
        raise ConfigurationError(f"auxiliary_artifacts must be a list: {path}")
    for artifact in auxiliary_artifacts:
        if not isinstance(artifact, dict):
            raise ConfigurationError(f"auxiliary artifact must be an object: {path}")
        required_artifact_fields = (
            "name",
            "repository",
            "revision",
            "filename",
            "path",
            "size_bytes",
            "sha256",
        )
        missing_artifact_fields = [
            key for key in required_artifact_fields if key not in artifact
        ]
        if missing_artifact_fields:
            raise ConfigurationError(
                f"auxiliary artifact is missing {missing_artifact_fields}: {path}"
            )
        if not Path(str(artifact["path"])).is_absolute():
            raise ConfigurationError(
                f"auxiliary artifact path must be absolute: {path}"
            )
        if not isinstance(artifact["size_bytes"], int) or artifact["size_bytes"] < 1:
            raise ConfigurationError(
                f"auxiliary artifact size must be positive: {path}"
            )
        digest = artifact["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in digest
        ):
            raise ConfigurationError(
                f"auxiliary artifact SHA-256 is invalid: {path}"
            )


def load_profile(name_or_path: str) -> dict[str, Any]:
    """Load one self-contained production deployment.

    A deployment owns its model, runtime, and coding-stack settings. Inheritance
    is deliberately rejected so that one reviewed file is the complete launch
    contract.
    """
    path = _profile_path(name_or_path)
    profile = load_json(path)
    if profile.get("schema_version") != 1:
        raise ConfigurationError(f"unsupported deployment profile schema: {path}")
    if _contains_extends(profile):
        raise ConfigurationError(f"deployment profiles cannot use extends: {path}")
    required = ("name", "status", "description", "model", "runtime", "stack")
    missing = [key for key in required if key not in profile]
    if missing:
        raise ConfigurationError(f"deployment profile is missing {missing}: {path}")
    if profile.get("status") != "production-ready":
        raise ConfigurationError(f"deployment is not production-ready: {path}")
    if profile.get("name") != path.stem:
        raise ConfigurationError(
            f"deployment name must match its filename: {profile.get('name')} != {path.stem}"
        )
    model = profile.get("model")
    runtime = profile.get("runtime")
    stack = profile.get("stack")
    if not isinstance(model, dict) or not isinstance(runtime, dict):
        raise ConfigurationError(f"deployment model and runtime must be objects: {path}")
    if not isinstance(stack, dict) or not isinstance(
        stack.get("claude_settings"), dict
    ):
        raise ConfigurationError(
            f"deployment stack must embed claude_settings: {path}"
        )
    aliases = stack.get("litellm_aliases")
    if not (
        isinstance(aliases, list)
        and aliases
        and len(aliases) == len(set(aliases))
        and all(isinstance(alias, str) and alias for alias in aliases)
    ):
        raise ConfigurationError(
            f"deployment stack must contain unique LiteLLM aliases: {path}"
        )
    _validate_model(model, path)
    validate_runtime(runtime)
    validate_compatibility(model, runtime)
    if runtime.get("llama_cpp", {}).get("speculative_type") == "draft-dflash":
        draft_model = runtime["llama_cpp"].get("draft_model")
        matches = [
            artifact
            for artifact in model.get("auxiliary_artifacts", [])
            if artifact.get("path") == draft_model
        ]
        if len(matches) != 1:
            raise ConfigurationError(
                f"DFlash runtime must bind exactly one auxiliary artifact: {path}"
            )
    result = dict(profile)
    result["model"] = dict(model)
    result["runtime"] = dict(runtime)
    result["_path"] = str(path)
    result["_sha256"] = canonical_sha256(profile)
    result["model"]["_path"] = str(path)
    result["model"]["_sha256"] = canonical_sha256(model)
    result["runtime"]["_path"] = str(path)
    result["runtime"]["_sha256"] = canonical_sha256(runtime)
    return result


def list_runtime_profiles(tier: str = "production") -> list[dict[str, Any]]:
    if tier not in ("production", "all"):
        raise ConfigurationError(
            "my-llm contains only self-contained production profiles"
        )

    paths = sorted(PRODUCTION_PROFILE_ROOT.glob("*.json"))
    records: list[dict[str, Any]] = []
    for path in paths:
        profile = load_profile(str(path))
        records.append(
            {
                "name": profile["name"],
                "tier": "production",
                "status": profile.get("status", "unknown"),
                "description": profile.get("description", ""),
                "path": str(path),
            }
        )
    return sorted(records, key=lambda value: (value["tier"], value["name"]))


def load_runtime(name_or_path: str) -> dict[str, Any]:
    return load_profile(name_or_path)["runtime"]


def load_model(name_or_path: str) -> dict[str, Any]:
    return load_profile(name_or_path)["model"]


def validate_runtime(profile: dict[str, Any]) -> None:
    required = (
        "name",
        "status",
        "recipe",
        "parallel",
        "limits",
        "cache",
        "scheduler",
    )
    missing = [key for key in required if key not in profile]
    if missing:
        raise ConfigurationError(f"runtime profile is missing fields: {missing}")
    recipe = profile.get("recipe")
    if not isinstance(recipe, str) or not recipe:
        raise ConfigurationError("runtime recipe must be a non-empty string")
    # Import lazily: manifest parsing uses the configuration JSON helpers.
    from .manifest import recipe_patch_ids, recipe_record

    recipe_metadata = recipe_record(recipe)
    backend = profile.get("backend", recipe_metadata["backend"])
    if backend not in ("vllm", "llama-cpp"):
        raise ConfigurationError(f"unsupported runtime backend: {backend}")
    if backend != recipe_metadata["backend"]:
        raise ConfigurationError(
            f"runtime backend {backend} differs from recipe {recipe} "
            f"backend {recipe_metadata['backend']}"
        )
    required_patches = profile.get("required_patches", [])
    if not isinstance(required_patches, list) or any(
        not isinstance(value, str) for value in required_patches
    ):
        raise ConfigurationError("required_patches must be a list of patch IDs")
    missing_patches = sorted(set(required_patches) - recipe_patch_ids(recipe))
    if missing_patches:
        raise ConfigurationError(
            f"runtime recipe {recipe} does not provide required patches: "
            + ",".join(missing_patches)
        )
    parallel = profile["parallel"]
    limits = profile["limits"]
    for label, value in {
        "parallel.tensor": parallel.get("tensor"),
        "parallel.pipeline": parallel.get("pipeline"),
        "parallel.data": parallel.get("data", 1),
        "limits.max_model_len": limits.get("max_model_len"),
        "limits.max_num_seqs": limits.get("max_num_seqs"),
        "limits.max_num_batched_tokens": limits.get("max_num_batched_tokens"),
    }.items():
        if not isinstance(value, int) or value < 1:
            raise ConfigurationError(f"{label} must be a positive integer")
    enable_expert_parallel = parallel.get("enable_expert_parallel", False)
    if not isinstance(enable_expert_parallel, bool):
        raise ConfigurationError("parallel.enable_expert_parallel must be boolean")
    world_size = parallel["tensor"] * parallel["pipeline"] * parallel.get("data", 1)
    gpu_order = profile.get("gpu_order")
    if not isinstance(gpu_order, list) or len(gpu_order) < world_size:
        raise ConfigurationError(
            f"gpu_order exposes fewer devices than TPxPPxDP world size {world_size}"
        )
    if len(set(gpu_order)) != len(gpu_order):
        raise ConfigurationError("gpu_order contains duplicate devices")
    cache = profile["cache"]
    if cache.get("cpu_offload_gb", 0) != 0 or "weight_offload" in profile:
        raise ConfigurationError("production profiles cannot use CPU offload")
    retention_interval = cache.get("prefix_cache_retention_interval")
    if retention_interval is not None:
        if not isinstance(retention_interval, int) or retention_interval < 1:
            raise ConfigurationError(
                "cache.prefix_cache_retention_interval must be a positive integer"
            )
        if not cache.get("prefix_cache"):
            raise ConfigurationError(
                "cache.prefix_cache_retention_interval requires prefix_cache"
            )
    gpu_bdfs = profile.get("gpu_bdfs")
    if gpu_bdfs is not None:
        if (
            not isinstance(gpu_bdfs, list)
            or len(gpu_bdfs) < world_size
            or len(set(gpu_bdfs)) != len(gpu_bdfs)
            or any(
                not isinstance(value, str)
                or len(value) != 12
                or value[4] != ":"
                or value[7] != ":"
                or value[10] != "."
                for value in gpu_bdfs
            )
        ):
            raise ConfigurationError(
                "gpu_bdfs must contain one unique PCI BDF per required GPU"
            )
    if backend == "llama-cpp":
        settings = profile.get("llama_cpp")
        if not isinstance(settings, dict):
            raise ConfigurationError("llama-cpp runtime requires llama_cpp settings")
        tensor_split = settings.get("tensor_split")
        if tensor_split is not None and (
            not isinstance(tensor_split, list)
            or len(tensor_split) != world_size
            or any(
                not isinstance(value, (int, float)) or value <= 0
                for value in tensor_split
            )
        ):
            raise ConfigurationError(
                "llama_cpp.tensor_split must contain one positive weight per GPU"
            )
        tensor_overrides = settings.get("tensor_overrides")
        if tensor_overrides is not None and not (
            isinstance(tensor_overrides, str)
            and bool(tensor_overrides)
            or isinstance(tensor_overrides, list)
            and bool(tensor_overrides)
            and all(isinstance(value, str) and value for value in tensor_overrides)
        ):
            raise ConfigurationError(
                "llama_cpp.tensor_overrides must be a non-empty string or list of strings"
            )
        if settings.get("speculative_type") == "draft-mtp":
            draft_n_max = settings.get("draft_n_max")
            draft_n_min = settings.get("draft_n_min", 0)
            draft_p_min = settings.get("draft_p_min")
            draft_ubatch_size = settings.get("draft_ubatch_size")
            draft_context_size = settings.get("draft_context_size")
            if not isinstance(draft_n_max, int) or draft_n_max < 1:
                raise ConfigurationError(
                    "llama_cpp.draft_n_max must be positive for draft-mtp"
                )
            if (
                not isinstance(draft_n_min, int)
                or draft_n_min < 0
                or draft_n_min > draft_n_max
            ):
                raise ConfigurationError(
                    "llama_cpp.draft_n_min must be between zero and draft_n_max"
                )
            if draft_p_min is not None and (
                not isinstance(draft_p_min, (int, float))
                or isinstance(draft_p_min, bool)
                or draft_p_min < 0
                or draft_p_min > 1
            ):
                raise ConfigurationError(
                    "llama_cpp.draft_p_min must be between zero and one"
                )
            if draft_ubatch_size is not None and (
                not isinstance(draft_ubatch_size, int) or draft_ubatch_size < 1
            ):
                raise ConfigurationError("llama_cpp.draft_ubatch_size must be positive")
            if draft_context_size is not None and (
                not isinstance(draft_context_size, int) or draft_context_size < 1
            ):
                raise ConfigurationError(
                    "llama_cpp.draft_context_size must be positive"
                )
        if settings.get("speculative_type") == "draft-dflash":
            draft_model = settings.get("draft_model")
            draft_n_max = settings.get("draft_n_max")
            draft_n_min = settings.get("draft_n_min", 0)
            draft_p_min = settings.get("draft_p_min")
            if not isinstance(draft_model, str) or not draft_model:
                raise ConfigurationError(
                    "llama_cpp.draft_model must be non-empty for draft-dflash"
                )
            if not isinstance(draft_n_max, int) or draft_n_max < 1:
                raise ConfigurationError(
                    "llama_cpp.draft_n_max must be positive for draft-dflash"
                )
            if (
                not isinstance(draft_n_min, int)
                or draft_n_min < 0
                or draft_n_min > draft_n_max
            ):
                raise ConfigurationError(
                    "llama_cpp.draft_n_min must be between zero and draft_n_max"
                )
            if draft_p_min is not None and (
                not isinstance(draft_p_min, (int, float))
                or isinstance(draft_p_min, bool)
                or draft_p_min < 0
                or draft_p_min > 1
            ):
                raise ConfigurationError(
                    "llama_cpp.draft_p_min must be between zero and one"
                )
        cache_ram_mib = settings.get("cache_ram_mib")
        if cache_ram_mib is not None and (
            not isinstance(cache_ram_mib, int) or cache_ram_mib < -1
        ):
            raise ConfigurationError(
                "llama_cpp.cache_ram_mib must be -1 or a non-negative integer"
            )
    for key in ("attention_backend", "moe_backend", "linear_backend"):
        value = profile.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise ConfigurationError(f"{key} must be a non-empty string")
    multimodal = profile.get("multimodal")
    if multimodal is not None:
        if not isinstance(multimodal, dict):
            raise ConfigurationError("multimodal must be an object")
        language_model_only = multimodal.get("language_model_only", False)
        if not isinstance(language_model_only, bool):
            raise ConfigurationError("multimodal.language_model_only must be boolean")
        limit_per_prompt = multimodal.get("limit_per_prompt")
        if limit_per_prompt is not None:
            if not isinstance(limit_per_prompt, dict) or not limit_per_prompt:
                raise ConfigurationError(
                    "multimodal.limit_per_prompt must be a non-empty object"
                )
            for modality, limit in limit_per_prompt.items():
                if not isinstance(modality, str) or not modality:
                    raise ConfigurationError(
                        "multimodal.limit_per_prompt keys must be non-empty strings"
                    )
                if isinstance(limit, int):
                    valid_limit = limit >= 0
                elif isinstance(limit, dict) and limit:
                    valid_limit = all(
                        isinstance(key, str)
                        and key
                        and isinstance(value, int)
                        and value >= 0
                        for key, value in limit.items()
                    )
                else:
                    valid_limit = False
                if not valid_limit:
                    raise ConfigurationError(
                        "multimodal.limit_per_prompt values must be non-negative "
                        "integers or non-empty objects of non-negative integers"
                    )
        for key in (
            "encoder_attention_backend",
            "processor_cache_type",
            "encoder_tp_mode",
        ):
            value = multimodal.get(key)
            if value is not None and (not isinstance(value, str) or not value):
                raise ConfigurationError(f"multimodal.{key} must be a non-empty string")
        processor_kwargs = multimodal.get("processor_kwargs")
        if processor_kwargs is not None:
            if not isinstance(processor_kwargs, dict):
                raise ConfigurationError(
                    "multimodal.processor_kwargs must be an object"
                )
            max_image_tokens = processor_kwargs.get("max_image_tokens")
            if max_image_tokens is not None and (
                not isinstance(max_image_tokens, int) or max_image_tokens < 16
            ):
                raise ConfigurationError(
                    "multimodal.processor_kwargs.max_image_tokens must be an "
                    "integer >= 16"
                )
    speculative = profile.get("speculative_config")
    if speculative:
        if not isinstance(speculative, dict):
            raise ConfigurationError("speculative_config must be an object")
        if (
            backend == "vllm"
            and parallel["pipeline"] > 1
            and profile.get("environment", {}).get("VLLM_USE_V2_MODEL_RUNNER")
            == "1"
            and profile["scheduler"].get("async") is not True
            and "0023" not in required_patches
        ):
            raise ConfigurationError(
                "vLLM V2 speculative decoding with pipeline parallelism "
                "requires scheduler.async=true or the synchronous PP cadence patch"
            )
        method = speculative.get("method")
        if method == "eagle3":
            model_profile = speculative.get("model_profile")
            model_path = speculative.get("model")
            if not (
                isinstance(model_profile, str)
                and model_profile
                or isinstance(model_path, str)
                and model_path
            ):
                raise ConfigurationError(
                    "eagle3 requires a non-empty model_profile or model"
                )
        k = speculative.get("num_speculative_tokens")
        if not isinstance(k, int) or k < 1:
            raise ConfigurationError("num_speculative_tokens must be positive")
        target_budget = limits["max_num_batched_tokens"] - (
            (k - 1) * limits["max_num_seqs"]
        )
        if target_budget < 1:
            raise ConfigurationError("raw batch leaves no target-token budget")


def validate_compatibility(model: dict[str, Any], runtime: dict[str, Any]) -> None:
    from .manifest import recipe_record

    backend = runtime.get("backend", recipe_record(runtime["recipe"])["backend"])
    model_key = {
        "llama-cpp": "llama_cpp",
    }.get(backend, "vllm")
    if not isinstance(model.get(model_key), dict):
        raise ConfigurationError(
            f"model {model['name']} does not support backend {backend}"
        )
    required_family = runtime.get("required_model_family")
    if required_family and model["family"] != required_family:
        raise ConfigurationError(
            f"runtime {runtime['name']} requires model family {required_family}, "
            f"found {model['family']}"
        )
    if (
        backend == "llama-cpp"
        and runtime.get("llama_cpp", {}).get("speculative_type") == "draft-mtp"
        and not model.get("supports_mtp", False)
    ):
        raise ConfigurationError(
            f"runtime {runtime['name']} enables MTP for an incompatible model"
        )
    speculative = runtime.get("speculative_config")
    if speculative:
        method = speculative.get("method")
        capability = {
            "dspark": "supports_dspark",
            "mtp": "supports_mtp",
            "eagle3": "supports_eagle3",
        }.get(method)
        if capability is None:
            raise ConfigurationError(
                f"runtime {runtime['name']} uses unsupported speculative method "
                f"{method!r}"
            )
        if not model.get(capability, False):
            raise ConfigurationError(
                f"runtime {runtime['name']} enables {method.upper()} for an "
                f"incompatible model"
            )


def canonical_sha256(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if not key.startswith("_")}
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_dotenv(path: Path | None = None) -> dict[str, str]:
    path = path or ROOT / ".env"
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfigurationError(f"invalid .env line {number}: {raw!r}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key.replace("_", "").isalnum() or key[0].isdigit():
            raise ConfigurationError(f"invalid .env name on line {number}: {key!r}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = os.path.expandvars(os.path.expanduser(value))
    return values


def resolve_model_directory(model: dict[str, Any], explicit: str | None = None) -> Path:
    raw = explicit
    if not raw:
        dotenv = read_dotenv()
        normalized_name = "".join(
            character if character.isalnum() else "_"
            for character in str(model["name"]).upper()
        )
        directory_environment = model.get(
            "directory_environment", f"MODEL_DIR_{normalized_name}"
        )
        if directory_environment:
            raw = os.environ.get(directory_environment) or dotenv.get(
                directory_environment
            )
    raw = raw or model.get("default_directory")
    if not raw:
        raise ConfigurationError(
            f"model directory is not configured for {model['name']}"
        )
    return Path(os.path.expandvars(os.path.expanduser(str(raw)))).resolve()
