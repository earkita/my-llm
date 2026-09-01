from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import load_model, resolve_model_directory
from ..manifest import recipe_venv
from .common import base_environment, rocm_root, visible_devices


def environment(runtime: dict[str, Any]) -> dict[str, str]:
    env = base_environment(runtime)
    rocm_home = rocm_root(runtime)
    venv = recipe_venv(runtime["recipe"])
    devices = visible_devices(runtime)
    transport = runtime["transport"]
    shutdown = runtime["shutdown"]
    env.update(
        {
            "PATH": f"{rocm_home / 'bin'}:{venv / 'bin'}:{env.get('PATH', '')}",
            "TRITON_DEFAULT_BACKEND": "amd",
            "HIP_VISIBLE_DEVICES": ",".join(devices),
            "NCCL_P2P_DISABLE": str(transport["p2p_disable"]),
            "NCCL_SHM_DISABLE": str(transport["shm_disable"]),
            "NCCL_SOCKET_IFNAME": str(transport["socket_ifname"]),
            "NCCL_RUNTIME_CONNECT": str(transport["runtime_connect"]),
            "HSA_ENABLE_IPC_MODE_LEGACY": str(transport["hsa_legacy_ipc"]),
            "NCCL_DEBUG": env.get("NCCL_DEBUG", "WARN"),
            "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS": str(
                shutdown["worker_timeout_seconds"]
            ),
            "VLLM_ENGINE_PROCESS_SHUTDOWN_GRACE_SECONDS": str(
                shutdown["process_grace_seconds"]
            ),
        }
    )
    env.update(
        {key: str(value) for key, value in runtime.get("environment", {}).items()}
    )
    partition = runtime["parallel"].get("pipeline_layers")
    if partition:
        env["VLLM_PP_LAYER_PARTITION"] = str(partition)
    return env


def command(
    model: dict[str, Any],
    runtime: dict[str, Any],
    model_directory: Path,
    host: str,
    port: int,
) -> list[str]:
    limits = runtime["limits"]
    cache = runtime["cache"]
    scheduler = runtime["scheduler"]
    shutdown = runtime["shutdown"]
    args = [
        str(recipe_venv(runtime["recipe"]) / "bin" / "vllm"),
        "serve",
        str(model_directory),
        "--host",
        host,
        "--port",
        str(port),
        "--served-model-name",
        model["served_name"],
        "--tensor-parallel-size",
        str(runtime["parallel"]["tensor"]),
        "--pipeline-parallel-size",
        str(runtime["parallel"]["pipeline"]),
        "--distributed-executor-backend",
        "mp",
        "--max-model-len",
        str(limits["max_model_len"]),
        "--max-num-seqs",
        str(limits["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(limits["max_num_batched_tokens"]),
        "--shutdown-timeout",
        str(shutdown["request_timeout_seconds"]),
        "--gpu-memory-utilization",
        str(limits["gpu_memory_utilization"]),
        "--kv-cache-dtype",
        str(cache["dtype"]),
        "--block-size",
        str(cache["block_size"]),
    ]
    data_parallel_size = runtime["parallel"].get("data", 1)
    if data_parallel_size > 1:
        args += ["--data-parallel-size", str(data_parallel_size)]
    if runtime["parallel"].get("enable_expert_parallel"):
        args.append("--enable-expert-parallel")
    if runtime["parallel"].get("disable_custom_all_reduce"):
        args.append("--disable-custom-all-reduce")
    if limits.get("kv_cache_memory_bytes") is not None:
        args += ["--kv-cache-memory-bytes", str(limits["kv_cache_memory_bytes"])]
    args.append(
        "--enable-prefix-caching"
        if cache.get("prefix_cache")
        else "--no-enable-prefix-caching"
    )
    if cache.get("prefix_cache_retention_interval") is not None:
        args += [
            "--prefix-cache-retention-interval",
            str(cache["prefix_cache_retention_interval"]),
        ]
    if scheduler.get("enforce_eager"):
        args.append("--enforce-eager")
    if scheduler.get("async") is True:
        args.append("--async-scheduling")
    elif scheduler.get("async") is False:
        args.append("--no-async-scheduling")
    if runtime.get("attention_backend"):
        args += ["--attention-backend", str(runtime["attention_backend"])]
    if runtime.get("moe_backend"):
        args += ["--moe-backend", str(runtime["moe_backend"])]
    if runtime.get("linear_backend"):
        args += ["--linear-backend", str(runtime["linear_backend"])]
    if runtime.get("attention_config"):
        args += [
            "--attention-config",
            json.dumps(runtime["attention_config"], separators=(",", ":")),
        ]
    if runtime.get("speculative_config"):
        speculative_config = dict(runtime["speculative_config"])
        draft_profile_name = speculative_config.pop("model_profile", None)
        if draft_profile_name:
            draft_model = load_model(str(draft_profile_name))
            speculative_config["model"] = str(resolve_model_directory(draft_model))
        args += [
            "--speculative-config",
            json.dumps(speculative_config, separators=(",", ":")),
        ]
    if runtime.get("profiler_config"):
        args += [
            "--profiler-config",
            json.dumps(runtime["profiler_config"], separators=(",", ":")),
        ]
    if runtime.get("compilation_config"):
        args += [
            "--compilation-config",
            json.dumps(runtime["compilation_config"], separators=(",", ":")),
        ]
    loading = runtime.get("loading", {})
    if loading.get("format"):
        args += ["--load-format", str(loading["format"])]
    if loading.get("max_parallel_workers") is not None:
        args += [
            "--max-parallel-loading-workers",
            str(loading["max_parallel_workers"]),
        ]
    model_args = model.get("vllm", {})
    for key, option in (
        ("tokenizer_mode", "--tokenizer-mode"),
        ("tool_call_parser", "--tool-call-parser"),
        ("reasoning_parser", "--reasoning-parser"),
        ("chat_template", "--chat-template"),
        ("quantization", "--quantization"),
    ):
        if model_args.get(key):
            args += [option, str(model_args[key])]
    if model_args.get("enable_auto_tool_choice"):
        args.append("--enable-auto-tool-choice")
    if model_args.get("trust_remote_code"):
        args.append("--trust-remote-code")
    multimodal = runtime.get("multimodal", {})
    language_model_only = multimodal.get(
        "language_model_only", model_args.get("language_model_only", False)
    )
    if language_model_only:
        args.append("--language-model-only")
    if multimodal.get("limit_per_prompt") is not None:
        args += [
            "--limit-mm-per-prompt",
            json.dumps(multimodal["limit_per_prompt"], separators=(",", ":")),
        ]
    if multimodal.get("encoder_attention_backend"):
        args += [
            "--mm-encoder-attn-backend",
            str(multimodal["encoder_attention_backend"]),
        ]
    if multimodal.get("processor_cache_type"):
        args += [
            "--mm-processor-cache-type",
            str(multimodal["processor_cache_type"]),
        ]
    if multimodal.get("processor_kwargs") is not None:
        args += [
            "--mm-processor-kwargs",
            json.dumps(multimodal["processor_kwargs"], separators=(",", ":")),
        ]
    if multimodal.get("encoder_tp_mode"):
        args += ["--mm-encoder-tp-mode", str(multimodal["encoder_tp_mode"])]
    return args
