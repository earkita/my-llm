#!/usr/bin/env python3
"""Tune the GLM-5.3-Flash TP8 W4A16 Triton MoE kernel on ROCm.

This intentionally avoids the upstream benchmark_moe.py Ray dependency.  Each
process owns one GPU and evaluates a deterministic shard of either the full
ROCm decode search space or a supplied finalist list.  Results are JSON so a
separate coordinator can rank candidates without sharing GPU state.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path
from typing import Any


BLOCK_SIZES = (16, 32, 64, 128, 256)
NUM_WARPS = (1, 2, 4, 8)
WAVES_PER_EU = (0, 1, 2, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configs", type=Path)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 3, 4, 8])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def full_decode_space() -> list[dict[str, int]]:
    configs: list[dict[str, int]] = []
    for block_n in BLOCK_SIZES:
        for block_k in BLOCK_SIZES:
            for num_warps in NUM_WARPS:
                for waves_per_eu in WAVES_PER_EU:
                    # Matches the supported/pruned ROCm W4A16 runtime space
                    # for the smallest decode bucket: BM=16, GROUP_M=1,
                    # SPLIT_K=1, and the upstream ROCm tuner's two stages.
                    lds_bytes = block_k * (16 + block_n)
                    if lds_bytes > 65536:
                        continue
                    configs.append(
                        {
                            "BLOCK_SIZE_M": 16,
                            "BLOCK_SIZE_N": block_n,
                            "BLOCK_SIZE_K": block_k,
                            "GROUP_SIZE_M": 1,
                            "SPLIT_K": 1,
                            "num_warps": num_warps,
                            "num_stages": 2,
                            "waves_per_eu": waves_per_eu,
                        }
                    )
    return configs


def load_configs(path: Path | None) -> list[dict[str, int]]:
    if path is None:
        return full_decode_space()
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = payload.get("configs", payload.get("results"))
    if not isinstance(payload, list):
        raise ValueError("config input must be a list or contain configs/results")
    configs: list[dict[str, int]] = []
    for item in payload:
        if isinstance(item, dict) and "config" in item:
            item = item["config"]
        if not isinstance(item, dict):
            raise ValueError("each config must be an object")
        configs.append({str(key): int(value) for key, value in item.items()})
    return configs


def max_relative_error(actual: Any, expected: Any) -> float:
    import torch

    denominator = torch.maximum(expected.abs(), torch.tensor(1e-3, device="cuda"))
    return float(((actual - expected).abs() / denominator).max().item())


def main() -> None:
    args = parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard selection")
    if any(token < 1 for token in args.tokens):
        raise ValueError("tokens must be positive")

    # vLLM otherwise sees both CUDA and HIP Triton drivers on this mixed host.
    from r9700.triton_backend import prefer_explicit_rocm_driver

    prefer_explicit_rocm_driver()

    import torch

    from vllm.model_executor.layers.fused_moe import fused_topk, override_config
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts,
        get_default_config,
    )

    torch.manual_seed(args.seed)
    torch.set_default_device("cuda")
    device_name = torch.cuda.get_device_name(0)

    # Exact GLM-5.3-Flash W4A16 shapes after TP8, without expert parallelism.
    experts = 288
    hidden_size = 4096
    shard_intermediate_size = 512
    top_k = 8
    group_size = 128
    max_tokens = max(args.tokens)

    hidden = torch.randn(max_tokens, hidden_size, dtype=torch.bfloat16) * 0.1
    gating = torch.randn(max_tokens, experts, dtype=torch.float32)
    w1 = torch.randint(
        0,
        255,
        (experts, shard_intermediate_size, hidden_size // 2),
        dtype=torch.uint8,
    )
    intermediate_after_activation = shard_intermediate_size // 2
    w2 = torch.randint(
        0,
        255,
        (experts, hidden_size, intermediate_after_activation // 2),
        dtype=torch.uint8,
    )
    w1_scale = torch.rand(
        (experts, shard_intermediate_size, hidden_size // group_size),
        dtype=torch.bfloat16,
    ) * 0.01
    w2_scale = torch.rand(
        (experts, hidden_size, intermediate_after_activation // group_size),
        dtype=torch.bfloat16,
    ) * 0.01
    quant_config = FusedMoEQuantConfig.make(
        weight_dtype="int4",
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_shape=[0, group_size],
    )

    routed: dict[int, tuple[Any, Any]] = {}
    references: dict[int, Any] = {}
    default_configs: dict[int, dict[str, int]] = {}
    for tokens in args.tokens:
        topk_weights, topk_ids, _ = fused_topk(
            hidden[:tokens], gating[:tokens], top_k, renormalize=True
        )
        routed[tokens] = (topk_weights, topk_ids)
        default_config = get_default_config(
            tokens,
            experts,
            shard_intermediate_size,
            hidden_size,
            top_k,
            "int4_w4a16",
            [0, group_size],
        )
        default_configs[tokens] = default_config
        with override_config(default_config):
            references[tokens] = fused_experts(
                hidden[:tokens],
                w1,
                w2,
                topk_weights,
                topk_ids,
                quant_config=quant_config,
            ).detach()
    torch.cuda.synchronize()

    configs = load_configs(args.configs)
    random.Random(args.seed).shuffle(configs)
    configs = configs[args.shard_index :: args.shard_count]
    if args.limit is not None:
        configs = configs[: args.limit]

    results: list[dict[str, Any]] = []
    started = time.monotonic()
    for position, config in enumerate(configs, start=1):
        record: dict[str, Any] = {"config": config, "times_us": {}}
        try:
            outputs: dict[int, Any] = {}
            with override_config(config):
                for tokens in args.tokens:
                    topk_weights, topk_ids = routed[tokens]
                    output = fused_experts(
                        hidden[:tokens],
                        w1,
                        w2,
                        topk_weights,
                        topk_ids,
                        quant_config=quant_config,
                    )
                    outputs[tokens] = output
            torch.cuda.synchronize()

            finite = all(bool(torch.isfinite(output).all().item()) for output in outputs.values())
            max_abs = max(
                float((outputs[tokens] - references[tokens]).abs().max().item())
                for tokens in args.tokens
            )
            max_rel = max(
                max_relative_error(outputs[tokens], references[tokens])
                for tokens in args.tokens
            )
            record.update(finite=finite, max_abs_error=max_abs, max_relative_error=max_rel)
            if not finite or max_abs > 1.0 or max_rel > 0.25:
                record["error"] = "numerical_mismatch"
                results.append(record)
                continue

            for tokens in args.tokens:
                topk_weights, topk_ids = routed[tokens]

                def run() -> Any:
                    with override_config(config):
                        return fused_experts(
                            hidden[:tokens],
                            w1,
                            w2,
                            topk_weights,
                            topk_ids,
                            quant_config=quant_config,
                        )

                for _ in range(args.warmup):
                    run()
                torch.cuda.synchronize()
                begin = torch.Event(enable_timing=True)
                end = torch.Event(enable_timing=True)
                begin.record()
                for _ in range(args.iterations):
                    run()
                end.record()
                end.synchronize()
                record["times_us"][str(tokens)] = (
                    float(begin.elapsed_time(end)) * 1000.0 / args.iterations
                )

            values = list(record["times_us"].values())
            record["geomean_us"] = math.exp(
                sum(math.log(value) for value in values) / len(values)
            )
        except Exception as exc:  # invalid Triton configs must not stop a shard
            record["error"] = f"{type(exc).__name__}: {exc}"
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        results.append(record)

        if position % 10 == 0 or position == len(configs):
            elapsed = time.monotonic() - started
            valid = sum("error" not in item for item in results)
            print(
                f"shard={args.shard_index} progress={position}/{len(configs)} "
                f"valid={valid} elapsed={elapsed:.1f}s",
                flush=True,
            )
        if position % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    payload = {
        "schema_version": 1,
        "device": device_name,
        "dtype": "bfloat16/int4_w4a16",
        "shape": {
            "experts": experts,
            "hidden_size": hidden_size,
            "shard_intermediate_size": shard_intermediate_size,
            "top_k": top_k,
            "group_size": group_size,
        },
        "tokens": args.tokens,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "default_configs": default_configs,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
