#!/usr/bin/env python3
"""Tune MiMo block-scaled W8A8 FP8 Triton GEMMs on R9700 GPUs."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _w8a8_triton_block_scaled_mm,
)
from vllm.triton_utils import triton
from vllm.utils.platform_utils import get_device_name_as_file_name


SHAPES = ((1856, 4096), (4096, 4096), (4096, 2048))
BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
BLOCK_SIZE = (128, 128)
DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 32,
    "num_warps": 4,
    "num_stages": 2,
}


def candidates(m: int) -> list[dict[str, int]]:
    if m <= 16:
        block_ms = (16, 32, 64)
    elif m <= 128:
        block_ms = (32, 64, 128)
    else:
        block_ms = (64, 128, 256)
    return [
        {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": group_m,
            "num_warps": warps,
            "num_stages": stages,
        }
        for block_m in block_ms
        for block_n in (128, 256)
        for block_k in (64, 128)
        for group_m in (1, 32)
        for warps in (2, 4, 8)
        for stages in (1, 2)
    ]


def make_inputs(m: int, n: int, k: int, device: int):
    generator = torch.Generator(device=f"cuda:{device}")
    generator.manual_seed(m * 1_000_003 + n * 101 + k)
    a = torch.randn(
        (m, k), dtype=torch.float32, device=f"cuda:{device}", generator=generator
    ).clamp_(-4, 4).to(torch.float8_e4m3fn)
    b = torch.randn(
        (n, k), dtype=torch.float32, device=f"cuda:{device}", generator=generator
    ).clamp_(-4, 4).to(torch.float8_e4m3fn)
    a_scale = torch.rand(
        (m, triton.cdiv(k, 128)),
        dtype=torch.float32,
        device=f"cuda:{device}",
        generator=generator,
    ).mul_(0.02)
    b_scale = torch.rand(
        (triton.cdiv(n, 128), triton.cdiv(k, 128)),
        dtype=torch.float32,
        device=f"cuda:{device}",
        generator=generator,
    ).mul_(0.02)
    return a, b, a_scale, b_scale


def run_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    config: dict[str, int],
) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[0]
    output = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)

    def grid(meta):
        return (
            triton.cdiv(m, meta["BLOCK_SIZE_M"])
            * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
        )

    _w8a8_triton_block_scaled_mm[grid](
        a,
        b,
        output,
        a_scale,
        b_scale,
        m,
        n,
        k,
        128,
        128,
        a.stride(0),
        a.stride(1),
        b.stride(1),
        b.stride(0),
        output.stride(0),
        output.stride(1),
        a_scale.stride(0),
        a_scale.stride(1),
        b_scale.stride(1),
        b_scale.stride(0),
        **config,
    )
    return output


def latency_us(inputs, config: dict[str, int], warmups: int, iterations: int) -> float:
    for _ in range(warmups):
        run_kernel(*inputs, config)
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_kernel(*inputs, config)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples)


def tune_one(
    device: int,
    m: int,
    n: int,
    k: int,
    warmups: int,
    iterations: int,
    require_bitwise_default: bool,
) -> dict[str, Any]:
    inputs = make_inputs(m, n, k, device)
    default_output = run_kernel(*inputs, DEFAULT_CONFIG)
    torch.cuda.synchronize()
    default_us = latency_us(inputs, DEFAULT_CONFIG, warmups, iterations)
    best_config = DEFAULT_CONFIG
    best_us = default_us
    tested = 0
    failed = 0
    bitwise_rejected = 0
    for config in candidates(m):
        try:
            if require_bitwise_default:
                candidate_output = run_kernel(*inputs, config)
                torch.cuda.synchronize()
                if not torch.equal(candidate_output, default_output):
                    bitwise_rejected += 1
                    continue
            measured = latency_us(inputs, config, warmups, iterations)
        except Exception:
            failed += 1
            continue
        tested += 1
        if measured < best_us:
            best_config = config
            best_us = measured
    return {
        "M": m,
        "N": n,
        "K": k,
        "default_us": default_us,
        "best_us": best_us,
        "speedup": default_us / best_us,
        "config": best_config,
        "tested": tested,
        "failed": failed,
        "bitwise_rejected": bitwise_rejected,
    }


def worker(job: tuple[int, list[int], int, int, int, int, bool]):
    device, batch_sizes, warmups, iterations, n, k, require_bitwise_default = job
    torch.cuda.set_device(device)
    rows = []
    for m in batch_sizes:
        row = tune_one(
            device, m, n, k, warmups, iterations, require_bitwise_default
        )
        print(
            f"gpu={device} M={m} N={n} K={k} "
            f"{row['default_us']:.2f}->{row['best_us']:.2f} us "
            f"x{row['speedup']:.3f}",
            flush=True,
        )
        rows.append(row)
    return rows


def expand_scales(scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    return scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)[
        :rows, :cols
    ]


def validate(rows: list[dict[str, Any]]) -> None:
    torch.cuda.set_device(0)
    for row in rows:
        m, n, k = row["M"], row["N"], row["K"]
        inputs = make_inputs(m, n, k, 0)
        actual = run_kernel(*inputs, row["config"]).float()
        a, b, a_scale, b_scale = inputs
        a_dequant = a.float() * a_scale.repeat_interleave(128, dim=1)[:, :k]
        b_dequant = b.float() * expand_scales(b_scale, n, k)
        expected = a_dequant @ b_dequant.T
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument(
        "--require-bitwise-default",
        action="store_true",
        help="Only accept kernels whose BF16 output exactly matches the default kernel",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("No ROCm GPU is visible")
    jobs = []
    for shape_index, (n, k) in enumerate(SHAPES):
        buckets = [[] for _ in range(gpu_count)]
        for index, m in enumerate(BATCH_SIZES):
            buckets[index % gpu_count].append(m)
        jobs.extend(
            (
                device,
                bucket,
                args.warmups,
                args.iterations,
                n,
                k,
                args.require_bitwise_default,
            )
            for device, bucket in enumerate(buckets)
            if bucket
        )

    started = time.perf_counter()
    with mp.get_context("spawn").Pool(min(len(jobs), gpu_count)) as pool:
        nested = pool.map(worker, jobs)
    rows = [row for group in nested for row in group]
    rows.sort(key=lambda row: (row["N"], row["K"], row["M"]))
    validate(rows)

    device_name = get_device_name_as_file_name()
    for n, k in SHAPES:
        selected = [row for row in rows if row["N"] == n and row["K"] == k]
        configs = {str(row["M"]): row["config"] for row in selected}
        filename = (
            f"N={n},K={k},device_name={device_name},dtype=fp8_w8a8,"
            "block_shape=[128,128].json"
        )
        (args.output_dir / filename).write_text(
            json.dumps(configs, indent=2) + "\n", encoding="utf-8"
        )
    report = {
        "device": device_name,
        "gpu_count": gpu_count,
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"validated=true output={args.output_dir} elapsed={report['elapsed_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())
