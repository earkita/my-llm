from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist


def _sizes(value: str) -> list[int]:
    try:
        sizes = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers") from exc
    if not sizes or any(size < 4 for size in sizes):
        raise argparse.ArgumentTypeError("every size must be at least four bytes")
    return sizes


def _allreduce(tensor: torch.Tensor) -> None:
    dist.all_reduce(tensor)


def _ping_pong(tensor: torch.Tensor, rank: int) -> None:
    peer = 1 - rank
    if rank == 0:
        dist.batch_isend_irecv([dist.P2POp(dist.isend, tensor, peer)])[0].wait()
        dist.batch_isend_irecv([dist.P2POp(dist.irecv, tensor, peer)])[0].wait()
    else:
        dist.batch_isend_irecv([dist.P2POp(dist.irecv, tensor, peer)])[0].wait()
        dist.batch_isend_irecv([dist.P2POp(dist.isend, tensor, peer)])[0].wait()


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure one isolated two-GPU RCCL link")
    parser.add_argument("--mode", choices=("allreduce", "ping-pong"), required=True)
    parser.add_argument(
        "--sizes",
        type=_sizes,
        default=_sizes("4096,1048576,8388608,67108864"),
        help="comma-separated payload sizes in bytes",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup must be non-negative and iterations must be positive")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 2:
        raise RuntimeError("link benchmark requires exactly two ranks")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        operation = _allreduce if args.mode == "allreduce" else None
        for requested_bytes in args.sizes:
            elements = (requested_bytes + 3) // 4
            tensor = torch.zeros(elements, dtype=torch.float32, device="cuda")
            payload_bytes = tensor.numel() * tensor.element_size()

            for _ in range(args.warmup):
                operation(tensor) if operation else _ping_pong(tensor, rank)
            torch.cuda.synchronize()
            dist.barrier()

            started = time.perf_counter()
            for _ in range(args.iterations):
                operation(tensor) if operation else _ping_pong(tensor, rank)
            torch.cuda.synchronize()
            dist.barrier()
            elapsed = time.perf_counter() - started

            if rank == 0:
                seconds = elapsed / args.iterations
                transferred = payload_bytes if args.mode == "allreduce" else 2 * payload_bytes
                print(
                    json.dumps(
                        {
                            "mode": args.mode,
                            "payload_bytes": payload_bytes,
                            "iterations": args.iterations,
                            "mean_microseconds": seconds * 1e6,
                            "effective_gigabytes_per_second": transferred / seconds / 1e9,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
