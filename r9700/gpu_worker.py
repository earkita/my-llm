from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist


def allreduce(rank: int, world: int) -> None:
    value = torch.tensor([rank + 1.0], device="cuda")
    dist.all_reduce(value)
    expected = world * (world + 1) / 2
    if value.item() != expected:
        raise RuntimeError(f"rank {rank}: allreduce {value.item()} != {expected}")


def pipeline(rank: int, world: int) -> None:
    if world % 2 or world < 2:
        raise RuntimeError("pipeline smoke requires an even world size >=2")
    lane = rank % 2
    stage = rank // 2
    stages = world // 2
    value = torch.tensor([float(lane + 1)], device="cuda")
    if stage:
        source = rank - 2
        dist.batch_isend_irecv([dist.P2POp(dist.irecv, value, source)])[0].wait()
        value.add_(1)
    if stage + 1 < stages:
        target = rank + 2
        dist.batch_isend_irecv([dist.P2POp(dist.isend, value, target)])[0].wait()
    else:
        expected = lane + stages
        if value.item() != expected:
            raise RuntimeError(f"rank {rank}: pipeline {value.item()} != {expected}")
    dist.barrier()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("allreduce", "pipeline"), required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        allreduce(rank, world) if args.mode == "allreduce" else pipeline(rank, world)
        print(f"PASS mode={args.mode} rank={rank} world={world}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
