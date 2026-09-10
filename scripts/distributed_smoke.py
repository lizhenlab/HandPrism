#!/usr/bin/env python3
"""Small NCCL all-reduce smoke; launch with torchrun."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    value = torch.tensor(float(rank + 1), device=f"cuda:{local_rank}")
    dist.all_reduce(value)
    expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
    if value.item() != expected:
        raise RuntimeError(f"rank {rank}: all-reduce {value.item()} != {expected}")
    if rank == 0:
        print(json.dumps({"world_size": dist.get_world_size(), "all_reduce": value.item()}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

