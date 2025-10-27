from __future__ import annotations
import torch
import torch.distributed as dist

def is_main_process() -> bool:
    return (not dist.is_initialized()) or (dist.get_rank() == 0)

def dist_mean_scalar(x: float, device: torch.device) -> float:
    if not dist.is_initialized():
        return float(x)
    t = torch.tensor(x, dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t = t / dist.get_world_size()
    return float(t.item())
