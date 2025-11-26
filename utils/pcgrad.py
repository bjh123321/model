# surrogate_model/utils/pcgrad.py
import torch
import torch.distributed as dist
from contextlib import contextmanager

def is_dist():
    return dist.is_available() and dist.is_initialized()

@contextmanager
def maybe_no_sync(module):
    """If module is DDP, suppress grad sync for a block."""
    no_sync = getattr(module, "no_sync", None)
    if callable(no_sync):
        with module.no_sync():
            yield
    else:
        yield

def clone_grads(params):
    """Clone .grad of params into a list, None -> None."""
    out = []
    for p in params:
        if p.grad is None:
            out.append(None)
        else:
            out.append(p.grad.detach().clone())
    return out

def zero_grads(params):
    for p in params:
        p.grad = None

def pcgrad_two_tasks(g1, g2):
    """
    PCGrad for two tasks.
    g1, g2: lists of tensors (same param order). Tensors are *unscaled* fp32.
    Return: combined grads (list of tensors) or None if both None.
    """
    out = []
    for a, b in zip(g1, g2):
        if a is None and b is None:
            out.append(None); continue
        if a is None:
            out.append(b); continue
        if b is None:
            out.append(a); continue

        va = a.view(-1); vb = b.view(-1)
        dot = torch.dot(va, vb)
        if dot < 0:
            # project a onto the normal of b: a <- a - proj_b(a)
            denom = vb.norm()**2 + 1e-12
            a = a - (dot / denom) * b
        out.append(a + b)
    return out

def set_grads(params, grads):
    for p, g in zip(params, grads):
        if g is None:
            p.grad = None
        else:
            if p.grad is None:
                p.grad = g
            else:
                p.grad.copy_(g)

def allreduce_grads_inplace(params, avg=True):
    """All-reduce grads if DDP, so each rank uses the same combined grads."""
    if not is_dist():
        return
    world = dist.get_world_size()
    for p in params:
        if p.grad is None:
            continue
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        if avg and world > 1:
            p.grad.div_(world)
