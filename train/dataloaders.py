from __future__ import annotations
from pathlib import Path
from typing import List
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import KFold
from ..config import Config
from ..data.io import ParsedRow
from ..data.normalizer import (
    compute_input_normalizer_sim,
    compute_output_normalizer_masked,
)
from ..data.datasets import ModalDataset, collate_modal
from ..utils.dist_utils import is_main_process


def _ddp_ready() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def build_dataloaders(sim_parsed: List[ParsedRow], real_parsed: List[ParsedRow], cfg: Config,
                      kfold: int, fold_idx: int, device: torch.device, out_dir: Path):
    in_norm_sim = compute_input_normalizer_sim(sim_parsed)
    if is_main_process():
        in_norm_sim.save(out_dir / 'Normalizer_Input_Sim.pth')

    out_norm_sim = compute_output_normalizer_masked(sim_parsed)
    if is_main_process():
        out_norm_sim.save(out_dir / 'Normalizer_Output_Sim.pth')

    # === 如果 kfold == 1，直接使用全部数据，不分折 ===
    if kfold <= 1:
        real_train = real_parsed
        real_val   = []    # 或者 real_val = [] 根据你是否需要验证
    else:
        # 正常 kfold 流程
        kf = KFold(n_splits=kfold, shuffle=True, random_state=42)
        indices = list(range(len(real_parsed)))
        train_idx, val_idx = list(kf.split(indices))[fold_idx]

        real_train = [real_parsed[i] for i in train_idx]
        real_val   = [real_parsed[i] for i in val_idx]


    out_norm_real = compute_output_normalizer_masked(real_train)
    if is_main_process():
        out_norm_real.save(out_dir / f'Normalizer_Output_Real_Fold{fold_idx}.pth')

    sim_ds  = ModalDataset(sim_parsed, cfg.K_max, in_norm_sim, out_norm_sim)
    real_tr = ModalDataset(real_train, cfg.K_max, in_norm_sim, out_norm_real)
    real_va = ModalDataset(real_val,   cfg.K_max, in_norm_sim, out_norm_real)

    if _ddp_ready():
        sim_sampler  = DistributedSampler(sim_ds, shuffle=True, drop_last=True)
        real_sampler = DistributedSampler(real_tr, shuffle=True, drop_last=True)
        val_sampler  = DistributedSampler(real_va, shuffle=False, drop_last=False)
    else:
        sim_sampler  = None
        real_sampler = None
        val_sampler  = None


    # ---- DataLoader (DDP compatible + non-DDP safe) ----
    sim_shuffle  = (sim_sampler is None)   # non-DDP: shuffle; DDP: sampler controls shuffle
    real_shuffle = (real_sampler is None)

    # pin_memory only helps on CUDA; keep it safe for CPU/Windows
    _pin = bool(torch.cuda.is_available())

    # prefetch_factor / persistent_workers only valid when num_workers > 0
    _nw = int(cfg.num_workers)
    _dl_kwargs = dict(
        num_workers=_nw,
        pin_memory=_pin,
        collate_fn=collate_modal,
    )
    if _nw > 0:
        _dl_kwargs.update(dict(
            prefetch_factor=int(cfg.prefetch),
            persistent_workers=True,
        ))

    sim_dl = DataLoader(
        sim_ds,
        batch_size=cfg.batch_sim,
        shuffle=sim_shuffle,
        sampler=sim_sampler,
        **_dl_kwargs
    )

    real_dl = DataLoader(
        real_tr,
        batch_size=cfg.batch_real,
        shuffle=real_shuffle,
        sampler=real_sampler,
        **_dl_kwargs
    )

    val_dl = DataLoader(
        real_va,
        batch_size=cfg.batch_real,
        shuffle=False,
        sampler=val_sampler,
        **_dl_kwargs
    )


    return sim_dl, real_dl, val_dl, len(sim_ds), len(real_tr)
