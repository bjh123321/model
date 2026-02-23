from pathlib import Path
import numpy as np
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

    # ========= Train/Val split =========
    # cfg.use_val_split=False: 使用全部数据训练（不划分验证集），保持原有训练逻辑
    # cfg.use_val_split=True : 划分验证集；当 kfold>1 时沿用 KFold，否则按 cfg.val_ratio 随机划分
    use_val_split = bool(getattr(cfg, "use_val_split", False))
    val_ratio = float(getattr(cfg, "val_ratio", 0.2))
    seed = int(getattr(cfg, "val_seed", 42))

    # --- Sim split (给 Stage-1 用) ---
    if use_val_split and (0.0 < val_ratio < 1.0) and (len(sim_parsed) > 1):
        rng = np.random.default_rng(seed)
        idx = np.arange(len(sim_parsed))
        rng.shuffle(idx)
        n_val = max(1, int(round(len(sim_parsed) * val_ratio)))
        val_idx = idx[:n_val]
        tr_idx  = idx[n_val:]
        sim_train = [sim_parsed[i] for i in tr_idx]
        sim_val   = [sim_parsed[i] for i in val_idx]
    else:
        sim_train = sim_parsed
        sim_val   = []

    # --- Real split (给 Stage-2 & final validation 用) ---
    if use_val_split:
        if kfold > 1 and len(real_parsed) > 1:
            kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)
            indices = list(range(len(real_parsed)))
            train_idx, val_idx = list(kf.split(indices))[fold_idx]
            real_train = [real_parsed[i] for i in train_idx]
            real_val   = [real_parsed[i] for i in val_idx]
        elif (0.0 < val_ratio < 1.0) and (len(real_parsed) > 1):
            rng = np.random.default_rng(seed + 1000)
            idx = np.arange(len(real_parsed))
            rng.shuffle(idx)
            n_val = max(1, int(round(len(real_parsed) * val_ratio)))
            val_idx = idx[:n_val]
            tr_idx  = idx[n_val:]
            real_train = [real_parsed[i] for i in tr_idx]
            real_val   = [real_parsed[i] for i in val_idx]
        else:
            real_train = real_parsed
            real_val   = []
    else:
        # 不划分验证集：全部用于训练
        real_train = real_parsed
        real_val   = []

    out_norm_real = compute_output_normalizer_masked(real_train)
    if is_main_process():
        out_norm_real.save(out_dir / f'Normalizer_Output_Real_Fold{fold_idx}.pth')

    sim_ds  = ModalDataset(sim_train, cfg.K_max, in_norm_sim, out_norm_sim)
    sim_va  = ModalDataset(sim_val,   cfg.K_max, in_norm_sim, out_norm_sim)
    real_tr = ModalDataset(real_train, cfg.K_max, in_norm_sim, out_norm_real)
    real_va = ModalDataset(real_val,   cfg.K_max, in_norm_sim, out_norm_real)

    if _ddp_ready():
        sim_sampler  = DistributedSampler(sim_ds,  shuffle=True,  drop_last=True) if len(sim_ds)  > 0 else None
        real_sampler = DistributedSampler(real_tr, shuffle=True,  drop_last=True) if len(real_tr) > 0 else None
        sim_val_sampler = DistributedSampler(sim_va, shuffle=False, drop_last=False) if len(sim_va) > 0 else None
        val_sampler  = DistributedSampler(real_va, shuffle=False, drop_last=False) if len(real_va) > 0 else None
    else:
        sim_sampler  = None
        real_sampler = None
        sim_val_sampler = None
        val_sampler  = None

    sim_shuffle  = (sim_sampler is None)
    real_shuffle = (real_sampler is None)

    _pin = bool(torch.cuda.is_available())

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

    sim_val_dl = DataLoader(
        sim_va,
        batch_size=cfg.batch_sim,
        shuffle=False,
        sampler=sim_val_sampler,
        **_dl_kwargs
    )

    val_dl = DataLoader(
        real_va,
        batch_size=cfg.batch_real,
        shuffle=False,
        sampler=val_sampler,
        **_dl_kwargs
    )

    return sim_dl, real_dl, sim_val_dl, val_dl, len(sim_ds), len(real_tr)