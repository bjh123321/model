from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import torch
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

def build_dataloaders(sim_parsed: List[ParsedRow], real_parsed: List[ParsedRow], cfg: Config,
                      kfold: int, fold_idx: int, device: torch.device, out_dir: Path):
    in_norm_sim = compute_input_normalizer_sim(sim_parsed)
    if is_main_process():
        in_norm_sim.save(out_dir / 'Normalizer_Input_Sim.pth')

    out_norm_sim = compute_output_normalizer_masked(sim_parsed)
    if is_main_process():
        out_norm_sim.save(out_dir / 'Normalizer_Output_Sim.pth')

    kf = KFold(n_splits=kfold, shuffle=True, random_state=42)
    indices = list(range(len(real_parsed)))
    folds = list(kf.split(indices))
    train_idx, val_idx = folds[fold_idx]

    real_train = [real_parsed[i] for i in train_idx]
    real_val   = [real_parsed[i] for i in val_idx]

    out_norm_real = compute_output_normalizer_masked(real_train)
    if is_main_process():
        out_norm_real.save(out_dir / f'Normalizer_Output_Real_Fold{fold_idx}.pth')

    sim_ds  = ModalDataset(sim_parsed, cfg.K_max, in_norm_sim, out_norm_sim)
    real_tr = ModalDataset(real_train, cfg.K_max, in_norm_sim, out_norm_real)
    real_va = ModalDataset(real_val,   cfg.K_max, in_norm_sim, out_norm_real)

    sim_sampler  = DistributedSampler(sim_ds, shuffle=True, drop_last=True)
    real_sampler = DistributedSampler(real_tr, shuffle=True, drop_last=True)
    val_sampler  = DistributedSampler(real_va, shuffle=False, drop_last=False)

    sim_dl = DataLoader(sim_ds, batch_size=cfg.batch_sim, sampler=sim_sampler,
                        num_workers=cfg.num_workers, pin_memory=True, prefetch_factor=cfg.prefetch, persistent_workers=True, collate_fn=collate_modal)
    real_dl = DataLoader(real_tr, batch_size=cfg.batch_real, sampler=real_sampler,
                        num_workers=cfg.num_workers, pin_memory=True, prefetch_factor=cfg.prefetch, persistent_workers=True, collate_fn=collate_modal)
    val_dl = DataLoader(real_va, batch_size=cfg.batch_real, sampler=val_sampler,
                        num_workers=cfg.num_workers, pin_memory=True, prefetch_factor=cfg.prefetch, persistent_workers=True, collate_fn=collate_modal)

    return sim_dl, real_dl, val_dl, len(sim_ds), len(real_tr)
