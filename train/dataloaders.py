from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.distributed as dist
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from surrogate_model.config import Config
from surrogate_model.data.io import ParsedRow
from surrogate_model.data.normalizer import compute_input_normalizer_sim, compute_output_normalizer_masked
from surrogate_model.data.datasets import ModalDataset, collate_modal
from surrogate_model.utils.dist_utils import is_main_process


def _ddp_ready() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _target_modes(cfg: Config) -> tuple[str, str]:
    pole_target = str(getattr(cfg, "pole_target", "alpha_omega"))
    residue_target = str(getattr(cfg, "residue_target", "polar"))
    return pole_target, residue_target


def _output_normalizer(parsed: List[ParsedRow], cfg: Config):
    pole_target, residue_target = _target_modes(cfg)
    return compute_output_normalizer_masked(parsed, pole_target=pole_target, residue_target=residue_target)


def _split_real_by_height_max_mach(real_parsed: List[ParsedRow], cfg: Config):
    heights = [float(h) for h in getattr(cfg, "val_heights", [3.0, 8.23])]
    h_tol = float(getattr(cfg, "val_height_tol", 1e-6))
    m_tol = float(getattr(cfg, "val_mach_tol", 1e-6))
    val_indices = set()
    split_info = []

    for target_h in heights:
        h_indices = [i for i, p in enumerate(real_parsed) if abs(float(p.H) - target_h) <= h_tol]
        if not h_indices:
            raise RuntimeError(
                f"No real samples found for validation height H={target_h:g}. "
                f"Available heights: {sorted({float(p.H) for p in real_parsed})}"
            )
        max_m = max(float(real_parsed[i].M) for i in h_indices)
        hm_indices = [i for i in h_indices if abs(float(real_parsed[i].M) - max_m) <= m_tol]
        val_indices.update(hm_indices)
        split_info.append((target_h, max_m, len(hm_indices)))

    real_train = [p for i, p in enumerate(real_parsed) if i not in val_indices]
    real_val = [p for i, p in enumerate(real_parsed) if i in val_indices]
    if not real_train:
        raise RuntimeError("Validation split consumed all real samples; no real training data remains.")
    if not real_val:
        raise RuntimeError("Validation split produced no real validation samples.")
    return real_train, real_val, split_info


def _hm_key(p: ParsedRow, h_tol: float, m_tol: float):
    h = round(float(p.H) / h_tol) if h_tol > 0 else float(p.H)
    m = round(float(p.M) / m_tol) if m_tol > 0 else float(p.M)
    return h, m


def _split_by_height_mach_groups(parsed: List[ParsedRow], cfg: Config, kfold: int, fold_idx: int, tag: str):
    if len(parsed) <= 1:
        return parsed, [], []
    h_tol = float(getattr(cfg, "val_height_tol", 1e-6))
    m_tol = float(getattr(cfg, "val_mach_tol", 1e-6))
    groups: dict[tuple, list[int]] = {}
    for i, p in enumerate(parsed):
        groups.setdefault(_hm_key(p, h_tol, m_tol), []).append(i)
    keys = sorted(groups.keys())
    if len(keys) <= 1:
        return parsed, [], []

    seed = int(getattr(cfg, "val_seed", 42))
    if kfold and kfold > 1:
        if kfold > len(keys):
            raise RuntimeError(f"{tag}: kfold={kfold} is larger than height-Mach groups={len(keys)}")
        splits = list(KFold(n_splits=kfold, shuffle=True, random_state=seed).split(keys))
        _, val_group_idx = splits[fold_idx % kfold]
        val_keys = {keys[i] for i in val_group_idx}
    else:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(keys))
        rng.shuffle(idx)
        n_val = max(1, int(round(len(keys) * float(getattr(cfg, "val_ratio", 0.2)))))
        val_keys = {keys[i] for i in idx[:n_val]}

    val_indices = {idx for key in val_keys for idx in groups[key]}
    train = [p for i, p in enumerate(parsed) if i not in val_indices]
    val = [p for i, p in enumerate(parsed) if i in val_indices]
    if not train or not val:
        raise RuntimeError(f"{tag}: height_mach_kfold split produced empty train or val set")
    return train, val, [(key, len(groups[key])) for key in sorted(val_keys)]


def _make_loader(ds, batch_size: int, shuffle: bool, sampler):
    _pin = bool(torch.cuda.is_available())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=0, pin_memory=_pin, collate_fn=collate_modal)


def _loader_kwargs(cfg: Config):
    _pin = bool(torch.cuda.is_available())
    _nw = int(cfg.num_workers)
    kwargs = dict(num_workers=_nw, pin_memory=_pin, collate_fn=collate_modal)
    if _nw > 0:
        kwargs.update(dict(prefetch_factor=int(cfg.prefetch), persistent_workers=True))
    return kwargs


def build_dataloaders(sim_parsed: List[ParsedRow], real_parsed: List[ParsedRow], cfg: Config,
                      kfold: int, fold_idx: int, device: torch.device, out_dir: Path):
    in_norm_sim = compute_input_normalizer_sim(sim_parsed)
    if is_main_process():
        in_norm_sim.save(out_dir / "Normalizer_Input_Sim.pth")

    out_norm_sim = _output_normalizer(sim_parsed, cfg)
    if is_main_process():
        out_norm_sim.save(out_dir / "Normalizer_Output_Sim.pth")

    use_val_split = bool(getattr(cfg, "use_val_split", False))
    val_ratio = float(getattr(cfg, "val_ratio", 0.2))
    seed = int(getattr(cfg, "val_seed", 42))
    val_strategy = str(getattr(cfg, "val_split_strategy", "")).lower()
    use_height_max_mach_val = use_val_split and val_strategy == "height_max_mach"
    use_height_mach_kfold = use_val_split and val_strategy == "height_mach_kfold"

    if use_height_mach_kfold:
        sim_train, sim_val, sim_info = _split_by_height_mach_groups(sim_parsed, cfg, kfold, fold_idx, "sim")
        if is_main_process():
            print(f"[VAL] sim height_mach_kfold: val_groups={len(sim_info)}, train_rows={len(sim_train)}, val_rows={len(sim_val)}", flush=True)
    elif (not use_height_max_mach_val) and use_val_split and (0.0 < val_ratio < 1.0) and (len(sim_parsed) > 1):
        rng = np.random.default_rng(seed)
        idx = np.arange(len(sim_parsed))
        rng.shuffle(idx)
        n_val = max(1, int(round(len(sim_parsed) * val_ratio)))
        sim_train = [sim_parsed[i] for i in idx[n_val:]]
        sim_val = [sim_parsed[i] for i in idx[:n_val]]
    else:
        sim_train = sim_parsed
        sim_val = []

    if use_val_split:
        if use_height_mach_kfold:
            real_train, real_val, real_info = _split_by_height_mach_groups(real_parsed, cfg, kfold, fold_idx, "real")
            if is_main_process():
                print(f"[VAL] real height_mach_kfold: val_groups={len(real_info)}, train_rows={len(real_train)}, val_rows={len(real_val)}", flush=True)
        elif use_height_max_mach_val:
            real_train, real_val, split_info = _split_real_by_height_max_mach(real_parsed, cfg)
            if is_main_process():
                detail = ", ".join(f"H={h:g}, maxM={m:g}, val_rows={n}" for h, m, n in split_info)
                print(f"[VAL] real split by height max Mach: {detail}; train_rows={len(real_train)}, val_rows={len(real_val)}", flush=True)
        elif kfold > 1 and len(real_parsed) > 1:
            kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)
            indices = list(range(len(real_parsed)))
            train_idx, val_idx = list(kf.split(indices))[fold_idx]
            real_train = [real_parsed[i] for i in train_idx]
            real_val = [real_parsed[i] for i in val_idx]
        elif (0.0 < val_ratio < 1.0) and (len(real_parsed) > 1):
            rng = np.random.default_rng(seed + 1000)
            idx = np.arange(len(real_parsed))
            rng.shuffle(idx)
            n_val = max(1, int(round(len(real_parsed) * val_ratio)))
            real_train = [real_parsed[i] for i in idx[n_val:]]
            real_val = [real_parsed[i] for i in idx[:n_val]]
        else:
            real_train = real_parsed
            real_val = []
    else:
        real_train = real_parsed
        real_val = []

    out_norm_real = _output_normalizer(real_train, cfg)
    if is_main_process():
        out_norm_real.save(out_dir / f"Normalizer_Output_Real_Fold{fold_idx}.pth")

    sim_ds = ModalDataset(sim_train, cfg.K_max, in_norm_sim, out_norm_sim)
    sim_va = ModalDataset(sim_val, cfg.K_max, in_norm_sim, out_norm_sim)
    real_tr = ModalDataset(real_train, cfg.K_max, in_norm_sim, out_norm_real)
    real_va = ModalDataset(real_val, cfg.K_max, in_norm_sim, out_norm_real)

    if _ddp_ready():
        sim_sampler = DistributedSampler(sim_ds, shuffle=True, drop_last=True) if len(sim_ds) > 0 else None
        real_sampler = DistributedSampler(real_tr, shuffle=True, drop_last=True) if len(real_tr) > 0 else None
        sim_val_sampler = DistributedSampler(sim_va, shuffle=False, drop_last=False) if len(sim_va) > 0 else None
        val_sampler = DistributedSampler(real_va, shuffle=False, drop_last=False) if len(real_va) > 0 else None
    else:
        sim_sampler = real_sampler = sim_val_sampler = val_sampler = None

    kwargs = _loader_kwargs(cfg)
    sim_dl = DataLoader(sim_ds, batch_size=int(getattr(cfg, "batch_sim", getattr(cfg, "batch_real", 64))), shuffle=(sim_sampler is None), sampler=sim_sampler, **kwargs)
    real_dl = DataLoader(real_tr, batch_size=cfg.batch_real, shuffle=(real_sampler is None), sampler=real_sampler, **kwargs)
    sim_val_dl = DataLoader(sim_va, batch_size=int(getattr(cfg, "batch_sim", getattr(cfg, "batch_real", 64))), shuffle=False, sampler=sim_val_sampler, **kwargs)
    val_dl = DataLoader(real_va, batch_size=cfg.batch_real, shuffle=False, sampler=val_sampler, **kwargs)

    return sim_dl, real_dl, sim_val_dl, val_dl, len(sim_ds), len(real_tr), out_norm_sim, out_norm_real, in_norm_sim


def build_single_fidelity_dataloaders(
    real_parsed: List[ParsedRow],
    cfg: Config,
    kfold: int,
    fold_idx: int,
    device: torch.device,
    out_dir: Path,
):
    use_val_split = bool(getattr(cfg, "use_val_split", False))
    val_ratio = float(getattr(cfg, "val_ratio", 0.2))
    seed = int(getattr(cfg, "val_seed", 42))
    val_strategy = str(getattr(cfg, "val_split_strategy", "")).lower()

    if use_val_split:
        if val_strategy == "height_mach_kfold":
            real_train, real_val, real_info = _split_by_height_mach_groups(real_parsed, cfg, kfold, fold_idx, "real")
            if is_main_process():
                print(f"[VAL] real height_mach_kfold: val_groups={len(real_info)}, train_rows={len(real_train)}, val_rows={len(real_val)}", flush=True)
        elif val_strategy == "height_max_mach":
            real_train, real_val, split_info = _split_real_by_height_max_mach(real_parsed, cfg)
            if is_main_process():
                detail = ", ".join(f"H={h:g}, maxM={m:g}, val_rows={n}" for h, m, n in split_info)
                print(f"[VAL] real split by height max Mach: {detail}; train_rows={len(real_train)}, val_rows={len(real_val)}", flush=True)
        elif kfold > 1 and len(real_parsed) > 1:
            kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)
            indices = list(range(len(real_parsed)))
            train_idx, val_idx = list(kf.split(indices))[fold_idx]
            real_train = [real_parsed[i] for i in train_idx]
            real_val = [real_parsed[i] for i in val_idx]
        elif (0.0 < val_ratio < 1.0) and (len(real_parsed) > 1):
            rng = np.random.default_rng(seed + 1000)
            idx = np.arange(len(real_parsed))
            rng.shuffle(idx)
            n_val = max(1, int(round(len(real_parsed) * val_ratio)))
            real_train = [real_parsed[i] for i in idx[n_val:]]
            real_val = [real_parsed[i] for i in idx[:n_val]]
        else:
            real_train = real_parsed
            real_val = []
    else:
        real_train = real_parsed
        real_val = []

    in_norm_real = compute_input_normalizer_sim(real_train)
    out_norm_real = _output_normalizer(real_train, cfg)
    if is_main_process():
        in_norm_real.save(out_dir / "Normalizer_Input_Real.pth")
        out_norm_real.save(out_dir / f"Normalizer_Output_Real_Fold{fold_idx}.pth")

    real_tr = ModalDataset(real_train, cfg.K_max, in_norm_real, out_norm_real)
    real_va = ModalDataset(real_val, cfg.K_max, in_norm_real, out_norm_real)

    if _ddp_ready():
        real_sampler = DistributedSampler(real_tr, shuffle=True, drop_last=True) if len(real_tr) > 0 else None
        val_sampler = DistributedSampler(real_va, shuffle=False, drop_last=False) if len(real_va) > 0 else None
    else:
        real_sampler = None
        val_sampler = None

    kwargs = _loader_kwargs(cfg)
    train_dl = DataLoader(real_tr, batch_size=cfg.batch_real, shuffle=(real_sampler is None), sampler=real_sampler, **kwargs)
    val_dl = DataLoader(real_va, batch_size=cfg.batch_real, shuffle=False, sampler=val_sampler, **kwargs)
    return train_dl, val_dl, len(real_tr), len(real_va), out_norm_real, in_norm_real
