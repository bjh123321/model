from pathlib import Path
from typing import List
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from surrogate_model.config import Config
from surrogate_model.data.io import ParsedRow
from surrogate_model.data.normalizer import Normalizer, compute_input_normalizer_sim, compute_output_normalizer_masked
from surrogate_model.data.datasets import ModalDataset, collate_modal
from surrogate_model.utils.dist_utils import is_main_process
from surrogate_model.utils.model_artifacts import group_key_from_parsed
from surrogate_model.utils.model_artifacts import atomic_write_json
from surrogate_model.utils.fdr_target_transforms import build_target_transform_specs


def _ddp_ready() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _target_modes(cfg: Config) -> tuple[str, str]:
    pole_target = str(getattr(cfg, "pole_target", "alpha_omega"))
    residue_target = str(getattr(cfg, "residue_target", "polar"))
    return pole_target, residue_target


def _output_normalizer(parsed: List[ParsedRow], cfg: Config):
    pole_target, residue_target = _target_modes(cfg)
    target_names = getattr(cfg, "model_target_names", None)
    transforms = build_target_transform_specs(
        target_names or [],
        (row.exc_id for row in parsed),
        enabled=bool(target_names) and bool(getattr(cfg, "fdr_physical_parameterization", False)),
    )
    return compute_output_normalizer_masked(
        parsed,
        pole_target=pole_target,
        residue_target=residue_target,
        target_names=target_names,
        target_transforms=transforms,
    )


def _slot_values(rows: List[ParsedRow], slot: int):
    values = []
    for row in rows:
        mask = np.asarray(row.mask, dtype=np.float64)
        damping = np.asarray(row.damping, dtype=np.float64)
        if slot >= len(mask) or slot >= len(damping) or mask[slot] <= 0.5:
            continue
        value = float(damping[slot])
        if np.isfinite(value):
            values.append((float(row.H), float(row.M), value))
    return values


def _observed_mach_slopes(values):
    groups: dict[float, list[tuple[float, float]]] = {}
    for height, mach, damping in values:
        groups.setdefault(round(float(height), 8), []).append((float(mach), float(damping)))
    slopes = []
    for points in groups.values():
        points.sort()
        for (m0, d0), (m1, d1) in zip(points, points[1:]):
            delta = m1 - m0
            if delta > 1e-8:
                slopes.append(abs((d1 - d0) / delta))
    return slopes


def _build_slot_support(real_train: List[ParsedRow], real_val: List[ParsedRow], cfg: Config):
    kmax = int(cfg.K_max)
    minimum_count = int(getattr(cfg, "min_slot_train_samples", 8))
    multiplier = float(getattr(cfg, "damping_slope_limit_multiplier", 1.5))
    floor = float(getattr(cfg, "damping_slope_limit_floor", 0.03))
    train_values = [_slot_values(real_train, slot) for slot in range(kmax)]
    validation_values = [_slot_values(real_val, slot) for slot in range(kmax)]
    pooled_slopes = [slope for values in train_values for slope in _observed_mach_slopes(values)]
    pooled_limit = max(
        floor,
        (float(np.quantile(pooled_slopes, 0.90)) * multiplier) if pooled_slopes else floor,
    )
    slots = []
    for slot, values in enumerate(train_values):
        slopes = _observed_mach_slopes(values)
        slope_limit = max(
            floor,
            (float(np.quantile(slopes, 0.90)) * multiplier) if slopes else pooled_limit,
        )
        slot_data = {
            "slot_index": slot,
            "slot_number": slot + 1,
            "train_count": len(values),
            "validation_count": len(validation_values[slot]),
            "low_confidence": len(values) < minimum_count,
            "slope_limit": slope_limit,
        }
        if values:
            slot_data.update({
                "height_min": min(item[0] for item in values),
                "height_max": max(item[0] for item in values),
                "mach_min": min(item[1] for item in values),
                "mach_max": max(item[1] for item in values),
                "damping_min": min(item[2] for item in values),
                "damping_max": max(item[2] for item in values),
            })
        slots.append(slot_data)
    return {
        "schema_version": 1,
        "minimum_train_samples": minimum_count,
        "slots": slots,
    }


def _attach_and_write_slot_support(
    out_norm: Normalizer, real_train: List[ParsedRow], real_val: List[ParsedRow], cfg: Config, out_dir: Path
):
    support = _build_slot_support(real_train, real_val, cfg)
    out_norm.support = support
    cfg._slot_support_metadata = support
    cfg._damping_slope_limits = [float(item["slope_limit"]) for item in support["slots"]]
    if is_main_process():
        atomic_write_json(Path(out_dir) / "slot_support.json", support)
        summary = ", ".join(
            f"slot{item['slot_number']}={item['train_count']}"
            + ("(low-confidence)" if item["low_confidence"] else "")
            for item in support["slots"]
        )
        print(f"[SLOT-SUPPORT] {summary}", flush=True)


def _filter_damping_slots(parsed: List[ParsedRow], split_name: str):
    """Mask only non-finite or non-positive damping slots."""
    filtered = []
    dropped = []
    for source in parsed:
        row = deepcopy(source)
        mask = np.asarray(row.mask, dtype=np.float32).copy()
        damping = np.asarray(row.damping, dtype=np.float64)
        width = min(len(mask), len(damping))
        for slot in range(width):
            invalid = (
                not np.isfinite(damping[slot])
                or float(damping[slot]) <= 0.0
            )
            if mask[slot] > 0.5 and invalid:
                mask[slot] = 0.0
                dropped.append({
                    "split": split_name,
                    "H": float(row.H),
                    "M": float(row.M),
                    "exc_id": int(row.exc_id),
                    "ch_id": int(row.ch_id),
                    "slot_number": slot + 1,
                    "slot_index": slot,
                    "damping": float(damping[slot]),
                    "reason": (
                        "non_finite" if not np.isfinite(damping[slot])
                        else "non_positive"
                    ),
                })
        row.mask = mask
        row.K_valid = int(np.count_nonzero(mask > 0.5))
        filtered.append(row)
    return filtered, dropped


def _write_damping_filter_log(out_dir: Path, dropped: list[dict]):
    if not is_main_process():
        return
    path = Path(out_dir) / "damping_slot_filter.log"
    train_count = sum(item["split"] == "train" for item in dropped)
    val_count = sum(item["split"] == "validation" for item in dropped)
    lines = [
        "DAMPING SLOT FILTER",
        "Stable-mode filter: discard non-finite or damping <= 0",
        "No excitation-specific upper damping limit is applied.",
        f"dropped_total={len(dropped)} dropped_train={train_count} dropped_validation={val_count}",
        "",
    ]
    for index, item in enumerate(dropped, 1):
        lines.append(
            f"[{index:05d}] split={item['split']} H={item['H']:.9g} M={item['M']:.9g} "
            f"exc_id={item['exc_id']} ch_id={item['ch_id']} "
            f"slot_number={item['slot_number']} slot_index={item['slot_index']} "
            f"damping={item['damping']:.9g} reason={item['reason']}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"[DAMPING-FILTER] dropped slots: total={len(dropped)}, "
        f"train={train_count}, validation={val_count}; log={path}",
        flush=True,
    )


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


def _split_real_by_height_mach_points(real_parsed: List[ParsedRow], cfg: Config):
    raw_points = list(getattr(cfg, "val_height_mach_points", []))
    if not raw_points:
        raise RuntimeError(
            'val_split_strategy="height_mach_points" requires non-empty '
            "Config.val_height_mach_points, for example [(3.0, 0.7), (8.23, 0.85)]."
        )

    points = []
    for point in raw_points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"Each validation operating point must be (H, M), got {point!r}")
        points.append((float(point[0]), float(point[1])))

    h_tol = float(getattr(cfg, "val_height_tol", 1e-6))
    m_tol = float(getattr(cfg, "val_mach_tol", 1e-6))
    val_indices = set()
    split_info = []
    available_points = sorted({(float(p.H), float(p.M)) for p in real_parsed})

    for target_h, target_m in points:
        hm_indices = [
            i
            for i, p in enumerate(real_parsed)
            if abs(float(p.H) - target_h) <= h_tol and abs(float(p.M) - target_m) <= m_tol
        ]
        if not hm_indices:
            raise RuntimeError(
                f"No real samples found for validation point H={target_h:g}, M={target_m:g}. "
                f"Available (H, M) points: {available_points}"
            )
        val_indices.update(hm_indices)
        split_info.append((target_h, target_m, len(hm_indices)))

    real_train = [p for i, p in enumerate(real_parsed) if i not in val_indices]
    real_val = [p for i, p in enumerate(real_parsed) if i in val_indices]
    if not real_train:
        raise RuntimeError("Validation operating points consumed all real samples; no real training data remains.")
    if not real_val:
        raise RuntimeError("Specified validation operating points produced no real validation samples.")
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


def split_single_fidelity_rows(
    real_parsed: List[ParsedRow], cfg: Config, kfold: int = 0, fold_idx: int = 0
):
    """Resolve the exact train/validation membership used by fixed-SF training."""
    explicit_train = getattr(cfg, "explicit_train_group_keys", None)
    explicit_validation = getattr(cfg, "explicit_validation_group_keys", None)
    if explicit_train is not None or explicit_validation is not None:
        train_keys = {tuple(map(int, key)) for key in (explicit_train or [])}
        validation_keys = {tuple(map(int, key)) for key in (explicit_validation or [])}
        overlap = train_keys & validation_keys
        if overlap:
            raise ValueError(f"Explicit train/validation group keys overlap: {sorted(overlap)}")
        available = {group_key_from_parsed(row) for row in real_parsed}
        unknown = (train_keys | validation_keys) - available
        if unknown:
            raise ValueError(f"Explicit split references missing groups: {sorted(unknown)}")
        unassigned = available - train_keys - validation_keys
        train_keys.update(unassigned)
        real_train = [row for row in real_parsed if group_key_from_parsed(row) in train_keys]
        real_val = [row for row in real_parsed if group_key_from_parsed(row) in validation_keys]
        if not real_train:
            raise RuntimeError("Explicit split produced an empty training set")
        if bool(getattr(cfg, "require_validation", False)) and not real_val:
            raise RuntimeError(
                "Correction groups removed every validation group; refusing to switch checkpoint selection silently."
            )
        return real_train, real_val, "inherited_manifest"

    use_val_split = bool(getattr(cfg, "use_val_split", False))
    val_ratio = float(getattr(cfg, "val_ratio", 0.2))
    seed = int(getattr(cfg, "val_seed", 42))
    val_strategy = str(getattr(cfg, "val_split_strategy", "")).lower()
    if not use_val_split:
        return real_parsed, [], "disabled"
    if val_strategy == "height_mach_kfold":
        train, val, _ = _split_by_height_mach_groups(real_parsed, cfg, kfold, fold_idx, "real")
        return train, val, val_strategy
    if val_strategy == "height_max_mach":
        train, val, _ = _split_real_by_height_max_mach(real_parsed, cfg)
        return train, val, val_strategy
    if val_strategy == "height_mach_points":
        train, val, _ = _split_real_by_height_mach_points(real_parsed, cfg)
        return train, val, val_strategy
    if kfold > 1 and len(real_parsed) > 1:
        kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)
        indices = list(range(len(real_parsed)))
        train_idx, val_idx = list(kf.split(indices))[fold_idx]
        return [real_parsed[i] for i in train_idx], [real_parsed[i] for i in val_idx], "row_kfold"
    if 0.0 < val_ratio < 1.0 and len(real_parsed) > 1:
        rng = np.random.default_rng(seed + 1000)
        indices = np.arange(len(real_parsed))
        rng.shuffle(indices)
        n_val = max(1, int(round(len(real_parsed) * val_ratio)))
        return (
            [real_parsed[i] for i in indices[n_val:]],
            [real_parsed[i] for i in indices[:n_val]],
            "random_rows",
        )
    return real_parsed, [], "disabled"


def build_dataloaders(sim_parsed: List[ParsedRow], real_parsed: List[ParsedRow], cfg: Config,
                      kfold: int, fold_idx: int, device: torch.device, out_dir: Path):
    use_val_split = bool(getattr(cfg, "use_val_split", False))
    val_ratio = float(getattr(cfg, "val_ratio", 0.2))
    seed = int(getattr(cfg, "val_seed", 42))
    val_strategy = str(getattr(cfg, "val_split_strategy", "")).lower()
    use_height_max_mach_val = use_val_split and val_strategy == "height_max_mach"
    use_height_mach_points_val = use_val_split and val_strategy == "height_mach_points"
    use_height_mach_kfold = use_val_split and val_strategy == "height_mach_kfold"

    if use_height_mach_kfold:
        sim_train, sim_val, sim_info = _split_by_height_mach_groups(sim_parsed, cfg, kfold, fold_idx, "sim")
        if is_main_process():
            print(f"[VAL] sim height_mach_kfold: val_groups={len(sim_info)}, train_rows={len(sim_train)}, val_rows={len(sim_val)}", flush=True)
    elif (not use_height_max_mach_val) and (not use_height_mach_points_val) and use_val_split and (0.0 < val_ratio < 1.0) and (len(sim_parsed) > 1):
        rng = np.random.default_rng(seed)
        idx = np.arange(len(sim_parsed))
        rng.shuffle(idx)
        n_val = max(1, int(round(len(sim_parsed) * val_ratio)))
        sim_train = [sim_parsed[i] for i in idx[n_val:]]
        sim_val = [sim_parsed[i] for i in idx[:n_val]]
    else:
        sim_train = sim_parsed
        sim_val = []

    real_train, real_val, real_split_strategy = split_single_fidelity_rows(
        real_parsed, cfg, kfold, fold_idx
    )
    cfg._last_split_metadata = {
        "split_strategy": real_split_strategy,
        "train_group_keys": sorted({group_key_from_parsed(row) for row in real_train}),
        "validation_group_keys": sorted({group_key_from_parsed(row) for row in real_val}),
    }
    if is_main_process():
        print(
            f"[VAL] real strategy={real_split_strategy} "
            f"train_rows={len(real_train)} val_rows={len(real_val)}",
            flush=True,
        )

    sim_train, dropped_sim_train = _filter_damping_slots(sim_train, "sim_train")
    sim_val, dropped_sim_val = _filter_damping_slots(sim_val, "sim_validation")
    real_train, dropped_train = _filter_damping_slots(real_train, "train")
    real_val, dropped_val = _filter_damping_slots(real_val, "validation")
    _write_damping_filter_log(
        out_dir,
        dropped_sim_train + dropped_sim_val + dropped_train + dropped_val,
    )

    if not sim_train:
        raise RuntimeError("No valid simulation training rows remain after filtering")
    if not real_train:
        raise RuntimeError("No valid real training rows remain after filtering")

    # Both fidelities use the simulation input coordinates.  Output statistics
    # remain separate because simulation and real targets can have different
    # offsets and scales.
    in_norm_sim = compute_input_normalizer_sim(sim_train)
    out_norm_sim = _output_normalizer(sim_train, cfg)
    if is_main_process():
        in_norm_sim.save(out_dir / "Normalizer_Input_Sim.pth")
        out_norm_sim.save(out_dir / "Normalizer_Output_Sim.pth")

    out_norm_real = _output_normalizer(real_train, cfg)
    _attach_and_write_slot_support(out_norm_real, real_train, real_val, cfg, out_dir)
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


def build_low_fidelity_dataloader(
    sim_parsed: List[ParsedRow], cfg: Config, out_dir: Path
):
    """Build a simulation-only component loader when no valid real rows exist."""
    sim_train, dropped = _filter_damping_slots(sim_parsed, "sim_train")
    _write_damping_filter_log(out_dir, dropped)
    if not sim_train:
        raise RuntimeError("Low-fidelity training requires valid simulation rows")
    in_norm_sim = compute_input_normalizer_sim(sim_train)
    out_norm_sim = _output_normalizer(sim_train, cfg)
    _attach_and_write_slot_support(out_norm_sim, sim_train, [], cfg, out_dir)
    if is_main_process():
        in_norm_sim.save(out_dir / "Normalizer_Input_Sim.pth")
        out_norm_sim.save(out_dir / "Normalizer_Output_Sim.pth")
    dataset = ModalDataset(sim_train, cfg.K_max, in_norm_sim, out_norm_sim)
    sampler = (
        DistributedSampler(dataset, shuffle=True, drop_last=True)
        if _ddp_ready()
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=int(getattr(cfg, "batch_sim", cfg.batch_real)),
        shuffle=(sampler is None),
        sampler=sampler,
        **_loader_kwargs(cfg),
    )
    cfg._last_split_metadata = {
        "split_strategy": "sim_all_rows_training",
        "train_group_keys": sorted({group_key_from_parsed(row) for row in sim_train}),
        "validation_group_keys": [],
    }
    return loader, len(dataset), out_norm_sim, in_norm_sim


def build_single_fidelity_dataloaders(
    real_parsed: List[ParsedRow],
    cfg: Config,
    kfold: int,
    fold_idx: int,
    device: torch.device,
    out_dir: Path,
):
    real_train, real_val, split_strategy = split_single_fidelity_rows(
        real_parsed, cfg, kfold, fold_idx
    )
    cfg._last_split_metadata = {
        "split_strategy": split_strategy,
        "train_group_keys": sorted({group_key_from_parsed(row) for row in real_train}),
        "validation_group_keys": sorted({group_key_from_parsed(row) for row in real_val}),
    }
    if is_main_process():
        print(
            f"[VAL] strategy={split_strategy} train_rows={len(real_train)} val_rows={len(real_val)}",
            flush=True,
        )

    real_train, dropped_train = _filter_damping_slots(real_train, "train")
    real_val, dropped_val = _filter_damping_slots(real_val, "validation")
    _write_damping_filter_log(out_dir, dropped_train + dropped_val)

    in_norm_real = compute_input_normalizer_sim(real_train)
    out_norm_real = _output_normalizer(real_train, cfg)
    _attach_and_write_slot_support(out_norm_real, real_train, real_val, cfg, out_dir)

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
