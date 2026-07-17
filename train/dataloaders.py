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
        validation_slot_values = validation_values[slot]
        support_values = values + validation_slot_values
        slopes = _observed_mach_slopes(values)
        slope_limit = max(
            floor,
            (float(np.quantile(slopes, 0.90)) * multiplier) if slopes else pooled_limit,
        )
        slot_data = {
            "slot_index": slot,
            "slot_number": slot + 1,
            "train_count": len(values),
            "validation_count": len(validation_slot_values),
            "support_count": len(support_values),
            "low_confidence": len(values) < minimum_count,
            "slope_limit": slope_limit,
        }
        if support_values:
            slot_data.update({
                "height_min": min(item[0] for item in support_values),
                "height_max": max(item[0] for item in support_values),
                "mach_min": min(item[1] for item in support_values),
                "mach_max": max(item[1] for item in support_values),
            })
        if values:
            slot_data.update({
                "damping_min": min(item[2] for item in values),
                "damping_max": max(item[2] for item in values),
            })
        slots.append(slot_data)
    return {
        "schema_version": 2,
        "minimum_train_samples": minimum_count,
        "coordinate_support_scope": "training_and_validation_rows",
        "slope_fit_scope": "training_rows_only",
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


def _real_damping_upper_rules(cfg: Config | None):
    if cfg is None or not bool(
        getattr(cfg, "real_damping_upper_filter_enabled", True)
    ):
        return {}
    thresholds = {
        int(excitation_id): float(threshold)
        for excitation_id, threshold in dict(
            getattr(cfg, "real_damping_upper_thresholds", {}) or {}
        ).items()
    }
    channel_ids = {
        int(excitation_id): {int(channel_id) for channel_id in channels}
        for excitation_id, channels in dict(
            getattr(cfg, "real_damping_upper_channel_ids", {}) or {}
        ).items()
    }
    rules = {}
    for excitation_id, threshold in thresholds.items():
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError(
                "real_damping_upper_thresholds values must be finite and positive"
            )
        channels = channel_ids.get(excitation_id)
        if not channels:
            raise ValueError(
                "Missing real damping channel IDs for excitation_id="
                f"{excitation_id}"
            )
        rules[excitation_id] = {
            "threshold": threshold,
            "channel_ids": channels,
        }
    return rules


def _filter_damping_slots(
    parsed: List[ParsedRow],
    split_name: str,
    cfg: Config | None = None,
    apply_real_upper_limits: bool = False,
):
    """Mask invalid damping slots and optionally apply real-data upper limits."""
    upper_rules = _real_damping_upper_rules(cfg) if apply_real_upper_limits else {}
    filtered = []
    dropped = []
    for source in parsed:
        row = deepcopy(source)
        mask = np.asarray(row.mask, dtype=np.float32).copy()
        damping = np.asarray(row.damping, dtype=np.float64)
        width = min(len(mask), len(damping))
        for slot in range(width):
            damping_value = float(damping[slot])
            rule = upper_rules.get(int(row.exc_id))
            upper_threshold = (
                float(rule["threshold"])
                if rule is not None and int(row.ch_id) in rule["channel_ids"]
                else None
            )
            tolerance = (
                max(1e-8, abs(upper_threshold) * 1e-6)
                if upper_threshold is not None
                else 0.0
            )
            above_upper_limit = (
                upper_threshold is not None
                and np.isfinite(damping_value)
                and damping_value > upper_threshold + tolerance
            )
            invalid = (
                not np.isfinite(damping_value)
                or damping_value <= 0.0
                or above_upper_limit
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
                    "damping": damping_value,
                    "upper_threshold": upper_threshold,
                    "reason": (
                        "non_finite" if not np.isfinite(damping_value)
                        else "non_positive" if damping_value <= 0.0
                        else "above_real_upper_limit"
                    ),
                })
        row.mask = mask
        row.K_valid = int(np.count_nonzero(mask > 0.5))
        filtered.append(row)
    return filtered, dropped


def _robust_metric_stats(values, threshold: float, scale_floor_ratio: float):
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    robust_scale = 1.4826 * mad
    scale_floor = max(abs(median) * scale_floor_ratio, 1e-12)
    scale = max(robust_scale, scale_floor)
    return {
        "median": median,
        "mad": mad,
        "robust_scale": robust_scale,
        "scale_used": scale,
        "lower": median - threshold * scale,
        "upper": median + threshold * scale,
    }


def _filter_sim_outlier_slots(
    sim_train: List[ParsedRow], sim_val: List[ParsedRow], cfg: Config
):
    """Mask a whole sim slot when its frequency or damping is a robust outlier.

    Robust ranges are fitted only on the simulation training split, separately
    for every excitation/channel/slot.  The resulting ranges are then applied
    to both sim train and sim validation rows, avoiding validation leakage.
    """
    enabled = bool(getattr(cfg, "sim_slot_outlier_filter_enabled", True))
    min_samples = int(getattr(cfg, "sim_slot_outlier_min_samples", 8))
    threshold = float(getattr(cfg, "sim_slot_outlier_mad_threshold", 12.0))
    scale_floor_ratio = float(
        getattr(cfg, "sim_slot_outlier_scale_floor_ratio", 0.02)
    )
    if min_samples <= 0:
        raise ValueError("sim_slot_outlier_min_samples must be positive")
    if threshold <= 0.0:
        raise ValueError("sim_slot_outlier_mad_threshold must be positive")
    if scale_floor_ratio <= 0.0:
        raise ValueError("sim_slot_outlier_scale_floor_ratio must be positive")

    grouped: dict[tuple[int, int, int], list[tuple[float, float]]] = {}
    for row in sim_train:
        mask = np.asarray(row.mask, dtype=np.float64)
        frequency = np.asarray(row.freq, dtype=np.float64)
        damping = np.asarray(row.damping, dtype=np.float64)
        width = min(len(mask), len(frequency), len(damping))
        for slot in range(width):
            if mask[slot] <= 0.5:
                continue
            freq_value = float(frequency[slot])
            damping_value = float(damping[slot])
            if (
                np.isfinite(freq_value)
                and freq_value > 0.0
                and np.isfinite(damping_value)
                and damping_value > 0.0
            ):
                key = (int(row.exc_id), int(row.ch_id), slot)
                grouped.setdefault(key, []).append((freq_value, damping_value))

    fitted = {}
    group_reports = []
    for key, pairs in sorted(grouped.items()):
        count = len(pairs)
        group_enabled = enabled and count >= min_samples
        report = {
            "excitation_id": key[0],
            "channel_id": key[1],
            "slot_index": key[2],
            "slot_number": key[2] + 1,
            "fit_sample_count": count,
            "filter_active": group_enabled,
        }
        if group_enabled:
            frequencies = [pair[0] for pair in pairs]
            damping_values = [pair[1] for pair in pairs]
            report["frequency"] = _robust_metric_stats(
                frequencies, threshold, scale_floor_ratio
            )
            report["damping"] = _robust_metric_stats(
                damping_values, threshold, scale_floor_ratio
            )
            fitted[key] = report
        group_reports.append(report)

    dropped = []
    empty_rows_after_filter = []

    def apply(rows, split_name: str):
        filtered = []
        for source in rows:
            row = deepcopy(source)
            mask = np.asarray(row.mask, dtype=np.float32).copy()
            frequency = np.asarray(row.freq, dtype=np.float64)
            damping = np.asarray(row.damping, dtype=np.float64)
            width = min(len(mask), len(frequency), len(damping))
            for slot in range(width):
                if mask[slot] <= 0.5:
                    continue
                stats = fitted.get((int(row.exc_id), int(row.ch_id), slot))
                if stats is None:
                    continue
                freq_value = float(frequency[slot])
                damping_value = float(damping[slot])
                freq_stats = stats["frequency"]
                damping_stats = stats["damping"]
                frequency_outlier = (
                    not np.isfinite(freq_value)
                    or freq_value <= 0.0
                    or freq_value < freq_stats["lower"]
                    or freq_value > freq_stats["upper"]
                )
                damping_outlier = (
                    not np.isfinite(damping_value)
                    or damping_value <= 0.0
                    or damping_value < damping_stats["lower"]
                    or damping_value > damping_stats["upper"]
                )
                if not frequency_outlier and not damping_outlier:
                    continue
                mask[slot] = 0.0
                dropped.append(
                    {
                        "split": split_name,
                        "H": float(row.H),
                        "M": float(row.M),
                        "excitation_id": int(row.exc_id),
                        "channel_id": int(row.ch_id),
                        "slot_index": slot,
                        "slot_number": slot + 1,
                        "frequency": freq_value,
                        "damping": damping_value,
                        "frequency_robust_score": (
                            None
                            if not np.isfinite(freq_value)
                            else abs(freq_value - freq_stats["median"])
                            / freq_stats["scale_used"]
                        ),
                        "damping_robust_score": (
                            None
                            if not np.isfinite(damping_value)
                            else abs(damping_value - damping_stats["median"])
                            / damping_stats["scale_used"]
                        ),
                        "reasons": [
                            reason
                            for reason, active in (
                                ("frequency_outlier", frequency_outlier),
                                ("damping_outlier", damping_outlier),
                            )
                            if active
                        ],
                    }
                )
            row.mask = mask
            row.K_valid = int(np.count_nonzero(mask > 0.5))
            filtered.append(row)
            if row.K_valid == 0:
                empty_rows_after_filter.append(
                    {
                        "split": split_name,
                        "H": float(row.H),
                        "M": float(row.M),
                        "excitation_id": int(row.exc_id),
                        "channel_id": int(row.ch_id),
                    }
                )
        return filtered

    filtered_train = apply(sim_train, "sim_train")
    filtered_val = apply(sim_val, "sim_validation")
    report = {
        "schema_version": 1,
        "filter": "joint_frequency_damping_robust_mad",
        "enabled": enabled,
        "fit_scope": "simulation_training_split_per_excitation_channel_slot",
        "min_samples": min_samples,
        "mad_threshold": threshold,
        "scale_floor_ratio": scale_floor_ratio,
        "input_train_rows": len(sim_train),
        "output_train_rows": len(filtered_train),
        "input_validation_rows": len(sim_val),
        "output_validation_rows": len(filtered_val),
        "masked_slot_count": len(dropped),
        "empty_row_after_filter_count": len(empty_rows_after_filter),
        "groups": group_reports,
        "masked_slots": dropped,
        "empty_rows_after_filter": empty_rows_after_filter,
    }
    return filtered_train, filtered_val, report


def _write_sim_outlier_filter_report(out_dir: Path, report: dict):
    if not is_main_process():
        return
    path = Path(out_dir) / "sim_slot_outlier_filter.json"
    atomic_write_json(path, report)
    print(
        f"[SIM-OUTLIER-FILTER] masked_slots={report['masked_slot_count']} "
        f"empty_rows_after_filter={report['empty_row_after_filter_count']} "
        f"train_rows={report['output_train_rows']}/{report['input_train_rows']} "
        f"validation_rows={report['output_validation_rows']}/"
        f"{report['input_validation_rows']} report={path}",
        flush=True,
    )


def _write_damping_filter_log(out_dir: Path, dropped: list[dict], cfg: Config):
    if not is_main_process():
        return
    path = Path(out_dir) / "damping_slot_filter.log"
    train_count = sum(item["split"] == "train" for item in dropped)
    val_count = sum(item["split"] == "validation" for item in dropped)
    upper_count = sum(
        item["reason"] == "above_real_upper_limit" for item in dropped
    )
    rules = _real_damping_upper_rules(cfg)
    rule_lines = [
        f"  excitation_id={excitation_id} "
        f"channel_ids={','.join(str(value) for value in sorted(rule['channel_ids']))} "
        f"damping_upper_threshold={rule['threshold']:.9g}"
        for excitation_id, rule in sorted(rules.items())
    ]
    lines = [
        "DAMPING SLOT FILTER",
        "Stable-mode filter: discard non-finite or damping <= 0",
        "Real-data upper filter: discard a whole slot when damping is greater than its excitation/channel threshold.",
        f"real_upper_filter_enabled={bool(rules)}",
        *rule_lines,
        f"dropped_total={len(dropped)} dropped_train={train_count} "
        f"dropped_validation={val_count} dropped_real_above_upper={upper_count}",
        "",
    ]
    for index, item in enumerate(dropped, 1):
        lines.append(
            f"[{index:05d}] split={item['split']} H={item['H']:.9g} M={item['M']:.9g} "
            f"exc_id={item['exc_id']} ch_id={item['ch_id']} "
            f"slot_number={item['slot_number']} slot_index={item['slot_index']} "
            f"damping={item['damping']:.9g} "
            f"upper_threshold={item.get('upper_threshold')} reason={item['reason']}"
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

    sim_train, sim_val, sim_outlier_report = _filter_sim_outlier_slots(
        sim_train, sim_val, cfg
    )
    _write_sim_outlier_filter_report(out_dir, sim_outlier_report)

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
    real_train, dropped_train = _filter_damping_slots(
        real_train, "train", cfg, apply_real_upper_limits=True
    )
    real_val, dropped_val = _filter_damping_slots(
        real_val, "validation", cfg, apply_real_upper_limits=True
    )
    _write_damping_filter_log(
        out_dir,
        dropped_sim_train + dropped_sim_val + dropped_train + dropped_val,
        cfg,
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
    sim_train, _sim_val, sim_outlier_report = _filter_sim_outlier_slots(
        sim_parsed, [], cfg
    )
    _write_sim_outlier_filter_report(out_dir, sim_outlier_report)
    sim_train, dropped = _filter_damping_slots(sim_train, "sim_train")
    _write_damping_filter_log(out_dir, dropped, cfg)
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

    real_train, dropped_train = _filter_damping_slots(
        real_train, "train", cfg, apply_real_upper_limits=True
    )
    real_val, dropped_val = _filter_damping_slots(
        real_val, "validation", cfg, apply_real_upper_limits=True
    )
    _write_damping_filter_log(out_dir, dropped_train + dropped_val, cfg)

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
