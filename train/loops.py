from __future__ import annotations

import csv
import math
import os
import time
from copy import deepcopy
from itertools import cycle
from pathlib import Path

import torch
import torch.distributed as dist

from surrogate_model.config import Config
from surrogate_model.data.io import OMEGA_N
from surrogate_model.models.blocks import (
    MFDNN,
    TargetMFDNN,
    TargetMultifidelityMFDNN,
    count_parameter_groups,
)
from surrogate_model.train.dataloaders import (
    build_dataloaders,
    build_low_fidelity_dataloader,
    build_single_fidelity_dataloaders,
)
from surrogate_model.utils.dist_utils import is_main_process


BASE_OUTPUT_NAMES = ("alpha", "omega", "rho", "sin_theta", "cos_theta")
EXCITATION_LOSS_SCOPES = (
    ("all", None),
    ("exc_FXD_FDC", (0,)),
    ("exc_FXD_DC", (1,)),
    ("exc_PW_FDC", (2,)),
    ("exc_PW_DC", (3,)),
    ("exc_JFY_FDC", (4,)),
    ("exc_JFY_DC", (5,)),
    ("group_FXD", (0, 1)),
    ("group_PW", (2, 3)),
    ("group_JFY", (4, 5)),
)


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _elapsed_and_eta(started_at: float, completed: int, total: int) -> tuple[str, str]:
    elapsed = time.perf_counter() - started_at
    eta = None
    if completed > 0:
        eta = elapsed / completed * max(total - completed, 0)
    return _format_duration(elapsed), _format_duration(eta)


def _select_device(args) -> torch.device:
    req = str(getattr(args, "device", "auto")).lower()
    if req == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(rank)
        return torch.device("cuda", rank)
    return torch.device("cpu")


def _to_device(batch, device):
    H, M, e, c, mask, y = batch
    H = H.to(device)
    M = M.to(device)
    e = e.to(device)
    c = c.to(device)
    mask = mask.to(device)
    for k in y:
        y[k] = y[k].to(device)
    return H, M, e, c, mask, y


def _part_dict_zero(names):
    return {k: 0.0 for k in names}


def _mean_parts(sums, count, names):
    den = max(int(count), 1)
    return {k: float(sums[k]) / den for k in names}


def _model_output_as_targets(pred, target_names):
    """Convert model's native alpha/omega/rho/sin/cos output to selected target variables."""
    alpha = pred["alpha"]
    omega_norm = pred["omega"]
    omega_phys = omega_norm * float(OMEGA_N)
    rho_log = pred["rho"]
    sin_theta = pred["sin_theta"]
    cos_theta = pred["cos_theta"]

    denom = torch.sqrt(alpha.pow(2) + omega_phys.pow(2) + 1e-12)
    damping = -alpha / denom
    freq = omega_phys / (2.0 * torch.pi)
    rho_mag = torch.exp(torch.clamp(rho_log, min=-80.0, max=80.0))

    converted = {
        "alpha": alpha,
        "omega": omega_norm,
        "freq": freq,
        "damping": damping,
        "rho": rho_log,
        "sin_theta": sin_theta,
        "cos_theta": cos_theta,
        "residue_re": rho_mag * cos_theta,
        "residue_im": rho_mag * sin_theta,
    }
    return {name: converted[name] for name in target_names}


def _representation_loss(
    pred, tgt, mask, mean, std, target_names, *, outputs_normalized: bool = False
):
    reference = pred[target_names[0]] if outputs_normalized else pred["alpha"]
    width = min(mask.size(1), reference.size(1), min(tgt[name].size(1) for name in target_names))
    w = mask[:, :width].to(dtype=reference.dtype)
    den = w.sum().clamp_min(1.0)
    pred_raw = pred if outputs_normalized else _model_output_as_targets(pred, target_names)
    total = reference.new_tensor(0.0)
    parts = {}
    for i, name in enumerate(target_names):
        pred_cmp = pred_raw[name][:, :width]
        if not outputs_normalized:
            pred_cmp = (pred_cmp - mean[i]) / std[i].clamp_min(1e-8)
        val = (w * (pred_cmp - tgt[name][:, :width]).pow(2)).sum() / den
        parts[name] = val
        total = total + val
    return total, parts


def _representation_slot_sums(
    pred, tgt, mask, mean, std, target_names, *, outputs_normalized: bool = False
):
    reference = pred[target_names[0]] if outputs_normalized else pred["alpha"]
    width = min(mask.size(1), reference.size(1), min(tgt[name].size(1) for name in target_names))
    pred_raw = pred if outputs_normalized else _model_output_as_targets(pred, target_names)
    errors = reference.new_zeros(width)
    counts = reference.new_zeros(width)
    for slot in range(width):
        slot_mask = mask[:, slot].to(dtype=reference.dtype)
        counts[slot] = slot_mask.sum()
        for index, name in enumerate(target_names):
            pred_cmp = pred_raw[name][:, slot]
            if not outputs_normalized:
                pred_cmp = (pred_cmp - mean[index]) / std[index].clamp_min(1e-8)
            errors[slot] += (slot_mask * (pred_cmp - tgt[name][:, slot]).pow(2)).sum()
    return errors, counts


def _damping_smoothness_losses(pred, M, mask, out_norm, in_norm, cfg):
    zero = M.new_tensor(0.0)
    if "damping" not in pred or not bool(getattr(cfg, "fdr_physical_parameterization", False)):
        return zero, zero
    slope_weight = float(getattr(cfg, "damping_slope_weight", 0.0))
    curvature_weight = float(getattr(cfg, "damping_curvature_weight", 0.0))
    if slope_weight <= 0.0 and curvature_weight <= 0.0:
        return zero, zero

    damping = out_norm.decode_normalized_target_torch("damping", pred["damping"])
    width = min(int(damping.size(1)), int(mask.size(1)))
    limits = list(getattr(cfg, "_damping_slope_limits", []))
    fallback_limit = float(getattr(cfg, "damping_slope_limit_floor", 0.03))
    mach_std = max(float(in_norm.std[1]), 1e-8)
    slope_terms = []
    curvature_terms = []
    weights = []
    for slot in range(width):
        grad_normalized = torch.autograd.grad(
            damping[:, slot].sum(), M, create_graph=True, retain_graph=True
        )[0][:, 0]
        grad_physical = grad_normalized / mach_std
        limit = float(limits[slot]) if slot < len(limits) else fallback_limit
        slope_terms.append(torch.relu(grad_physical.abs() - limit).pow(2))
        if curvature_weight > 0.0:
            second_normalized = torch.autograd.grad(
                grad_normalized.sum(), M, create_graph=True, retain_graph=True
            )[0][:, 0]
            curvature_terms.append((second_normalized / (mach_std * mach_std)).pow(2))
        weights.append(mask[:, slot].to(dtype=damping.dtype))

    weight = torch.stack(weights, dim=1)
    denominator = weight.sum().clamp_min(1.0)
    slope_loss = (weight * torch.stack(slope_terms, dim=1)).sum() / denominator
    if curvature_terms:
        curvature_loss = (weight * torch.stack(curvature_terms, dim=1)).sum() / denominator
    else:
        curvature_loss = zero
    return slope_loss, curvature_loss


def _damping_collocation_batch(model, H, M, e, c, mask, in_norm, cfg):
    support_slots = list(dict(getattr(cfg, "_slot_support_metadata", {}) or {}).get("slots", []))
    mach_minima = [float(item["mach_min"]) for item in support_slots if "mach_min" in item]
    mach_maxima = [float(item["mach_max"]) for item in support_slots if "mach_max" in item]
    if not mach_minima or not mach_maxima:
        collocation_m = M.detach().requires_grad_(True)
        return model(H, collocation_m, e, c), collocation_m, mask

    physical_min = min(mach_minima)
    physical_max = max(mach_maxima)
    random_physical = physical_min + (physical_max - physical_min) * torch.rand_like(M)
    mach_mean = float(in_norm.mean[1])
    mach_std = max(float(in_norm.std[1]), 1e-8)
    collocation_m = ((random_physical - mach_mean) / mach_std).detach().requires_grad_(True)
    active_slots = torch.as_tensor(
        [float(int(item.get("train_count", 0)) > 0) for item in support_slots],
        dtype=mask.dtype,
        device=mask.device,
    )
    collocation_mask = active_slots.reshape(1, -1).expand(mask.size(0), -1)
    return model(H, collocation_m, e, c), collocation_m, collocation_mask


def _filter_by_excitation(H, M, e, c, mask, y, excitation_ids):
    if excitation_ids is None:
        return H, M, e, c, mask, y
    keep = torch.zeros_like(e, dtype=torch.bool)
    for exc_id in excitation_ids:
        keep |= (e == int(exc_id))
    if not bool(keep.any().item()):
        return None
    y_sel = {k: v[keep] for k, v in y.items()}
    return H[keep], M[keep], e[keep], c[keep], mask[keep], y_sel


def _empty_scope_accum(target_names):
    return {
        scope: {"sum": 0.0, "count": 0, "parts": _part_dict_zero(target_names)}
        for scope, _ids in EXCITATION_LOSS_SCOPES
    }


def _scope_keep(e: torch.Tensor, excitation_ids):
    if excitation_ids is None:
        return torch.ones_like(e, dtype=torch.bool)
    keep = torch.zeros_like(e, dtype=torch.bool)
    for exc_id in excitation_ids:
        keep |= (e == int(exc_id))
    return keep


def _select_pred(pred: dict, keep: torch.Tensor):
    return {k: v[keep] for k, v in pred.items()}


def _accumulate_scope_losses(
    accum, pred, e, mask, y, mean, std, target_names,
    base_loss=None, base_parts=None, *, outputs_normalized: bool = False,
):
    pred_detached = {k: v.detach() for k, v in pred.items()}
    for scope, excitation_ids in EXCITATION_LOSS_SCOPES:
        keep = _scope_keep(e, excitation_ids)
        if not bool(keep.any().item()):
            continue
        if excitation_ids is None and base_loss is not None and base_parts is not None:
            loss = base_loss.detach()
            parts = {k: v.detach() for k, v in base_parts.items()}
        else:
            y_sel = {k: v[keep] for k, v in y.items()}
            loss, parts = _representation_loss(
                _select_pred(pred_detached, keep), y_sel, mask[keep], mean, std,
                target_names, outputs_normalized=outputs_normalized,
            )
        bs = int(keep.sum().item())
        accum[scope]["sum"] += float(loss.item()) * bs
        accum[scope]["count"] += bs
        for name in target_names:
            accum[scope]["parts"][name] += float(parts[name].item()) * bs


def _finalize_scope_accum(accum, target_names):
    out = {}
    for scope, data in accum.items():
        count = int(data["count"])
        if count <= 0:
            out[scope] = None
            continue
        out[scope] = {
            "loss": float(data["sum"]) / count,
            "parts": _mean_parts(data["parts"], count, target_names),
        }
    return out


def _allreduce_scope_accum(accum, device):
    if not (dist.is_available() and dist.is_initialized()):
        return accum
    for data in accum.values():
        t = torch.tensor([float(data["sum"]), float(data["count"])], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        data["sum"] = float(t[0].item())
        data["count"] = int(t[1].item())
        for name, value in list(data["parts"].items()):
            p = torch.tensor(float(value), device=device)
            dist.all_reduce(p, op=dist.ReduceOp.SUM)
            data["parts"][name] = float(p.item())
    return accum


def _add_scope_fields(row: dict, prefix: str, scope_losses: dict):
    for scope, _ids in EXCITATION_LOSS_SCOPES:
        entry = scope_losses.get(scope)
        row[f"{prefix}_{scope}"] = "" if entry is None else entry["loss"]


def _eval_model(model, dl, device, mean, std, target_names, eval_excitation_ids=None):
    if dl is None or len(dl.dataset) == 0:
        return None, _part_dict_zero(target_names)
    model.eval()
    total_sum = 0.0
    count = 0
    part_sum = _part_dict_zero(target_names)
    with torch.no_grad():
        for batch in dl:
            H, M, e, c, mask, y = _to_device(batch, device)
            filtered = _filter_by_excitation(H, M, e, c, mask, y, eval_excitation_ids)
            if filtered is None:
                continue
            H, M, e, c, mask, y = filtered
            loss, parts = _representation_loss(
                model(H, M, e, c), y, mask, mean, std, target_names,
                outputs_normalized=bool(getattr(model, "outputs_normalized", False)),
            )
            bs = H.size(0)
            total_sum += float(loss.item()) * bs
            count += bs
            for name in target_names:
                part_sum[name] += float(parts[name].item()) * bs
    if count == 0:
        return None, _part_dict_zero(target_names)
    return total_sum / count, _mean_parts(part_sum, count, target_names)


def _eval_model_scopes(model, dl, device, mean, std, target_names):
    accum = _empty_scope_accum(target_names)
    if dl is None or len(dl.dataset) == 0:
        return _finalize_scope_accum(accum, target_names)
    model.eval()
    with torch.no_grad():
        for batch in dl:
            H, M, e, c, mask, y = _to_device(batch, device)
            pred = model(H, M, e, c)
            outputs_normalized = bool(getattr(model, "outputs_normalized", False))
            loss, parts = _representation_loss(
                pred, y, mask, mean, std, target_names,
                outputs_normalized=outputs_normalized,
            )
            _accumulate_scope_losses(
                accum, pred, e, mask, y, mean, std, target_names, loss, parts,
                outputs_normalized=outputs_normalized,
            )
    return _finalize_scope_accum(_allreduce_scope_accum(accum, device), target_names)


def _eval_model_slots(model, dl, device, mean, std, target_names, kmax: int):
    sums = torch.zeros(kmax, dtype=torch.float64, device=device)
    counts = torch.zeros(kmax, dtype=torch.float64, device=device)
    if dl is None or len(dl.dataset) == 0:
        return [None] * kmax
    model.eval()
    with torch.no_grad():
        for batch in dl:
            H, M, e, c, mask, y = _to_device(batch, device)
            pred = model(H, M, e, c)
            slot_sums, slot_counts = _representation_slot_sums(
                pred, y, mask, mean, std, target_names,
                outputs_normalized=bool(getattr(model, "outputs_normalized", False)),
            )
            width = min(kmax, int(slot_sums.numel()))
            sums[:width] += slot_sums[:width].to(dtype=torch.float64)
            counts[:width] += slot_counts[:width].to(dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    return [
        None if float(counts[slot].item()) <= 0.0
        else float((sums[slot] / counts[slot]).item())
        for slot in range(kmax)
    ]


def _write_train_log(path: Path, rows, target_names):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["epoch", "loss_train_total", "loss_train_objective"]
    fields += [f"loss_train_{n}" for n in target_names]
    fields += ["loss_val_total"] + [f"loss_val_{n}" for n in target_names] + ["lr"]
    fields += ["loss_train_slope", "loss_train_curvature"]
    fields += [f"loss_train_slot_{slot}" for slot in range(1, 4)]
    fields += [f"loss_val_slot_{slot}" for slot in range(1, 4)]
    fields += [f"loss_train_{scope}" for scope, _ids in EXCITATION_LOSS_SCOPES]
    fields += [f"loss_val_{scope}" for scope, _ids in EXCITATION_LOSS_SCOPES]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _save_deploy_artifacts(args, out_dir: Path, model_core, in_norm, out_norm, epoch: int, metric: float):
    model_save_path = Path(getattr(args, "model_save_path", "") or (out_dir / f"{out_dir.name}.pth"))
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    normalizer_dir = Path(getattr(args, "normalizer_dir", "") or out_dir)
    normalizer_dir.mkdir(parents=True, exist_ok=True)
    fidelity_normalizer_names = bool(
        getattr(args, "fidelity_normalizer_names", False)
    )
    if not fidelity_normalizer_names:
        in_norm.save(normalizer_dir / "Normalizer_Input.pth")
        out_norm.save(normalizer_dir / "Normalizer_Output.pth")
    payload = {
        "stage": "fixed_single_fidelity",
        "epoch": int(epoch),
        "best_metric": float(metric),
        "checkpoint_selection": str(getattr(model_core.cfg, "checkpoint_selection", "training")),
        "best_metric_name": str(getattr(model_core.cfg, "_best_metric_name", "loss_val_group_FXD")),
        "model_state_dict": model_core.state_dict(),
        "model_type": type(model_core).__name__,
        "target_component": str(getattr(model_core.cfg, "target_component", "")),
        "pole_target": getattr(out_norm, "pole_target", "alpha_omega"),
        "residue_target": getattr(out_norm, "residue_target", "polar"),
        "target_names": out_norm.target_names(),
        "target_transforms": dict(out_norm.target_transforms or {}),
        "slot_support": dict(out_norm.support or {}),
        "K_max": int(model_core.cfg.K_max),
        "low_hidden": list(getattr(model_core.cfg, "low_hidden", [])),
        "adapt_hidden": list(getattr(model_core.cfg, "adapt_hidden", [])),
        "dropout_prob": float(getattr(model_core.cfg, "dropout_prob", 0.0)),
        "multifidelity_real_loss_weight": float(
            getattr(model_core.cfg, "multifidelity_real_loss_weight", 1.0)
        ),
        "multifidelity_sim_loss_weight": float(
            getattr(model_core.cfg, "multifidelity_sim_loss_weight", 0.2)
        ),
        "use_lora": bool(getattr(model_core.cfg, "use_lora", False)),
        "use_modulation": bool(getattr(model_core.cfg, "use_modulation", False)),
        "modulation_mode": "legacy",
        "mod_gamma": float(getattr(model_core.cfg, "mod_gamma", 0.0)),
        "embed_excitation_num": int(getattr(model_core.cfg, "embed_excitation_num", 0)),
        "embed_excitation_dim": int(getattr(model_core.cfg, "embed_excitation_dim", 0)),
        "embed_channel_num": int(getattr(model_core.cfg, "embed_channel_num", 0)),
        "embed_channel_dim": int(getattr(model_core.cfg, "embed_channel_dim", 0)),
        "fold": getattr(args, "fold_index", None),
        "num_folds": getattr(args, "num_folds", None),
    }
    if payload["use_lora"]:
        payload.update(
            {
                "lora_rank": int(getattr(model_core.cfg, "lora_rank", 0)),
                "lora_alpha": float(getattr(model_core.cfg, "lora_alpha", 1.0)),
                "lora_dropout": float(getattr(model_core.cfg, "lora_dropout", 0.0)),
            }
        )
    payload.update(dict(getattr(args, "checkpoint_metadata", {}) or {}))
    torch.save(payload, model_save_path)
    deploy_model_save_path = getattr(args, "deploy_model_save_path", "")
    if deploy_model_save_path:
        deploy_model_save_path = Path(deploy_model_save_path)
        if deploy_model_save_path != model_save_path:
            deploy_model_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, deploy_model_save_path)
    deploy_normalizer_dir = getattr(args, "deploy_normalizer_dir", "")
    if deploy_normalizer_dir:
        deploy_normalizer_dir = Path(deploy_normalizer_dir)
        if deploy_normalizer_dir != normalizer_dir and not fidelity_normalizer_names:
            deploy_normalizer_dir.mkdir(parents=True, exist_ok=True)
            in_norm.save(deploy_normalizer_dir / "Normalizer_Input.pth")
            out_norm.save(deploy_normalizer_dir / "Normalizer_Output.pth")
    return model_save_path, normalizer_dir


class _TargetOutputView(torch.nn.Module):
    outputs_normalized = True

    def __init__(self, model: TargetMultifidelityMFDNN, output: str):
        super().__init__()
        self.model = model
        self.output = str(output)

    def forward(self, H, M, e, c):
        if self.output == "low":
            return self.model.forward_low(H, M, e, c)
        _low, high = self.model(H, M, e, c)
        return high


def _write_multifidelity_log(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "stage", "epoch", "loss_sim", "loss_real", "loss_total",
        "loss_val_sim", "loss_val_real", "lr",
    ]
    extra = sorted({key for row in rows for key in row} - set(preferred))
    fields = preferred + extra
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def train_multifidelity_component(
    fold: int,
    cfg: Config,
    sim_parsed,
    real_parsed,
    args,
):
    """Train one FDR component: simulation low head then real correction head."""
    sim_parsed = list(sim_parsed or [])
    real_parsed = list(real_parsed or [])
    if not sim_parsed:
        raise RuntimeError("Multifidelity component training requires simulation rows")
    if int(cfg.epochs_stage1) <= 0:
        raise ValueError("epochs_stage1 must be positive")
    if real_parsed and int(cfg.epochs_stage2) <= 0:
        raise ValueError("epochs_stage2 must be positive for multifidelity training")

    device = _select_device(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fidelity_mode = "multifidelity" if real_parsed else "sim_only"
    prediction_output = "high" if real_parsed else "low"

    if real_parsed:
        (
            sim_dl,
            real_dl,
            sim_val_dl,
            real_val_dl,
            n_sim,
            n_real,
            out_norm_sim,
            out_norm_real,
            in_norm,
        ) = build_dataloaders(
            sim_parsed, real_parsed, cfg, args.kfold, fold, device, out_dir
        )
        deploy_norm = out_norm_real
    else:
        sim_dl, n_sim, out_norm_sim, in_norm = build_low_fidelity_dataloader(
            sim_parsed, cfg, out_dir
        )
        sim_val_dl = None
        real_dl = real_val_dl = None
        n_real = 0
        out_norm_real = None
        deploy_norm = out_norm_sim

    cfg._target_transforms = dict(deploy_norm.target_transforms or {})
    cfg.embed_excitation_num = max(
        int(getattr(cfg, "embed_excitation_num", 0)),
        max(int(row.exc_id) for row in sim_parsed + real_parsed) + 1,
    )
    cfg.embed_channel_num = max(
        int(getattr(cfg, "embed_channel_num", 0)),
        max(int(row.ch_id) for row in sim_parsed + real_parsed) + 1,
    )
    target_names = out_norm_sim.target_names()
    if real_parsed and tuple(out_norm_real.target_names()) != tuple(target_names):
        raise RuntimeError(
            "Simulation/real target mismatch: "
            f"sim={target_names}, real={out_norm_real.target_names()}"
        )

    model = TargetMultifidelityMFDNN(cfg).to(device)
    low_view = _TargetOutputView(model, "low")
    high_view = _TargetOutputView(model, "high")
    mean_sim = torch.as_tensor(out_norm_sim.mean, dtype=torch.float32, device=device)
    std_sim = torch.as_tensor(out_norm_sim.std, dtype=torch.float32, device=device).clamp_min(1e-8)
    mean_real = (
        None
        if out_norm_real is None
        else torch.as_tensor(out_norm_real.mean, dtype=torch.float32, device=device)
    )
    std_real = (
        None
        if out_norm_real is None
        else torch.as_tensor(out_norm_real.std, dtype=torch.float32, device=device).clamp_min(1e-8)
    )
    print(
        f"[MULTIFIDELITY-FDR] device={device} mode={fidelity_mode} "
        f"sim_rows={n_sim} real_rows={n_real} targets={target_names} "
        f"parameters={count_parameter_groups(model)}",
        flush=True,
    )

    logs: list[dict] = []
    train_hist: list[float] = []
    val_hist: list[float] = []
    min_delta = float(getattr(cfg, "early_stop_min_delta", 0.0))
    checkpoint_selection = str(
        getattr(cfg, "checkpoint_selection", "training")
    ).lower()
    if checkpoint_selection not in {"training", "validation"}:
        raise ValueError(f"Unsupported checkpoint_selection={checkpoint_selection!r}")

    # Stage 1: train only the simulation low head.  Embeddings remain fixed,
    # matching user_for_win7_fixed_exc_channel's per-combination behavior.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.low.parameters():
        parameter.requires_grad_(True)
    optimizer1 = torch.optim.AdamW(
        model.low.parameters(), lr=cfg.lr_low_adam, weight_decay=cfg.weight_decay
    )
    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer1, mode="min", factor=0.5, patience=20, min_lr=1e-6
    )
    best_stage1 = float("inf")
    best_stage1_epoch = 0
    best_low_state = None
    bad = 0
    stage1_started_at = time.perf_counter()
    for epoch in range(int(cfg.epochs_stage1)):
        model.train()
        total, count = 0.0, 0
        for batch in sim_dl:
            H, M, e, c, mask, target = _to_device(batch, device)
            optimizer1.zero_grad(set_to_none=True)
            prediction = model.forward_low(H, M, e, c)
            loss, _parts = _representation_loss(
                prediction,
                target,
                mask,
                mean_sim,
                std_sim,
                target_names,
                outputs_normalized=True,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.low.parameters(), 5.0)
            optimizer1.step()
            total += float(loss.detach().item()) * H.size(0)
            count += H.size(0)
        train_loss = total / max(count, 1)
        val_loss, _ = _eval_model(
            low_view, sim_val_dl, device, mean_sim, std_sim, target_names
        )
        metric = (
            train_loss
            if checkpoint_selection == "training" or val_loss is None
            else val_loss
        )
        scheduler1.step(train_loss)
        logs.append(
            {
                "stage": 1,
                "epoch": epoch + 1,
                "loss_sim": train_loss,
                "loss_val_sim": "" if val_loss is None else val_loss,
                "lr": optimizer1.param_groups[0]["lr"],
            }
        )
        if metric < best_stage1 - min_delta:
            best_stage1 = float(metric)
            best_stage1_epoch = epoch + 1
            best_low_state = deepcopy(model.low.state_dict())
            bad = 0
        else:
            bad += 1
        if (epoch + 1) == 1 or (epoch + 1) % 10 == 0:
            val_text = "nan" if val_loss is None else f"{val_loss:.6f}"
            elapsed_text, eta_text = _elapsed_and_eta(
                stage1_started_at, epoch + 1, int(cfg.epochs_stage1)
            )
            print(
                f"[S1][{cfg.target_component}] epoch={epoch + 1}/{cfg.epochs_stage1} "
                f"sim={train_loss:.6f} val={val_text} "
                f"elapsed={elapsed_text} eta_to_max={eta_text}",
                flush=True,
            )
        if cfg.early_stop_patience_s1 > 0 and bad >= cfg.early_stop_patience_s1:
            break
    stage1_epochs_run = sum(int(row.get("stage", 0)) == 1 for row in logs)
    print(
        f"[S1-DONE][{cfg.target_component}] epochs={stage1_epochs_run}/"
        f"{cfg.epochs_stage1} elapsed="
        f"{_format_duration(time.perf_counter() - stage1_started_at)} "
        f"best={best_stage1:.6f}@{best_stage1_epoch}",
        flush=True,
    )
    if best_low_state is not None:
        model.low.load_state_dict(best_low_state)

    if not real_parsed:
        cfg._best_metric_name = "loss_sim"
        cfg._last_training_result = {
            "best_epoch": int(best_stage1_epoch),
            "best_metric": float(best_stage1),
            "best_metric_name": "loss_sim",
            "final_train_loss": float(logs[-1]["loss_sim"]),
            "final_validation_loss": None,
            "fidelity_mode": fidelity_mode,
            "prediction_output": prediction_output,
        }
        if is_main_process():
            _save_deploy_artifacts(
                args,
                out_dir,
                model,
                in_norm,
                deploy_norm,
                best_stage1_epoch,
                best_stage1,
            )
            _write_multifidelity_log(Path(args.train_log_path), logs)
        return best_stage1, [float(row["loss_sim"]) for row in logs], []

    # Stage 2: learn the real correction while retaining a soft sim constraint.
    freeze_low = bool(getattr(cfg, "freeze_low_stage2", False))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.lin.parameters():
        parameter.requires_grad_(True)
    for parameter in model.adaptation.parameters():
        parameter.requires_grad_(True)
    if not freeze_low:
        for parameter in model.low.parameters():
            parameter.requires_grad_(True)
    stage2_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer2 = torch.optim.AdamW(
        stage2_parameters, lr=cfg.lr_high_adam, weight_decay=cfg.weight_decay
    )
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer2, mode="min", factor=0.5, patience=20, min_lr=1e-6
    )
    real_weight = float(getattr(cfg, "multifidelity_real_loss_weight", 1.0))
    sim_weight = float(getattr(cfg, "multifidelity_sim_loss_weight", 0.2))
    for name, value in (
        ("multifidelity_real_loss_weight", real_weight),
        ("multifidelity_sim_loss_weight", sim_weight),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    ceiling = getattr(cfg, "checkpoint_train_loss_ceiling", None)
    ceiling = None if ceiling is None else float(ceiling)
    cfg._best_metric_name = (
        "loss_val_real" if checkpoint_selection == "validation" else "loss_real"
    )
    parent_loss_target = getattr(cfg, "parent_train_loss_target", None)
    parent_loss_target = (
        None if parent_loss_target is None else float(parent_loss_target)
    )
    epochs_after_parent_loss = int(getattr(cfg, "epochs_after_parent_loss", 0))
    parent_loss_reached_epoch = None
    best = float("inf")
    best_epoch = 0
    best_state = None
    fallback_best = float("inf")
    fallback_state = None
    fallback_epoch = 0
    bad = 0
    sim_iterator = cycle(sim_dl)
    stage2_started_at = time.perf_counter()
    for epoch in range(int(cfg.epochs_stage2)):
        model.train()
        if freeze_low:
            model.low.eval()
        total_real = total_sim = total_combined = 0.0
        count = 0
        for real_batch in real_dl:
            H_r, M_r, e_r, c_r, mask_r, target_r = _to_device(real_batch, device)
            optimizer2.zero_grad(set_to_none=True)
            _out_low_r, out_high_r = model(H_r, M_r, e_r, c_r)
            loss_real, _ = _representation_loss(
                out_high_r,
                target_r,
                mask_r,
                mean_real,
                std_real,
                target_names,
                outputs_normalized=True,
            )
            if freeze_low or sim_weight <= 0.0:
                loss_sim = loss_real.new_tensor(0.0)
            else:
                H_s, M_s, e_s, c_s, mask_s, target_s = _to_device(
                    next(sim_iterator), device
                )
                out_low_s = model.forward_low(H_s, M_s, e_s, c_s)
                loss_sim, _ = _representation_loss(
                    out_low_s,
                    target_s,
                    mask_s,
                    mean_sim,
                    std_sim,
                    target_names,
                    outputs_normalized=True,
                )
            loss_total = real_weight * loss_real + sim_weight * loss_sim
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(stage2_parameters, 5.0)
            optimizer2.step()
            batch_size = H_r.size(0)
            total_real += float(loss_real.detach().item()) * batch_size
            total_sim += float(loss_sim.detach().item()) * batch_size
            total_combined += float(loss_total.detach().item()) * batch_size
            count += batch_size
        real_loss = total_real / max(count, 1)
        sim_loss = total_sim / max(count, 1)
        combined_loss = total_combined / max(count, 1)
        val_loss, _ = _eval_model(
            high_view, real_val_dl, device, mean_real, std_real, target_names
        )
        raw_metric = (
            real_loss
            if checkpoint_selection == "training" or val_loss is None
            else val_loss
        )
        eligible = val_loss is None or ceiling is None or real_loss <= ceiling
        if raw_metric < fallback_best - min_delta:
            fallback_best = float(raw_metric)
            fallback_epoch = epoch + 1
            fallback_state = deepcopy(model.state_dict())
        metric = raw_metric if eligible else None
        scheduler2.step(real_loss)
        train_hist.append(real_loss)
        val_hist.append(float("nan") if val_loss is None else val_loss)
        logs.append(
            {
                "stage": 2,
                "epoch": epoch + 1,
                "loss_sim": sim_loss,
                "loss_real": real_loss,
                "loss_total": combined_loss,
                "loss_val_real": "" if val_loss is None else val_loss,
                "lr": optimizer2.param_groups[0]["lr"],
            }
        )
        if (
            parent_loss_target is not None
            and parent_loss_reached_epoch is None
            and real_loss <= parent_loss_target
        ):
            parent_loss_reached_epoch = epoch + 1
        if metric is not None and metric < best - min_delta:
            best = float(metric)
            best_epoch = epoch + 1
            best_state = deepcopy(model.state_dict())
            bad = 0
            if is_main_process():
                _save_deploy_artifacts(
                    args, out_dir, model, in_norm, deploy_norm, best_epoch, best
                )
        elif eligible:
            bad += 1
        if (epoch + 1) == 1 or (epoch + 1) % 10 == 0:
            val_text = "nan" if val_loss is None else f"{val_loss:.6f}"
            elapsed_text, eta_text = _elapsed_and_eta(
                stage2_started_at, epoch + 1, int(cfg.epochs_stage2)
            )
            print(
                f"[S2][{cfg.target_component}] epoch={epoch + 1}/{cfg.epochs_stage2} "
                f"sim={sim_loss:.6f} real={real_loss:.6f} val={val_text} "
                f"elapsed={elapsed_text} eta_to_max={eta_text}",
                flush=True,
            )
        if cfg.early_stop_patience_s2 > 0 and bad >= cfg.early_stop_patience_s2:
            break
        if (
            parent_loss_reached_epoch is not None
            and (epoch + 1) - parent_loss_reached_epoch >= epochs_after_parent_loss
        ):
            break

    stage2_epochs_run = sum(int(row.get("stage", 0)) == 2 for row in logs)
    print(
        f"[S2-DONE][{cfg.target_component}] epochs={stage2_epochs_run}/"
        f"{cfg.epochs_stage2} elapsed="
        f"{_format_duration(time.perf_counter() - stage2_started_at)}",
        flush=True,
    )

    if best_state is None:
        best, best_epoch, best_state = fallback_best, fallback_epoch, fallback_state
    if best_state is None:
        raise RuntimeError("Multifidelity training did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    cfg._last_training_result = {
        "best_epoch": int(best_epoch),
        "best_metric": float(best),
        "best_metric_name": str(cfg._best_metric_name),
        "final_train_loss": None if not train_hist else float(train_hist[-1]),
        "final_validation_loss": (
            None
            if not val_hist or val_hist[-1] != val_hist[-1]
            else float(val_hist[-1])
        ),
        "fidelity_mode": fidelity_mode,
        "prediction_output": prediction_output,
        "parent_train_loss_target": parent_loss_target,
        "parent_loss_reached_epoch": parent_loss_reached_epoch,
        "epochs_after_parent_loss": (
            epochs_after_parent_loss if parent_loss_target is not None else None
        ),
    }
    if is_main_process():
        _save_deploy_artifacts(
            args, out_dir, model, in_norm, deploy_norm, best_epoch, best
        )
        _write_multifidelity_log(Path(args.train_log_path), logs)
    return best, train_hist, val_hist


def train_one_fold(fold: int, cfg: Config, real_parsed, args):
    debug_n = int(getattr(args, "debug_overfit_n", 0) or 0)
    debug_mode = debug_n > 0
    if debug_mode:
        cfg = deepcopy(cfg)
        cfg.use_val_split = False
        cfg.early_stop_patience_s1 = 0
        offset = int(getattr(args, "debug_overfit_offset", 0) or 0)
        real_parsed = list(real_parsed)[offset:offset + debug_n]
        if is_main_process():
            print(f"[DEBUG] overfit samples={len(real_parsed)} offset={offset}", flush=True)

    device = _select_device(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dl, val_dl, n_train, n_val, out_norm, in_norm = build_single_fidelity_dataloaders(real_parsed, cfg, args.kfold, fold, device, out_dir)
    cfg._target_transforms = dict(out_norm.target_transforms or {})
    cfg.embed_excitation_num = max(int(getattr(cfg, "embed_excitation_num", 0)), max(int(p.exc_id) for p in real_parsed) + 1)
    cfg.embed_channel_num = max(int(getattr(cfg, "embed_channel_num", 0)), max(int(p.ch_id) for p in real_parsed) + 1)

    target_names = out_norm.target_names()
    if getattr(cfg, "model_target_names", None):
        model_core = TargetMFDNN(cfg).to(device)
    else:
        model_core = MFDNN(cfg).to(device)
    for parameter in model_core.parameters():
        parameter.requires_grad_(True)
    model = model_core
    mean = torch.as_tensor(out_norm.mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(out_norm.std, dtype=torch.float32, device=device).clamp_min(1e-8)
    opt = torch.optim.AdamW(model_core.parameters(), lr=cfg.lr_low_adam, weight_decay=cfg.weight_decay)
    sched = None if debug_mode else torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=20, min_lr=1e-6)

    best = float("inf")
    best_epoch = 0
    best_state = None
    fallback_best = float("inf")
    fallback_best_epoch = 0
    fallback_best_state = None
    bad = 0
    patience = int(getattr(cfg, "early_stop_patience_s1", 500))
    min_delta = float(getattr(cfg, "early_stop_min_delta", 0.0))
    checkpoint_selection = str(getattr(cfg, "checkpoint_selection", "training")).lower()
    if checkpoint_selection not in ("validation", "training"):
        raise ValueError(f"Unsupported checkpoint_selection={checkpoint_selection!r}")
    checkpoint_train_loss_ceiling = getattr(
        cfg, "checkpoint_train_loss_ceiling", None
    )
    if checkpoint_train_loss_ceiling is not None:
        checkpoint_train_loss_ceiling = float(checkpoint_train_loss_ceiling)
        if checkpoint_train_loss_ceiling <= 0.0:
            raise ValueError("checkpoint_train_loss_ceiling must be positive")
    parent_loss_target = getattr(cfg, "parent_train_loss_target", None)
    epochs_after_parent_loss = int(getattr(cfg, "epochs_after_parent_loss", 0))
    if parent_loss_target is not None:
        parent_loss_target = float(parent_loss_target)
        if checkpoint_selection != "training":
            raise ValueError("parent_train_loss_target requires checkpoint_selection='training'")
        if epochs_after_parent_loss < 0:
            raise ValueError("epochs_after_parent_loss must be non-negative")
    parent_loss_reached_epoch = None
    logs = []
    train_hist = []
    val_hist = []

    if is_main_process():
        modulation_fields = ("use_modulation", "mod_gamma", "pole_target", "residue_target")
        print("[MODULATION] " + " ".join(f"{name}={getattr(cfg, name)}" for name in modulation_fields), flush=True)
        print("[PARAMETERS] " + " ".join(f"{name}={value}" for name, value in count_parameter_groups(model_core).items()), flush=True)
        print(f"[INITIALIZATION] random; all_parameters_trainable={all(p.requires_grad for p in model_core.parameters())}", flush=True)
        print(
            f"[Fixed-SF] train_rows={n_train}, val_rows={n_val}, "
            f"pole_target={out_norm.pole_target}, residue_target={out_norm.residue_target}, targets={target_names}",
            flush=True,
        )
        validation_scope = str(getattr(cfg, "validation_metric_scope", "group_FXD"))
        selection_text = "loss_train_total" if checkpoint_selection == "training" else f"loss_val_{validation_scope}"
        print(f"[CHECKPOINT] selection={checkpoint_selection} metric={selection_text}", flush=True)
        if checkpoint_selection == "validation" and checkpoint_train_loss_ceiling is not None:
            print(
                "[CHECKPOINT] validation candidates require "
                f"loss_train_total <= {checkpoint_train_loss_ceiling:.6g}",
                flush=True,
            )
        if parent_loss_target is not None:
            print(
                f"[EARLY-STOP] target parent train_loss={parent_loss_target:.9g}; "
                f"stop after {epochs_after_parent_loss} additional epochs",
                flush=True,
            )
        print("[VAL] all excitation scopes are still reported.", flush=True)
        if sched is not None:
            print("[LR] ReduceLROnPlateau monitors train_loss: factor=0.5 patience=20 min_lr=1e-6", flush=True)

    for epoch in range(int(cfg.epochs_stage1)):
        sampler = getattr(train_dl, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        model.train()
        total_sum = 0.0
        objective_sum = 0.0
        count = 0
        part_sum = _part_dict_zero(target_names)
        slot_sum = torch.zeros(int(cfg.K_max), dtype=torch.float64, device=device)
        slot_count = torch.zeros(int(cfg.K_max), dtype=torch.float64, device=device)
        slope_sum = 0.0
        curvature_sum = 0.0
        train_scope_accum = _empty_scope_accum(target_names)
        for batch in train_dl:
            H, M, e, c, mask, y = _to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            pred = model(H, M, e, c)
            outputs_normalized = bool(getattr(model_core, "outputs_normalized", False))
            data_loss, parts = _representation_loss(
                pred, y, mask, mean, std, target_names,
                outputs_normalized=outputs_normalized,
            )
            batch_slot_sum, batch_slot_count = _representation_slot_sums(
                pred, y, mask, mean, std, target_names,
                outputs_normalized=outputs_normalized,
            )
            slot_width = min(int(cfg.K_max), int(batch_slot_sum.numel()))
            slot_sum[:slot_width] += batch_slot_sum[:slot_width].detach().to(dtype=torch.float64)
            slot_count[:slot_width] += batch_slot_count[:slot_width].detach().to(dtype=torch.float64)
            if str(getattr(cfg, "target_component", "")) == "damping":
                smooth_pred, smooth_m, smooth_mask = _damping_collocation_batch(
                    model, H, M, e, c, mask, in_norm, cfg
                )
                slope_loss, curvature_loss = _damping_smoothness_losses(
                    smooth_pred, smooth_m, smooth_mask, out_norm, in_norm, cfg
                )
            else:
                slope_loss = curvature_loss = M.new_tensor(0.0)
            loss = (
                data_loss
                + float(getattr(cfg, "damping_slope_weight", 0.0)) * slope_loss
                + float(getattr(cfg, "damping_curvature_weight", 0.0)) * curvature_loss
            )
            _accumulate_scope_losses(
                train_scope_accum, pred, e, mask, y, mean, std, target_names,
                data_loss, parts, outputs_normalized=outputs_normalized,
            )
            loss.backward()
            if not debug_mode:
                torch.nn.utils.clip_grad_norm_(model_core.parameters(), max_norm=5.0)
            opt.step()
            bs = H.size(0)
            # Keep checkpoint selection and LR scheduling tied to data fit.  The
            # smoothness terms shape the damping curve but must not make an
            # under-fitted model look preferable merely because it is flatter.
            total_sum += float(data_loss.detach().item()) * bs
            objective_sum += float(loss.detach().item()) * bs
            count += bs
            slope_sum += float(slope_loss.detach().item()) * bs
            curvature_sum += float(curvature_loss.detach().item()) * bs
            for name in target_names:
                part_sum[name] += float(parts[name].detach().item()) * bs

        train_loss = total_sum / max(count, 1)
        train_objective = objective_sum / max(count, 1)
        train_parts = _mean_parts(part_sum, count, target_names)
        train_slope = slope_sum / max(count, 1)
        train_curvature = curvature_sum / max(count, 1)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(slot_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(slot_count, op=dist.ReduceOp.SUM)
        train_slot_losses = [
            None if float(slot_count[slot].item()) <= 0.0
            else float((slot_sum[slot] / slot_count[slot]).item())
            for slot in range(int(cfg.K_max))
        ]
        train_scope_losses = _finalize_scope_accum(_allreduce_scope_accum(train_scope_accum, device), target_names)
        val_scope_losses = _eval_model_scopes(model, val_dl, device, mean, std, target_names)
        val_slot_losses = _eval_model_slots(
            model, val_dl, device, mean, std, target_names, int(cfg.K_max)
        )
        validation_scope = str(getattr(cfg, "validation_metric_scope", "group_FXD"))
        val_selection = val_scope_losses.get(validation_scope)
        val_loss = None if val_selection is None else val_selection["loss"]
        val_parts = _part_dict_zero(target_names) if val_selection is None else val_selection["parts"]
        if checkpoint_selection == "training":
            raw_metric = train_loss
            candidate_eligible = True
            cfg._best_metric_name = "loss_train_total"
        else:
            raw_metric = val_loss if val_loss is not None else train_loss
            candidate_eligible = (
                val_loss is None
                or checkpoint_train_loss_ceiling is None
                or train_loss <= checkpoint_train_loss_ceiling
            )
            cfg._best_metric_name = (
                f"loss_val_{validation_scope}_after_train_fit"
                if val_loss is not None and checkpoint_train_loss_ceiling is not None
                else f"loss_val_{validation_scope}"
                if val_loss is not None
                else "loss_train_total_fallback"
            )
            if raw_metric < fallback_best - min_delta:
                fallback_best = float(raw_metric)
                fallback_best_epoch = epoch + 1
                fallback_best_state = deepcopy(model_core.state_dict())
        metric = raw_metric if candidate_eligible else None
        if sched is not None:
            sched.step(train_loss)
        train_hist.append(train_loss)
        val_hist.append(float("nan") if val_loss is None else val_loss)

        row = {
            "epoch": epoch + 1,
            "loss_train_total": train_loss,
            "loss_train_objective": train_objective,
            "loss_val_total": "" if val_loss is None else val_loss,
            "loss_train_slope": train_slope,
            "loss_train_curvature": train_curvature,
            "lr": opt.param_groups[0]["lr"],
        }
        for name in target_names:
            row[f"loss_train_{name}"] = train_parts[name]
            row[f"loss_val_{name}"] = "" if val_loss is None else val_parts[name]
        for slot in range(int(cfg.K_max)):
            row[f"loss_train_slot_{slot + 1}"] = (
                "" if train_slot_losses[slot] is None else train_slot_losses[slot]
            )
            row[f"loss_val_slot_{slot + 1}"] = (
                "" if val_slot_losses[slot] is None else val_slot_losses[slot]
            )
        _add_scope_fields(row, "loss_train", train_scope_losses)
        _add_scope_fields(row, "loss_val", val_scope_losses)
        logs.append(row)

        improved = metric is not None and metric < best - min_delta
        if improved:
            best = float(metric)
            best_epoch = epoch + 1
            best_state = deepcopy(model_core.state_dict())
            bad = 0
            if is_main_process():
                _save_deploy_artifacts(args, out_dir, model_core, in_norm, out_norm, best_epoch, best)
        elif candidate_eligible:
            bad += 1

        if parent_loss_target is not None and parent_loss_reached_epoch is None and train_loss <= parent_loss_target:
            parent_loss_reached_epoch = epoch + 1
            if is_main_process():
                print(
                    f"[EARLY-STOP] parent train-loss target reached at epoch "
                    f"{parent_loss_reached_epoch}; countdown started.",
                    flush=True,
                )

        if is_main_process() and ((epoch + 1) == 1 or (epoch + 1) % 10 == 0 or improved):
            vt = "nan" if val_loss is None else f"{val_loss:.6f}"
            best_text = "pending-fit" if best_state is None else f"{best:.6f}@{best_epoch}"
            print(f"[Fold {fold + 1}] Epoch {epoch + 1}/{cfg.epochs_stage1} train={train_loss:.6f} val={vt} best={best_text} lr={opt.param_groups[0]['lr']:.3e}", flush=True)

        if (
            parent_loss_reached_epoch is not None
            and (epoch + 1) - parent_loss_reached_epoch >= epochs_after_parent_loss
        ):
            if is_main_process():
                print(
                    f"[EARLY-STOP] completed {epochs_after_parent_loss} epochs after reaching "
                    "the parent train-loss target.",
                    flush=True,
                )
            break
        if parent_loss_target is None and patience > 0 and bad >= patience:
            break

    if best_state is None and fallback_best_state is not None:
        best = fallback_best
        best_epoch = fallback_best_epoch
        best_state = fallback_best_state
        if is_main_process():
            print(
                "[WARN] No checkpoint reached checkpoint_train_loss_ceiling; "
                "falling back to the best validation checkpoint.",
                flush=True,
            )
    if best_state is not None:
        model_core.load_state_dict(best_state)
    if is_main_process():
        model_path, norm_dir = _save_deploy_artifacts(args, out_dir, model_core, in_norm, out_norm, best_epoch, best)
        _write_train_log(Path(getattr(args, "train_log_path", "") or (out_dir / "train_log.csv")), logs, target_names)
        print(f"[OK] model saved to: {model_path}", flush=True)
        print(f"[OK] normalizers saved to: {norm_dir}", flush=True)
    cfg._last_training_result = {
        "best_epoch": int(best_epoch),
        "best_metric": float(best),
        "best_metric_name": str(getattr(cfg, "_best_metric_name", "")),
        "final_train_loss": None if not train_hist else float(train_hist[-1]),
        "final_validation_loss": None if not val_hist or val_hist[-1] != val_hist[-1] else float(val_hist[-1]),
        "parent_train_loss_target": parent_loss_target,
        "parent_loss_reached_epoch": parent_loss_reached_epoch,
        "epochs_after_parent_loss": epochs_after_parent_loss if parent_loss_target is not None else None,
    }
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return best, train_hist, val_hist
