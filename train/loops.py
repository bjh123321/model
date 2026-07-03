from __future__ import annotations

import csv
import os
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist

from surrogate_model.config import Config
from surrogate_model.models.blocks import MFDNN
from surrogate_model.train.dataloaders import build_single_fidelity_dataloaders
from surrogate_model.utils.dist_utils import is_main_process


PART_NAMES = ("alpha", "omega", "rho", "sin_theta", "cos_theta")


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


def _part_dict_zero():
    return {k: 0.0 for k in PART_NAMES}


def _mean_parts(sums, count):
    den = max(int(count), 1)
    return {k: float(sums[k]) / den for k in PART_NAMES}


def _polar_loss(pred, tgt, mask, mean, std, direct_target: bool = False):
    width = min(mask.size(1), pred["alpha"].size(1), tgt["alpha"].size(1))
    w = mask[:, :width].to(dtype=pred["alpha"].dtype)
    den = w.sum().clamp_min(1.0)
    if direct_target:
        pred_cmp = {name: pred[name][:, :width] for name in PART_NAMES}
    else:
        pred_cmp = {
            "alpha": (pred["alpha"][:, :width] - mean[0]) / std[0].clamp_min(1e-8),
            "omega": (pred["omega"][:, :width] - mean[1]) / std[1].clamp_min(1e-8),
            "rho": (pred["rho"][:, :width] - mean[2]) / std[2].clamp_min(1e-8),
            "sin_theta": pred["sin_theta"][:, :width],
            "cos_theta": pred["cos_theta"][:, :width],
        }
    total = pred["alpha"].new_tensor(0.0)
    parts = {}
    for name in PART_NAMES:
        val = (w * (pred_cmp[name] - tgt[name][:, :width]).pow(2)).sum() / den
        parts[name] = val
        total = total + val
    return total, parts


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


def _eval_model(model, dl, device, mean, std, direct_target: bool = False, eval_excitation_ids=None):
    if dl is None or len(dl.dataset) == 0:
        return None, _part_dict_zero()
    model.eval()
    total_sum = 0.0
    count = 0
    part_sum = _part_dict_zero()
    with torch.no_grad():
        for batch in dl:
            H, M, e, c, mask, y = _to_device(batch, device)
            filtered = _filter_by_excitation(H, M, e, c, mask, y, eval_excitation_ids)
            if filtered is None:
                continue
            H, M, e, c, mask, y = filtered
            loss, parts = _polar_loss(model(H, M, e, c), y, mask, mean, std, direct_target=direct_target)
            bs = H.size(0)
            total_sum += float(loss.item()) * bs
            count += bs
            for name in PART_NAMES:
                part_sum[name] += float(parts[name].item()) * bs
    if count == 0:
        return None, _part_dict_zero()
    return total_sum / count, _mean_parts(part_sum, count)


def _write_train_log(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["epoch", "loss_train_total"] + [f"loss_train_{n}" for n in PART_NAMES]
    fields += ["loss_val_total"] + [f"loss_val_{n}" for n in PART_NAMES] + ["lr"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _save_deploy_artifacts(args, out_dir: Path, model_core: MFDNN, in_norm, out_norm, epoch: int, metric: float):
    model_save_path = Path(getattr(args, "model_save_path", "") or (out_dir / f"{out_dir.name}.pth"))
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    normalizer_dir = Path(getattr(args, "normalizer_dir", "") or out_dir.parent)
    normalizer_dir.mkdir(parents=True, exist_ok=True)
    in_norm.save(normalizer_dir / "Normalizer_Input.pth")
    out_norm.save(normalizer_dir / "Normalizer_Output.pth")
    torch.save({"stage": "fixed_single_fidelity", "epoch": int(epoch), "best_metric": float(metric), "model_state_dict": model_core.state_dict(), "residue_target": "polar", "K_max": int(model_core.cfg.K_max)}, model_save_path)
    return model_save_path, normalizer_dir


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
            print(f"[DEBUG] overfit samples={len(real_parsed)} offset={offset} direct_label_loss=True", flush=True)

    device = _select_device(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dl, val_dl, n_train, n_val, out_norm, in_norm = build_single_fidelity_dataloaders(real_parsed, cfg, args.kfold, fold, device, out_dir)
    cfg.embed_excitation_num = max(int(getattr(cfg, "embed_excitation_num", 0)), max(int(p.exc_id) for p in real_parsed) + 1)
    cfg.embed_channel_num = max(int(getattr(cfg, "embed_channel_num", 0)), max(int(p.ch_id) for p in real_parsed) + 1)

    model_core = MFDNN(cfg).to(device)
    model = model_core
    mean = torch.as_tensor(out_norm.mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(out_norm.std, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model_core.parameters(), lr=cfg.lr_low_adam, weight_decay=cfg.weight_decay)
    sched = None if debug_mode else torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=20, min_lr=1e-6)
    val_eval_excitation_ids = None if debug_mode else (0, 1)

    best = float("inf")
    best_epoch = 0
    best_state = None
    bad = 0
    patience = int(getattr(cfg, "early_stop_patience_s1", 500))
    min_delta = float(getattr(cfg, "early_stop_min_delta", 0.0))
    logs = []
    train_hist = []
    val_hist = []

    if is_main_process():
        print(f"[Fixed-SF] train_rows={n_train}, val_rows={n_val}, target=polar", flush=True)
        if val_eval_excitation_ids is not None:
            print(f"[VAL] loss is computed only on validation samples with exc_id in {val_eval_excitation_ids}", flush=True)
        if sched is not None:
            print("[LR] ReduceLROnPlateau monitors train_loss: factor=0.5 patience=20 min_lr=1e-6", flush=True)

    for epoch in range(int(cfg.epochs_stage1)):
        sampler = getattr(train_dl, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        model.train()
        total_sum = 0.0
        count = 0
        part_sum = _part_dict_zero()
        for batch in train_dl:
            H, M, e, c, mask, y = _to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            loss, parts = _polar_loss(model(H, M, e, c), y, mask, mean, std, direct_target=debug_mode)
            loss.backward()
            if not debug_mode:
                torch.nn.utils.clip_grad_norm_(model_core.parameters(), max_norm=5.0)
            opt.step()
            bs = H.size(0)
            total_sum += float(loss.detach().item()) * bs
            count += bs
            for name in PART_NAMES:
                part_sum[name] += float(parts[name].detach().item()) * bs

        train_loss = total_sum / max(count, 1)
        train_parts = _mean_parts(part_sum, count)
        val_loss, val_parts = _eval_model(model, val_dl, device, mean, std, direct_target=debug_mode, eval_excitation_ids=val_eval_excitation_ids)
        metric = val_loss if val_loss is not None else train_loss
        if sched is not None:
            sched.step(train_loss)
        train_hist.append(train_loss)
        val_hist.append(float("nan") if val_loss is None else val_loss)

        row = {"epoch": epoch + 1, "loss_train_total": train_loss, "loss_val_total": "" if val_loss is None else val_loss, "lr": opt.param_groups[0]["lr"]}
        for name in PART_NAMES:
            row[f"loss_train_{name}"] = train_parts[name]
            row[f"loss_val_{name}"] = "" if val_loss is None else val_parts[name]
        logs.append(row)

        if metric < best - min_delta:
            best = float(metric)
            best_epoch = epoch + 1
            best_state = deepcopy(model_core.state_dict())
            bad = 0
            if is_main_process():
                _save_deploy_artifacts(args, out_dir, model_core, in_norm, out_norm, best_epoch, best)
        else:
            bad += 1

        if is_main_process() and ((epoch + 1) == 1 or (epoch + 1) % 10 == 0 or bad == 0):
            vt = "nan" if val_loss is None else f"{val_loss:.6f}"
            print(f"[Fold {fold + 1}] Epoch {epoch + 1}/{cfg.epochs_stage1} train={train_loss:.6f} val={vt} best={best:.6f}@{best_epoch} lr={opt.param_groups[0]['lr']:.3e}", flush=True)

        if patience > 0 and bad >= patience:
            break

    if best_state is not None:
        model_core.load_state_dict(best_state)
    if is_main_process():
        model_path, norm_dir = _save_deploy_artifacts(args, out_dir, model_core, in_norm, out_norm, best_epoch, best)
        _write_train_log(Path(getattr(args, "train_log_path", "") or (out_dir / "train_log.csv")), logs)
        print(f"[OK] model saved to: {model_path}", flush=True)
        print(f"[OK] normalizers saved to: {norm_dir}", flush=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return best, train_hist, val_hist
