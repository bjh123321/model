from __future__ import annotations

import csv
import os
from copy import deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

from surrogate_model.config import Config
from surrogate_model.losses.masked_mse import MaskedMSE
from surrogate_model.models.blocks import MFDNN
from surrogate_model.train.dataloaders import build_single_fidelity_dataloaders
from surrogate_model.utils.dist_utils import is_main_process


PART_NAMES = ("freq", "damping", "residue_re", "residue_im")


def _select_device(args) -> torch.device:
    requested = str(getattr(args, "device", "auto")).lower()
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"Unsupported device option: {requested}")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Device was set to cuda, but torch.cuda.is_available() is False.")
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _cur_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def _maybe_set_epoch(dl, epoch: int):
    sampler = getattr(dl, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def _batch_to_device(batch, device):
    H, M, e, c, mask, yN = batch
    H = H.to(device, non_blocking=True)
    M = M.to(device, non_blocking=True)
    e = e.to(device, non_blocking=True)
    c = c.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    for key in yN:
        yN[key] = yN[key].to(device, non_blocking=True)
    return H, M, e, c, mask, yN


def _empty_parts():
    return {name: 0.0 for name in PART_NAMES}


def _parts_from_criterion(crit) -> dict[str, float]:
    stats = getattr(crit, "last_reg_stats", None) or {}
    return {name: float(stats.get(name, 0.0)) for name in PART_NAMES}


def _accum_parts(accum: dict[str, float], parts: dict[str, float], weight: int):
    for name in PART_NAMES:
        accum[name] += float(parts.get(name, 0.0)) * int(weight)


def _mean_parts(accum: dict[str, float], count: int) -> dict[str, float]:
    denom = max(int(count), 1)
    return {name: accum[name] / denom for name in PART_NAMES}


def _eval_loss(model, dl, crit, device):
    if dl is None or len(dl.dataset) == 0:
        return None, _empty_parts()
    model.eval()
    use_cuda = device.type == "cuda"
    value_sum, value_count = 0.0, 0
    part_sum = _empty_parts()
    with torch.no_grad():
        for batch in dl:
            H, M, e, c, mask, yN = _batch_to_device(batch, device)
            if use_cuda:
                with torch.amp.autocast(device_type="cuda"):
                    loss = crit(model(H, M, e, c), yN, mask)
            else:
                loss = crit(model(H, M, e, c), yN, mask)
            bs = H.size(0)
            value_sum += float(loss.item()) * bs
            value_count += bs
            _accum_parts(part_sum, _parts_from_criterion(crit), bs)

    t_sum = torch.tensor(value_sum, device=device)
    t_count = torch.tensor(value_count, device=device, dtype=torch.float32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_count, op=dist.ReduceOp.SUM)
    if t_count.item() <= 0:
        return None, _empty_parts()
    return (t_sum / t_count).item(), _mean_parts(part_sum, value_count)


def _write_train_log(path: Path, rows: list[dict[str, float]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "epoch",
        "loss_train_total",
        "loss_train_freq",
        "loss_train_damping",
        "loss_train_residue_re",
        "loss_train_residue_im",
        "loss_val_total",
        "loss_val_freq",
        "loss_val_damping",
        "loss_val_residue_re",
        "loss_val_residue_im",
        "lr",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _make_criterion(out_norm, device):
    return MaskedMSE(
        mean=torch.as_tensor(out_norm.mean, dtype=torch.float32, device=device),
        std=torch.as_tensor(out_norm.std, dtype=torch.float32, device=device),
        w_alpha=1.0,
        w_omega=1.0,
        w_rho=1.0,
        w_sin_theta=1.0,
        w_cos_theta=1.0,
        presence_weight=0.0,
        cardinality_weight=0.0,
        residue_target="fixed",
    )


def _save_deploy_artifacts(args, out_dir: Path, model_core: MFDNN, in_norm, out_norm, epoch: int, metric: float):
    model_save_path = Path(getattr(args, "model_save_path", "") or (out_dir / f"{out_dir.name}.pth"))
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    normalizer_dir = Path(getattr(args, "normalizer_dir", "") or out_dir.parent)
    normalizer_dir.mkdir(parents=True, exist_ok=True)

    in_norm.save(normalizer_dir / "Normalizer_Input.pth")
    out_norm.save(normalizer_dir / "Normalizer_Output.pth")
    torch.save(
        {
            "stage": "fixed_single_fidelity",
            "epoch": int(epoch),
            "best_metric": float(metric),
            "model_state_dict": model_core.state_dict(),
            "residue_target": "fixed",
            "K_max": int(model_core.cfg.K_max),
        },
        model_save_path,
    )
    return model_save_path, normalizer_dir


def train_one_fold(fold: int, cfg: Config, real_parsed, args):
    debug_n = int(getattr(args, "debug_overfit_n", 0) or 0)
    if debug_n > 0:
        cfg = deepcopy(cfg)
        cfg.use_val_split = False
        cfg.early_stop_patience_s1 = 0
        real_parsed = list(real_parsed)[:debug_n]
        if is_main_process():
            print(f"[DEBUG] Overfit mode enabled: using first {len(real_parsed)} samples, validation disabled.", flush=True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = _select_device(args)
    if is_main_process():
        if device.type == "cuda":
            print(f"[Device] Using GPU: {torch.cuda.get_device_name(device)} ({device})", flush=True)
        else:
            print("[Device] Using CPU.", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dl, val_dl, n_train, n_val, out_norm, in_norm = build_single_fidelity_dataloaders(
        real_parsed, cfg, args.kfold, fold, device, out_dir
    )
    if is_main_process():
        print(
            f"[Fixed-SF] real-only low training: train_rows={n_train}, val_rows={n_val}, "
            f"K_max={cfg.K_max}, target=fixed",
            flush=True,
        )

    cfg.embed_excitation_num = max(
        int(getattr(cfg, "embed_excitation_num", 0)),
        max(int(p.exc_id) for p in real_parsed) + 1,
    )
    cfg.embed_channel_num = max(
        int(getattr(cfg, "embed_channel_num", 0)),
        max(int(p.ch_id) for p in real_parsed) + 1,
    )

    model_core = MFDNN(cfg).to(device)
    for p in model_core.parameters():
        p.requires_grad_(True)

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1 and device.type == "cuda":
        model = nn.parallel.DistributedDataParallel(
            model_core,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    else:
        model = model_core

    crit = _make_criterion(out_norm, device)
    optimizer = torch.optim.AdamW(model_core.parameters(), lr=cfg.lr_low_adam, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=getattr(cfg, "lr_factor_s1", 0.5),
        patience=getattr(cfg, "lr_patience_s1", 20),
        threshold=getattr(cfg, "early_stop_min_delta", 0.0),
        threshold_mode="abs",
        min_lr=getattr(cfg, "lr_min_s1", 1e-6),
        eps=1e-12,
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    use_cuda = device.type == "cuda"

    best_metric = float("inf")
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    log_rows: list[dict[str, float]] = []
    train_hist: list[float] = []
    val_hist: list[float] = []
    patience = int(getattr(cfg, "early_stop_patience_s1", 500))
    min_delta = float(getattr(cfg, "early_stop_min_delta", 0.0))

    for epoch in range(int(cfg.epochs_stage1)):
        _maybe_set_epoch(train_dl, epoch)
        model.train()
        loss_sum, sample_count = 0.0, 0
        train_part_sum = _empty_parts()
        for batch in train_dl:
            H, M, e, c, mask, yN = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            if use_cuda:
                with torch.amp.autocast(device_type="cuda"):
                    loss = crit(model(H, M, e, c), yN, mask)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_core.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = crit(model(H, M, e, c), yN, mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_core.parameters(), max_norm=5.0)
                optimizer.step()

            bs = H.size(0)
            loss_sum += float(loss.detach().item()) * bs
            sample_count += bs
            _accum_parts(train_part_sum, _parts_from_criterion(crit), bs)

        train_loss = loss_sum / max(sample_count, 1)
        train_parts = _mean_parts(train_part_sum, sample_count)
        val_loss, val_parts = _eval_loss(model, val_dl, crit, device)
        metric = val_loss if val_loss is not None else train_loss
        scheduler.step(metric)

        train_hist.append(train_loss)
        val_hist.append(float("nan") if val_loss is None else val_loss)
        row = {
            "epoch": epoch + 1,
            "loss_train_total": train_loss,
            "loss_train_freq": train_parts["freq"],
            "loss_train_damping": train_parts["damping"],
            "loss_train_residue_re": train_parts["residue_re"],
            "loss_train_residue_im": train_parts["residue_im"],
            "loss_val_total": "" if val_loss is None else val_loss,
            "loss_val_freq": "" if val_loss is None else val_parts["freq"],
            "loss_val_damping": "" if val_loss is None else val_parts["damping"],
            "loss_val_residue_re": "" if val_loss is None else val_parts["residue_re"],
            "loss_val_residue_im": "" if val_loss is None else val_parts["residue_im"],
            "lr": _cur_lr(optimizer),
        }
        log_rows.append(row)

        improved = metric < (best_metric - min_delta)
        if improved:
            best_metric = float(metric)
            best_epoch = epoch + 1
            best_state = deepcopy(model_core.state_dict())
            bad_epochs = 0
            if is_main_process():
                _save_deploy_artifacts(args, out_dir, model_core, in_norm, out_norm, best_epoch, best_metric)
        else:
            bad_epochs += 1

        if is_main_process() and ((epoch + 1) == 1 or (epoch + 1) % 10 == 0 or improved):
            val_text = "nan" if val_loss is None else f"{val_loss:.6f}"
            print(
                f"[Fold {fold + 1}] Fixed-SF Epoch {epoch + 1}/{cfg.epochs_stage1} "
                f"loss_real={train_loss:.6f} val_real={val_text} "
                f"best={best_metric:.6f}@{best_epoch} lr={_cur_lr(optimizer):.3e} "
                f"train_parts=freq:{train_parts['freq']:.4f},damp:{train_parts['damping']:.4f},"
                f"re:{train_parts['residue_re']:.4f},im:{train_parts['residue_im']:.4f} "
                f"val_parts=freq:{val_parts['freq']:.4f},damp:{val_parts['damping']:.4f},"
                f"re:{val_parts['residue_re']:.4f},im:{val_parts['residue_im']:.4f}",
                flush=True,
            )

        if patience > 0 and bad_epochs >= patience:
            if is_main_process():
                print(
                    f"[Fixed-SF] Early stop at epoch {epoch + 1}; best_epoch={best_epoch}, "
                    f"best_metric={best_metric:.6f}",
                    flush=True,
                )
            break

    if best_state is not None:
        model_core.load_state_dict(best_state)

    if is_main_process():
        model_path, norm_dir = _save_deploy_artifacts(
            args, out_dir, model_core, in_norm, out_norm, best_epoch, best_metric
        )
        train_log_path = Path(getattr(args, "train_log_path", "") or (out_dir / "train_log.csv"))
        _write_train_log(train_log_path, log_rows)
        print(f"[OK] Fixed single-fidelity model saved to: {model_path}", flush=True)
        print(f"[OK] Normalizers saved to: {norm_dir}", flush=True)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    return best_metric, train_hist, val_hist
