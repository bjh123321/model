from __future__ import annotations

import csv
import os
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from surrogate_model.config import Config
from surrogate_model.losses.masked_mse import MaskedHungarianMSE, MaskedMSE
from surrogate_model.models.blocks import MFDNN
from surrogate_model.train.dataloaders import build_dataloaders
from surrogate_model.utils.dist_utils import dist_mean_scalar, is_main_process
from surrogate_model.utils.pcgrad import (
    allreduce_grads_inplace,
    clone_grads,
    maybe_no_sync,
    pcgrad_two_tasks,
    set_grads,
    zero_grads,
)


def _cur_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


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


def _eval_loss(model, dl, crit, device, forward_high: bool):
    if dl is None:
        return None
    model.eval()
    use_cuda = device.type == "cuda"
    value_sum, value_count = 0.0, 0
    with torch.no_grad():
        for H, M, e, c, mask, yN in dl:
            H = H.to(device, non_blocking=True)
            M = M.to(device, non_blocking=True)
            e = e.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            for k in yN:
                yN[k] = yN[k].to(device, non_blocking=True)

            if use_cuda:
                with torch.amp.autocast(device_type="cuda"):
                    outL, outH = model(H, M, e, c)
                    out = outH if forward_high else outL
                    loss = crit(out, yN, mask)
            else:
                outL, outH = model(H, M, e, c)
                out = outH if forward_high else outL
                loss = crit(out, yN, mask)

            bs = H.size(0)
            value_sum += float(loss.item()) * bs
            value_count += bs

    t_sum = torch.tensor(value_sum, device=device)
    t_count = torch.tensor(value_count, device=device, dtype=torch.float32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_count, op=dist.ReduceOp.SUM)

    if t_count.item() <= 0:
        return None
    return (t_sum / t_count).item()


def _accum_stat(stat_accum: dict, new_stat: dict):
    for key, value in new_stat.items():
        stat_accum[key] = stat_accum.get(key, 0.0) + float(value)


def _crit_part_stats(crit) -> dict:
    stat = {}
    reg = getattr(crit, "last_reg_stats", None)
    cls = getattr(crit, "last_cls_stats", None)
    if reg:
        stat.update(reg)
    if cls:
        stat.update(cls)
    if stat:
        stat["cls_weighted"] = (
            float(getattr(crit, "pres_w", 0.0)) * float(stat.get("bce", 0.0))
            + float(getattr(crit, "card_w", 0.0)) * float(stat.get("card", 0.0))
        )
    return stat


def _make_weights(std_np: np.ndarray):
    eps = 1e-8
    w = 1.0 / ((np.asarray(std_np, dtype=np.float64) ** 2) + eps)
    w = w / (w.mean() + eps)
    return float(w[0]), float(w[1]), float(w[2]), float(w[3])


def _load_stage1_low_from_fold0(model_core: nn.Module, ckpt_dir: Path) -> bool:
    candidates = [
        ckpt_dir / "base.pth",
        *sorted(ckpt_dir.glob("best_stage1_fold0*.pth")),
        *sorted(ckpt_dir.glob("checkpoint_stage1_fold0*.pth")),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            state = torch.load(path, map_location="cpu")
            state = state.get("model_state_dict", state)
            low_state = {k: v for k, v in state.items() if k.startswith("low.")}
            if not low_state:
                continue
            missing, unexpected = model_core.load_state_dict(low_state, strict=False)
            if is_main_process():
                print(
                    f"  [S1] Reuse low.* from {path.name} "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})",
                    flush=True,
                )
            return True
        except Exception as exc:
            if is_main_process():
                print(f"  [S1] Failed loading {path.name}: {exc}", flush=True)
    if is_main_process():
        print("  [S1] No reusable low.* checkpoint found; will train Stage-1.", flush=True)
    return False


def _write_train_log(path: Path, sim_hist, real_hist, total_hist, val_hist, lr_hist):
    path.parent.mkdir(parents=True, exist_ok=True)
    max_len = max(len(sim_hist), len(real_hist), len(total_hist), len(val_hist), len(lr_hist), 0)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss_total", "loss_sim", "loss_real", "loss_val_real", "lr"])
        for i in range(max_len):
            writer.writerow(
                [
                    i + 1,
                    total_hist[i] if i < len(total_hist) else "",
                    sim_hist[i] if i < len(sim_hist) else "",
                    real_hist[i] if i < len(real_hist) else "",
                    val_hist[i] if i < len(val_hist) else "",
                    lr_hist[i] if i < len(lr_hist) else "",
                ]
            )


def _build_criteria(cfg: Config, mean_sim, std_sim, mean_real, std_real):
    cls = MaskedHungarianMSE if cfg.use_hungarian_loss else MaskedMSE
    common = dict(
        w_alpha=1.0,
        w_omega=1.0,
        w_rho=1.0,
        w_sin_theta=1.0,
        w_cos_theta=1.0,
        presence_weight=cfg.presence_loss_weight,
        cardinality_weight=cfg.cardinality_loss_weight,
        residue_target=getattr(cfg, "residue_target", "polar"),
    )
    return cls(mean=mean_sim, std=std_sim, **common), cls(mean=mean_real, std=std_real, **common)


def train_one_fold(fold: int, cfg: Config, sim_parsed, real_parsed, args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = _select_device(args)
    if is_main_process():
        if device.type == "cuda":
            print(f"[Device] Using GPU: {torch.cuda.get_device_name(device)} ({device})", flush=True)
        else:
            print("[Device] Using CPU.", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (
        sim_dl,
        real_dl,
        sim_val_dl,
        val_dl,
        _n_sim,
        _n_real,
        out_norm_sim,
        out_norm_real,
        in_norm_deploy,
    ) = build_dataloaders(sim_parsed, real_parsed, cfg, args.kfold, fold, device, out_dir)

    def _maybe_set_epoch(dl, epoch: int):
        sampler = getattr(dl, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    if sim_parsed or real_parsed:
        cfg.embed_excitation_num = max(
            int(getattr(cfg, "embed_excitation_num", 0)),
            max(int(p.exc_id) for p in (sim_parsed + real_parsed)) + 1,
        )

    model_core = MFDNN(cfg).to(device)
    for name, param in model_core.named_parameters():
        if not name.startswith("low."):
            param.requires_grad_(False)

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1 and device.type == "cuda":
        model = nn.parallel.DistributedDataParallel(
            model_core,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    else:
        model = model_core

    mean_sim = torch.tensor(out_norm_sim.mean, dtype=torch.float32, device=device)
    std_sim = torch.tensor(out_norm_sim.std, dtype=torch.float32, device=device)
    mean_real = torch.tensor(out_norm_real.mean, dtype=torch.float32, device=device)
    std_real = torch.tensor(out_norm_real.std, dtype=torch.float32, device=device)
    crit_s1, crit_s2 = _build_criteria(cfg, mean_sim, std_sim, mean_real, std_real)

    stage1_once = bool(getattr(args, "stage1_once", 1))
    did_reuse = stage1_once and _load_stage1_low_from_fold0(model_core, out_dir)

    use_cuda = device.type == "cuda"
    loss_hist_s1: list[float] = []
    lr_hist_s1: list[float] = []

    if not did_reuse:
        for p in model_core.lin.parameters():
            p.requires_grad_(False)
        for p in model_core.adaptation.parameters():
            p.requires_grad_(False)

        opt1 = torch.optim.AdamW(model_core.low.parameters(), lr=cfg.lr_low_adam, weight_decay=cfg.weight_decay)
        scaler1 = torch.amp.GradScaler("cuda") if use_cuda else None
        sched1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt1,
            mode="min",
            factor=getattr(cfg, "lr_factor_s1", 0.5),
            patience=getattr(cfg, "lr_patience_s1", 20),
            threshold=getattr(cfg, "early_stop_min_delta", 0.0),
            threshold_mode="abs",
            min_lr=getattr(cfg, "lr_min_s1", 1e-6),
            eps=1e-12,
        )

        best_sim = float("inf")
        no_improve = 0
        for epoch in range(cfg.epochs_stage1):
            model.train()
            _maybe_set_epoch(sim_dl, epoch)
            sum_loss, seen = 0.0, 0
            comp_accum, comp_count = {}, 0

            for H, M, e, c, mask, yN in sim_dl:
                H = H.to(device, non_blocking=True)
                M = M.to(device, non_blocking=True)
                e = e.to(device, non_blocking=True)
                c = c.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                for k in yN:
                    yN[k] = yN[k].to(device, non_blocking=True)

                opt1.zero_grad(set_to_none=True)
                if use_cuda:
                    with torch.amp.autocast(device_type="cuda"):
                        outL, _ = model(H, M, e, c)
                        loss = crit_s1(outL, yN, mask)
                    stat = _crit_part_stats(crit_s1)
                    if stat:
                        _accum_stat(comp_accum, stat)
                        comp_count += 1
                    scaler1.scale(loss).backward()
                    scaler1.unscale_(opt1)
                    torch.nn.utils.clip_grad_norm_(model_core.low.parameters(), max_norm=1.0)
                    scaler1.step(opt1)
                    scaler1.update()
                else:
                    outL, _ = model(H, M, e, c)
                    loss = crit_s1(outL, yN, mask)
                    stat = _crit_part_stats(crit_s1)
                    if stat:
                        _accum_stat(comp_accum, stat)
                        comp_count += 1
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model_core.low.parameters(), max_norm=1.0)
                    opt1.step()

                bs = H.size(0)
                sum_loss += float(loss.item()) * bs
                seen += bs

            epoch_loss = sum_loss / max(1, seen)
            epoch_loss_w = dist_mean_scalar(epoch_loss, device)
            loss_hist_s1.append(float(epoch_loss_w))
            lr_hist_s1.append(_cur_lr(opt1))

            val_loss_w = None
            if bool(getattr(cfg, "use_val_split", False)):
                val_loss = _eval_loss(model, sim_val_dl, crit_s1, device, forward_high=False)
                if val_loss is not None:
                    val_loss_w = dist_mean_scalar(val_loss, device)
            metric_s1 = float(val_loss_w) if val_loss_w is not None else float(epoch_loss_w)

            if is_main_process():
                prev_lr = _cur_lr(opt1)
            sched1.step(metric_s1)
            if is_main_process():
                val_text = f" val_sim={float(val_loss_w):.6f}" if val_loss_w is not None else ""
                print(
                    f"[Fold {fold + 1}] S1 Epoch {epoch + 1}/{cfg.epochs_stage1} "
                    f"sim_loss={epoch_loss_w:.6f}{val_text} lr={_cur_lr(opt1):.3e}",
                    flush=True,
                )
                if _cur_lr(opt1) < prev_lr:
                    print(f"  [S1] LR reduced: {prev_lr:.3e} -> {_cur_lr(opt1):.3e}", flush=True)
                if comp_count > 0:
                    avg = {k: v / comp_count for k, v in comp_accum.items()}
                    print(
                        f"[S1 parts] alpha={avg.get('alpha', 0.0):.4f}, "
                        f"omega={avg.get('omega', 0.0):.4f}, rho={avg.get('rho', 0.0):.4f}, "
                        f"sin_theta={avg.get('sin_theta', 0.0):.4f}, "
                        f"cos_theta={avg.get('cos_theta', 0.0):.4f}, "
                        f"BCE={avg.get('bce', 0.0):.4f}, CARD={avg.get('card', 0.0):.4f}",
                        flush=True,
                    )

            if metric_s1 < best_sim - cfg.early_stop_min_delta:
                best_sim = metric_s1
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg.early_stop_patience_s1:
                    if is_main_process():
                        print("  Early stop Stage1", flush=True)
                    break

        if is_main_process():
            torch.save(
                {
                    "fold": fold,
                    "stage": 1,
                    "epoch": int(locals().get("epoch", -1)) + 1,
                    "model_state_dict": model_core.state_dict(),
                    "optimizer_state_dict": opt1.state_dict(),
                    "loss_sim": float(loss_hist_s1[-1]) if loss_hist_s1 else float("nan"),
                    "loss_history_sim": loss_hist_s1,
                    "lr_history": lr_hist_s1,
                },
                out_dir / "base.pth",
            )
    elif is_main_process():
        print(f"[Fold {fold + 1}] S1 skipped (reuse fold0).", flush=True)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if is_main_process():
        normalizer_dir = Path(getattr(args, "normalizer_dir", out_dir.parent))
        normalizer_dir.mkdir(parents=True, exist_ok=True)
        in_norm_deploy.save(normalizer_dir / "Normalizer_Input.pth")
        out_norm_real.save(normalizer_dir / "Normalizer_Output.pth")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    crit_s1.pres_w = 0.0
    crit_s1.card_w = 0.0

    for p in model_core.low.parameters():
        p.requires_grad_(True)
    for p in model_core.lin.parameters():
        p.requires_grad_(True)
    for p in model_core.adaptation.parameters():
        p.requires_grad_(True)

    stage2_params = [
        p for p in (
            list(model_core.low.parameters())
            + list(model_core.lin.parameters())
            + list(model_core.adaptation.parameters())
        )
        if p.requires_grad
    ]
    param_groups = [{"params": stage2_params, "lr": cfg.lr_high_adam, "name": "low_lin_adaptation"}]
    opt2 = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    if is_main_process():
        group_text = ", ".join(
            f"{group.get('name', 'group')}={sum(p.numel() for p in group['params'])}@{group['lr']:.3e}"
            for group in param_groups
        )
        print(
            f"[S2] low+lin+adaptation trainable; {group_text}",
            flush=True,
        )

    use_sim_grad_stage2 = any(p.requires_grad for p in model_core.low.parameters())
    sched2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt2,
        mode="min",
        factor=getattr(cfg, "lr_factor_s2", 0.5),
        patience=getattr(cfg, "lr_patience_s2", 20),
        threshold=getattr(cfg, "early_stop_min_delta", 0.0),
        threshold_mode="abs",
        min_lr=getattr(cfg, "lr_min_s2", 1e-6),
        eps=1e-12,
    )

    w_sim, w_real = 0.2, 1.0
    best_metric = float("inf")
    best_state_dict = None
    best_epoch_s2 = 0
    no_improve = 0
    sim_iter = None
    total_hist, sim_hist, real_hist, val_hist, lr_hist_s2 = [], [], [], [], []
    scaler2 = torch.amp.GradScaler("cuda") if use_cuda else None

    for epoch in range(cfg.epochs_stage2):
        model.train()
        _maybe_set_epoch(real_dl, epoch)
        _maybe_set_epoch(sim_dl, epoch)
        if sim_iter is None:
            sim_iter = cycle(sim_dl)

        sum_sim, sum_real, cnt_sim, cnt_real = 0.0, 0.0, 0, 0
        real_part_accum, real_part_count = {}, 0
        params = [p for p in model_core.parameters() if p.requires_grad]

        for H_r, M_r, e_r, c_r, mask_r, yNr in real_dl:
            H_s, M_s, e_s, c_s, mask_s, yNs = next(sim_iter)

            H_r = H_r.to(device, non_blocking=True)
            M_r = M_r.to(device, non_blocking=True)
            e_r = e_r.to(device, non_blocking=True)
            c_r = c_r.to(device, non_blocking=True)
            mask_r = mask_r.to(device, non_blocking=True)
            for k in yNr:
                yNr[k] = yNr[k].to(device, non_blocking=True)

            H_s = H_s.to(device, non_blocking=True)
            M_s = M_s.to(device, non_blocking=True)
            e_s = e_s.to(device, non_blocking=True)
            c_s = c_s.to(device, non_blocking=True)
            mask_s = mask_s.to(device, non_blocking=True)
            for k in yNs:
                yNs[k] = yNs[k].to(device, non_blocking=True)

            opt2.zero_grad(set_to_none=True)

            if use_cuda:
                with torch.amp.autocast(device_type="cuda"):
                    _, outH_r = model(H_r, M_r, e_r, c_r)
                    loss_real = crit_s2(outH_r, yNr, mask_r)
                    real_stat = _crit_part_stats(crit_s2)
                    if use_sim_grad_stage2:
                        outL_s = model_core.forward_low(H_s, M_s, e_s, c_s)
                    else:
                        with torch.no_grad():
                            outL_s = model_core.forward_low(H_s, M_s, e_s, c_s)
                    loss_sim = crit_s1(outL_s, yNs, mask_s)
            else:
                _, outH_r = model(H_r, M_r, e_r, c_r)
                loss_real = crit_s2(outH_r, yNr, mask_r)
                real_stat = _crit_part_stats(crit_s2)
                if use_sim_grad_stage2:
                    outL_s = model_core.forward_low(H_s, M_s, e_s, c_s)
                else:
                    with torch.no_grad():
                        outL_s = model_core.forward_low(H_s, M_s, e_s, c_s)
                loss_sim = crit_s1(outL_s, yNs, mask_s)

            if not torch.isfinite(loss_real):
                print("[WARN] loss_real is not finite. Skip step.", flush=True)
                continue

            if not use_sim_grad_stage2:
                if use_cuda:
                    scaler2.scale(w_real * loss_real).backward()
                    scaler2.unscale_(opt2)
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    scaler2.step(opt2)
                    scaler2.update()
                else:
                    (w_real * loss_real).backward()
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    opt2.step()
            elif use_cuda:
                with maybe_no_sync(model):
                    scaler2.scale(w_sim * loss_sim).backward(retain_graph=True)
                g_sim_scaled = clone_grads(params)
                zero_grads(params)
                with maybe_no_sync(model):
                    scaler2.scale(w_real * loss_real).backward(retain_graph=True)
                g_real_scaled = clone_grads(params)

                scale = scaler2.get_scale()
                g_sim = [None if g is None else (g.float() / scale) for g in g_sim_scaled]
                g_real = [None if g is None else (g.float() / scale) for g in g_real_scaled]
                g_comb = pcgrad_two_tasks(g_sim, g_real)

                total_sq = 0.0
                for grad in g_comb:
                    if grad is not None:
                        total_sq += grad.pow(2).sum()
                total_norm = total_sq.sqrt()
                clip_coef = (1.0 / (total_norm + 1e-12)).clamp(max=1.0)
                if clip_coef < 1.0:
                    g_comb = [None if g is None else g * clip_coef for g in g_comb]

                set_grads(params, [None if g is None else g * scale for g in g_comb])
                allreduce_grads_inplace(params)
                scaler2.step(opt2)
                scaler2.update()
            else:
                with maybe_no_sync(model):
                    (w_sim * loss_sim).backward(retain_graph=True)
                g_sim = clone_grads(params)
                zero_grads(params)
                with maybe_no_sync(model):
                    (w_real * loss_real).backward(retain_graph=True)
                g_real = clone_grads(params)
                set_grads(params, pcgrad_two_tasks(g_sim, g_real))
                allreduce_grads_inplace(params)
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                opt2.step()

            bs_r, bs_s = H_r.size(0), H_s.size(0)
            sum_real += float(loss_real.item()) * bs_r
            cnt_real += bs_r
            sum_sim += float(loss_sim.item()) * bs_s
            cnt_sim += bs_s
            if real_stat:
                _accum_stat(real_part_accum, real_stat)
                real_part_count += 1

        t_sim = torch.tensor(sum_sim, device=device)
        t_cnt_s = torch.tensor(cnt_sim, device=device, dtype=torch.float32)
        t_real = torch.tensor(sum_real, device=device)
        t_cnt_r = torch.tensor(cnt_real, device=device, dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(t_sim, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_cnt_s, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_real, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_cnt_r, op=dist.ReduceOp.SUM)

        sim_epoch = (t_sim / t_cnt_s).item() if t_cnt_s.item() > 0 else 0.0
        real_epoch = (t_real / t_cnt_r).item() if t_cnt_r.item() > 0 else 0.0
        total_epoch = float(w_sim * sim_epoch + w_real * real_epoch)

        val_real_epoch = None
        if bool(getattr(cfg, "use_val_split", False)):
            val_real_epoch = _eval_loss(model, val_dl, crit_s2, device, forward_high=True)
        metric_s2 = float(val_real_epoch) if val_real_epoch is not None else float(total_epoch)

        if is_main_process():
            prev_lr2 = _cur_lr(opt2)
        sched2.step(metric_s2)
        if is_main_process() and _cur_lr(opt2) < prev_lr2:
            print(f"  [S2] LR reduced: {prev_lr2:.3e} -> {_cur_lr(opt2):.3e}", flush=True)

        sim_hist.append(float(sim_epoch))
        real_hist.append(float(real_epoch))
        total_hist.append(float(total_epoch))
        val_hist.append(float(val_real_epoch) if val_real_epoch is not None else float("nan"))
        lr_hist_s2.append(_cur_lr(opt2))

        if is_main_process():
            val_text = f" val_real={float(val_real_epoch):.6f}" if val_real_epoch is not None else " val_real=nan"
            print(
                f"[Fold {fold + 1}] S2 Epoch {epoch + 1}/{cfg.epochs_stage2} "
                f"total={total_epoch:.6f} sim={sim_epoch:.6f} real={real_epoch:.6f}"
                f"{val_text} lr={_cur_lr(opt2):.3e}",
                flush=True,
            )
            if real_part_count > 0:
                avg = {k: v / real_part_count for k, v in real_part_accum.items()}
                print(
                    f"[S2 train-real parts] alpha={avg.get('alpha', 0.0):.4f}, "
                    f"omega={avg.get('omega', 0.0):.4f}, rho={avg.get('rho', 0.0):.4f}, "
                    f"sin_theta={avg.get('sin_theta', 0.0):.4f}, "
                    f"cos_theta={avg.get('cos_theta', 0.0):.4f}, "
                    f"BCE={avg.get('bce', 0.0):.4f}, CARD={avg.get('card', 0.0):.4f}, "
                    f"CLSw={avg.get('cls_weighted', 0.0):.4f}",
                    flush=True,
                )

        if metric_s2 < best_metric - cfg.early_stop_min_delta:
            best_metric = metric_s2
            best_epoch_s2 = epoch + 1
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model_core.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.early_stop_patience_s2:
                if is_main_process():
                    print("  Early stop Stage2", flush=True)
                break

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if best_state_dict is not None:
        model_core.load_state_dict(best_state_dict)

    if is_main_process():
        save_path = Path(getattr(args, "model_save_path", out_dir / (out_dir.name + ".pth")))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "fold": fold,
                "stage": 2,
                "epoch": int(locals().get("epoch", -1)) + 1,
                "best_epoch": best_epoch_s2,
                "best_metric": float(best_metric),
                "model_state_dict": model_core.state_dict(),
                "optimizer_state_dict": opt2.state_dict(),
                "loss_total": float(total_hist[-1]) if total_hist else float("nan"),
                "loss_sim": float(sim_hist[-1]) if sim_hist else float("nan"),
                "loss_real": float(real_hist[-1]) if real_hist else float("nan"),
                "loss_val_real": float(val_hist[-1]) if val_hist else float("nan"),
                "loss_history_total": total_hist,
                "loss_history_sim": sim_hist,
                "loss_history_real": real_hist,
                "loss_history_val_real": val_hist,
                "lr_history": lr_hist_s2,
            },
            save_path,
        )
        train_log_path = getattr(args, "train_log_path", None)
        if train_log_path:
            _write_train_log(Path(train_log_path), sim_hist, real_hist, total_hist, val_hist, lr_hist_s2)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    eval_cls = MaskedHungarianMSE if cfg.use_hungarian_loss else MaskedMSE
    crit_eval = eval_cls(mean=mean_real, std=std_real)
    vloss = _eval_loss(model, val_dl, crit_eval, device, forward_high=True)
    if vloss is None:
        vloss = float("nan")
    return vloss, best_metric, sim_hist, real_hist, total_hist
