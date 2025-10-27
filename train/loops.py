from __future__ import annotations
import os, json
from pathlib import Path
from itertools import cycle
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from ..config import Config
from ..models.blocks import MFDNN
from ..losses.masked_mse import MaskedMSE, MaskedHungarianMSE  # [已含]
from ..utils.dist_utils import is_main_process, dist_mean_scalar
from ..data.normalizer import Normalizer
from .dataloaders import build_dataloaders


def _cur_lr(optimizer):  # [Added]
    return optimizer.param_groups[0]['lr']

def _mse_per_param_z(pred, yN, mask, mean, std, use_presence: bool):
    """
    统计四个参数在 z-score 空间的 MSE 分项；可选统计 presence BCE 与 cardinality。
    返回 dict：{'alpha':..., 'omega':..., 'gain':..., 'zero':..., 'bce':..., 'card':...}
    """
    # 取预测（物理域）-> z-score
    # pred['alpha'/'omega'/...] 形状：[B, K_max]
    eps = 1e-8
    P = torch.stack([pred['alpha'], pred['omega'], pred['gain'], pred['zero']], dim=1)  # [B,4,K]
    T = torch.stack([yN['alpha'], yN['omega'], yN['gain'], yN['zero']], dim=1)          # [B,4,K]
    # mean/std 形状：[4]，广播到 [B,4,K]
    PN = (P - mean.view(1,4,1)) / (std.view(1,4,1).clamp_min(1e-8))
    # TN = (T - mean.view(1,4,1)) / (std.view(1,4,1).clamp_min(1e-8))
    # PN = P
    TN = T

    # mask: [B,K] -> [B,1,K] 参与四个通道
    m = (mask > 0.5).float().unsqueeze(1)  # [B,1,K]
    denom = m.sum(dim=(0,2)).clamp_min(1.0)  # [1]每通道一样，用总数归一

    err = (PN - TN).pow(2) * m  # [B,4,K]
    mse_ch = err.sum(dim=(0,2)) / denom  # [4] -> α,ω,gain,zero

    out = {
        'alpha': mse_ch[0].item(),
        'omega': mse_ch[1].item(),
        'gain' : mse_ch[2].item(),
        'zero' : mse_ch[3].item(),
    }

    if use_presence:
        # BCE: 用 u_pres 做 logits（若没提供 u_pres，就退化用 p_exist 的 logit 近似）
        if 'u_pres' in pred:
            logits = pred['u_pres']              # [B,K]
            target = (mask > 0.5).float()
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target, reduction='none'
            )
            bce = bce.mean().item()
        else:
            # 退化：p_exist∈(0,1)，避免数值问题
            p = pred.get('p_exist', torch.sigmoid(torch.zeros_like(mask)))
            p = p.clamp(1e-6, 1-1e-6)
            target = (mask > 0.5).float()
            bce = -(target*torch.log(p) + (1-target)*torch.log(1-p)).mean().item()
        # Cardinality：预测个数 vs 目标个数（用 MSE）
        pred_cnt = (torch.sigmoid(pred['u_pres']) if 'u_pres' in pred else pred.get('p_exist', 0.0)).sum(dim=1)  # [B]
        true_cnt = (mask > 0.5).float().sum(dim=1)  # [B]
        card = (pred_cnt - true_cnt).pow(2).mean().item()

        out['bce'] = bce
        out['card'] = card

    return out


def _accum_stat(stat_accum: dict, new_stat: dict):
    for k, v in new_stat.items():
        stat_accum[k] = stat_accum.get(k, 0.0) + float(v)




def _make_weights(std_np: np.ndarray):
    eps = 1e-8
    w = 1.0 / ((np.asarray(std_np, dtype=np.float64) ** 2) + eps)
    w = w / (w.mean() + eps)
    return float(w[0]), float(w[1]), float(w[2]), float(w[3])

def train_one_fold(fold: int, cfg: Config, sim_parsed, real_parsed, args):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True) # parents=True表示如果上级目录不存在，会自动一并创建，exist_ok=True表示如果目录已存在，不会报错

    sim_dl, real_dl, val_dl, N_sim, N_real = build_dataloaders(
        sim_parsed, real_parsed, cfg, args.kfold, fold, device, out_dir
    )

    # excitation vocab size
    max_exc = 0
    for p in (sim_parsed + real_parsed):
        if p.exc_id > max_exc:
            max_exc = p.exc_id
    cfg.embed_excitation_num = int(max_exc + 1)

    model_core = MFDNN(cfg).to(device)
    for name, p in model_core.named_parameters():
        if not name.startswith("low."):
            p.requires_grad = False
    model = nn.parallel.DistributedDataParallel(
        model_core, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
    )

    norm_sim  = Normalizer.load(out_dir / 'Normalizer_Output_Sim.pth')
    norm_real = Normalizer.load(out_dir / f'Normalizer_Output_Real_Fold{fold}.pth')

    mean_sim  = torch.tensor(norm_sim.mean,  dtype=torch.float32, device=device)
    std_sim   = torch.tensor(norm_sim.std,   dtype=torch.float32, device=device)
    mean_real = torch.tensor(norm_real.mean, dtype=torch.float32, device=device)
    std_real  = torch.tensor(norm_real.std,  dtype=torch.float32, device=device)

    # w_s1_alpha, w_s1_omega, w_s1_gain, w_s1_zero = _make_weights(norm_sim.std)
    # w_s2_alpha, w_s2_omega, w_s2_gain, w_s2_zero = _make_weights(norm_real.std)
    # 替换 _make_weights 的使用结果为等权
    w_s1_alpha = w_s1_omega = w_s1_gain = w_s1_zero = 1.0
    w_s2_alpha = w_s2_omega = w_s2_gain = w_s2_zero = 1.0

    pres_w = cfg.presence_loss_weight
    card_w = cfg.cardinality_loss_weight

    if cfg.use_hungarian_loss:
        crit_s1 = MaskedHungarianMSE(mean=mean_sim, std=std_sim,
                                     w_alpha=w_s1_alpha, w_omega=w_s1_omega,
                                     w_gain=w_s1_gain, w_zero=w_s1_zero,
                                     presence_weight=pres_w, cardinality_weight=card_w)
        crit_s2 = MaskedHungarianMSE(mean=mean_real, std=std_real,
                                     w_alpha=w_s2_alpha, w_omega=w_s2_omega,
                                     w_gain=w_s2_gain, w_zero=w_s2_zero,
                                     presence_weight=pres_w, cardinality_weight=card_w)
    else:
        crit_s1 = MaskedMSE(mean=mean_sim, std=std_sim,
                            w_alpha=w_s1_alpha, w_omega=w_s1_omega,
                            w_gain=w_s1_gain, w_zero=w_s1_zero,
                            presence_weight=pres_w, cardinality_weight=card_w)
        crit_s2 = MaskedMSE(mean=mean_real, std=std_real,
                            w_alpha=w_s2_alpha, w_omega=w_s2_omega,
                            w_gain=w_s2_gain, w_zero=w_s2_zero,
                            presence_weight=pres_w, cardinality_weight=card_w)

    # ============ Stage 1 ============
    for p in model_core.lin.parameters(): p.requires_grad_(False)
    for p in model_core.res.parameters(): p.requires_grad_(False)

    opt1 = torch.optim.Adam(model_core.low.parameters(), lr=cfg.lr_low_adam, weight_decay=cfg.weight_decay)
    use_cuda = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda') if use_cuda else None

    pat1   = getattr(cfg, 'lr_patience_s1', 50)         # 连续多少个 epoch 无提升就降 LR
    fac1   = getattr(cfg, 'lr_factor_s1', 0.5)         # 降低倍数（乘以该系数）
    minlr1 = getattr(cfg, 'lr_min_s1', 1e-6)           # LR 下限
    thr    = getattr(cfg, 'early_stop_min_delta', 0.0) # 认为“有提升”的阈值
    sched1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt1, mode='min', factor=fac1, patience=pat1,
    threshold=thr, threshold_mode='abs', cooldown=0,
    min_lr=minlr1, eps=1e-12
    )

    best_sim = float('inf'); no_improve = 0
    loss_hist = []
    lr_hist = []

    for epoch in range(cfg.epochs_stage1):
        model.train()
        sim_dl.sampler.set_epoch(epoch)
        sum_loss = 0.0; seen = 0
        comp_accum = {}   # 新增：分项损失累加
        comp_count = 0    # 新增：样本计数（以 batch 为单位或以样本为单位都可，这里用 batch 计数做均值）


        for H, M, e, c, mask, yN in sim_dl:
            H = H.to(device, non_blocking=True)
            M = M.to(device, non_blocking=True)
            e = e.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            for k in yN: yN[k] = yN[k].to(device, non_blocking=True)

            opt1.zero_grad(set_to_none=True)
            if use_cuda:
                with torch.amp.autocast(device_type='cuda'):
                    outL, _ = model(H, M, e, c)
                    loss = crit_s1(outL, yN, mask)
                    # === 新增分项统计（S1，低保真） ===
                with torch.no_grad():
                    stat = _mse_per_param_z(
                        pred=outL, yN=yN, mask=mask,
                        mean=mean_sim, std=std_sim,
                        use_presence=cfg.use_presence_head
                    )
                    _accum_stat(comp_accum, stat)
                    comp_count += 1
                scaler.scale(loss).backward()
                scaler.step(opt1)
                scaler.update()
            else:
                outL, _ = model(H, M, e, c)
                loss = crit_s1(outL, yN, mask)
                with torch.no_grad():
                    stat = _mse_per_param_z(
                        pred=outL, yN=yN, mask=mask,
                        mean=mean_sim, std=std_sim,
                        use_presence=cfg.use_presence_head
                    )
                    _accum_stat(comp_accum, stat)
                    comp_count += 1
                loss.backward()
                opt1.step()

            bs = H.size(0)
            sum_loss += loss.item() * bs; seen += bs

        epoch_loss = sum_loss / max(1, seen)
        epoch_loss_w = dist_mean_scalar(epoch_loss, device)
        loss_hist.append(float(epoch_loss_w))
        lr_hist.append(_cur_lr(opt1))

        if is_main_process():
            prev_lr = _cur_lr(opt1)
        sched1.step(epoch_loss_w)
        if is_main_process():
            now_lr = _cur_lr(opt1)
            if now_lr < prev_lr:
                print(f"  [S1] LR reduced: {prev_lr:.3e} -> {now_lr:.3e}")

        if is_main_process():
            print(f'[Fold {fold+1}] S1 Epoch {epoch+1}/{cfg.epochs_stage1} '
                  f'sim_loss={epoch_loss_w:.6f} lr={_cur_lr(opt1):.3e}', flush=True)
            if (epoch + 1) % 1000 == 0:
                torch.save({
                    'fold': fold, 'stage': 1, 'epoch': epoch + 1,
                    'model_state_dict': model_core.state_dict(),
                    'optimizer_state_dict': opt1.state_dict(),
                    'loss': float(epoch_loss_w),
                    'loss_history': loss_hist,
                    'lr_history': lr_hist,
                }, out_dir / f'checkpoint_stage1_fold{fold}_epoch{epoch+1}.pth')

        if is_main_process() and comp_count > 0:
            avg = {k: v/comp_count for k,v in comp_accum.items()}
            if cfg.use_presence_head:
                print(f"    [S1 z-MSE] alpha={avg['alpha']:.4f}, omega={avg['omega']:.4f}, gain={avg['gain']:.4f}, zero={avg['zero']:.4f}; "
                    f"BCE={avg['bce']:.4f}, CARD={avg.get('card',0.0):.4f}", flush=True)
            else:
                print(f"    [S1 z-MSE] alpha={avg['alpha']:.4f}, omega={avg['omega']:.4f}, gain={avg['gain']:.4f}, zero={avg['zero']:.4f}", flush=True)



        if epoch_loss_w < best_sim - cfg.early_stop_min_delta:
            best_sim = epoch_loss_w; no_improve = 0
            if is_main_process():
                torch.save({
                    'fold': fold, 'stage': 1, 'epoch': epoch + 1,
                    'model_state_dict': model_core.state_dict(),
                    'optimizer_state_dict': opt1.state_dict(),
                    'loss': float(epoch_loss_w),
                    'loss_history': loss_hist,
                    'lr_history': lr_hist,
                }, out_dir / f'best_stage1_fold{fold}.pth')
        else:
            no_improve += 1
            if no_improve >= cfg.early_stop_patience_s1:
                if is_main_process():
                    print('  Early stop Stage1')
                    torch.save({
                        'fold': fold, 'stage': 1, 'epoch': epoch + 1,
                        'model_state_dict': model_core.state_dict(),
                        'optimizer_state_dict': opt1.state_dict(),
                        'loss': float(epoch_loss_w),
                        'loss_history': loss_hist,
                        'lr_history': lr_hist,
                    }, out_dir / f'checkpoint_stage1_fold{fold}_earlystop_epoch{epoch+1}.pth')
                break

    # ============ Stage 2 ============
    for p in model_core.lin.parameters(): p.requires_grad_(True)
    for p in model_core.res.parameters(): p.requires_grad_(True)
    opt2 = torch.optim.Adam(model_core.parameters(), lr=cfg.lr_high_adam, weight_decay=cfg.weight_decay)

    pat2   = getattr(cfg, 'lr_patience_s2', 50)
    fac2   = getattr(cfg, 'lr_factor_s2', 0.5)
    minlr2 = getattr(cfg, 'lr_min_s2', 1e-6)
    sched2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt2, mode='min', factor=fac2, patience=pat2,
        threshold=thr, threshold_mode='abs', cooldown=0,
        min_lr=minlr2, eps=1e-12
    )

    w_sim, w_real = 0.3, 1.0
    best_total = float('inf'); no_improve = 0
    sim_iter = None
    total_hist, sim_hist, real_hist = [], [], []
    lr_hist_s2 = []

    for epoch in range(cfg.epochs_stage2):
        model.train()
        real_dl.sampler.set_epoch(epoch)
        sim_dl.sampler.set_epoch(epoch)
        if sim_iter is None:
            sim_iter = cycle(sim_dl)

        sum_sim, sum_real, cnt_sim, cnt_real = 0.0, 0.0, 0, 0

        for H_r, M_r, e_r, c_r, mask_r, yNr in real_dl:
            H_s, M_s, e_s, c_s, mask_s, yNs = next(sim_iter)

            H_r = H_r.to(device, non_blocking=True); M_r = M_r.to(device, non_blocking=True)
            e_r = e_r.to(device, non_blocking=True); c_r = c_r.to(device, non_blocking=True)
            mask_r = mask_r.to(device, non_blocking=True)
            for k in yNr: yNr[k] = yNr[k].to(device, non_blocking=True)

            H_s = H_s.to(device, non_blocking=True); M_s = M_s.to(device, non_blocking=True)
            e_s = e_s.to(device, non_blocking=True); c_s = c_s.to(device, non_blocking=True)
            mask_s = mask_s.to(device, non_blocking=True)
            for k in yNs: yNs[k] = yNs[k].to(device, non_blocking=True)

            opt2.zero_grad(set_to_none=True)
            if use_cuda:
                with torch.amp.autocast(device_type='cuda'):
                    _, outH_r = model(H_r, M_r, e_r, c_r)
                    loss_real = crit_s2(outH_r, yNr, mask_r)
                    outL_s, _ = model(H_s, M_s, e_s, c_s)
                    loss_sim = crit_s1(outL_s, yNs, mask_s)
                    loss = w_real*loss_real + w_sim*loss_sim
                scaler.scale(loss).backward(); scaler.step(opt2); scaler.update()
            else:
                _, outH_r = model(H_r, M_r, e_r, c_r)
                loss_real = crit_s2(outH_r, yNr, mask_r)
                outL_s, _ = model(H_s, M_s, e_s, c_s)
                loss_sim = crit_s1(outL_s, yNs, mask_s)
                loss = w_real*loss_real + w_sim*loss_sim
                loss.backward(); opt2.step()

            bs_r = H_r.size(0); bs_s = H_s.size(0)
            sum_real += loss_real.item() * bs_r; cnt_real += bs_r
            sum_sim  += loss_sim.item()  * bs_s; cnt_sim  += bs_s

        t_sim   = torch.tensor(sum_sim, device=device)
        t_cnt_s = torch.tensor(cnt_sim,  device=device, dtype=torch.float32)
        t_real  = torch.tensor(sum_real, device=device)
        t_cnt_r = torch.tensor(cnt_real, device=device, dtype=torch.float32)
        if dist.is_initialized():
            dist.all_reduce(t_sim, op=dist.ReduceOp.SUM);   dist.all_reduce(t_cnt_s, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_real, op=dist.ReduceOp.SUM);  dist.all_reduce(t_cnt_r, op=dist.ReduceOp.SUM)

        sim_epoch   = (t_sim  / t_cnt_s).item() if t_cnt_s.item() > 0 else 0.0
        real_epoch  = (t_real / t_cnt_r).item() if t_cnt_r.item() > 0 else 0.0
        total_epoch = float(w_sim*sim_epoch + w_real*real_epoch)

        if is_main_process():
            prev_lr2 = _cur_lr(opt2)
        sched2.step(total_epoch)
        if is_main_process():
            now_lr2 = _cur_lr(opt2)
            if now_lr2 < prev_lr2:
                print(f"  [S2] LR reduced: {prev_lr2:.3e} -> {now_lr2:.3e}")

        sim_hist.append(float(sim_epoch))
        real_hist.append(float(real_epoch))
        total_hist.append(float(total_epoch))
        lr_hist_s2.append(_cur_lr(opt2))

        if is_main_process():
            print(f'[Fold {fold+1}] S2 Epoch {epoch+1}/{cfg.epochs_stage2} ' 
                  f'total={total_epoch:.6f} sim={sim_epoch:.6f} real={real_epoch:.6f} '
                  f'lr={_cur_lr(opt2):.3e}', flush=True)
            if (epoch + 1) % 1000 == 0:
                save_path = out_dir / f'checkpoint_stage2_fold{fold}_epoch{epoch+1}.pth'
                torch.save({
                    'fold': fold, 'stage': 2, 'epoch': epoch + 1,
                    'model_state_dict': model_core.state_dict(),
                    'optimizer_state_dict': opt2.state_dict(),
                    'loss_total': float(total_epoch),
                    'loss_sim':   float(sim_epoch),
                    'loss_real':  float(real_epoch),
                    'loss_history_total': total_hist,
                    'loss_history_sim':   sim_hist,
                    'loss_history_real':  real_hist,
                    'lr_history': lr_hist_s2,
                }, save_path)
                print(f"  ⏺️ Saved checkpoint: {save_path.name}")

        if total_epoch < best_total - cfg.early_stop_min_delta:
            best_total = total_epoch; no_improve = 0
            if is_main_process():
                torch.save({
                    'fold': fold, 'stage': 2, 'epoch': epoch + 1,
                    'model_state_dict': model_core.state_dict(),
                    'optimizer_state_dict': opt2.state_dict(),
                    'loss_total': float(total_epoch),
                    'loss_sim':   float(sim_epoch),
                    'loss_real':  float(real_epoch),
                    'loss_history_total': total_hist,
                    'loss_history_sim':   sim_hist,
                    'loss_history_real':  real_hist,
                    'lr_history': lr_hist_s2,
                }, out_dir / f'best_stage2_fold{fold}.pth')
        else:
            no_improve += 1
            if no_improve >= cfg.early_stop_patience_s2:
                if is_main_process():
                    print('  Early stop Stage2')
                    torch.save({
                        'fold': fold, 'stage': 2, 'epoch': epoch + 1,
                        'model_state_dict': model_core.state_dict(),
                        'optimizer_state_dict': opt2.state_dict(),
                        'loss_total': float(total_epoch),
                        'loss_sim':   float(sim_epoch),
                        'loss_real':  float(real_epoch),
                        'loss_history_total': total_hist,
                        'loss_history_sim':   sim_hist,
                        'loss_history_real':  real_hist,
                        'lr_history': lr_hist_s2,
                    }, out_dir / f'checkpoint_stage2_fold{fold}_earlystop_epoch{epoch+1}.pth')
                break

    # ============ Validation ============
    model.eval()
    if cfg.use_hungarian_loss:
        crit_eval = MaskedHungarianMSE(mean=mean_real, std=std_real,
                                       w_alpha=w_s2_alpha, w_omega=w_s2_omega,
                                       w_gain=w_s2_gain, w_zero=w_s2_zero)
    else:
        crit_eval = MaskedMSE(mean=mean_real, std=std_real,
                               w_alpha=w_s2_alpha, w_omega=w_s2_omega,
                               w_gain=w_s2_gain, w_zero=w_s2_zero)
    with torch.no_grad():
        vsum, vcnt = 0.0, 0
        for H, M, e, c, mask, yN in val_dl:
            H = H.to(device, non_blocking=True)
            M = M.to(device, non_blocking=True)
            e = e.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            for k in yN: yN[k] = yN[k].to(device, non_blocking=True)
            outL, outH = model(H, M, e, c)
            loss = crit_eval(outH, yN, mask)
            bs = H.size(0)
            vsum += loss.item() * bs; vcnt += bs
        vloss = vsum / max(1, vcnt)
    return vloss, best_total, sim_hist, real_hist, total_hist
