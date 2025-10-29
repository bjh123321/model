from __future__ import annotations
from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np
from ..config import Config


class MaskedMSE(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor,
                 w_alpha=1.0, w_omega=0.5, w_gain=0.5, w_zero=0.5,
                 presence_weight: float = 0.0, cardinality_weight: float = 0.0):
        super().__init__()
        self.register_buffer('mean', mean.view(4,1))
        self.register_buffer('std',  std.view(4,1))
        self.w_alpha, self.w_omega, self.w_gain, self.w_zero = w_alpha, w_omega, w_gain, w_zero
        self.pres_w = presence_weight
        self.card_w = cardinality_weight

    def norm_pred(self, pred: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        P = torch.stack([pred['alpha'], pred['omega'], pred['gain'], pred['zero']], dim=1)  # (B,4,K)
        PN = (P - self.mean) / self.std.clamp_min(1e-8)
        return {'alpha': PN[:,0,:], 'omega': PN[:,1,:], 'gain': PN[:,2,:], 'zero': PN[:,3,:]}

    def _presence_losses(self, pred, mask, assign_rows=None):
        cfg = Config()
        if self.pres_w <= 0 and self.card_w <= 0:
            # 权重都为0时直接返回0，避免不必要计算
            return torch.zeros((), device=mask.device, dtype=mask.dtype)
            # 1) 取 logits（兼容两种命名）
        logits = pred.get('u_pres', None)
        if logits is None:
            logits = pred.get('p_exist_logits', None)
        if logits is None:
            # 没有存在头就不计算这部分loss
            return torch.zeros((), device=mask.device, dtype=mask.dtype)

        # 2) 形状对齐：支持 K_out != K_tgt 的情况
        #    - 如果模型更宽 (K_out > K_tgt)：给 target 右侧补 0（多余槽位视作“负样本”）
        #    - 如果目标更宽 (K_out < K_tgt)：裁掉 target 的多余槽位
        B1, K_out = logits.shape
        B2, K_tgt = mask.shape
        assert B1 == B2, f"presence loss batch mismatch: logits={logits.shape}, mask={mask.shape}"
        if cfg.use_hungarian_loss:
            target = torch.zeros_like(logits)
            for i, b in enumerate(assign_rows):
                target[i, b] = 1.0
        else:
            target = mask.to(logits.dtype)
            if K_out > K_tgt:
                pad = target.new_zeros(B1, K_out - K_tgt)
                target = torch.cat([target, pad], dim=1)
            elif K_out < K_tgt:
                target = target[:, :K_out]

        # 3) BCEWithLogits
        bce = F.binary_cross_entropy_with_logits(
            logits, target, reduction='sum'
        ) / (target.numel() + 1e-8)

        # 4) 基数损失 (Cardinality)：预测存在数量 ≈ 真实数量
        p = torch.sigmoid(logits)
        card = F.mse_loss(p.sum(dim=1), target.sum(dim=1))

        return self.pres_w * bce + self.card_w * card

    def _sort_by_omega(self, alpha, omega, gain, zero, mask):
        freq = torch.sqrt(alpha**2 + omega**2) / (2 * math.pi)
        if mask.shape[1] != freq.shape[1]:
            B, K_pred = freq.shape
            K_mask = mask.shape[1]
            if K_mask < K_pred:
                pad = mask.new_zeros(B, K_pred - K_mask)
                mask = torch.cat([mask, pad], dim=1)
            else:
                mask = mask[:, :K_pred]
        big  = torch.full_like(freq, float('inf')) if freq.dtype.is_floating_point else freq.new_full(freq.shape, float('inf'))
        freq_eff = torch.where(mask > 0.5, freq, big)
        idx = torch.argsort(freq_eff, dim=1)
        def re(t):
            return torch.gather(t, 1, idx)
        return re(alpha), re(omega), re(gain), re(zero), re(mask)

    def forward(self, pred, tgt, mask):
        predN = self.norm_pred(pred)
        aP, wP, gP, zP, _ = self._sort_by_omega(predN['alpha'], predN['omega'], predN['gain'], predN['zero'], mask)
        aT, wT, gT, zT, mT = self._sort_by_omega(tgt['alpha'],  tgt['omega'],  tgt['gain'],  tgt['zero'],  mask)
        w = mT; num = w.sum().clamp_min(1.0)
        l_alpha = (aP - aT).pow(2); l_omega = (wP - wT).pow(2); l_gain = (gP - gT).pow(2); l_zero = (zP - zT).pow(2)
        loss_reg = (self.w_alpha*(w*l_alpha).sum() + self.w_omega*(w*l_omega).sum() +
                    self.w_gain *(w*l_gain ).sum() + self.w_zero *(w*l_zero ).sum()) / num
        loss_cls = self._presence_losses(pred, mask)
        return loss_reg + loss_cls

class MaskedHungarianMSE(MaskedMSE):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor,
                 w_alpha=1.0, w_omega=0.5, w_gain=0.5, w_zero=0.5,
                 presence_weight: float = 0.0, cardinality_weight: float = 0.0):
        super().__init__(mean, std, w_alpha=w_alpha, w_omega=w_omega, w_gain=w_gain, w_zero=w_zero,
                         presence_weight=presence_weight, cardinality_weight=cardinality_weight)
    
    def forward(self, pred, tgt, mask):
        device = mask.device
        predN = self.norm_pred(pred)
        # predN = pred
        PA, PO, PG, PZ = predN['alpha'], predN['omega'], predN['gain'], predN['zero']  # [B, K_pred]
        B = PA.size(0)  # batch大小
        K_pred = PA.size(1) # 总模态数
        K_tgt  = mask.size(1)

        # === 标签标准化到 z-score 空间 ===
        T  = torch.stack([tgt['alpha'], tgt['omega'], tgt['gain'], tgt['zero']], dim=1)  # [B,4,K_tgt]
        TA, TO, TG, TZ = T[:,0,:], T[:,1,:], T[:,2,:], T[:,3,:]                       # [B, K_tgt]

        # === 代价矩阵（no_grad）：[B, K_pred, K_tgt] ===
        with torch.no_grad():
            dA = (PA.unsqueeze(2) - TA.unsqueeze(1))**2
            dO = (PO.unsqueeze(2) - TO.unsqueeze(1))**2
            dG = (PG.unsqueeze(2) - TG.unsqueeze(1))**2
            dZ = (PZ.unsqueeze(2) - TZ.unsqueeze(1))**2
            cost = self.w_alpha*dA + self.w_omega*dO + self.w_gain*dG + self.w_zero*dZ    # [B,K_pred,K_tgt]
            cost.masked_fill_(((mask > 0.5).unsqueeze(1) == 0), 1e6)

            # 合法配对掩码：预测侧全1，目标侧来自 mask
            pm = torch.ones(B, K_pred, 1, dtype=cost.dtype, device=cost.device)           # [B,K_pred,1]
            tm = (mask > 0.5).to(cost.dtype).unsqueeze(1)                                  # [B,1,K_tgt]
            valid_pair = pm * tm                                                           # [B,K_pred,K_tgt]

            big = cost.new_full(cost.shape, 1e6)    # cost.new_full表示以cost的dtype、device作为模板，创建一个新张量，形状是cost.shape，所有元素值都是1e6
            cost = torch.where(valid_pair > 0.5, cost, big)    # torch.where(condition)表示返回非0元素的坐标，torch.where(condition, x, y)表示x if condition else y
            cost = torch.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=1e6)     # 去除异常值

        cost_cpu = cost.detach().cpu().numpy()
        cost_cpu = np.nan_to_num(cost_cpu, nan=1e6, posinf=1e6, neginf=1e6)     # 去除异常值
        assign_cols, assign_rows, m_eff_list = [], [], []
        for b in range(B):
            cb = cost_cpu[b]       # (K_pred, K_tgt)
            row_ind, col_ind = linear_sum_assignment(cb)    # 选中的行索引和列索引，计算得到cost最小
            # 过滤掉 big 惩罚的无效配对
            keep = (cb[row_ind, col_ind] < 1e5)
            if keep.any():  # keep.any()表示这个布尔数组中是否存在至少一个True，
                r = row_ind[keep]     # 布尔索引也是numpy和pytorch的一个核心特性，True时取出，False时舍弃
                c = col_ind[keep]
                assign_rows.append(torch.from_numpy(r).to(device))
                assign_cols.append(torch.from_numpy(c).to(device))
                m_eff_list.append(int(len(c)))
            else:
                assign_rows.append(torch.empty(0, device=device, dtype=torch.long))
                assign_cols.append(torch.empty(0, device=device, dtype=torch.long))
                m_eff_list.append(0)

        # === 在图里计算回归 MSE（z-score 空间），保留梯度 ===
        loss_sum = 0.0
        pair_cnt = 0
        for b in range(B):
            m_eff = m_eff_list[b]
            if m_eff == 0:
                continue
            rows = assign_rows[b]    # 预测侧索引（行）  ∈ [0, K_pred)
            cols = assign_cols[b]    # 目标侧索引（列）  ∈ [0, K_tgt)

            da = PA[b, rows] - TA[b, cols]
            do = PO[b, rows] - TO[b, cols]
            dg = PG[b, rows] - TG[b, cols]
            dz = PZ[b, rows] - TZ[b, cols]

            # 等权：也可改为 pair-weighted；此处每样本先均值，再跨样本均值
            loss_b = (da.pow(2).mean() + do.pow(2).mean() + dg.pow(2).mean() + dz.pow(2).mean())
            loss_sum += loss_b
            pair_cnt += 1

        loss_reg = (loss_sum / pair_cnt) if pair_cnt > 0 else PA.new_tensor(0.0, requires_grad=True)

        # === presence + cardinality（原样） ===
        loss_cls = self._presence_losses(pred, mask, assign_rows)

        return loss_reg + loss_cls
    
    
