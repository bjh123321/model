from __future__ import annotations

import itertools
import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


_PERM_CACHE: dict[tuple[int, int], torch.Tensor] = {}


def _assignment_perms(k_pred: int, k_tgt: int, device: torch.device) -> torch.Tensor:
    key = (int(k_pred), int(k_tgt))
    if key not in _PERM_CACHE:
        perms = list(itertools.permutations(range(k_pred), k_tgt))
        _PERM_CACHE[key] = torch.tensor(perms, dtype=torch.long)
    return _PERM_CACHE[key].to(device=device)


class MaskedMSE(nn.Module):
    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        w_alpha: float = 1.0,
        w_omega: float = 1.0,
        w_rho: float = 1.0,
        w_sin_theta: float = 1.0,
        w_cos_theta: float = 1.0,
        presence_weight: float = 0.0,
        cardinality_weight: float = 0.0,
    ):
        super().__init__()
        self.register_buffer("mean", mean.view(3))
        self.register_buffer("std", std.view(3))
        self.w_alpha = w_alpha
        self.w_omega = w_omega
        self.w_rho = w_rho
        self.w_sin_theta = w_sin_theta
        self.w_cos_theta = w_cos_theta
        self.pres_w = presence_weight
        self.card_w = cardinality_weight
        self.last_reg_stats = None
        self.last_cls_stats = None

    def norm_pred(self, pred: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        p = torch.stack([pred["alpha"], pred["omega"], pred["rho"]], dim=1)
        mean = self.mean.view(3, 1)
        std = self.std.view(3, 1).clamp_min(1e-8)
        pn = (p - mean) / std
        return {
            "alpha": pn[:, 0, :],
            "omega": pn[:, 1, :],
            "rho": pn[:, 2, :],
            "sin_theta": pred["sin_theta"],
            "cos_theta": pred["cos_theta"],
        }

    def _presence_losses(self, pred, mask, assign_rows=None):
        if self.pres_w <= 0 and self.card_w <= 0:
            self.last_cls_stats = {}
            return torch.zeros((), device=mask.device, dtype=mask.dtype)

        logits = pred.get("u_pres", None)
        if logits is None:
            logits = pred.get("p_exist_logits", None)
        if logits is None:
            self.last_cls_stats = {}
            return torch.zeros((), device=mask.device, dtype=mask.dtype)

        bsz, k_out = logits.shape
        b_mask, k_tgt = mask.shape
        assert bsz == b_mask, f"presence loss batch mismatch: logits={logits.shape}, mask={mask.shape}"

        if assign_rows is not None:
            target = torch.zeros_like(logits)
            if torch.is_tensor(assign_rows):
                if assign_rows.dtype == torch.bool:
                    target = assign_rows.to(device=logits.device, dtype=logits.dtype)
                else:
                    for i, rows in enumerate(assign_rows):
                        rows = rows[rows >= 0].to(device=logits.device, dtype=torch.long)
                        target[i, rows] = 1.0
            else:
                for i, rows in enumerate(assign_rows):
                    rows = rows.to(device=logits.device, dtype=torch.long)
                    target[i, rows] = 1.0
        else:
            target = mask.to(logits.dtype)
            if k_out > k_tgt:
                target = torch.cat([target, target.new_zeros(bsz, k_out - k_tgt)], dim=1)
            elif k_out < k_tgt:
                target = target[:, :k_out]

        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="sum") / (target.numel() + 1e-8)
        p_exist = torch.sigmoid(logits)
        card = F.mse_loss(p_exist.sum(dim=1), target.sum(dim=1))
        loss_cls = self.pres_w * bce + self.card_w * card
        with torch.no_grad():
            self.last_cls_stats = {"bce": float(bce.detach()), "card": float(card.detach())}
        return loss_cls

    def _sort_by_omega(self, alpha, omega, rho, sin_theta, cos_theta, mask):
        freq = torch.sqrt(alpha**2 + omega**2) / (2 * math.pi)
        if mask.shape[1] != freq.shape[1]:
            bsz, k_pred = freq.shape
            k_mask = mask.shape[1]
            if k_mask < k_pred:
                mask = torch.cat([mask, mask.new_zeros(bsz, k_pred - k_mask)], dim=1)
            else:
                mask = mask[:, :k_pred]
        big = torch.full_like(freq, float("inf"))
        freq_eff = torch.where(mask > 0.5, freq, big)
        idx = torch.argsort(freq_eff, dim=1)

        def reorder(t):
            return torch.gather(t, 1, idx)

        return reorder(alpha), reorder(omega), reorder(rho), reorder(sin_theta), reorder(cos_theta), reorder(mask)

    def forward(self, pred, tgt, mask):
        pred_n = self.norm_pred(pred)
        a_p, w_p, r_p, s_p, c_p, _ = self._sort_by_omega(
            pred_n["alpha"],
            pred_n["omega"],
            pred_n["rho"],
            pred_n["sin_theta"],
            pred_n["cos_theta"],
            mask,
        )
        a_t, w_t, r_t, s_t, c_t, m_t = self._sort_by_omega(
            tgt["alpha"],
            tgt["omega"],
            tgt["rho"],
            tgt["sin_theta"],
            tgt["cos_theta"],
            mask,
        )
        w = m_t
        num = w.sum().clamp_min(1.0)
        l_alpha = (a_p - a_t).pow(2)
        l_omega = (w_p - w_t).pow(2)
        l_rho = (r_p - r_t).pow(2)
        l_sin_theta = (s_p - s_t).pow(2)
        l_cos_theta = (c_p - c_t).pow(2)
        loss_reg = (
            self.w_alpha * (w * l_alpha).sum()
            + self.w_omega * (w * l_omega).sum()
            + self.w_rho * (w * l_rho).sum()
            + self.w_sin_theta * (w * l_sin_theta).sum()
            + self.w_cos_theta * (w * l_cos_theta).sum()
        ) / num
        loss_cls = self._presence_losses(pred, mask)

        with torch.no_grad():
            denom = float(num.detach().cpu().item())
            self.last_reg_stats = {
                "alpha": float((w * l_alpha).sum().detach().cpu().item() / denom),
                "omega": float((w * l_omega).sum().detach().cpu().item() / denom),
                "rho": float((w * l_rho).sum().detach().cpu().item() / denom),
                "sin_theta": float((w * l_sin_theta).sum().detach().cpu().item() / denom),
                "cos_theta": float((w * l_cos_theta).sum().detach().cpu().item() / denom),
            }

        return loss_reg + loss_cls


class MaskedHungarianMSE(MaskedMSE):
    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        w_alpha: float = 1.0,
        w_omega: float = 1.0,
        w_rho: float = 1.0,
        w_sin_theta: float = 1.0,
        w_cos_theta: float = 1.0,
        presence_weight: float = 0.0,
        cardinality_weight: float = 0.0,
    ):
        super().__init__(
            mean,
            std,
            w_alpha=w_alpha,
            w_omega=w_omega,
            w_rho=w_rho,
            w_sin_theta=w_sin_theta,
            w_cos_theta=w_cos_theta,
            presence_weight=presence_weight,
            cardinality_weight=cardinality_weight,
        )

    def forward(self, pred, tgt, mask):
        device = mask.device
        pred_n = self.norm_pred(pred)
        pa, po, pr = pred_n["alpha"], pred_n["omega"], pred_n["rho"]
        ps, pc = pred_n["sin_theta"], pred_n["cos_theta"]
        bsz = pa.size(0)
        k_pred = pa.size(1)
        k_tgt = mask.size(1)

        tgt_stack = torch.stack(
            [tgt["alpha"], tgt["omega"], tgt["rho"], tgt["sin_theta"], tgt["cos_theta"]],
            dim=1,
        )
        ta, to, tr, ts, tc = tgt_stack[:, 0, :], tgt_stack[:, 1, :], tgt_stack[:, 2, :], tgt_stack[:, 3, :], tgt_stack[:, 4, :]

        with torch.no_grad():
            cost = (
                self.w_alpha * (pa.unsqueeze(2) - ta.unsqueeze(1)).pow(2)
                + self.w_omega * (po.unsqueeze(2) - to.unsqueeze(1)).pow(2)
                + self.w_rho * (pr.unsqueeze(2) - tr.unsqueeze(1)).pow(2)
                + self.w_sin_theta * (ps.unsqueeze(2) - ts.unsqueeze(1)).pow(2)
                + self.w_cos_theta * (pc.unsqueeze(2) - tc.unsqueeze(1)).pow(2)
            )
            cost = cost.masked_fill((mask <= 0.5).unsqueeze(1), 1e6)
            cost = torch.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=1e6)

            assigned_mask = torch.zeros(bsz, k_pred, dtype=torch.bool, device=device)
            row_index = torch.full((bsz, k_tgt), -1, dtype=torch.long, device=device)
            col_index = torch.full((bsz, k_tgt), -1, dtype=torch.long, device=device)

            if device.type == "cpu":
                cost_cpu = np.nan_to_num(cost.detach().numpy(), nan=1e6, posinf=1e6, neginf=1e6)
                for b in range(bsz):
                    rows_np, cols_np = linear_sum_assignment(cost_cpu[b])
                    keep = cost_cpu[b][rows_np, cols_np] < 1e5
                    if not keep.any():
                        continue
                    rows = torch.as_tensor(rows_np[keep], device=device, dtype=torch.long)
                    cols = torch.as_tensor(cols_np[keep], device=device, dtype=torch.long)
                    n = rows.numel()
                    row_index[b, :n] = rows
                    col_index[b, :n] = cols
                    assigned_mask[b, rows] = True
            else:
                valid_counts = (mask > 0.5).sum(dim=1).to(torch.long)
                for m in range(1, min(k_pred, k_tgt) + 1):
                    batch_idx = torch.nonzero(valid_counts == m, as_tuple=False).flatten()
                    if batch_idx.numel() == 0:
                        continue
                    perms = _assignment_perms(k_pred, m, device)
                    cst = cost.index_select(0, batch_idx)[:, :, :m]
                    totals = cst.new_zeros((batch_idx.numel(), perms.size(0)))
                    for col in range(m):
                        totals += cst[:, perms[:, col], col]
                    best = totals.argmin(dim=1)
                    rows = perms.index_select(0, best)
                    cols = torch.arange(m, device=device).expand(batch_idx.numel(), m)
                    row_index[batch_idx, :m] = rows
                    col_index[batch_idx, :m] = cols
                    assigned_mask[batch_idx.unsqueeze(1), rows] = True

        valid_pair_mask = row_index >= 0
        pair_cnt = int(valid_pair_mask.sum().detach().item())
        if pair_cnt > 0:
            safe_rows = row_index.clamp_min(0)
            safe_cols = col_index.clamp_min(0)
            da = torch.gather(pa, 1, safe_rows) - torch.gather(ta, 1, safe_cols)
            do = torch.gather(po, 1, safe_rows) - torch.gather(to, 1, safe_cols)
            dr = torch.gather(pr, 1, safe_rows) - torch.gather(tr, 1, safe_cols)
            ds = torch.gather(ps, 1, safe_rows) - torch.gather(ts, 1, safe_cols)
            dc = torch.gather(pc, 1, safe_rows) - torch.gather(tc, 1, safe_cols)

            valid_f = valid_pair_mask.to(dtype=pa.dtype)
            per_sample_cnt = valid_f.sum(dim=1).clamp_min(1.0)
            per_sample_loss = ((da.pow(2) + do.pow(2) + dr.pow(2) + ds.pow(2) + dc.pow(2)) * valid_f).sum(dim=1) / per_sample_cnt
            loss_reg = per_sample_loss[valid_pair_mask.any(dim=1)].mean()
        else:
            da = do = dr = ds = dc = pa.new_zeros((bsz, k_tgt))
            valid_f = pa.new_zeros((bsz, k_tgt))
            loss_reg = pa.new_tensor(0.0, requires_grad=True)

        loss_cls = self._presence_losses(pred, mask, assigned_mask)
        with torch.no_grad():
            if pair_cnt > 0:
                denom = float(pair_cnt)
                self.last_reg_stats = {
                    "alpha": float((da.pow(2) * valid_f).sum().detach().cpu().item() / denom),
                    "omega": float((do.pow(2) * valid_f).sum().detach().cpu().item() / denom),
                    "rho": float((dr.pow(2) * valid_f).sum().detach().cpu().item() / denom),
                    "sin_theta": float((ds.pow(2) * valid_f).sum().detach().cpu().item() / denom),
                    "cos_theta": float((dc.pow(2) * valid_f).sum().detach().cpu().item() / denom),
                }
            else:
                self.last_reg_stats = {
                    "alpha": 0.0,
                    "omega": 0.0,
                    "rho": 0.0,
                    "sin_theta": 0.0,
                    "cos_theta": 0.0,
                }

        return loss_reg + loss_cls
