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


def _match_width(t: torch.Tensor, width: int) -> torch.Tensor:
    width = int(width)
    cur = int(t.size(-1))
    if cur == width:
        return t
    if cur > width:
        return t[..., :width]
    pad_shape = list(t.shape)
    pad_shape[-1] = width - cur
    pad = t.new_zeros(pad_shape)
    return torch.cat([t, pad], dim=-1)


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
        residue_target: str = "polar",
    ):
        super().__init__()
        mean = mean.flatten()
        std = std.flatten()
        if mean.numel() < 2 or std.numel() < 2:
            raise ValueError("mean/std must contain at least alpha and omega statistics")
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.w_alpha = w_alpha
        self.w_omega = w_omega
        self.w_rho = w_rho
        self.w_sin_theta = w_sin_theta
        self.w_cos_theta = w_cos_theta
        self.residue_target = str(residue_target).lower()
        self.pres_w = presence_weight
        self.card_w = cardinality_weight
        self.last_reg_stats = None
        self.last_cls_stats = None

    def norm_pred(self, pred: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        p = torch.stack([pred["alpha"], pred["omega"]], dim=1)
        mean = self.mean[:2].view(2, 1)
        std = self.std[:2].view(2, 1).clamp_min(1e-8)
        pn = (p - mean) / std
        return {
            "alpha": pn[:, 0, :],
            "omega": pn[:, 1, :],
        }

    def _residue_names(self):
        if self.residue_target == "cartesian":
            return ["residue_re", "residue_im"]
        return ["rho", "sin_theta", "cos_theta"]

    def _residue_weights(self):
        if self.residue_target == "cartesian":
            return [self.w_rho, self.w_sin_theta]
        return [self.w_rho, self.w_sin_theta, self.w_cos_theta]

    def _normalized_pred_dict(self, pred, tgt):
        pred_n = self.norm_pred(pred)
        names = self._residue_names()
        res_mean = self.mean[2:2 + len(names)]
        res_std = self.std[2:2 + len(names)]
        if res_mean.numel() < len(names):
            res_mean = torch.cat([res_mean, res_mean.new_zeros(len(names) - res_mean.numel())], dim=0)
        if res_std.numel() < len(names):
            res_std = torch.cat([res_std, res_std.new_ones(len(names) - res_std.numel())], dim=0)
        res_mean = res_mean.view(1, len(names), 1).to(device=pred_n["alpha"].device, dtype=pred_n["alpha"].dtype)
        res_std = res_std.view(1, len(names), 1).to(device=pred_n["alpha"].device, dtype=pred_n["alpha"].dtype).clamp_min(1e-8)
        pred_width = pred_n["alpha"].size(1)
        res_mean = _match_width(res_mean, pred_width)
        res_std = _match_width(res_std, pred_width).clamp_min(1e-8)
        for i, name in enumerate(names):
            pred_n[name] = (pred[name] - res_mean[:, i, :]) / res_std[:, i, :]
        return pred_n

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

    def _sort_by_omega(self, values: dict, mask):
        alpha = values["alpha"]
        omega = values["omega"]
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

        out = {name: reorder(tensor) for name, tensor in values.items()}
        return out, reorder(mask)

    def forward(self, pred, tgt, mask):
        names = ["alpha", "omega", *self._residue_names()]
        weights = [self.w_alpha, self.w_omega, *self._residue_weights()]
        pred_n = self._normalized_pred_dict(pred, tgt)
        pred_s, _ = self._sort_by_omega({name: pred_n[name] for name in names}, mask)
        tgt_s, m_t = self._sort_by_omega({name: tgt[name] for name in names}, mask)
        width = max(pred_s[names[0]].size(1), tgt_s[names[0]].size(1), m_t.size(1))
        pred_s = {name: _match_width(value, width) for name, value in pred_s.items()}
        tgt_s = {name: _match_width(value, width) for name, value in tgt_s.items()}
        m_t = _match_width(m_t, width)
        w = m_t
        num = w.sum().clamp_min(1.0)
        losses = {name: (pred_s[name] - tgt_s[name]).pow(2) for name in names}
        loss_reg = sum(weight * (w * losses[name]).sum() for name, weight in zip(names, weights)) / num
        loss_cls = self._presence_losses(pred, mask)

        with torch.no_grad():
            denom = float(num.detach().cpu().item())
            self.last_reg_stats = {
                name: float((w * losses[name]).sum().detach().cpu().item() / denom)
                for name in names
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
        residue_target: str = "polar",
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
            residue_target=residue_target,
        )

    def forward(self, pred, tgt, mask):
        device = mask.device
        names = ["alpha", "omega", *self._residue_names()]
        weights = [self.w_alpha, self.w_omega, *self._residue_weights()]
        pred_n = self._normalized_pred_dict(pred, tgt)
        pred_vals = [pred_n[name] for name in names]
        tgt_width = tgt[names[0]].size(1)
        tgt_vals = [_match_width(tgt[name], tgt_width) for name in names]
        mask = _match_width(mask, tgt_width)
        pa = pred_n["alpha"]
        bsz = pa.size(0)
        k_pred = pa.size(1)
        k_tgt = tgt_width

        with torch.no_grad():
            cost = pa.new_zeros((bsz, k_pred, k_tgt))
            for weight, p_val, t_val in zip(weights, pred_vals, tgt_vals):
                cost = cost + weight * (p_val.unsqueeze(2) - t_val.unsqueeze(1)).pow(2)
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
            diffs = [
                torch.gather(p_val, 1, safe_rows) - torch.gather(t_val, 1, safe_cols)
                for p_val, t_val in zip(pred_vals, tgt_vals)
            ]

            valid_f = valid_pair_mask.to(dtype=pa.dtype)
            per_sample_cnt = valid_f.sum(dim=1).clamp_min(1.0)
            per_pair = sum(weight * diff.pow(2) for weight, diff in zip(weights, diffs))
            per_sample_loss = (per_pair * valid_f).sum(dim=1) / per_sample_cnt
            loss_reg = per_sample_loss[valid_pair_mask.any(dim=1)].mean()
        else:
            diffs = [pa.new_zeros((bsz, k_tgt)) for _ in names]
            valid_f = pa.new_zeros((bsz, k_tgt))
            loss_reg = pa.new_tensor(0.0, requires_grad=True)

        loss_cls = self._presence_losses(pred, mask, assigned_mask)
        with torch.no_grad():
            if pair_cnt > 0:
                denom = float(pair_cnt)
                self.last_reg_stats = {
                    name: float((diff.pow(2) * valid_f).sum().detach().cpu().item() / denom)
                    for name, diff in zip(names, diffs)
                }
            else:
                self.last_reg_stats = {name: 0.0 for name in names}

        return loss_reg + loss_cls
