from __future__ import annotations

from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMSE(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor,
                 w_alpha: float = 1.0, w_omega: float = 1.0,
                 w_rho: float = 1.0, w_sin_theta: float = 1.0,
                 w_cos_theta: float = 1.0,
                 presence_weight: float = 0.0, cardinality_weight: float = 0.0):
        super().__init__()
        self.register_buffer('mean', mean.view(3))
        self.register_buffer('std', std.view(3))
        self.w_alpha = float(w_alpha)
        self.w_omega = float(w_omega)
        self.w_rho = float(w_rho)
        self.w_sin_theta = float(w_sin_theta)
        self.w_cos_theta = float(w_cos_theta)
        self.pres_w = float(presence_weight)
        self.card_w = float(cardinality_weight)
        self.last_reg_stats = None
        self.last_cls_stats = {}

    def norm_pred(self, pred: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        P = torch.stack([pred['alpha'], pred['omega'], pred['rho']], dim=1)
        mean = self.mean.view(3, 1)
        std = self.std.view(3, 1).clamp_min(1e-8)
        PN = (P - mean) / std
        return {
            'alpha': PN[:, 0, :],
            'omega': PN[:, 1, :],
            'rho': PN[:, 2, :],
            'sin_theta': pred['sin_theta'],
            'cos_theta': pred['cos_theta'],
        }

    def _presence_losses(self, pred, mask):
        if self.pres_w <= 0.0 and self.card_w <= 0.0:
            self.last_cls_stats = {}
            return torch.zeros((), device=mask.device, dtype=mask.dtype)
        logits = pred.get('u_pres', None)
        if logits is None:
            logits = pred.get('p_exist_logits', None)
        if logits is None:
            self.last_cls_stats = {}
            return torch.zeros((), device=mask.device, dtype=mask.dtype)
        target = mask.to(logits.dtype)
        if logits.shape[1] > target.shape[1]:
            pad = target.new_zeros(target.shape[0], logits.shape[1] - target.shape[1])
            target = torch.cat([target, pad], dim=1)
        elif logits.shape[1] < target.shape[1]:
            target = target[:, :logits.shape[1]]
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='sum') / (target.numel() + 1e-8)
        p = torch.sigmoid(logits)
        card = F.mse_loss(p.sum(dim=1), target.sum(dim=1))
        self.last_cls_stats = {'bce': float(bce.detach()), 'card': float(card.detach())}
        return self.pres_w * bce + self.card_w * card

    def _sort_by_omega(self, alpha, omega, rho, sin_theta, cos_theta, mask):
        freq = torch.sqrt(alpha.pow(2) + omega.pow(2)) / (2.0 * math.pi)
        if mask.shape[1] != freq.shape[1]:
            if mask.shape[1] < freq.shape[1]:
                pad = mask.new_zeros(mask.shape[0], freq.shape[1] - mask.shape[1])
                mask = torch.cat([mask, pad], dim=1)
            else:
                mask = mask[:, :freq.shape[1]]
        big = torch.full_like(freq, float('inf'))
        freq_eff = torch.where(mask > 0.5, freq, big)
        idx = torch.argsort(freq_eff, dim=1)

        def re(t):
            return torch.gather(t, 1, idx)

        return re(alpha), re(omega), re(rho), re(sin_theta), re(cos_theta), re(mask)

    def forward(self, pred, tgt, mask):
        predN = self.norm_pred(pred)
        aP, wP, rP, sP, cP, _ = self._sort_by_omega(
            predN['alpha'], predN['omega'], predN['rho'], predN['sin_theta'], predN['cos_theta'], mask
        )
        aT, wT, rT, sT, cT, mT = self._sort_by_omega(
            tgt['alpha'], tgt['omega'], tgt['rho'], tgt['sin_theta'], tgt['cos_theta'], mask
        )
        w = mT
        num = w.sum().clamp_min(1.0)
        l_alpha = (aP - aT).pow(2)
        l_omega = (wP - wT).pow(2)
        l_rho = (rP - rT).pow(2)
        l_sin_theta = (sP - sT).pow(2)
        l_cos_theta = (cP - cT).pow(2)
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
            if denom > 0:
                self.last_reg_stats = {
                    'alpha': float((w * l_alpha).sum().detach().cpu().item() / denom),
                    'omega': float((w * l_omega).sum().detach().cpu().item() / denom),
                    'rho': float((w * l_rho).sum().detach().cpu().item() / denom),
                    'sin_theta': float((w * l_sin_theta).sum().detach().cpu().item() / denom),
                    'cos_theta': float((w * l_cos_theta).sum().detach().cpu().item() / denom),
                }
            else:
                self.last_reg_stats = {'alpha': 0.0, 'omega': 0.0, 'rho': 0.0, 'sin_theta': 0.0, 'cos_theta': 0.0}
        return loss_reg + loss_cls


class MaskedHungarianMSE(MaskedMSE):
    pass
