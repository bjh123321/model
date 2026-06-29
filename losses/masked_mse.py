from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class MaskedMSE(nn.Module):
    """Fixed-slot masked MSE for [freq, damping_percent, residue_re, residue_im]."""

    names = ("freq", "damping", "residue_re", "residue_im")

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
        residue_target: str = "fixed",
    ):
        super().__init__()
        mean = mean.flatten().float()
        std = std.flatten().float()
        if mean.numel() < 4 or std.numel() < 4:
            raise ValueError("fixed-slot mean/std must contain freq, damping, residue_re and residue_im statistics")
        self.register_buffer("mean", mean[:4])
        self.register_buffer("std", std[:4].clamp_min(1e-8))
        self.weights = {
            "freq": float(w_alpha),
            "damping": float(w_omega),
            "residue_re": float(w_rho),
            "residue_im": float(w_sin_theta),
        }
        self.pres_w = 0.0
        self.card_w = 0.0
        self.last_reg_stats = None
        self.last_cls_stats = {}

    def _normalized_pred_dict(self, pred: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for i, name in enumerate(self.names):
            mean = self.mean[i].to(device=pred[name].device, dtype=pred[name].dtype)
            std = self.std[i].to(device=pred[name].device, dtype=pred[name].dtype).clamp_min(1e-8)
            out[name] = (pred[name] - mean) / std
        return out

    def forward(self, pred, tgt, mask):
        pred_n = self._normalized_pred_dict(pred)
        width = min(mask.size(1), pred_n["freq"].size(1), tgt["freq"].size(1))
        mask = mask[:, :width].to(dtype=pred_n["freq"].dtype)
        num = mask.sum().clamp_min(1.0)

        loss = pred_n["freq"].new_tensor(0.0)
        per_name = {}
        for name in self.names:
            diff2 = (pred_n[name][:, :width] - tgt[name][:, :width]).pow(2)
            part = (mask * diff2).sum() / num
            per_name[name] = part
            loss = loss + self.weights[name] * part

        with torch.no_grad():
            self.last_reg_stats = {name: float(part.detach().cpu()) for name, part in per_name.items()}
            self.last_cls_stats = {}
        return loss
