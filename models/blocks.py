from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


SLOT_PARAM_COUNT = 5


class MLP(nn.Module):
    def __init__(self, dims: Iterable[int], dropout: float = 0.0):
        super().__init__()
        dims = list(dims)
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LoRALinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.base = nn.Linear(in_dim, out_dim)
        self.A = nn.Linear(in_dim, rank, bias=False)
        self.B = nn.Linear(rank, out_dim, bias=False)
        nn.init.zeros_(self.B.weight)
        self.scale = alpha / max(rank, 1)

    def forward(self, x):
        return self.base(x) + self.B(self.A(x)) * self.scale


class LowFidelityHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        in_dim = 2 + cfg.embed_excitation_dim + cfg.embed_channel_dim
        hidden = list(cfg.low_hidden)
        out_dim = cfg.K_max * SLOT_PARAM_COUNT
        if getattr(cfg, "use_lora", False):
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden[0]),
                nn.GELU(),
                *[
                    layer
                    for a, b in zip(hidden[:-1], hidden[1:])
                    for layer in (LoRALinear(a, b, cfg.lora_rank), nn.GELU())
                ],
                nn.Linear(hidden[-1], out_dim),
            )
        else:
            self.mlp = MLP([in_dim] + hidden + [out_dim], dropout=cfg.dropout_prob)

        self.mod = None
        if getattr(cfg, "use_modulation", False):
            self.mod = nn.Sequential(
                nn.Linear(2, cfg.K_max * SLOT_PARAM_COUNT),
                nn.Tanh(),
            )

    def forward(self, Hn, Mn, exc_emb, ch_emb):
        x = torch.cat([Hn, Mn, exc_emb, ch_emb], dim=1)
        out = self.mlp(x).view(-1, self.cfg.K_max, SLOT_PARAM_COUNT)
        if self.mod is not None:
            hm = torch.cat([Hn, Mn], dim=1)
            out = out + self.cfg.mod_gamma * self.mod(hm).view(-1, self.cfg.K_max, SLOT_PARAM_COUNT)
        return {
            "u_alpha": out[..., 0],
            "u_omega": out[..., 1],
            "u_rho": out[..., 2],
            "u_sin_theta": out[..., 3],
            "u_cos_theta": out[..., 4],
        }


class ParamPolar(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.K_max = int(cfg.K_max)

    def to_physical(self, u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta):
        if bool(getattr(self.cfg, "debug_direct_output", False)):
            alpha = u_alpha
            omega = u_omega
            rho = u_rho
            sin_theta = u_sin_theta
            cos_theta = u_cos_theta
        else:
            alpha = -F.softplus(u_alpha)
            omega = torch.sigmoid(u_omega)
            rho = u_rho
            norm = torch.sqrt(u_sin_theta.pow(2) + u_cos_theta.pow(2) + 1e-8)
            sin_theta = u_sin_theta / norm
            cos_theta = u_cos_theta / norm
        return {
            "alpha": alpha,
            "omega": omega,
            "rho": rho,
            "sin_theta": sin_theta,
            "cos_theta": cos_theta,
            "u_alpha": u_alpha,
            "u_omega": u_omega,
            "u_rho": u_rho,
            "u_sin_theta": u_sin_theta,
            "u_cos_theta": u_cos_theta,
        }


class MFDNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb_exc = nn.Embedding(cfg.embed_excitation_num, cfg.embed_excitation_dim)
        self.emb_ch = nn.Embedding(cfg.embed_channel_num, cfg.embed_channel_dim)
        self.low = LowFidelityHead(cfg)
        self.param = ParamPolar(cfg)
        self.register_buffer("online_gain", torch.ones(1, cfg.K_max, SLOT_PARAM_COUNT))
        self.register_buffer("online_bias", torch.zeros(1, cfg.K_max, SLOT_PARAM_COUNT))
        self.enable_online_fix = False

    def forward_low(self, Hn, Mn, exc_id, ch_id):
        exc_emb = self.emb_exc(exc_id)
        ch_emb = self.emb_ch(ch_id)
        raw = self.low(Hn, Mn, exc_emb, ch_emb)
        U = torch.stack(
            [raw["u_alpha"], raw["u_omega"], raw["u_rho"], raw["u_sin_theta"], raw["u_cos_theta"]],
            dim=-1,
        )
        if self.enable_online_fix:
            U = U * self.online_gain + self.online_bias
        return self.param.to_physical(U[..., 0], U[..., 1], U[..., 2], U[..., 3], U[..., 4])

    def forward_high(self, Hn, Mn, exc_id, ch_id, outL):
        raise RuntimeError("high head disabled")

    def forward(self, Hn, Mn, exc_id, ch_id):
        return self.forward_low(Hn, Mn, exc_id, ch_id)

    def flatten_u(self, u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta, *unused):
        return torch.cat([u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta], dim=-1)
