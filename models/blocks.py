from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


SLOT_PARAM_COUNT = 4


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
        return self._split(out)

    @staticmethod
    def _split(out):
        return {
            "u_freq": out[..., 0],
            "u_damping": out[..., 1],
            "u_residue_re": out[..., 2],
            "u_residue_im": out[..., 3],
        }


class ParamFixed(nn.Module):
    def __init__(self, K_max: int):
        super().__init__()
        self.K_max = int(K_max)

    def to_physical(self, u_freq, u_damping, u_residue_re, u_residue_im):
        freq = F.softplus(u_freq).clamp_min(1e-9)
        damping_percent = torch.sigmoid(u_damping) * 100.0
        residue_re = u_residue_re
        residue_im = u_residue_im
        return {
            "freq": freq,
            "damping": damping_percent,
            "damping_percent": damping_percent,
            "residue_re": residue_re,
            "residue_im": residue_im,
            "u_freq": u_freq,
            "u_damping": u_damping,
            "u_residue_re": u_residue_re,
            "u_residue_im": u_residue_im,
        }


class MFDNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb_exc = nn.Embedding(cfg.embed_excitation_num, cfg.embed_excitation_dim)
        self.emb_ch = nn.Embedding(cfg.embed_channel_num, cfg.embed_channel_dim)
        self.low = LowFidelityHead(cfg)
        self.param = ParamFixed(cfg.K_max)
        self.register_buffer("online_gain", torch.ones(1, cfg.K_max, SLOT_PARAM_COUNT))
        self.register_buffer("online_bias", torch.zeros(1, cfg.K_max, SLOT_PARAM_COUNT))
        self.enable_online_fix = False

    def forward_low(self, Hn, Mn, exc_id, ch_id):
        exc_emb = self.emb_exc(exc_id)
        ch_emb = self.emb_ch(ch_id)
        raw = self.low(Hn, Mn, exc_emb, ch_emb)
        U = torch.stack(
            [raw["u_freq"], raw["u_damping"], raw["u_residue_re"], raw["u_residue_im"]],
            dim=-1,
        )
        if self.enable_online_fix:
            U = U * self.online_gain + self.online_bias
        return self.param.to_physical(
            U[..., 0],
            U[..., 1],
            U[..., 2],
            U[..., 3],
        )

    def forward_high(self, Hn, Mn, exc_id, ch_id, outL):
        raise RuntimeError("High-fidelity head is disabled in user_for_win7_fixed_sf.")

    def forward(self, Hn, Mn, exc_id, ch_id):
        return self.forward_low(Hn, Mn, exc_id, ch_id)

    @torch.no_grad()
    def quick_fix_biasgain(self, delta_or_ab: dict):
        if "delta" in delta_or_ab:
            delta = delta_or_ab["delta"].to(self.online_bias.device).view_as(self.online_bias)
            self.online_bias.copy_(delta)
            self.online_gain.fill_(1.0)
        elif "a" in delta_or_ab and "b" in delta_or_ab:
            self.online_gain.copy_(delta_or_ab["a"].to(self.online_gain.device).view_as(self.online_gain))
            self.online_bias.copy_(delta_or_ab["b"].to(self.online_bias.device).view_as(self.online_bias))
        else:
            raise ValueError("expect {'delta': ...} or {'a': ..., 'b': ...}")

    @torch.no_grad()
    def reset_quick_fix(self):
        self.online_gain.fill_(1.0)
        self.online_bias.zero_()
        self.enable_online_fix = False

    @torch.no_grad()
    def enable_quick_fix(self, flag: bool = True):
        self.enable_online_fix = bool(flag)

    def flatten_u(self, u_freq, u_damping, u_residue_re, u_residue_im, *unused) -> torch.Tensor:
        return torch.cat([u_freq, u_damping, u_residue_re, u_residue_im], dim=-1)
