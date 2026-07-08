from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


SLOT_PARAM_COUNT = 5


class MLP(nn.Module):
    def __init__(
        self,
        dims: Iterable[int],
        dropout: float = 0.0,
        use_lora: bool = False,
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        dims = list(dims)
        layers = []
        for i in range(len(dims) - 2):
            linear = (
                LoRALinear(dims[i], dims[i + 1], lora_rank, lora_alpha, lora_dropout)
                if use_lora
                else nn.Linear(dims[i], dims[i + 1])
            )
            layers += [linear, nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if int(rank) <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base = nn.Linear(in_dim, out_dim)
        self.A = nn.Linear(in_dim, int(rank), bias=False)
        self.B = nn.Linear(int(rank), out_dim, bias=False)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        nn.init.zeros_(self.B.weight)
        self.scale = float(alpha) / max(int(rank), 1)

    def forward(self, x):
        return self.base(x) + self.B(self.A(self.dropout(x))) * self.scale


class LowFidelityHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        in_dim = 2 + cfg.embed_excitation_dim + cfg.embed_channel_dim
        hidden = list(cfg.low_hidden)
        out_dim = cfg.K_max * SLOT_PARAM_COUNT
        self.mlp = MLP(
            [in_dim] + hidden + [out_dim],
            dropout=cfg.dropout_prob,
            use_lora=bool(getattr(cfg, "use_lora", False)),
            lora_rank=int(getattr(cfg, "lora_rank", 4)),
            lora_alpha=float(getattr(cfg, "lora_alpha", 1.0)),
            lora_dropout=float(getattr(cfg, "lora_dropout", 0.0)),
        )

        self.mod = None
        if getattr(cfg, "use_modulation", False):
            mod_in_dim = cfg.embed_excitation_dim + cfg.embed_channel_dim
            self.mod = MLP(
                [mod_in_dim] + list(cfg.mod_hidden) + [cfg.K_max * SLOT_PARAM_COUNT],
                dropout=cfg.dropout_prob,
            )

    def forward(self, Hn, Mn, exc_emb, ch_emb):
        x = torch.cat([Hn, Mn, exc_emb, ch_emb], dim=1)
        out = self.mlp(x).view(-1, self.cfg.K_max, SLOT_PARAM_COUNT)
        if self.mod is not None:
            mod_input = torch.cat([exc_emb, ch_emb], dim=1)
            scale = 1.0 + self.cfg.mod_gamma * torch.tanh(
                self.mod(mod_input).view(-1, self.cfg.K_max, SLOT_PARAM_COUNT)
            )
            out = out * scale
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


def freeze_base_train_lora_only(model: nn.Module):
    for name, param in model.named_parameters():
        param.requires_grad_((".A." in name) or (".B." in name))


def get_lora_parameters(model: nn.Module):
    return [
        param
        for name, param in model.named_parameters()
        if param.requires_grad and ((".A." in name) or (".B." in name))
    ]


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def _copy_tensor(dst: torch.Tensor, src: torch.Tensor, key: str):
    if tuple(dst.shape) != tuple(src.shape):
        raise RuntimeError(f"Shape mismatch for {key}: checkpoint {tuple(src.shape)} vs model {tuple(dst.shape)}")
    dst.copy_(src)


def load_base_weights_into_lora_model(model: nn.Module, state_dict: dict):
    model_state = model.state_dict()
    consumed: set[str] = set()
    missing = []

    with torch.no_grad():
        for key, dst in model_state.items():
            if ".A." in key or ".B." in key:
                missing.append(key)
                continue
            src_key = key
            if ".base." in key:
                plain_key = key.replace(".base.", ".")
                if plain_key in state_dict:
                    src_key = plain_key
            if src_key in state_dict:
                _copy_tensor(dst, state_dict[src_key], src_key)
                consumed.add(src_key)
            else:
                missing.append(key)

    unexpected = [key for key in state_dict.keys() if key not in consumed]
    model.load_state_dict(model_state, strict=False)

    hard_missing = [key for key in missing if ".A." not in key and ".B." not in key]
    if hard_missing:
        raise RuntimeError(f"Missing non-LoRA keys when loading base checkpoint: {hard_missing}")
    print(
        f"[LoRA load] missing_lora={len(missing) - len(hard_missing)}, "
        f"unexpected={len(unexpected)}",
        flush=True,
    )
    if unexpected:
        print(f"[LoRA load] unexpected keys: {unexpected}", flush=True)
    return missing, unexpected
