from __future__ import annotations

from typing import Dict, Iterable, Optional, Set, Tuple

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
        lora_target_hidden_indices: Optional[Iterable[int]] = None,
    ):
        super().__init__()
        dims = list(dims)
        hidden_count = len(dims) - 2
        if lora_target_hidden_indices is None:
            targets = tuple(range(1, hidden_count))
        else:
            targets = tuple(sorted({int(index) for index in lora_target_hidden_indices}))
        invalid = [index for index in targets if index < 0 or index >= hidden_count]
        if use_lora and invalid:
            raise ValueError(
                f"LoRA target hidden indices {invalid} are outside [0, {hidden_count - 1}]"
            )
        self.lora_target_hidden_indices = targets
        self.hidden_linear_module_indices = []
        layers = []
        for i in range(hidden_count):
            self.hidden_linear_module_indices.append(len(layers))
            linear = (
                LoRALinear(dims[i], dims[i + 1], lora_rank, lora_alpha, lora_dropout)
                if use_lora and i in targets
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
            lora_target_hidden_indices=getattr(cfg, "lora_target_hidden_indices", (2,)),
        )

        self.mod = None
        if getattr(cfg, "use_modulation", False):
            self.mod = nn.Sequential(nn.Linear(2, out_dim), nn.Tanh())

    def forward(self, Hn, Mn, exc_emb, ch_emb):
        x = torch.cat([Hn, Mn, exc_emb, ch_emb], dim=1)
        out = self.mlp(x).view(-1, self.cfg.K_max, SLOT_PARAM_COUNT)
        if self.mod is not None:
            hm = torch.cat([Hn, Mn], dim=1)
            correction = self.cfg.mod_gamma * self.mod(hm).view(-1, self.cfg.K_max, SLOT_PARAM_COUNT)
            out = out + correction
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


class TargetHead(nn.Module):
    """One independent head for one FDR target component."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.target_names = tuple(str(name) for name in cfg.model_target_names)
        if not self.target_names:
            raise ValueError("model_target_names must not be empty")
        in_dim = 2 + cfg.embed_excitation_dim + cfg.embed_channel_dim
        out_dim = cfg.K_max * len(self.target_names)
        self.mlp = MLP(
            [in_dim] + list(cfg.low_hidden) + [out_dim],
            dropout=cfg.dropout_prob,
            use_lora=bool(getattr(cfg, "use_lora", False)),
            lora_rank=int(getattr(cfg, "lora_rank", 4)),
            lora_alpha=float(getattr(cfg, "lora_alpha", 1.0)),
            lora_dropout=float(getattr(cfg, "lora_dropout", 0.0)),
            lora_target_hidden_indices=getattr(cfg, "lora_target_hidden_indices", (2,)),
        )
        self.mod = None
        if getattr(cfg, "use_modulation", False):
            self.mod = nn.Sequential(nn.Linear(2, out_dim), nn.Tanh())

    def forward(self, Hn, Mn, exc_emb, ch_emb):
        x = torch.cat([Hn, Mn, exc_emb, ch_emb], dim=1)
        out = self.mlp(x)
        if self.mod is not None:
            hm = torch.cat([Hn, Mn], dim=1)
            out = out + self.cfg.mod_gamma * self.mod(hm)
        out = out.view(-1, self.cfg.K_max, len(self.target_names))
        values = {name: out[..., index] for index, name in enumerate(self.target_names)}
        if self.target_names == ("sin_theta", "cos_theta"):
            norm = torch.sqrt(
                values["sin_theta"].pow(2) + values["cos_theta"].pow(2) + 1e-8
            )
            values["sin_theta"] = values["sin_theta"] / norm
            values["cos_theta"] = values["cos_theta"] / norm
        return values


class TargetMFDNN(nn.Module):
    """A standalone network that predicts one normalized FDR target component."""

    outputs_normalized = True

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.target_names = tuple(str(name) for name in cfg.model_target_names)
        self.emb_exc = nn.Embedding(cfg.embed_excitation_num, cfg.embed_excitation_dim)
        self.emb_ch = nn.Embedding(cfg.embed_channel_num, cfg.embed_channel_dim)
        self.low = TargetHead(cfg)

    def forward(self, Hn, Mn, exc_id, ch_id):
        return self.low(Hn, Mn, self.emb_exc(exc_id), self.emb_ch(ch_id))


class TargetLinearHighFidelityHead(nn.Module):
    """Per-slot affine map from simulation-normalized to real-normalized targets."""

    def __init__(self, target_count: int):
        super().__init__()
        self.affine = nn.Linear(int(target_count), int(target_count))
        with torch.no_grad():
            self.affine.weight.copy_(torch.eye(int(target_count)))
            self.affine.bias.zero_()

    def forward(self, low_values: torch.Tensor) -> torch.Tensor:
        return self.affine(low_values)


class TargetAdaptationHead(nn.Module):
    """Nonlinear real-data correction conditioned on H/M and the low prediction."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.target_count = len(tuple(cfg.model_target_names))
        in_dim = (
            2
            + cfg.embed_excitation_dim
            + cfg.embed_channel_dim
            + self.target_count
        )
        self.mlp = MLP(
            [in_dim]
            + list(getattr(cfg, "adapt_hidden", [32, 32]))
            + [cfg.K_max * self.target_count],
            dropout=cfg.dropout_prob,
        )

    def forward(self, Hn, Mn, exc_emb, ch_emb, low_values):
        low_pool = low_values.mean(dim=1)
        x = torch.cat([Hn, Mn, exc_emb, ch_emb, low_pool], dim=1)
        return self.mlp(x).view(-1, self.cfg.K_max, self.target_count)


class TargetMultifidelityMFDNN(nn.Module):
    """Independent FDR component with a sim low head and real correction head."""

    outputs_normalized = True

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.target_names = tuple(str(name) for name in cfg.model_target_names)
        if not self.target_names:
            raise ValueError("model_target_names must not be empty")
        self.emb_exc = nn.Embedding(cfg.embed_excitation_num, cfg.embed_excitation_dim)
        self.emb_ch = nn.Embedding(cfg.embed_channel_num, cfg.embed_channel_dim)
        self.low = TargetHead(cfg)
        self.lin = TargetLinearHighFidelityHead(len(self.target_names))
        self.adaptation = TargetAdaptationHead(cfg)

    def _as_tensor(self, values: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack([values[name] for name in self.target_names], dim=-1)

    def _as_dict(self, values: torch.Tensor) -> Dict[str, torch.Tensor]:
        result = {
            name: values[..., index]
            for index, name in enumerate(self.target_names)
        }
        if self.target_names == ("sin_theta", "cos_theta"):
            norm = torch.sqrt(
                result["sin_theta"].pow(2)
                + result["cos_theta"].pow(2)
                + 1e-8
            )
            result["sin_theta"] = result["sin_theta"] / norm
            result["cos_theta"] = result["cos_theta"] / norm
        return result

    def forward_low(self, Hn, Mn, exc_id, ch_id):
        return self.low(Hn, Mn, self.emb_exc(exc_id), self.emb_ch(ch_id))

    def forward_high(self, Hn, Mn, exc_id, ch_id, out_low):
        exc_emb = self.emb_exc(exc_id)
        ch_emb = self.emb_ch(ch_id)
        low_values = self._as_tensor(out_low)
        high_values = self.lin(low_values) + self.adaptation(
            Hn, Mn, exc_emb, ch_emb, low_values
        )
        return self._as_dict(high_values)

    def forward(self, Hn, Mn, exc_id, ch_id):
        out_low = self.forward_low(Hn, Mn, exc_id, ch_id)
        out_high = self.forward_high(Hn, Mn, exc_id, ch_id, out_low)
        return out_low, out_high


def freeze_base_train_lora_only(model: nn.Module):
    for name, param in model.named_parameters():
        param.requires_grad_((".A." in name) or (".B." in name))


def get_lora_parameters(model: nn.Module):
    return [
        param
        for name, param in model.named_parameters()
        if param.requires_grad and ((".A." in name) or (".B." in name))
    ]


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def count_parameter_groups(model: nn.Module) -> Dict[str, int]:
    low = model.low
    groups = {
        "main_mlp_parameters": sum(p.numel() for p in low.mlp.parameters()),
        "mod_parameters": sum(p.numel() for p in low.mod.parameters()) if low.mod is not None else 0,
        "embedding_parameters": sum(p.numel() for p in model.emb_exc.parameters()) + sum(p.numel() for p in model.emb_ch.parameters()),
        "total_parameters": sum(p.numel() for p in model.parameters()),
    }
    if hasattr(model, "lin"):
        groups["linear_high_parameters"] = sum(p.numel() for p in model.lin.parameters())
    if hasattr(model, "adaptation"):
        groups["adaptation_parameters"] = sum(
            p.numel() for p in model.adaptation.parameters()
        )
    return groups


def _copy_tensor(dst: torch.Tensor, src: torch.Tensor, key: str):
    if tuple(dst.shape) != tuple(src.shape):
        raise RuntimeError(f"Shape mismatch for {key}: checkpoint {tuple(src.shape)} vs model {tuple(dst.shape)}")
    dst.copy_(src)


def load_base_weights_into_lora_model(model: nn.Module, state_dict: dict):
    model_state = model.state_dict()
    consumed: Set[str] = set()
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
