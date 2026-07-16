from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from surrogate_model.data.io import ParsedRow


POLE_ALPHA_OMEGA = "alpha_omega"
POLE_FREQ_DAMPING = "freq_damping"
RESIDUE_POLAR = "polar"
RESIDUE_CARTESIAN = "cartesian"


def canonical_pole_target(value: str | None) -> str:
    v = str(value or POLE_ALPHA_OMEGA).strip().lower()
    aliases = {
        "pole": POLE_ALPHA_OMEGA,
        "alpha_omega": POLE_ALPHA_OMEGA,
        "alphaomega": POLE_ALPHA_OMEGA,
        "real_imag": POLE_ALPHA_OMEGA,
        "re_im": POLE_ALPHA_OMEGA,
        "freq_damping": POLE_FREQ_DAMPING,
        "freq_damp": POLE_FREQ_DAMPING,
        "frequency_damping": POLE_FREQ_DAMPING,
        "modal": POLE_FREQ_DAMPING,
    }
    if v not in aliases:
        raise ValueError(f"Unsupported pole_target={value!r}; expected alpha_omega or freq_damping")
    return aliases[v]


def canonical_residue_target(value: str | None) -> str:
    v = str(value or RESIDUE_POLAR).strip().lower()
    aliases = {
        "polar": RESIDUE_POLAR,
        "rho_phase": RESIDUE_POLAR,
        "rho_sincos": RESIDUE_POLAR,
        "rho_sin_cos": RESIDUE_POLAR,
        "cartesian": RESIDUE_CARTESIAN,
        "residue_re_im": RESIDUE_CARTESIAN,
        "residue_real_imag": RESIDUE_CARTESIAN,
        "re_im": RESIDUE_CARTESIAN,
        "real_imag": RESIDUE_CARTESIAN,
        "fixed": RESIDUE_CARTESIAN,
    }
    if v not in aliases:
        raise ValueError(f"Unsupported residue_target={value!r}; expected polar or cartesian")
    return aliases[v]


def target_names_for(pole_target: str | None = None, residue_target: str | None = None) -> list[str]:
    pole = canonical_pole_target(pole_target)
    residue = canonical_residue_target(residue_target)
    names: list[str] = []
    if pole == POLE_ALPHA_OMEGA:
        names += ["alpha", "omega"]
    else:
        names += ["freq", "damping"]
    if residue == RESIDUE_POLAR:
        names += ["rho", "sin_theta", "cos_theta"]
    else:
        names += ["residue_re", "residue_im"]
    return names


def normalize_mask_for(names: list[str]) -> np.ndarray:
    # Keep the historical alpha/omega/rho/sin/cos path unchanged:
    # sin_theta and cos_theta are already bounded direction cosines and are not z-scored.
    return np.asarray([name not in {"sin_theta", "cos_theta"} for name in names], dtype=bool)


def row_target_values(p: ParsedRow, pole_target: str | None = None, residue_target: str | None = None) -> dict[str, np.ndarray]:
    pole = canonical_pole_target(pole_target)
    residue = canonical_residue_target(residue_target)
    out: dict[str, np.ndarray] = {}
    if pole == POLE_ALPHA_OMEGA:
        out["alpha"] = np.asarray(p.alpha, dtype=np.float32)
        out["omega"] = np.asarray(p.omega, dtype=np.float32)
    else:
        out["freq"] = np.asarray(p.freq, dtype=np.float32)
        out["damping"] = np.asarray(p.damping, dtype=np.float32)
    if residue == RESIDUE_POLAR:
        out["rho"] = np.asarray(p.rho, dtype=np.float32)
        out["magnitude"] = np.exp(
            np.clip(np.asarray(p.rho, dtype=np.float64), -80.0, 80.0)
        ).astype(np.float32)
        out["sin_theta"] = np.asarray(p.sin_theta, dtype=np.float32)
        out["cos_theta"] = np.asarray(p.cos_theta, dtype=np.float32)
    else:
        out["residue_re"] = np.asarray(p.residue_re, dtype=np.float32)
        out["residue_im"] = np.asarray(p.residue_im, dtype=np.float32)
    return out


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray
    residue_target: str = RESIDUE_POLAR
    pole_target: str = POLE_ALPHA_OMEGA
    names: list[str] | None = None

    def __post_init__(self):
        self.pole_target = canonical_pole_target(self.pole_target)
        self.residue_target = canonical_residue_target(self.residue_target)
        self.mean = np.asarray(self.mean, dtype=np.float32)
        self.std = np.asarray(self.std, dtype=np.float32)
        if self.names is None:
            self.names = target_names_for(self.pole_target, self.residue_target)
        else:
            self.names = [str(x) for x in self.names]

    def save(self, path):
        import torch as _torch
        d = {
            "format": "list_v2",
            "mean": self.mean.astype(np.float32).tolist(),
            "std": self.std.astype(np.float32).tolist(),
            "residue_target": self.residue_target,
            "pole_target": self.pole_target,
            "target_names": list(self.names or target_names_for(self.pole_target, self.residue_target)),
        }
        _torch.save(d, path)

    @staticmethod
    def load(path) -> "Normalizer":
        import torch as _torch
        import numpy as _np
        d = _torch.load(path, map_location="cpu")
        mean = d.get("mean")
        std = d.get("std")
        residue_target = d.get("residue_target", RESIDUE_POLAR)
        pole_target = d.get("pole_target", POLE_ALPHA_OMEGA)
        names = d.get("target_names", None)
        if isinstance(mean, _torch.Tensor):
            mean = mean.detach().cpu().numpy()
        elif not isinstance(mean, _np.ndarray):
            mean = _np.asarray(mean, dtype=_np.float32)
        if isinstance(std, _torch.Tensor):
            std = std.detach().cpu().numpy()
        elif not isinstance(std, _np.ndarray):
            std = _np.asarray(std, dtype=_np.float32)
        return Normalizer(mean=mean, std=std, residue_target=str(residue_target), pole_target=str(pole_target), names=names)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        m = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        s = torch.as_tensor(self.std, dtype=x.dtype, device=x.device).clamp_min(1e-8)
        return (x - m) / s

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        m = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        s = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        return x * s + m

    def residue_names(self) -> list[str]:
        return target_names_for(self.pole_target, self.residue_target)[2:]

    def target_names(self) -> list[str]:
        return list(self.names or target_names_for(self.pole_target, self.residue_target))


# Backward-compatible name kept for older imports.
def compute_input_normalizer_sim(sim_parsed: List[ParsedRow]) -> Normalizer:
    HM = np.array([[p.H, p.M] for p in sim_parsed], dtype=np.float64)
    return Normalizer(
        mean=HM.mean(axis=0),
        std=np.maximum(HM.std(axis=0), np.array([1e-8, 1e-8], dtype=np.float64)),
        residue_target=RESIDUE_POLAR,
        pole_target=POLE_ALPHA_OMEGA,
        names=["H", "M"],
    )


def compute_output_normalizer_masked(
    parsed: List[ParsedRow],
    residue_target: str = RESIDUE_POLAR,
    pole_target: str = POLE_ALPHA_OMEGA,
    target_names: list[str] | tuple[str, ...] | None = None,
) -> Normalizer:
    pole_target = canonical_pole_target(pole_target)
    residue_target = canonical_residue_target(residue_target)
    default_names = target_names_for(pole_target, residue_target)
    available_names = list(default_names)
    if residue_target == RESIDUE_POLAR:
        available_names.append("magnitude")
    names = default_names if target_names is None else [str(name) for name in target_names]
    if not names:
        raise ValueError("target_names must not be empty")
    unsupported = [name for name in names if name not in available_names]
    if unsupported:
        raise ValueError(
            f"Targets {unsupported} are unavailable for pole_target={pole_target!r}, "
            f"residue_target={residue_target!r}; available={available_names}"
        )
    values = []
    for p in parsed:
        mask = np.asarray(p.mask, dtype=bool)
        if not mask.any():
            continue
        row_vals = row_target_values(p, pole_target, residue_target)
        mat = np.stack([row_vals[name] for name in names], axis=1)
        values.append(mat[mask])
    if not values:
        return Normalizer(
            mean=np.zeros(len(names), dtype=np.float64),
            std=np.ones(len(names), dtype=np.float64),
            residue_target=residue_target,
            pole_target=pole_target,
            names=names,
        )
    arr = np.concatenate(values, axis=0).astype(np.float64)
    mean = arr.mean(axis=0)
    std = np.maximum(arr.std(axis=0), np.full((len(names),), 1e-8, dtype=np.float64))
    norm_mask = normalize_mask_for(names)
    mean[~norm_mask] = 0.0
    std[~norm_mask] = 1.0
    return Normalizer(mean=mean, std=std, residue_target=residue_target, pole_target=pole_target, names=names)
