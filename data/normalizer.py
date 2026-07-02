from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
from surrogate_model.data.io import ParsedRow


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray
    residue_target: str = "polar"

    def save(self, path):
        import torch as _torch
        d = {
            'format': 'list_v1',
            'mean': self.mean.astype(np.float32).tolist(),
            'std': self.std.astype(np.float32).tolist(),
            'residue_target': self.residue_target,
        }
        _torch.save(d, path)

    @staticmethod
    def load(path) -> 'Normalizer':
        import torch as _torch
        import numpy as _np
        d = _torch.load(path, map_location='cpu')
        mean = d.get('mean')
        std = d.get('std')
        residue_target = d.get('residue_target', 'polar')
        if isinstance(mean, _torch.Tensor):
            mean = mean.detach().cpu().numpy()
        elif not isinstance(mean, _np.ndarray):
            mean = _np.asarray(mean, dtype=_np.float32)
        if isinstance(std, _torch.Tensor):
            std = std.detach().cpu().numpy()
        elif not isinstance(std, _np.ndarray):
            std = _np.asarray(std, dtype=_np.float32)
        return Normalizer(mean=mean, std=std, residue_target=str(residue_target))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        m = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        s = torch.as_tensor(self.std, dtype=x.dtype, device=x.device).clamp_min(1e-8)
        return (x - m) / s

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        m = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        s = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        return x * s + m

    def residue_names(self) -> list[str]:
        return ["rho", "sin_theta", "cos_theta"]

    def target_names(self) -> list[str]:
        return ["alpha", "omega", "rho", "sin_theta", "cos_theta"]


def compute_input_normalizer_sim(sim_parsed: List[ParsedRow]) -> Normalizer:
    HM = np.array([[p.H, p.M] for p in sim_parsed], dtype=np.float64)
    return Normalizer(mean=HM.mean(axis=0), std=HM.std(axis=0))


def compute_output_normalizer_masked(parsed: List[ParsedRow], residue_target: str = "polar") -> Normalizer:
    values = []
    for p in parsed:
        mask = np.asarray(p.mask, dtype=bool)
        if not mask.any():
            continue
        mat = np.stack([p.alpha, p.omega, p.rho], axis=1)
        values.append(mat[mask])
    if not values:
        return Normalizer(mean=np.zeros(3, dtype=np.float64), std=np.ones(3, dtype=np.float64), residue_target="polar")
    arr = np.concatenate(values, axis=0).astype(np.float64)
    mean = arr.mean(axis=0)
    std = np.maximum(arr.std(axis=0), np.array([1e-8, 1e-8, 1e-8], dtype=np.float64))
    return Normalizer(mean=mean, std=std, residue_target="polar")
