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
            'std':  self.std.astype(np.float32).tolist(),
            'residue_target': self.residue_target,
        }
        _torch.save(d, path)
    @staticmethod
    def load(path) -> 'Normalizer':
        import torch as _torch
        import numpy as _np
        try:
            d = _torch.load(path, map_location='cpu')
        except Exception:
            # 兼容旧文件（含 numpy 对象）
            d = _torch.load(path, map_location='cpu')

        mean = d.get('mean')
        std  = d.get('std')
        residue_target = d.get('residue_target', 'polar')

        # 统一转成 numpy
        if isinstance(mean, _torch.Tensor):
            mean = mean.detach().cpu().numpy()
        elif not isinstance(mean, _np.ndarray):
            mean = _np.asarray(mean, dtype=_np.float32)

        if isinstance(std, _torch.Tensor):
            std = std.detach().cpu().numpy()
        elif not isinstance(std, _np.ndarray):
            std = _np.asarray(std, dtype=_np.float32)

        return Normalizer(
            mean=mean,
            std=std,
            residue_target=str(residue_target),
        )
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        m = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        s = torch.as_tensor(self.std, dtype=x.dtype, device=x.device).clamp_min(1e-8)
        return (x - m) / s
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        m = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        s = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        return x * s + m
    def residue_names(self) -> list[str]:
        target = str(getattr(self, "residue_target", "polar")).lower()
        if target == "cartesian":
            return ["residue_re", "residue_im"]
        return ["rho", "sin_theta", "cos_theta"]

def compute_input_normalizer_sim(sim_parsed: List[ParsedRow]) -> Normalizer:
    HM = np.array([[p.H, p.M] for p in sim_parsed], dtype=np.float64)
    mean = HM.mean(axis=0)
    std = HM.std(axis=0)
    return Normalizer(mean=mean, std=std)

def _residue_values(p: ParsedRow, target: str) -> np.ndarray:
    if target == "cartesian":
        return np.stack([p.residue_re, p.residue_im], axis=1)
    return np.stack([p.rho, p.sin_theta, p.cos_theta], axis=1)


def compute_output_normalizer_masked(parsed: List[ParsedRow], residue_target: str = "polar") -> Normalizer:
    residue_target = str(residue_target).lower()
    if residue_target not in {"polar", "cartesian"}:
        raise ValueError(f"Unsupported residue_target: {residue_target}")

    al, om, residue_all = [], [], []
    for p in parsed:
        k = int(p.K_valid)
        if k <= 0:
            continue
        al.append(p.alpha[:k]); om.append(p.omega[:k]); residue_all.append(_residue_values(p, residue_target)[:k])
    if len(al) == 0:
        r_count = 2 if residue_target == "cartesian" else 3
        mean = np.zeros(2 + r_count, dtype=np.float64)
        std  = np.ones(2 + r_count, dtype=np.float64)
        return Normalizer(mean=mean, std=std, residue_target=residue_target)

    al_arr = np.concatenate(al)
    om_arr = np.concatenate(om)
    residue_mat = np.concatenate(residue_all, axis=0)
    residue_param_mean = residue_mat.mean(axis=0)
    residue_param_std = np.maximum(residue_mat.std(axis=0), 1e-8)
    mean = np.concatenate([[al_arr.mean(), om_arr.mean()], residue_param_mean], axis=0)
    std = np.concatenate([[al_arr.std(), om_arr.std()], residue_param_std], axis=0)
    std = np.maximum(std, 1e-8)
    return Normalizer(mean=mean, std=std, residue_target=residue_target)

if __name__ == '__main__':
    # 简单测试
    from surrogate_model.data.io import read_variable_txt, parse_rows_to_struct
    from pathlib import Path
    sim_txt = Path(__file__).parent / 'sim_dataset_new.txt'
    sim_rows  = read_variable_txt(sim_txt)
    sim_parsed, sim_stats   = parse_rows_to_struct(sim_rows)

    normalizer = compute_output_normalizer_masked(sim_parsed)

    print('sim_mean', normalizer.mean)
    print('sim_std ', normalizer.std)
