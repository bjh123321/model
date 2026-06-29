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
    residue_target: str = "fixed"
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
        residue_target = d.get('residue_target', 'fixed')

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
        return ["residue_re", "residue_im"]

    def target_names(self) -> list[str]:
        return ["freq", "damping", "residue_re", "residue_im"]

def compute_input_normalizer_sim(sim_parsed: List[ParsedRow]) -> Normalizer:
    HM = np.array([[p.H, p.M] for p in sim_parsed], dtype=np.float64)
    mean = HM.mean(axis=0)
    std = HM.std(axis=0)
    return Normalizer(mean=mean, std=std)

def compute_output_normalizer_masked(parsed: List[ParsedRow], residue_target: str = "fixed") -> Normalizer:
    values = []
    for p in parsed:
        mask = np.asarray(p.mask, dtype=bool)
        if not mask.any():
            continue
        mat = np.stack([p.freq, p.damping_percent, p.residue_re, p.residue_im], axis=1)
        values.append(mat[mask])
    if not values:
        return Normalizer(mean=np.zeros(4, dtype=np.float64), std=np.ones(4, dtype=np.float64), residue_target="fixed")

    arr = np.concatenate(values, axis=0).astype(np.float64)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    min_std = np.array([1e-6, 1e-6, 1e-8, 1e-8], dtype=np.float64)
    small = std < min_std
    if np.any(small):
        names = np.array(["freq", "damping_percent", "residue_re", "residue_im"])
        print(
            "[fixed-slots] output std too small; using safe floor for "
            + ", ".join(names[small].tolist())
        )
    std = np.maximum(std, min_std)
    return Normalizer(mean=mean, std=std, residue_target="fixed")

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
