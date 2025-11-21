from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
from .io import ParsedRow

@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray
    def save(self, path):
        import torch as _torch
        d = {
            'format': 'list_v1',
            'mean': self.mean.astype(np.float32).tolist(),
            'std':  self.std.astype(np.float32).tolist(),
        }
        _torch.save(d, path)
    @staticmethod
    def load(path) -> 'Normalizer':
        import torch as _torch
        import numpy as _np
        try:
            d = _torch.load(path, map_location='cpu', weights_only=True)
        except Exception:
            # 兼容旧文件（含 numpy 对象）
            d = _torch.load(path, map_location='cpu', weights_only=True)

        mean = d.get('mean')
        std  = d.get('std')

        # 统一转成 numpy
        if isinstance(mean, _torch.Tensor):
            mean = mean.detach().cpu().numpy()
        elif not isinstance(mean, _np.ndarray):
            mean = _np.asarray(mean, dtype=_np.float32)

        if isinstance(std, _torch.Tensor):
            std = std.detach().cpu().numpy()
        elif not isinstance(std, _np.ndarray):
            std = _np.asarray(std, dtype=_np.float32)

        return Normalizer(mean=mean, std=std)
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        m = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        s = torch.as_tensor(self.std, dtype=x.dtype, device=x.device).clamp_min(1e-8)
        return (x - m) / s
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        m = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        s = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        return x * s + m

def compute_input_normalizer_sim(sim_parsed: List[ParsedRow]) -> Normalizer:
    HM = np.array([[p.H, p.M] for p in sim_parsed], dtype=np.float64)
    mean = HM.mean(axis=0)
    std = HM.std(axis=0)
    return Normalizer(mean=mean, std=std)

def compute_output_normalizer_masked(parsed: List[ParsedRow]) -> Normalizer:
    al, om, rh = [], [], []
    for p in parsed:
        k = int(p.K_valid)
        if k <= 0:
            continue
        al.append(p.alpha[:k]); om.append(p.omega[:k]); rh.append(p.rho[:k])
    if len(al) == 0:
        mean = np.zeros(3, dtype=np.float64)
        std  = np.ones(3, dtype=np.float64)
        return Normalizer(mean=mean, std=std)
    al = np.concatenate(al); om = np.concatenate(om); rh = np.concatenate(rh)
    mat = np.stack([al, om, rh], axis=1)

    mean = mat.mean(axis=0)
    std  = mat.std(axis=0)
    return Normalizer(mean=mean, std=std)

if __name__ == '__main__':
    # 简单测试
    from .io import read_variable_txt, parse_rows_to_struct
    from pathlib import Path
    sim_txt = Path(__file__).parent / 'sim_dataset_new.txt'
    sim_rows  = read_variable_txt(sim_txt)
    sim_parsed, sim_stats   = parse_rows_to_struct(sim_rows)

    normalizer = compute_output_normalizer_masked(sim_parsed)

    print('sim_mean', normalizer.mean)
    print('sim_std ', normalizer.std)