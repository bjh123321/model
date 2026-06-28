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
    residue_mean: np.ndarray | None = None
    residue_std: np.ndarray | None = None
    residue_global_mean: np.ndarray | None = None
    residue_global_std: np.ndarray | None = None
    def save(self, path):
        import torch as _torch
        d = {
            'format': 'list_v1',
            'mean': self.mean.astype(np.float32).tolist(),
            'std':  self.std.astype(np.float32).tolist(),
            'residue_target': self.residue_target,
        }
        if self.residue_mean is not None:
            d['residue_mean'] = self.residue_mean.astype(np.float32).tolist()
        if self.residue_std is not None:
            d['residue_std'] = self.residue_std.astype(np.float32).tolist()
        if self.residue_global_mean is not None:
            d['residue_global_mean'] = self.residue_global_mean.astype(np.float32).tolist()
        if self.residue_global_std is not None:
            d['residue_global_std'] = self.residue_global_std.astype(np.float32).tolist()
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

        def _arr_or_none(name):
            value = d.get(name)
            if value is None:
                return None
            if isinstance(value, _torch.Tensor):
                return value.detach().cpu().numpy().astype(_np.float32)
            return _np.asarray(value, dtype=_np.float32)

        return Normalizer(
            mean=mean,
            std=std,
            residue_target=str(residue_target),
            residue_mean=_arr_or_none('residue_mean'),
            residue_std=_arr_or_none('residue_std'),
            residue_global_mean=_arr_or_none('residue_global_mean'),
            residue_global_std=_arr_or_none('residue_global_std'),
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
    def residue_stats_for_row(self, p: ParsedRow, K_max: int) -> tuple[np.ndarray, np.ndarray]:
        names = self.residue_names()
        r_count = len(names)
        mean = np.zeros((r_count, K_max), dtype=np.float32)
        std = np.ones((r_count, K_max), dtype=np.float32)
        if self.residue_mean is None or self.residue_std is None:
            g_mean = self.residue_global_mean
            g_std = self.residue_global_std
            if g_mean is None:
                g_mean = np.asarray(self.mean[2:2 + r_count], dtype=np.float32)
            if g_std is None:
                g_std = np.asarray(self.std[2:2 + r_count], dtype=np.float32)
            mean[:] = np.asarray(g_mean, dtype=np.float32).reshape(r_count, 1)
            std[:] = np.asarray(g_std, dtype=np.float32).reshape(r_count, 1)
            return mean, np.maximum(std, 1e-8)

        e = int(p.exc_id)
        c = int(p.ch_id)
        max_e, max_c, max_k, _ = self.residue_mean.shape
        g_mean = np.asarray(self.residue_global_mean, dtype=np.float32).reshape(r_count)
        g_std = np.asarray(self.residue_global_std, dtype=np.float32).reshape(r_count)
        for k in range(K_max):
            if 0 <= e < max_e and 0 <= c < max_c and 0 <= k < max_k:
                mean[:, k] = self.residue_mean[e, c, k, :]
                std[:, k] = self.residue_std[e, c, k, :]
            else:
                mean[:, k] = g_mean
                std[:, k] = g_std
        return mean, np.maximum(std, 1e-8)

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
    residue_global_mean = residue_mat.mean(axis=0)
    residue_global_std = np.maximum(residue_mat.std(axis=0), 1e-8)
    mean = np.concatenate([[al_arr.mean(), om_arr.mean()], residue_global_mean], axis=0)
    std = np.concatenate([[al_arr.std(), om_arr.std()], residue_global_std], axis=0)
    std = np.maximum(std, 1e-8)

    e_num = max(int(p.exc_id) for p in parsed) + 1
    c_num = max(int(p.ch_id) for p in parsed) + 1
    k_num = max(len(p.mask) for p in parsed)
    r_count = residue_mat.shape[1]
    sums = np.zeros((e_num, c_num, k_num, r_count), dtype=np.float64)
    sums2 = np.zeros_like(sums)
    counts = np.zeros((e_num, c_num, k_num, r_count), dtype=np.float64)

    for p in parsed:
        values = _residue_values(p, residue_target)
        for k in range(int(p.K_valid)):
            e = int(p.exc_id)
            c = int(p.ch_id)
            v = values[k]
            sums[e, c, k, :] += v
            sums2[e, c, k, :] += v * v
            counts[e, c, k, :] += 1.0

    group_mean = np.broadcast_to(residue_global_mean.reshape(1, 1, 1, r_count), sums.shape).copy()
    group_std = np.broadcast_to(residue_global_std.reshape(1, 1, 1, r_count), sums.shape).copy()
    valid = counts > 0
    group_mean[valid] = sums[valid] / counts[valid]
    var = np.zeros_like(sums)
    var[valid] = sums2[valid] / counts[valid] - group_mean[valid] ** 2
    group_std[valid] = np.sqrt(np.maximum(var[valid], 1e-12))

    return Normalizer(
        mean=mean,
        std=std,
        residue_target=residue_target,
        residue_mean=group_mean,
        residue_std=group_std,
        residue_global_mean=residue_global_mean,
        residue_global_std=residue_global_std,
    )

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
