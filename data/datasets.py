from __future__ import annotations
from typing import List
import torch
from torch.utils.data import Dataset
from surrogate_model.data.io import ParsedRow
from surrogate_model.data.normalizer import Normalizer


def _fixed_len_tensor(values, K_max: int) -> torch.Tensor:
    out = torch.zeros((K_max,), dtype=torch.float32)
    src = torch.as_tensor(values, dtype=torch.float32).flatten()
    n = min(int(K_max), int(src.numel()))
    if n > 0:
        out[:n] = src[:n]
    return out


def build_label_tensor(p: ParsedRow, K_max: int, out_norm: Normalizer):
    alpha = _fixed_len_tensor(p.alpha, K_max)
    omega = _fixed_len_tensor(p.omega, K_max)
    alpha = (alpha - float(out_norm.mean[0])) / max(float(out_norm.std[0]), 1e-8)
    omega = (omega - float(out_norm.mean[1])) / max(float(out_norm.std[1]), 1e-8)

    residue_target = str(getattr(out_norm, "residue_target", "polar")).lower()
    if residue_target == "cartesian":
        residue = torch.stack(
            [
                _fixed_len_tensor(p.residue_re, K_max),
                _fixed_len_tensor(p.residue_im, K_max),
            ],
            dim=0,
        )
    else:
        residue = torch.stack(
            [
                _fixed_len_tensor(p.rho, K_max),
                _fixed_len_tensor(p.sin_theta, K_max),
                _fixed_len_tensor(p.cos_theta, K_max),
            ],
            dim=0,
        )

    res_mean_np, res_std_np = out_norm.residue_stats_for_row(p, K_max)
    res_mean = torch.from_numpy(res_mean_np)
    res_std = torch.from_numpy(res_std_np).clamp_min(1e-8)
    residue_norm = (residue - res_mean) / res_std
    lab_norm = torch.cat([alpha.unsqueeze(0), omega.unsqueeze(0), residue_norm], dim=0)
    return lab_norm, res_mean, res_std


class ModalDataset(Dataset):
    def __init__(self, parsed: List[ParsedRow], K_max: int, input_norm: Normalizer, output_norm: Normalizer):
        self.parsed = parsed
        self.K_max = K_max
        self.in_norm = input_norm
        self.out_norm = output_norm
    def __len__(self):
        return len(self.parsed)
    def __getitem__(self, idx):
        p = self.parsed[idx]
        HM = torch.tensor([p.H, p.M], dtype=torch.float32)
        HM = self.in_norm.transform(HM)
        Hn = HM[0:1]; Mn = HM[1:2]
        exc_id = torch.tensor(p.exc_id, dtype=torch.long)
        ch_id  = torch.tensor(p.ch_id, dtype=torch.long)
        lab_norm, res_mean, res_std = build_label_tensor(p, self.K_max, self.out_norm)
        mask = _fixed_len_tensor(p.mask, self.K_max)
        return Hn, Mn, exc_id, ch_id, mask, lab_norm, res_mean, res_std

def collate_modal(batch):
    H  = torch.stack([b[0] for b in batch], dim=0)  # (B,1)
    M  = torch.stack([b[1] for b in batch], dim=0)  # (B,1)
    e  = torch.stack([b[2] for b in batch], dim=0)  # (B,)
    c  = torch.stack([b[3] for b in batch], dim=0)  # (B,)
    m  = torch.stack([b[4] for b in batch], dim=0)  # (B,K)
    yN = torch.stack([b[5] for b in batch], dim=0)  # (B,P,K)
    res_mean = torch.stack([b[6] for b in batch], dim=0)
    res_std = torch.stack([b[7] for b in batch], dim=0)
    y_dict = {
        'alpha':     yN[:,0,:],
        'omega':     yN[:,1,:],
        'residue_mean': res_mean,
        'residue_std': res_std,
    }
    if yN.size(1) == 5:
        y_dict.update({
            'rho':       yN[:,2,:],
            'sin_theta': yN[:,3,:],
            'cos_theta': yN[:,4,:],
        })
    elif yN.size(1) == 4:
        y_dict.update({
            'residue_re': yN[:,2,:],
            'residue_im': yN[:,3,:],
        })
    else:
        raise RuntimeError(f"Unexpected label tensor shape: {tuple(yN.shape)}")
    return H, M, e, c, m, y_dict
