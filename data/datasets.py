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


def _global_stats(out_norm: Normalizer, count: int):
    mean = torch.zeros((count,), dtype=torch.float32)
    std = torch.ones((count,), dtype=torch.float32)
    src_mean = torch.as_tensor(out_norm.mean, dtype=torch.float32).flatten()
    src_std = torch.as_tensor(out_norm.std, dtype=torch.float32).flatten()
    n = min(count, int(src_mean.numel()), int(src_std.numel()))
    if n > 0:
        mean[:n] = src_mean[:n]
        std[:n] = src_std[:n].clamp_min(1e-8)
    return mean, std.clamp_min(1e-8)


def build_label_tensor(p: ParsedRow, K_max: int, out_norm: Normalizer):
    lab = torch.stack(
        [
            _fixed_len_tensor(p.freq, K_max),
            _fixed_len_tensor(p.damping_percent, K_max),
            _fixed_len_tensor(p.residue_re, K_max),
            _fixed_len_tensor(p.residue_im, K_max),
        ],
        dim=0,
    )

    mean, std = _global_stats(out_norm, lab.size(0))
    return (lab - mean[:, None]) / std[:, None]


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
        lab_norm = build_label_tensor(p, self.K_max, self.out_norm)
        mask = _fixed_len_tensor(p.mask, self.K_max)
        return Hn, Mn, exc_id, ch_id, mask, lab_norm

def collate_modal(batch):
    H  = torch.stack([b[0] for b in batch], dim=0)  # (B,1)
    M  = torch.stack([b[1] for b in batch], dim=0)  # (B,1)
    e  = torch.stack([b[2] for b in batch], dim=0)  # (B,)
    c  = torch.stack([b[3] for b in batch], dim=0)  # (B,)
    m  = torch.stack([b[4] for b in batch], dim=0)  # (B,K)
    yN = torch.stack([b[5] for b in batch], dim=0)  # (B,P,K)
    y_dict = {
        'freq':       yN[:,0,:],
        'damping':    yN[:,1,:],
        'residue_re': yN[:,2,:],
        'residue_im': yN[:,3,:],
    }
    if yN.size(1) != 4:
        raise RuntimeError(f"Unexpected label tensor shape: {tuple(yN.shape)}")
    return H, M, e, c, m, y_dict
