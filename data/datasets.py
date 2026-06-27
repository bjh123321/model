from __future__ import annotations
from typing import List
import torch
from torch.utils.data import Dataset
from surrogate_model.data.io import ParsedRow
from surrogate_model.data.normalizer import Normalizer

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
        alpha = torch.from_numpy(p.alpha.copy())
        omega = torch.from_numpy(p.omega.copy())
        rho  = torch.from_numpy(p.rho.copy())
        sin_theta = torch.from_numpy(p.sin_theta.copy())
        cos_theta = torch.from_numpy(p.cos_theta.copy())
        lab = torch.stack([alpha, omega, rho], dim=0)  # (4, K) 
        lab_norm = (lab - torch.as_tensor(self.out_norm.mean[:,None], dtype=lab.dtype)) / torch.as_tensor(self.out_norm.std[:,None], dtype=lab.dtype).clamp_min(1e-8)
        sin_theta = sin_theta.unsqueeze(0)  # (1, K)
        cos_theta = cos_theta.unsqueeze(0)
        lab_norm = torch.cat([lab_norm, sin_theta, cos_theta], dim=0)
        mask = torch.from_numpy(p.mask.copy())
        return Hn, Mn, exc_id, ch_id, mask, lab_norm

def collate_modal(batch):
    H  = torch.stack([b[0] for b in batch], dim=0)  # (B,1)
    M  = torch.stack([b[1] for b in batch], dim=0)  # (B,1)
    e  = torch.stack([b[2] for b in batch], dim=0)  # (B,)
    c  = torch.stack([b[3] for b in batch], dim=0)  # (B,)
    m  = torch.stack([b[4] for b in batch], dim=0)  # (B,K)
    yN = torch.stack([b[5] for b in batch], dim=0)  # (B,4,K)
    y_dict = {
        'alpha':     yN[:,0,:],
        'omega':     yN[:,1,:],
        'rho':       yN[:,2,:],
        'sin_theta': yN[:,3,:],
        'cos_theta': yN[:,4,:]
    }
    return H, M, e, c, m, y_dict
