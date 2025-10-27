from __future__ import annotations
from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import Config

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: list):
        super().__init__()
        h: List[nn.Module] = []
        last = in_dim
        for layer in hidden:
            h += [nn.Linear(last, layer), nn.SiLU()]
            last = layer
        h += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*h)
    def forward(self, x):
        return self.net(x)

class LowFidelityHead(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.emb_exc = nn.Embedding(cfg.embed_excitation_num, cfg.embed_excitation_dim)
        self.emb_chn = nn.Embedding(cfg.embed_channel_num, cfg.embed_channel_dim)
        in_dim = 2 + cfg.embed_excitation_dim + cfg.embed_channel_dim
        out_dim = cfg.K_max * (5 if cfg.use_presence_head else 4)   # 如果输出存在头乘5
        self.mlp = MLP(in_dim=in_dim, out_dim=out_dim, hidden=cfg.low_hidden)
    def forward(self, H, M, exc_id, ch_id):
        e_exc = self.emb_exc(exc_id)
        e_chn = self.emb_chn(ch_id)
        x = torch.cat([H, M, e_exc, e_chn], dim=-1)
        out = self.mlp(x).view(-1, self.cfg.K_max, (5 if self.cfg.use_presence_head else 4))
        u_alpha = out[...,0]; u_omega = out[...,1]; u_gain = out[...,2]; u_zero = out[...,3]
        if self.cfg.use_presence_head:
            u_pres  = out[...,4]
            p_exist = torch.sigmoid(u_pres)     # 存在头输出u_pres转化为概率
            return {'u_alpha': u_alpha, 'u_omega': u_omega, 'u_gain': u_gain, 'u_zero': u_zero,
                    'u_pres': u_pres, 'p_exist': p_exist}
        return {'u_alpha': u_alpha, 'u_omega': u_omega, 'u_gain': u_gain, 'u_zero': u_zero}

class LinearHFHead(nn.Module):
    def __init__(self, K_max: int, use_presence: bool):
        super().__init__()
        self.use_presence = use_presence
        nout = 5 if use_presence else 4
        self.A = nn.Parameter(torch.ones(1, K_max, nout))
        self.b = nn.Parameter(torch.zeros(1, K_max, nout))
    def forward(self, u_alpha, u_omega, u_gain, u_zero, u_pres=None):
        if self.use_presence:
            u = torch.stack([u_alpha, u_omega, u_gain, u_zero, u_pres], dim=-1)
        else:
            u = torch.stack([u_alpha, u_omega, u_gain, u_zero], dim=-1)
        y = self.A * u + self.b
        if self.use_presence:
            return y[...,0], y[...,1], y[...,2], y[...,3], y[...,4]
        return y[...,0], y[...,1], y[...,2], y[...,3]

class ResidualHFHead(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.emb_exc = nn.Embedding(cfg.embed_excitation_num, cfg.embed_excitation_dim)
        self.emb_chn = nn.Embedding(cfg.embed_channel_num, cfg.embed_channel_dim)
        in_dim = 2 + cfg.embed_excitation_dim + cfg.embed_channel_dim + 4 + (1 if cfg.use_presence_head else 0)
        out_dim = cfg.K_max * (5 if cfg.use_presence_head else 4)
        self.mlp = MLP(in_dim, out_dim=out_dim, hidden=cfg.lin_high_hidden)
    def forward(self, H, M, exc_id, ch_id, uL_alpha, uL_omega, uL_gain, uL_zero, uL_pres=None):
        uL_pool = torch.stack([uL_alpha, uL_omega, uL_gain, uL_zero], dim=-1).mean(dim=1)
        e_exc = self.emb_exc(exc_id)
        e_chn = self.emb_chn(ch_id)
        if self.cfg.use_presence_head:
            x = torch.cat([H, M, e_exc, e_chn, uL_pool, uL_pres.mean(dim=1, keepdim=True)], dim=-1)
        else:
            x = torch.cat([H, M, e_exc, e_chn, uL_pool], dim=-1)
        out = self.mlp(x).view(-1, self.cfg.K_max, (5 if self.cfg.use_presence_head else 4))
        if self.cfg.use_presence_head:
            return out[...,0], out[...,1], out[...,2], out[...,3], out[...,4]
        return out[...,0], out[...,1], out[...,2], out[...,3]

class ParamPZK(nn.Module):
    def __init__(self, enforce_min_phase_zero: bool=True, enforce_positive_omega: bool=True):
        super().__init__()
        self.enforce_min_phase_zero = enforce_min_phase_zero
        self.enforce_positive_omega = enforce_positive_omega
    def to_physical(self, u_alpha, u_omega, u_gain, u_zero):
        alpha = -F.softplus(u_alpha)
        omega =  F.softplus(u_omega) if self.enforce_positive_omega else u_omega
        gain  =  u_gain
        zero  = -F.softplus(u_zero) if self.enforce_min_phase_zero else u_zero
        return alpha, omega, gain, zero

class MFDNN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.low = LowFidelityHead(cfg)
        self.lin = LinearHFHead(cfg.K_max, use_presence=cfg.use_presence_head)
        self.res = ResidualHFHead(cfg)
        self.param = ParamPZK(cfg.enforce_min_phase_zero, cfg.enforce_positive_omega)

    def forward_low(self, H, M, e, c):
        u = self.low(H, M, e, c)
        a, w, g, z = self.param.to_physical(u['u_alpha'], u['u_omega'], u['u_gain'], u['u_zero'])
        out = {**u, 'alpha': a, 'omega': w, 'gain': g, 'zero': z}
        if self.cfg.use_presence_head:
            out['p_exist'] = u['p_exist']
            out['u_pres']  = u['u_pres']
        return out
    
    def forward_high(self, H, M, e, c, outL):
        if self.cfg.use_presence_head:
            uA, uW, uG, uZ, uP = self.lin(outL['u_alpha'], outL['u_omega'], outL['u_gain'], outL['u_zero'], outL['u_pres'])
            dA, dW, dG, dZ, dP = self.res(H, M, e, c, outL['u_alpha'], outL['u_omega'], outL['u_gain'], outL['u_zero'], outL['u_pres'])
            uA, uW, uG, uZ, uP = uA+dA, uW+dW, uG+dG, uZ+dZ, uP+dP
            a, w, g, z = self.param.to_physical(uA, uW, uG, uZ)
            p = torch.sigmoid(uP)
            return {'alpha': a, 'omega': w, 'gain': g, 'zero': z,
                    'u_alpha': uA, 'u_omega': uW, 'u_gain': uG, 'u_zero': uZ,
                    'p_exist': p, 'u_pres': uP}
        else:
            uA, uW, uG, uZ = self.lin(outL['u_alpha'], outL['u_omega'], outL['u_gain'], outL['u_zero'])
            dA, dW, dG, dZ = self.res(H, M, e, c, outL['u_alpha'], outL['u_omega'], outL['u_gain'], outL['u_zero'])
            uA, uW, uG, uZ = uA+dA, uW+dW, uG+dG, uZ+dZ
            a, w, g, z = self.param.to_physical(uA, uW, uG, uZ)
            return {'alpha': a, 'omega': w, 'gain': g, 'zero': z,
                    'u_alpha': uA, 'u_omega': uW, 'u_gain': uG, 'u_zero': uZ}
        
    def forward(self, H, M, e, c):
        outL = self.forward_low(H, M, e, c)
        outH = self.forward_high(H, M, e, c, outL)
        return outL, outH