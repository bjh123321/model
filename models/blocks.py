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
        out_dim = cfg.K_max * (6 if cfg.use_presence_head else 5)   # 如果输出存在头乘6
        self.mlp = MLP(in_dim=in_dim, out_dim=out_dim, hidden=cfg.low_hidden)
    def forward(self, H, M, exc_id, ch_id):
        e_exc = self.emb_exc(exc_id)
        e_chn = self.emb_chn(ch_id)
        x = torch.cat([H, M, e_exc, e_chn], dim=-1)
        out = self.mlp(x).view(-1, self.cfg.K_max, (6 if self.cfg.use_presence_head else 5))
        u_alpha = out[...,0]; u_omega = out[...,1]; u_rho = out[...,2]; u_sin_theta = out[...,3]; u_cos_theta = out[...,4]
        if self.cfg.use_presence_head:
            u_pres  = out[...,5]
            p_exist = torch.sigmoid(u_pres)     # 存在头输出u_pres转化为概率
            return {'u_alpha': u_alpha, 'u_omega': u_omega, 'u_rho': u_rho, 'u_sin_theta': u_sin_theta, 'u_cos_theta': u_cos_theta,
                    'u_pres': u_pres, 'p_exist': p_exist}
        return {'u_alpha': u_alpha, 'u_omega': u_omega, 'u_rho': u_rho, 'u_sin_theta': u_sin_theta, 'u_cos_theta': u_cos_theta}

class LinearHFHead(nn.Module):
    def __init__(self, K_max: int, use_presence: bool):
        super().__init__()
        self.use_presence = use_presence
        nout = 6 if use_presence else 5
        self.A = nn.Parameter(torch.ones(1, K_max, nout))
        self.b = nn.Parameter(torch.zeros(1, K_max, nout))
    def forward(self, u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta, u_pres=None):
        if self.use_presence:
            u = torch.stack([u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta, u_pres], dim=-1)
        else:
            u = torch.stack([u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta], dim=-1)
        y = self.A * u + self.b
        if self.use_presence:
            return y[...,0], y[...,1], y[...,2], y[...,3], y[...,4], y[...,5]
        return y[...,0], y[...,1], y[...,2], y[...,3], y[...,4]

class ResidualHFHead(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.emb_exc = nn.Embedding(cfg.embed_excitation_num, cfg.embed_excitation_dim)
        self.emb_chn = nn.Embedding(cfg.embed_channel_num, cfg.embed_channel_dim)
        in_dim = 2 + cfg.embed_excitation_dim + cfg.embed_channel_dim + 5 + (1 if cfg.use_presence_head else 0)
        out_dim = cfg.K_max * (6 if cfg.use_presence_head else 5)
        self.mlp = MLP(in_dim, out_dim=out_dim, hidden=cfg.lin_high_hidden)
    def forward(self, H, M, exc_id, ch_id, uL_alpha, uL_omega, uL_rho, uL_sin_theta, uL_cos_theta, uL_pres=None):
        uL_pool = torch.stack([uL_alpha, uL_omega, uL_rho, uL_sin_theta, uL_cos_theta], dim=-1).mean(dim=1)
        e_exc = self.emb_exc(exc_id)
        e_chn = self.emb_chn(ch_id)
        if self.cfg.use_presence_head:
            x = torch.cat([H, M, e_exc, e_chn, uL_pool, uL_pres.mean(dim=1, keepdim=True)], dim=-1)
        else:
            x = torch.cat([H, M, e_exc, e_chn, uL_pool], dim=-1)
        out = self.mlp(x).view(-1, self.cfg.K_max, (6 if self.cfg.use_presence_head else 5))
        if self.cfg.use_presence_head:
            return out[...,0], out[...,1], out[...,2], out[...,3], out[...,4], out[...,5]
        return out[...,0], out[...,1], out[...,2], out[...,3], out[...,4]

class ParamPZK(nn.Module):
    def __init__(self):
        super().__init__()
    def to_physical(self, u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta):
        alpha = -F.softplus(u_alpha)
        omega =  F.sigmoid(u_omega)
        rho  =  u_rho
        norm = torch.sqrt(u_sin_theta.pow(2) + u_cos_theta.pow(2) + 1e-8)  # 防止除0
        sin_theta = u_sin_theta / norm
        cos_theta = u_cos_theta / norm
        return alpha, omega, rho, sin_theta, cos_theta

class MFDNN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.low = LowFidelityHead(cfg)
        self.lin = LinearHFHead(cfg.K_max, use_presence=cfg.use_presence_head)
        self.res = ResidualHFHead(cfg)
        self.param = ParamPZK()

    def forward_low(self, H, M, e, c):
        u = self.low(H, M, e, c)
        alpha, omega, rho, sin_theta, cos_theta = self.param.to_physical(u['u_alpha'], u['u_omega'], u['u_rho'], u['u_sin_theta'], u['u_cos_theta'])
        out = {**u, 'alpha': alpha, 'omega': omega, 'rho': rho, 'sin_theta': sin_theta, 'cos_theta': cos_theta}
        if self.cfg.use_presence_head:
            out['p_exist'] = u['p_exist']
            out['u_pres']  = u['u_pres']
        return out
    
    def forward_high(self, H, M, e, c, outL):
        if self.cfg.use_presence_head:
            uA, uW, uR, uS, uC, uP = self.lin(outL['u_alpha'], outL['u_omega'], outL['u_rho'], outL['u_sin_theta'], outL['u_cos_theta'], outL['u_pres'])
            dA, dW, dR, dS, dC, dP = self.res(H, M, e, c, outL['u_alpha'], outL['u_omega'], outL['u_rho'], outL['u_sin_theta'], outL['u_cos_theta'], outL['u_pres'])
            uA, uW, uR, uS, uC, uP = uA+dA, uW+dW, uR+dR, uS+dS, uC+dC, uP+dP
            alpha, omega, rho, sin_theta, cos_theta = self.param.to_physical(uA, uW, uR, uS, uC)
            p = torch.sigmoid(uP)
            return {'alpha': alpha, 'omega': omega, 'rho': rho, 'sin_theta': sin_theta, 'cos_theta': cos_theta,
                    'u_alpha': uA, 'u_omega': uW, 'u_rho': uR, 'u_sin_theta': uS, 'u_cos_theta': uC,
                    'p_exist': p, 'u_pres': uP}
        else:
            uA, uW, uR, uS, uC = self.lin(outL['u_alpha'], outL['u_omega'], outL['u_rho'], outL['u_sin_theta'], outL['u_cos_theta'])
            dA, dW, dR, dS, dC = self.res(H, M, e, c, outL['u_alpha'], outL['u_omega'], outL['u_rho'], outL['u_sin_theta'], outL['u_cos_theta'])
            uA, uW, uR, uS, uC = uA+dA, uW+dW, uR+dR, uS+dS, uC+dC
            alpha, omega, rho, sin_theta, cos_theta = self.param.to_physical(uA, uW, uR, uS, uC)
            return {'alpha': alpha, 'omega': omega, 'rho': rho, 'sin_theta': sin_theta, 'cos_theta': cos_theta,
                    'u_alpha': uA, 'u_omega': uW, 'u_rho': uR, 'u_sin_theta': uS, 'u_cos_theta': uC,}
        
    def forward(self, H, M, e, c):
        outL = self.forward_low(H, M, e, c)
        outH = self.forward_high(H, M, e, c, outL)
        return outL, outH