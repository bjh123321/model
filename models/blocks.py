from __future__ import annotations
from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import Config

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: list, dropout: float=0.0):
        super().__init__()
        h: List[nn.Module] = []
        last = in_dim
        for layer in hidden:
            h += [nn.Linear(last, layer), nn.SiLU()]
            if dropout > 0.0:
                h += [nn.Dropout(dropout)]
            last = layer
        h += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*h)
    def forward(self, x):
        return self.net(x)


class LoRALinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.base = nn.Linear(in_dim, out_dim)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        self.scale = alpha / max(rank, 1)
        # 初始化
        nn.init.xavier_uniform_(self.base.weight)
        nn.init.zeros_(self.base.bias)
        nn.init.normal_(self.lora_A, std=1e-3)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        y0 = self.base(x)
        if self.lora_A.numel() == 0:   # rank=0 保护
            return y0
        # y = W x + (B (A x)) * scale
        return y0 + self.scale * (x @ self.lora_A.T) @ self.lora_B.T


class ModulationHead(nn.Module):
    """
    调制头：根据离散激励/通道信息生成调制向量，对低保真预测进行条件调制。
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # 独立的离散嵌入
        self.emb_exc = nn.Embedding(cfg.embed_excitation_num, cfg.embed_excitation_dim)
        self.emb_chn = nn.Embedding(cfg.embed_channel_num, cfg.embed_channel_dim)
        in_dim = cfg.embed_excitation_dim + cfg.embed_channel_dim
        # 使用配置中的 mod_hidden 作为隐藏层
        out_dim = cfg.K_max * (6 if cfg.use_presence_head else 5)
        self.mlp = MLP(in_dim=in_dim, out_dim=out_dim, hidden=cfg.mod_hidden, dropout=cfg.dropout_prob)
        with torch.no_grad():
            last_linear = self.mlp.net[-1]
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)
        # 调制系数幅度
        self.gamma = getattr(cfg, 'mod_gamma', 0.2)

    def forward(self, exc_id: torch.Tensor, ch_id: torch.Tensor) -> torch.Tensor:
        e_exc = self.emb_exc(exc_id)
        e_chn = self.emb_chn(ch_id)
        x = torch.cat([e_exc, e_chn], dim=-1)
        h = self.mlp(x)
        mod = 1.0 + self.gamma * torch.tanh(h)
        return mod.view(-1, self.cfg.K_max, (6 if self.cfg.use_presence_head else 5))


class LowFidelityHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb_exc = nn.Embedding(cfg.embed_excitation_num, cfg.embed_excitation_dim)
        self.emb_chn = nn.Embedding(cfg.embed_channel_num, cfg.embed_channel_dim)

        in_dim  = 2 + cfg.embed_excitation_dim + cfg.embed_channel_dim
        nout    = 6 if cfg.use_presence_head else 5
        out_dim = cfg.K_max * nout

        # === 一定要无条件创建 layers，再无条件创建 self.mlp ===
        layers: List[nn.Module] = []
        last = in_dim
        # 允许 cfg.low_hidden 为空
        for h in (getattr(cfg, "low_hidden", []) or []):
            if getattr(cfg, "use_lora", False):
                layers.append(LoRALinear(last, h, rank=cfg.lora_rank))
            else:
                layers.append(nn.Linear(last, h))
            layers.append(nn.SiLU())
            if getattr(cfg, "dropout_prob", 0.0) > 0:
                layers.append(nn.Dropout(cfg.dropout_prob))
            last = h

        # 输出层（无条件添加）
        if getattr(cfg, "use_lora", False):
            layers.append(LoRALinear(last, out_dim, rank=cfg.lora_rank))
        else:
            layers.append(nn.Linear(last, out_dim))

        self.mlp = nn.Sequential(*layers)

        # FiLM 调制头：可选开启
        if getattr(cfg, "use_modulation", False):
            self.mod = ModulationHead(cfg)

        # 防合并/缩进错误：开局断言一次，训练期若缺失能及早发现
        assert hasattr(self, "mlp"), "LowFidelityHead: self.mlp was not created"


    def forward(self, H, M, exc_id, ch_id):
        # H, M: (B,1) float; exc_id, ch_id: (B,) long
        e_exc = self.emb_exc(exc_id)
        e_chn = self.emb_chn(ch_id)
        x = torch.cat([H, M, e_exc, e_chn], dim=-1)                  # (B, 2+emb_e+emb_c)
        out = self.mlp(x).view(-1, self.cfg.K_max, (6 if self.cfg.use_presence_head else 5))

        # FiLM 调制在 LF 输出上（若开启）
        if getattr(self.cfg, "use_modulation", False):                     # 形状应能广播到 (B,K,nout)
            out = out * self.mod(exc_id, ch_id)

        # 拆分 u 空间分量并按是否开启存在头返回
        u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta = out[...,0], out[...,1], out[...,2], out[...,3], out[...,4]
        if self.cfg.use_presence_head:
            u_pres = out[...,5]
            p_exist = torch.sigmoid(u_pres)
            return {
                'u_alpha':u_alpha, 'u_omega':u_omega, 'u_rho':u_rho,
                'u_sin_theta':u_sin_theta, 'u_cos_theta':u_cos_theta,
                'u_pres':u_pres, 'p_exist':p_exist
            }
        else:
            return {
                'u_alpha':u_alpha, 'u_omega':u_omega, 'u_rho':u_rho,
                'u_sin_theta':u_sin_theta, 'u_cos_theta':u_cos_theta
            }



class LinearHFHead(nn.Module):
    def __init__(self, K_max: int, use_presence: bool):
        super().__init__()
        self.use_presence = use_presence
        nout = 6 if use_presence else 5
        self.linear = nn.Linear(K_max * nout, K_max * nout)
        with torch.no_grad():
            nn.init.eye_(self.linear.weight)              # 近似恒等
            nn.init.zeros_(self.linear.bias)

    def forward(self, u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta, u_pres=None):
        if self.use_presence:
            u_vec = torch.cat([u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta, u_pres], dim=-1)  # (B, K*nout)
        else:
            u_vec = torch.cat([u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta], dim=-1)          # (B, K*nout)
        y_vec = self.linear(u_vec)  # (B, K*nout)
        K = u_alpha.size(1); nout = 6 if self.use_presence else 5
        Y = y_vec.view(-1, K, nout) # (B,K,nout)
        if self.use_presence:
            return Y[...,0],Y[...,1],Y[...,2],Y[...,3],Y[...,4],Y[...,5]
        else:
            return Y[...,0],Y[...,1],Y[...,2],Y[...,3],Y[...,4]


class AdaptationHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb_exc = nn.Embedding(cfg.embed_excitation_num, cfg.embed_excitation_dim)
        self.emb_chn = nn.Embedding(cfg.embed_channel_num, cfg.embed_channel_dim)
        in_dim  = 2 + cfg.embed_excitation_dim + cfg.embed_channel_dim + 5
        out_dim = cfg.K_max * (6 if cfg.use_presence_head else 5)

        layers: List[nn.Module] = []
        last = in_dim
        for h in cfg.adapt_hidden:
            if cfg.use_lora:
                layers.append(LoRALinear(last, h, rank=cfg.lora_rank))
            else:
                layers.append(nn.Linear(last, h))
            layers.append(nn.SiLU())
            if cfg.dropout_prob > 0:
                layers.append(nn.Dropout(cfg.dropout_prob))
            last = h
        if cfg.use_lora:
            layers.append(LoRALinear(last, out_dim, rank=cfg.lora_rank))
        else:
            layers.append(nn.Linear(last, out_dim))
        self.mlp = nn.Sequential(*layers)

        # # 以“δ=0”起步更稳
        # with torch.no_grad():
        #     for m in self.mlp:
        #         # LoRALinear 没有 weight/bias 属性，真实权重在 base 里
        #         if isinstance(m, LoRALinear):
        #             nn.init.zeros_(m.base.weight)
        #             nn.init.zeros_(m.base.bias)
        #             # 清零 LoRA 专用 A/B
        #             nn.init.zeros_(m.lora_A)
        #             nn.init.zeros_(m.lora_B)
        #         else:
        #             nn.init.zeros_(m.weight)
        #             nn.init.zeros_(m.bias)
        #     # NHF的FiLM
        if getattr(self.cfg, "use_modulation_in_high", False):
            self.mod = ModulationHead(cfg)
        self.zero_all_adaptation_params()

    def zero_all_adaptation_params(self):
        """
        将 AdaptationHead 的所有 Linear / LoRALinear 层全部置零。
        Stage1 时就不会对 outL 产生任何残差。
        """
        with torch.no_grad():
            for m in self.mlp:
                if isinstance(m, LoRALinear):
                    nn.init.zeros_(m.base.weight)
                    nn.init.zeros_(m.base.bias)
                    nn.init.zeros_(m.lora_A)
                    nn.init.zeros_(m.lora_B)
                elif isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)

            if hasattr(self, "mod"):
                for m in self.mod.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.zeros_(m.weight)
                        nn.init.zeros_(m.bias)

    def forward(self, H, M, exc_id, ch_id,
                uL_alpha, uL_omega, uL_rho, uL_sin_theta, uL_cos_theta, uL_pres=None):
        # 跨槽位的LF全局池化（稳定尺度）
        uL_pool = torch.stack([uL_alpha,uL_omega,uL_rho,uL_sin_theta,uL_cos_theta], dim=-1).mean(dim=1)  # (B,5)
        e_exc = self.emb_exc(exc_id); e_chn = self.emb_chn(ch_id)
        x = torch.cat([H, M, e_exc, e_chn, uL_pool], dim=-1)
        out = self.mlp(x).view(-1, self.cfg.K_max, (6 if self.cfg.use_presence_head else 5))
        # FiLM调制只作用于残差δ
        if getattr(self.cfg, "use_modulation_in_high", False):
            out = out * self.mod(exc_id, ch_id)
        # p_exist门控减少不确定槽位的残差影响
        if self.cfg.use_presence_head:
            u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta, u_pres = (
                out[...,0], out[...,1], out[...,2], out[...,3], out[...,4], out[...,5]
            )
            p_exist = torch.sigmoid(u_pres)
            return u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta, u_pres
        else:
            u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta = (
                out[...,0], out[...,1], out[...,2], out[...,3], out[...,4]
            )
            return u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta



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
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.low  = LowFidelityHead(cfg)
        self.lin  = LinearHFHead(cfg.K_max, use_presence=cfg.use_presence_head)
        self.adaptation = AdaptationHead(cfg)
        self.param = ParamPZK()
        nout = 6 if cfg.use_presence_head else 5
        self.register_buffer('online_gain', torch.ones(1, cfg.K_max, nout))
        self.register_buffer('online_bias', torch.zeros(1, cfg.K_max, nout))
        self.enable_online_fix = False
        self.log_sigma_sim  = nn.Parameter(torch.zeros(1))
        self.log_sigma_real = nn.Parameter(torch.zeros(1))

    def forward_low(self, H, M, exc_id, ch_id):
        outL = self.low(H, M, exc_id, ch_id)
        alpha, omega, rho, sin_theta, cos_theta = self.param.to_physical(
            outL['u_alpha'], outL['u_omega'], outL['u_rho'], outL['u_sin_theta'], outL['u_cos_theta']
        )
        out = {
            **outL,
            'alpha':alpha,'omega':omega,'rho':rho,'sin_theta':sin_theta,'cos_theta':cos_theta
        }
        if self.cfg.use_presence_head:
            out['p_exist'] = outL['p_exist']
            out['u_pres']  = outL['u_pres']
        return out


    def forward_high(self, H, M, exc_id, ch_id, outL: dict):
        if self.cfg.use_presence_head:
            uA,uW,uR,uS,uC,uP = self.lin(
                outL['u_alpha'], outL['u_omega'], outL['u_rho'], outL['u_sin_theta'], outL['u_cos_theta'], outL['u_pres']
            )
            dA,dW,dR,dS,dC,dP = self.adaptation(
                H, M, exc_id, ch_id,
                outL['u_alpha'], outL['u_omega'], outL['u_rho'], outL['u_sin_theta'], outL['u_cos_theta'], outL['u_pres']
            )
            # 合成 (LHF + NHF)
            U = torch.stack([uA+dA, uW+dW, uR+dR, uS+dS, uC+dC, uP+dP], dim=-1)  # (B,K,6)
            # 在线快修偏置
            if self.enable_online_fix:
                U = self.online_gain * U + self.online_bias
            # 物理域映射
            alpha, omega, rho, sin_theta, cos_theta = self.param.to_physical(
                U[...,0], U[...,1], U[...,2], U[...,3], U[...,4]
            )
            p_exist = torch.sigmoid(U[...,5])
            return {
                'alpha':alpha,'omega':omega,'rho':rho,'sin_theta':sin_theta,'cos_theta':cos_theta,
                'u_alpha':U[...,0],'u_omega':U[...,1],'u_rho':U[...,2],'u_sin_theta':U[...,3],'u_cos_theta':U[...,4],
                'p_exist':p_exist,'u_pres':U[...,5]
            }
        else:
            # === 没有存在头的情况（nout=5） ===
            uA,uW,uR,uS,uC = self.lin(
                outL['u_alpha'], outL['u_omega'], outL['u_rho'], outL['u_sin_theta'], outL['u_cos_theta'], None
            )
            dA,dW,dR,dS,dC = self.adaptation(
                H, M, exc_id, ch_id,
                outL['u_alpha'], outL['u_omega'], outL['u_rho'], outL['u_sin_theta'], outL['u_cos_theta'], None
            )
            U = torch.stack([uA+dA, uW+dW, uR+dR, uS+dS, uC+dC], dim=-1)        # (B,K,5)
            if self.enable_online_fix:
                U = self.online_gain * U + self.online_bias                    # 同样在线仿射
            alpha, omega, rho, sin_theta, cos_theta = self.param.to_physical(
                U[...,0], U[...,1], U[...,2], U[...,3], U[...,4]
            )
            return {
                'alpha':alpha,'omega':omega,'rho':rho,'sin_theta':sin_theta,'cos_theta':cos_theta,
                'u_alpha':U[...,0],'u_omega':U[...,1],'u_rho':U[...,2],'u_sin_theta':U[...,3],'u_cos_theta':U[...,4]
            }

    # 顶层 forward：给出低保真 + 高保真输出
    def forward(self, H, M, exc_id, ch_id):
        """
        返回一个 dict:
          - low:  低保真物理域 + u 域（含 p_exist/u_pres）
          - high: 高保真物理域 + u 域（含 p_exist/u_pres）
        """
        outL = self.forward_low(H, M, exc_id, ch_id)
        outH = self.forward_high(H, M, exc_id, ch_id, outL)
        return outL, outH

    # ===== 在线快修：偏置/仿射（简化为偏置） =====
    @torch.no_grad()
    def quick_fix_biasgain(self, delta_or_ab: dict):
        """
        [MOD] 在线快修入口：
        - {'delta': (1,K,nout)} : 只修偏置，等价于 a=1, b=delta
        - {'a':..., 'b':...}    : 完整仿射 U_new = a ⊙ U_old + b
        """
        if 'delta' in delta_or_ab:
            delta = delta_or_ab['delta'].to(self.online_bias.device).view_as(self.online_bias)
            self.online_bias.copy_(delta)
            self.online_gain.fill_(1.0)
        elif ('a' in delta_or_ab) and ('b' in delta_or_ab):
            a = delta_or_ab['a'].to(self.online_gain.device).view_as(self.online_gain)
            b = delta_or_ab['b'].to(self.online_bias.device).view_as(self.online_bias)
            self.online_gain.copy_(a)
            self.online_bias.copy_(b)
        else:
            raise ValueError("expect {'delta':...} or {'a':...,'b':...}")
        
    @torch.no_grad()
    def reset_quick_fix(self):
        """清空快修：gain=1, bias=0，并关闭 quick_fix。"""
        self.online_gain.fill_(1.0)
        self.online_bias.zero_()
        self.enable_online_fix = False

    @torch.no_grad()
    def enable_quick_fix(self, flag: bool = True):
        """显式打开/关闭 quick_fix."""
        self.enable_online_fix = flag


    # ===== 岭回归重标定 LHF：直接更新 LinearHFHead 的 W,b =====
    @torch.no_grad()
    def ridge_recalibrate_linear(self, X_low: torch.Tensor, Y_target_u: torch.Tensor, reg: float = 1e-2):
        """
        X_low: (N, D) = 由 LF 的 u 输出(含 LF-FiLM)展平得到
        Y_target_u: (N, D) = 目标物理值反算回 u 空间后展平
        仅更新 self.lin.linear 的 weight/bias（无激活线性），其它网络不动。
        """
        device = self.lin.linear.weight.device
        X = X_low.to(device); Y = Y_target_u.to(device)
        N, D = X.shape
        assert Y.shape == (N, D), f"Y shape {Y.shape} mismatch X shape {(N,D)}"

        # 带偏置的岭回归闭式解
        X_aug = torch.cat([X, torch.ones(N,1, device=device)], dim=1)   # (N, D+1)
        G = X_aug.T @ X_aug                                             # (D+1, D+1)
        # 只对权重部分加正则（不惩罚 bias）
        Lambda = torch.zeros_like(G); Lambda[:D,:D].fill_(reg)
        W_aug = torch.linalg.solve(G + Lambda, X_aug.T @ Y)             # (D+1, D)
        W, b = W_aug[:D,:], W_aug[D:,:].squeeze(0)                      # W:(D,D), b:(D,)

        # 写回 nn.Linear（weight: (D_out,D) = (D,D)）
        self.lin.linear.weight.copy_(W.T.contiguous())
        self.lin.linear.bias.copy_(b.contiguous())

    # ===== 辅助：把 u 分量展平为 (B, D) =====
    def flatten_u(self, u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta, u_pres=None) -> torch.Tensor:
        if self.cfg.use_presence_head:
            assert u_pres is not None, "use_presence_head=True 时必须给 u_pres"
            U = torch.cat([u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta, u_pres], dim=-1)  # (B, K*6)
        else:
            U = torch.cat([u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta], dim=-1)          # (B, K*5)
        return U
    
