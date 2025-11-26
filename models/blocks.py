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
        out_dim = cfg.K_max * 6
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
        return mod.view(-1, self.cfg.K_max, 6)


class LowFidelityHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb_exc = nn.Embedding(cfg.embed_excitation_num, cfg.embed_excitation_dim)
        self.emb_chn = nn.Embedding(cfg.embed_channel_num, cfg.embed_channel_dim)

        in_dim  = 2 + cfg.embed_excitation_dim + cfg.embed_channel_dim
        nout    = 6
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
        out = self.mlp(x).view(-1, self.cfg.K_max, 6)

        # FiLM 调制在 LF 输出上（若开启）
        if getattr(self.cfg, "use_modulation", False):                     # 形状应能广播到 (B,K,nout)
            out = out * self.mod(exc_id, ch_id)

        # 拆分 u 空间分量并按是否开启存在头返回
        u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta, u_pres = out[...,0], out[...,1], out[...,2], out[...,3], out[...,4], out[...,5]

        return {
            'u_alpha':u_alpha, 'u_omega':u_omega, 'u_rho':u_rho,
            'u_sin_theta':u_sin_theta, 'u_cos_theta':u_cos_theta,
            'u_pres':u_pres
        }


class LinearHFHead(nn.Module):
    """
    LHF = 槽内特征混合(5×5，针对 α,ω,ρ,sinθ,cosθ) + pres 独立 1×1 仿射
    （槽位混合如需以后加，再在外面接 K×K 即可）
    """
    def __init__(self, K_max: int):
        super().__init__()
        self.K_max = K_max
        # 物理 5 维
        self.aff_phys = nn.Linear(5, 5, bias=True)
        # pres 独立 1×1（在线标定合成进这里）
        self.aff_pres = nn.Linear(1, 1, bias=True)
        with torch.no_grad():
            nn.init.eye_(self.aff_phys.weight)
            nn.init.zeros_(self.aff_phys.bias)
            self.aff_pres.weight.fill_(1.0)
            self.aff_pres.bias.zero_()

    def forward(self, u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta, u_pres):
        # 物理 5 维：对最后一维做 5×5
        U5 = torch.stack([u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta], dim=-1)  # (B,K,5)
        B, K, _ = U5.shape
        Y5 = self.aff_phys(U5.view(-1, 5)).view(B, K, 5)
        # pres：1×1 仿射
        pres = self.aff_pres(u_pres.unsqueeze(-1)).squeeze(-1)                          # (B,K)
        return Y5[...,0], Y5[...,1], Y5[...,2], Y5[...,3], Y5[...,4], pres



# class LinearHFHead(nn.Module):
#     def __init__(self, K_max: int):
#         super().__init__()
#         nout = 6
#         self.linear = nn.Linear(K_max * nout, K_max * nout)
#         with torch.no_grad():
#             nn.init.eye_(self.linear.weight)              # 近似恒等
#             nn.init.zeros_(self.linear.bias)

#     def forward(self, u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta, u_pres=None):
#         u_vec = torch.cat([u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta, u_pres], dim=-1)  # (B, K*nout)
#         y_vec = self.linear(u_vec)  # (B, K*nout)
#         K = u_alpha.size(1); nout = 6
#         Y = y_vec.view(-1, K, nout) # (B,K,nout)
        
#         return Y[...,0],Y[...,1],Y[...,2],Y[...,3],Y[...,4],Y[...,5]




class AdaptationHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb_exc = nn.Embedding(cfg.embed_excitation_num, cfg.embed_excitation_dim)
        self.emb_chn = nn.Embedding(cfg.embed_channel_num, cfg.embed_channel_dim)
        in_dim  = 2 + cfg.embed_excitation_dim + cfg.embed_channel_dim + 5
        out_dim = cfg.K_max * 6

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
        out = self.mlp(x).view(-1, self.cfg.K_max, 6)
        # FiLM调制只作用于残差δ
        if getattr(self.cfg, "use_modulation_in_high", False):
            out = out * self.mod(exc_id, ch_id)
        # p_exist门控减少不确定槽位的残差影响
        u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta, u_pres = (
            out[...,0], out[...,1], out[...,2], out[...,3], out[...,4], out[...,5]
        )
        return u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta, u_pres



class ParamPZK(nn.Module):
    def __init__(self):
        super().__init__()
    def to_physical(self, u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta, u_pres):
        alpha = -F.softplus(u_alpha)
        omega =  F.sigmoid(u_omega)
        rho  =  u_rho
        norm = torch.sqrt(u_sin_theta.pow(2) + u_cos_theta.pow(2) + 1e-8)  # 防止除0
        sin_theta = u_sin_theta / norm
        cos_theta = u_cos_theta / norm
        p_exist = torch.sigmoid(u_pres)
        return alpha, omega, rho, sin_theta, cos_theta, p_exist


class MFDNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.low  = LowFidelityHead(cfg)
        self.lin  = LinearHFHead(cfg.K_max)
        self.adaptation = AdaptationHead(cfg)
        self.param = ParamPZK()
        nout = 6
        self.register_buffer('online_gain', torch.ones(1, cfg.K_max, nout))
        self.register_buffer('online_bias', torch.zeros(1, cfg.K_max, nout))
        self.enable_online_fix = False
        self.log_sigma_sim  = nn.Parameter(torch.zeros(1))
        self.log_sigma_real = nn.Parameter(torch.zeros(1))

    def forward_low(self, H, M, exc_id, ch_id):
        outL = self.low(H, M, exc_id, ch_id)
        alpha, omega, rho, sin_theta, cos_theta, p_exist = self.param.to_physical(
            outL['u_alpha'], outL['u_omega'], outL['u_rho'], outL['u_sin_theta'], outL['u_cos_theta'], outL['u_pres']
        )
        out = {
            **outL,
            'alpha':alpha,'omega':omega,'rho':rho,'sin_theta':sin_theta,'cos_theta':cos_theta, 'p_exist':p_exist
        }

        return out


    def forward_high(self, H, M, exc_id, ch_id, outL: dict):
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
        alpha, omega, rho, sin_theta, cos_theta, p_exist = self.param.to_physical(
            U[...,0], U[...,1], U[...,2], U[...,3], U[...,4], U[...,5]
        )
        return {
            'alpha':alpha,'omega':omega,'rho':rho,'sin_theta':sin_theta,'cos_theta':cos_theta,'p_exist':p_exist,
            'u_alpha':U[...,0],'u_omega':U[...,1],'u_rho':U[...,2],'u_sin_theta':U[...,3],'u_cos_theta':U[...,4],'u_pres':U[...,5]
        }

    # 顶层 forward：给出低保真 + 高保真输出
    def forward(self, H, M, exc_id, ch_id):
        """
        返回一个 dict:
          - low:  低保真物理域 + u 域
          - high: 高保真物理域 + u 域
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
    def ridge_recalibrate_linear(self, U_low: torch.Tensor, U_target: torch.Tensor, reg: float = 1e-2):
        """
        用岭回归“共享 5×5 + 偏置”重标定 LHF（只对前5维; pres 交由 calibrate_pres_platt）。
        U_low:    (N, K, 6)  -- LHF 输入侧的 u（网络空间）
        U_target: (N, K, 6)  -- 真实标签反推到 u 空间后的目标
        """
        device = self.lin.aff_phys.weight.device
        U_low    = U_low.to(device)
        U_target = U_target.to(device)

        N, K, nout = U_low.shape
        assert nout == 6 and U_target.shape == (N, K, 6)

        # 只对前5维做岭回归
        X = U_low[...,  :5].reshape(N*K, 5)    # (NK,5)
        Y = U_target[..., :5].reshape(N*K, 5)  # (NK,5)

        ones = torch.ones(N*K, 1, device=device, dtype=X.dtype)
        X_aug = torch.cat([X, ones], dim=1)    # (NK,6) = [X, 1]
        G = X_aug.T @ X_aug                    # (6,6)

        Lambda = torch.zeros_like(G)
        Lambda[:5, :5].fill_(reg)              # 只正则权重，不正则偏置

        W_aug = torch.linalg.solve(G + Lambda, X_aug.T @ Y)  # (6,5)
        W = W_aug[:5, :].T.contiguous()       # (5,5) -> nn.Linear 需要 out×in
        b = W_aug[5, :].contiguous()          # (5,)

        # 写回 LHF（物理 5 维）
        self.lin.aff_phys.weight.copy_(W)
        self.lin.aff_phys.bias.copy_(b)
        # pres（第6维）保持现状，后续由 calibrate_pres_platt 合成


    @torch.no_grad()
    def calibrate_pres_platt(self, a: torch.Tensor, b: torch.Tensor):
        """
        将 pres 的 logit 做 z' = a*z + b，直接折算进 LHF 的 aff_pres（1×1 仿射）。
        a,b: 标量（tensor 或 float）
        """
        if not torch.is_tensor(a):
            a = torch.tensor(a, dtype=self.lin.aff_pres.weight.dtype, device=self.lin.aff_pres.weight.device)
        if not torch.is_tensor(b):
            b = torch.tensor(b, dtype=self.lin.aff_pres.bias.dtype, device=self.lin.aff_pres.bias.device)

        # w_new = a * w_old,  b_new = a * b_old + b
        w_old = self.lin.aff_pres.weight.detach().clone()
        b_old = self.lin.aff_pres.bias.detach().clone()
        self.lin.aff_pres.weight.copy_(a * w_old)
        self.lin.aff_pres.bias.copy_(a * b_old + b)



    # ===== 辅助：把 u 分量展平为 (B, D) =====
    def flatten_u(self, u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta, u_pres=None) -> torch.Tensor:
        U = torch.cat([u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta, u_pres], dim=-1)  # (B, K*6)
        return U
    
