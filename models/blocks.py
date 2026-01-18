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
    def __init__(self, K_max: int):
        super().__init__()
        self.K_max = K_max
        
        # 原来的 5×5 线性层保持不变
        self.aff_phys = nn.Linear(5, 5, bias=True)
        
        # 新增：状态感知的修正器 (7->5)，可以修正所有5个参数
        # 输入: [H, M, u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta]
        self.state_corrector = nn.Linear(7, 5, bias=True)
        
        # pres 独立 1×1
        self.aff_pres = nn.Linear(1, 1, bias=True)
        
        with torch.no_grad():
            # 初始化原有层
            nn.init.eye_(self.aff_phys.weight)
            nn.init.zeros_(self.aff_phys.bias)
            
            # 初始化状态修正器为零修正
            nn.init.zeros_(self.state_corrector.weight)
            nn.init.zeros_(self.state_corrector.bias)
            
            self.aff_pres.weight.fill_(1.0)
            self.aff_pres.bias.zero_()
        
        # 修正掩码：决定修正哪些参数 [alpha, omega, rho, sin_theta, cos_theta]
        self.correction_mask = torch.ones(5, dtype=torch.bool)
        # 标志位：是否启用状态修正
        self.use_state_correction = False

    def forward(self, u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta, u_pres, 
                H=None, M=None):
        U5 = torch.stack([u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta], dim=-1)  # (B,K,5)
        B, K, _ = U5.shape
        
        # 基础线性变换
        Y5_base = self.aff_phys(U5.view(-1, 5)).view(B, K, 5)
        
        # 如果提供了H,M且启用了状态修正
        if self.use_state_correction and H is not None and M is not None:
            # 扩展H,M到槽位维度
            H_exp = H.unsqueeze(1).expand(B, K, 1)  # (B,K,1)
            M_exp = M.unsqueeze(1).expand(B, K, 1)  # (B,K,1)
            
            # 构建特征: [H, M, u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta]
            features = torch.cat([
                H_exp, M_exp, U5
            ], dim=-1)  # (B,K,7)
            
            # 计算所有参数的修正量
            all_corrections = self.state_corrector(features.view(-1, 7)).view(B, K, 5)
            
            # 应用掩码：只启用选定的参数修正
            masked_corrections = torch.zeros_like(all_corrections)
            for i in range(5):
                if self.correction_mask[i]:
                    masked_corrections[..., i] = all_corrections[..., i]
            
            # 应用修正
            Y5 = Y5_base + masked_corrections
        else:
            # 回退到原始行为
            Y5 = Y5_base
        
        # pres：1×1 仿射（保持不变）
        pres = self.aff_pres(u_pres.unsqueeze(-1)).squeeze(-1)  # (B,K)
        
        return Y5[...,0], Y5[...,1], Y5[...,2], Y5[...,3], Y5[...,4], pres
    
    def enable_state_correction(self, enable=True):
        """启用或禁用状态修正"""
        self.use_state_correction = enable
    
    def set_correction_mask(self, mask):
        """
        设置修正掩码
        mask: 长度为5的布尔列表或张量，对应[alpha, omega, rho, sin_theta, cos_theta]
        例如：[False, True, False, False, False] 表示只修正omega
            [True, True, False, False, False] 表示修正alpha和omega
        """
        if isinstance(mask, list):
            mask = torch.tensor(mask, dtype=torch.bool)
        self.correction_mask = mask.to(self.state_corrector.weight.device)
        print(f"设置修正掩码: {self.correction_mask.tolist()}")
    
    def calibrate_state_corrector(self, weight, bias):
        """设置状态修正器的权重"""
        with torch.no_grad():
            self.state_corrector.weight.copy_(weight)
            self.state_corrector.bias.copy_(bias)


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
        # 调用lin时传入H和M
        uA, uW, uR, uS, uC, uP = self.lin(
            outL['u_alpha'], outL['u_omega'], outL['u_rho'], 
            outL['u_sin_theta'], outL['u_cos_theta'], outL['u_pres'],
            H, M  # 新增：传入状态信息
        )
        
        dA, dW, dR, dS, dC, dP = self.adaptation(
            H, M, exc_id, ch_id,
            outL['u_alpha'], outL['u_omega'], outL['u_rho'], 
            outL['u_sin_theta'], outL['u_cos_theta'], outL['u_pres']
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
            'alpha': alpha, 'omega': omega, 'rho': rho, 
            'sin_theta': sin_theta, 'cos_theta': cos_theta, 'p_exist': p_exist,
            'u_alpha': U[...,0], 'u_omega': U[...,1], 'u_rho': U[...,2], 
            'u_sin_theta': U[...,3], 'u_cos_theta': U[...,4], 'u_pres': U[...,5]
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


    @torch.no_grad()
    def ridge_recalibrate_linear(self, U_low: torch.Tensor, U_target: torch.Tensor,
                                H_norm: torch.Tensor, M_norm: torch.Tensor,
                                reg: float = 1e-2,
                                target_params: str = "omega"):
        """
        新的岭回归函数：可选标定参数
        
        U_low:   (1, N_total, 6) - 所有有效样本的预测U值
        U_target: (1, N_total, 6) - 所有有效样本的真实U值
        H_norm: (1, N_total, 1) - 每个样本的高度（归一化）
        M_norm: (1, N_total, 1) - 每个样本的马赫数（归一化）
        reg:    正则化系数
        target_params: 标定参数选项
            "omega": 只标定omega
            "alpha_omega": 标定alpha和omega
            "all": 标定所有5个参数
        """
        device = self.lin.aff_phys.weight.device
        
        # 去掉批次维度
        U_low = U_low.squeeze(0)    # (N_total, 6)
        U_target = U_target.squeeze(0)  # (N_total, 6)
        H_norm = H_norm.squeeze(0)  # (N_total, 1)
        M_norm = M_norm.squeeze(0)  # (N_total, 1)
        
        N_total = U_low.shape[0]
        
        # 1. 构建特征矩阵 Phi: [1, H, M, u_alpha, u_omega, u_rho, u_sin, u_cos]
        ones = torch.ones(N_total, 1, device=device)
        Phi = torch.cat([
            ones,                    # 偏置项
            H_norm,                  # 高度
            M_norm,                  # 马赫数
            U_low[:, :5]             # 低保真预测的5个参数
        ], dim=-1)  # (N_total, 8)
        
        # 2. 构建目标矩阵 Delta: 真实值 - 预测值
        Delta = U_target[:, :5] - U_low[:, :5]  # (N_total, 5)
        
        # 3. 根据target_params确定要标定的参数索引
        param_names = ["alpha", "omega", "rho", "sin_theta", "cos_theta"]
        
        if target_params == "omega":
            target_indices = [1]  # 只标定omega
        elif target_params == "alpha_omega":
            target_indices = [0, 1]  # 标定alpha和omega
        elif target_params == "all":
            target_indices = list(range(5))  # 标定所有参数
        else:
            raise ValueError(f"未知的target_params: {target_params}")
        
        print(f"标定参数: {[param_names[i] for i in target_indices]}")
        
        # 4. 初始化完整的权重矩阵和偏置向量（5×7和5维）
        W_full = torch.zeros(5, 7, device=device)
        b_full = torch.zeros(5, device=device)
        
        # 5. 对每个目标参数进行岭回归
        for i in target_indices:
            y_i = Delta[:, i]  # (N_total,)
            
            # 岭回归闭式解: (Phi^T Phi + λI) θ = Phi^T y_i
            A = Phi.T @ Phi  # (8, 8)
            B = Phi.T @ y_i  # (8,)
            
            # 正则化矩阵：只对权重部分正则化，不对偏置项正则化
            I_reg = torch.eye(8, device=device)
            I_reg[0, 0] = 0  # 不对偏置项正则化
            
            theta_i = torch.linalg.solve(A + reg * I_reg, B)  # (8,)
            
            b_i = theta_i[0]       # 偏置
            w_i = theta_i[1:]      # 权重 (7,)
            
            W_full[i] = w_i
            b_full[i] = b_i
            
            # 输出该参数的统计信息
            param_name = param_names[i]
            error_mean = y_i.mean().item()
            error_std = y_i.std().item()
            weight_norm = torch.norm(w_i).item()
            
            print(f"  {param_name}: 平均误差={error_mean:.6f}, "
                f"标准差={error_std:.6f}, 权重范数={weight_norm:.4f}")
        
        # 6. 将权重写入修正器
        self.lin.calibrate_state_corrector(W_full, b_full)
        
        # 7. 设置修正掩码
        if target_params == "omega":
            self.lin.set_correction_mask([False, True, False, False, False])
        elif target_params == "alpha_omega":
            self.lin.set_correction_mask([True, True, False, False, False])
        elif target_params == "all":
            self.lin.set_correction_mask([True, True, True, True, True])
        
        # 8. 启用状态修正
        self.lin.enable_state_correction(True)
        
        # 9. 更新原来的 aff_phys 偏置（只更新被标定的参数）
        with torch.no_grad():
            for i in target_indices:
                error_mean = Delta[:, i].mean().item()
                self.lin.aff_phys.bias[i] += error_mean
        
        print(f"岭回归完成，使用 {N_total} 个有效样本")
        print(f"正则化系数 λ={reg}")
        
        # 10. 诊断几个样本的修正效果
        if N_total > 0:
            print("\n修正效果诊断（前3个样本）:")
            for sample_idx in range(min(3, N_total)):
                # 构建特征
                phi_i = torch.cat([
                    torch.ones(1, 1, device=device),
                    H_norm[sample_idx:sample_idx+1],
                    M_norm[sample_idx:sample_idx+1],
                    U_low[sample_idx:sample_idx+1, :5]
                ], dim=-1)
                
                # 计算预测修正量
                corrections_pred = (phi_i[:, 1:] @ W_full.T + b_full).squeeze()
                corrections_true = Delta[sample_idx]
                
                print(f"  样本{sample_idx}:")
                for i in range(5):
                    if i in target_indices:
                        pred = corrections_pred[i].item()
                        true = corrections_true[i].item()
                        ratio = pred / (true + 1e-8)
                        print(f"    {param_names[i]}: 预测修正={pred:.6f}, "
                            f"需要修正={true:.6f}, 比例={ratio:.2f}")


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
    
