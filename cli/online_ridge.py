# surrogate_model/cli/online_ridge.py
# %%
import argparse
import torch
import csv
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

from surrogate_model.config import Config
from surrogate_model.data.io import parse_rows_to_struct
from surrogate_model.data.normalizer import Normalizer
from surrogate_model.models.blocks import MFDNN


# ===========================
# 物理域 → u-space 反推
# ===========================
def invert_to_u_space(alpha, omega, rho, sin_t, cos_t, mask, r):
    neg_alpha = -alpha
    u_alpha = torch.log(torch.exp(neg_alpha) - 1.0 + 1e-12)

    # u_omega (inverse sigmoid)
    omega = omega.clamp(1e-6, 1 - 1e-6)
    u_omega = torch.log(omega / (1 - omega))
    u_rho = rho
    u_sin = r * sin_t
    u_cos = r * cos_t
    u_pres = torch.where(
        mask > 0.5,
        torch.full_like(mask, 10.0),
        torch.full_like(mask, -2.0),
    )

    return u_alpha, u_omega, u_rho, u_sin, u_cos, u_pres


def normlizer(pred: torch.Tensor, mean, std):
    # 转为 torch.Tensor 并对齐设备和精度
    mean_t = torch.as_tensor(mean, dtype=pred.dtype, device=pred.device)
    std_t  = torch.as_tensor(std,  dtype=pred.dtype, device=pred.device)

    # 只对前 3 维 [alpha, omega, rho] 做标准化
    core = pred[..., :3]       # (..., 3)
    others = pred[..., 3:]     # (..., 2) -> [sin_theta, cos_theta]

    # mean/std 长度 = 3，这里 reshape 成 (..., 3)，前面的维度用 1 方便广播
    assert mean_t.numel() == 3 and std_t.numel() == 3, "mean/std 必须是长度为 3"
    view_shape = [1] * (core.ndim - 1) + [3]   # 例如 core 是 (B,K,3) -> mean/std 变成 (1,1,3)
    mean_t = mean_t.view(*view_shape)
    std_t  = std_t.view(*view_shape)

    core_n = (core - mean_t) / std_t  # 标准化后的 (alpha, omega, rho)

    # 再把没归一化的 sinθ, cosθ 拼回去
    pred_norm = torch.cat([core_n, others], dim=-1)  # (..., 5)

    return pred_norm


# [MOD BEGIN] use numpy with linear_sum_assignment，去掉对全局 device 的依赖
def hungarian_match(
    predN: torch.Tensor,   # (N,K,5) after norm, 实际你传的是 (N,K,5) 或 (N,K,6) 取前5
    tgt: torch.Tensor,     # (N,K,6)，其中第5列是 mask
    w_alpha=1.0, w_omega=1.0, w_rho=1.0,
    big=1e6
):
    PA, PO, PR, PS, PC = predN[:,:,0], predN[:,:,1], predN[:,:,2], predN[:,:,3], predN[:,:,4]
    TA, TO, TR, TS, TC, mask = tgt[:,:,0], tgt[:,:,1], tgt[:,:,2], tgt[:,:,3], tgt[:,:,4], tgt[:,:,5]
    B = PA.size(0); K = PA.size(1)

    dA = (PA.unsqueeze(2) - TA.unsqueeze(1))**2
    dO = (PO.unsqueeze(2) - TO.unsqueeze(1))**2
    dR = (PR.unsqueeze(2) - TR.unsqueeze(1))**2
    dS = (PS.unsqueeze(2) - TS.unsqueeze(1))**2
    dC = (PC.unsqueeze(2) - TC.unsqueeze(1))**2
    cost = dA + dO + dR + dS + dC

    pm = torch.ones(B, K, 1, dtype=cost.dtype, device=cost.device)
    tm = (mask > 0.5).to(cost.dtype).unsqueeze(1)
    valid_pair = pm * tm

    cost = torch.where(valid_pair > 0.5, cost, cost.new_full(cost.shape, big))
    cost = torch.nan_to_num(cost, nan=big, posinf=big, neginf=big)

    assign_rows, assign_cols, m_eff_list = [], [], []
    for b in range(B):
        cb_np = cost[b].detach().cpu().numpy()   # <-- numpy
        row_ind, col_ind = linear_sum_assignment(cb_np)
        keep = (cb_np[row_ind, col_ind] < (big * 0.1))
        if keep.any():
            r = torch.from_numpy(row_ind[keep]).to(cost.device, dtype=torch.long)
            c = torch.from_numpy(col_ind[keep]).to(cost.device, dtype=torch.long)
            assign_rows.append(r)
            assign_cols.append(c)
            m_eff_list.append(int(keep.sum()))
        else:
            assign_rows.append(torch.empty(0, device=cost.device, dtype=torch.long))
            assign_cols.append(torch.empty(0, device=cost.device, dtype=torch.long))
            m_eff_list.append(0)

    return assign_rows, assign_cols, m_eff_list


ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", type=str, required=True)
ap.add_argument("--calib_csv", type=str, required=True)
ap.add_argument("--save_ckpt", type=str, default="")
ap.add_argument("--lambda_ridge", type=float, default=1e-2)
ap.add_argument("--mode", type=str,
                choices=["lhf", "bias", "affine"],
                default="lhf",
                help="lhf: 重标定 LHF | bias: 偏置修正 | affine: 仿射修正")
ap.add_argument("--ridge_reduce", type=str, choices=["slot", "mean"], default="mean",
                help="LHF 岭回归的数据聚合方式：slot=逐行堆叠；mean=按槽位均值（保持旧逻辑）")
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = Config()
K = cfg.K_max
nout = 6
D = K * nout

# --------------------------------------------------------
# 加载模型
# --------------------------------------------------------
model = MFDNN(cfg).to(device)
ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
state = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(state, strict=False)
model.eval()


# --------------------------------------------------------
# CSV 数据
# --------------------------------------------------------
rows_float = []
with open(args.calib_csv, "r", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for row in reader:
        cleaned = [float(x.strip().lstrip("\ufeff"))
                    for x in row if x.strip()]
        if cleaned:
            rows_float.append(cleaned)

parsed, stats = parse_rows_to_struct(rows_float)
N = len(parsed)
if N == 0:
    raise RuntimeError("校准 CSV 为空！")

# --------------------------------------------------------
# 载入 normalizer
# --------------------------------------------------------
in_norm_path = Path(args.ckpt).with_name("Normalizer_Input_Sim.pth")
in_norm = Normalizer.load(in_norm_path)
if args.mode == "lhf":
    out_norm_path = Path(args.ckpt).with_name("Normalizer_Output_Sim.pth")
else:
    out_norm_path = Path(args.ckpt).with_name("Normalizer_Output_Real_Fold0.pth")
out_norm = Normalizer.load(out_norm_path)

# --------------------------------------------------------
# 构建批量输入  (H,M,Exc,Ch)
# --------------------------------------------------------
H = torch.tensor([p.H for p in parsed], dtype=torch.float32, device=device).unsqueeze(-1)
M = torch.tensor([p.M for p in parsed], dtype=torch.float32, device=device).unsqueeze(-1)
E = torch.tensor([p.exc_id for p in parsed], dtype=torch.long, device=device)
Cc = torch.tensor([p.ch_id for p in parsed], dtype=torch.long, device=device)

# 输入归一化
H = (H - in_norm.mean[0]) / in_norm.std[0]
M = (M - in_norm.mean[1]) / in_norm.std[1]


# --------------------------------------------------------
# 1) 特征输入网络，构造网络输出
# --------------------------------------------------------
with torch.no_grad():
    outL = model.forward_low(H, M, E, Cc)
u_sin_theta = outL["u_sin_theta"]  # (N,K)
u_cos_theta = outL["u_cos_theta"]  # (N,K)
r = torch.sqrt(u_sin_theta**2 + u_cos_theta**2)  # (N,K) 拿到u空间的留数模长，真实物理参数反推回u空间时用
# 真实数据输入特征后得到网络输出与to_physical空间后的参数
U_low = torch.stack([outL["u_alpha"], outL["u_omega"], outL["u_rho"], outL["u_sin_theta"], outL["u_cos_theta"], outL["u_pres"]], dim=-1)        # (N,K,nout)
Pred_low = torch.stack([outL["alpha"], outL["omega"], outL["rho"], outL["sin_theta"], outL["cos_theta"]], dim=-1)   # (N,K,5)
# 对网络输出的物理参数做归一化
PredN_low = normlizer(Pred_low, out_norm.mean, out_norm.std)


# --------------------------------------------------------
# 2) 标签反推回 u空间，准备岭回归数据
# --------------------------------------------------------
U_target_list = []
target_list = []
for i, p in enumerate(parsed):# 每行真实数据，分别拿到所有物理参数
    alpha = torch.from_numpy(p.alpha.copy()).to(device)   # (K,)
    omega = torch.from_numpy(p.omega.copy()).to(device)   # (K,)
    rho   = torch.from_numpy(p.rho.copy()).to(device)     # (K,)
    sin_t = torch.from_numpy(p.sin_theta.copy()).to(device)  # (K,)
    cos_t = torch.from_numpy(p.cos_theta.copy()).to(device)  # (K,)
    mask  = torch.from_numpy(p.mask.copy()).to(device)    # (K,)

    # 将真实数据的物理参数反推回 u-space，岭回归备用
    u_alpha, u_omega, u_rho, u_sin, u_cos, u_pres = invert_to_u_space(
        alpha, omega, rho, sin_t, cos_t, mask, r[i, :]
    )

    U = torch.stack([u_alpha, u_omega, u_rho, u_sin, u_cos, u_pres], dim=-1)
    T = torch.stack([alpha, omega, rho, sin_t, cos_t, mask], dim=-1)  # (K,5)
    U_target_list.append(U)   # (K,nout)
    target_list.append(T)     # (K,5)

U_target = torch.stack(U_target_list, dim=0)   # (N,K,nout) 真实数据反推回u空间，包括mask
target = torch.stack(target_list, dim=0)     # (N,K,5) 真实数据物理参数
targetN = normlizer(target, out_norm.mean, out_norm.std)  # (N,5,K) 真实数据物理参数归一化


# --------------------------------------------------------
# 3) 对每个样本做训练同款匈牙利匹配，在物理空间对齐槽位
# --------------------------------------------------------
rows, cols, mlist = hungarian_match(PredN_low, targetN)


# --------------------------------------------------------
# 4) 根据模式执行校准（最关键部分）
# --------------------------------------------------------

# ========= 准备岭回归数据：用匈牙利匹配结果对齐 u-space =========

X_list = []   # u_pred
Y_list = []   # u_target

for b in range(N):
    rb = rows[b]   # 预测槽位
    cb = cols[b]   # 真实槽位

    if rb.numel() == 0:
        continue

    # u_pred 取自 U_low（LHF 输出的 u-space）
    up = U_low[b, rb, :]      # (m_eff, nout)

    # u_target 取自 U_target（真实反推回 u-space）
    ut = U_target[b, cb, :]   # (m_eff, nout)

    X_list.append(up)
    Y_list.append(ut)

if len(X_list) == 0:
    raise RuntimeError("没有任何有效匹配，无法进行岭回归！")

X_u = torch.cat(X_list, dim=0)   # (Nt, nout)  预测 u
Y_u = torch.cat(Y_list, dim=0)   # (Nt, nout)  真实 u


# ========= 根据模式重新计算  =========
if args.mode == "lhf":
    if args.ridge_reduce == "mean":
        # ====== 保持你原来的“按槽位均值”逻辑 ======
        # 建立 (K, nout) 的容器
        X_slot = [[] for _ in range(K)]
        Y_slot = [[] for _ in range(K)]

        idx_total = 0
        for b in range(N):
            rb = rows[b]
            for j in range(rb.numel()):
                slot_pred = rb[j].item()
                X_slot[slot_pred].append(X_u[idx_total])
                Y_slot[slot_pred].append(Y_u[idx_total])
                idx_total += 1

        # 转成矩阵形式
        X_slot = [torch.stack(v, dim=0) if len(v)>0 else torch.zeros(0, nout, device=device)
                for v in X_slot]
        Y_slot = [torch.stack(v, dim=0) if len(v)>0 else torch.zeros(0, nout, device=device)
                for v in Y_slot]


        # 构成 (K,6) 的目标结构（均值作为校准值）
        U_pred_slot = torch.zeros(K, 6, device=device)
        U_tgt_slot  = torch.zeros(K, 6, device=device)
        cnt_slot    = torch.zeros(K, dtype=torch.long, device=device)

        for k in range(K):
            if X_slot[k].size(0) > 0:
                U_pred_slot[k] = X_slot[k].mean(dim=0)
                U_tgt_slot[k]  = Y_slot[k].mean(dim=0)
                cnt_slot[k]    = X_slot[k].size(0)

        # 岭回归：只对前5维（pres 留给 Platt）
        model.ridge_recalibrate_linear(
            U_pred_slot.unsqueeze(0),   # (1,K,6)
            U_tgt_slot.unsqueeze(0),    # (1,K,6)
            reg=args.lambda_ridge
        )

    else:
        # ====== slot：逐行堆叠（不分槽、信息更全） ======
        model.ridge_recalibrate_linear(
            U_low    = X_u.view(1, -1, 6),   # (1,T,6)
            U_target = Y_u.view(1, -1, 6),   # (1,T,6)
            reg      = args.lambda_ridge
        )
        # pres 的 Platt：用逐行样本
        z = X_u[:, 5].detach().clone()                            # (T,)
        y = (Y_u[:, 5] > 0).to(z.dtype)

    # [MOD BEGIN] pres 的 Platt：用“全部预测槽位”，标签按匈牙利把预测索引置1
    z_chunks, y_chunks = [], []
    for b in range(N):
        z_b = U_low[b, :, 5]                               # 该样本的所有预测槽位 u_pres，shape=(K,)
        y_b = torch.zeros(K, device=z_b.device, dtype=z_b.dtype)
        if rows[b].numel() > 0:
            y_b[rows[b]] = 1.0                             # 预测索引(被匹配)置1，未匹配保持0
        z_chunks.append(z_b)
        y_chunks.append(y_b)

    z_all = torch.cat(z_chunks, dim=0)                     # (N*K,)
    y_all = torch.cat(y_chunks, dim=0)                     # (N*K,)

    # 2 参数 Platt（BCE + L2 到 a）
    if (y_all.sum() == 0) or (y_all.sum() == y_all.numel()):
        a = z_all.new_tensor(1.0); b = z_all.new_tensor(0.0)
    else:
        a = z_all.new_zeros(()).requires_grad_(True)
        b = z_all.new_zeros(()).requires_grad_(True)
        opt = torch.optim.Adam([a, b], lr=5e-2)
        lam = args.lambda_ridge
        for _ in range(200):
            p = torch.sigmoid(a * z_all + b)
            loss = -(y_all * torch.log(p + 1e-8) + (1 - y_all) * torch.log(1 - p + 1e-8)).mean() + 0.5 * lam * (a * a)
            if not torch.isfinite(loss): break
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        a = a.detach(); b = b.detach()

    model.calibrate_pres_platt(a, b)
    print(f"[OK] LHF 已重标定（reduce={args.ridge_reduce}），pres 使用全槽位+匈牙利索引完成 Platt 标定。")


else:
    # ------------------------------------------
    # bias | affine : 在线 quick-fix
    # ------------------------------------------

    # (N,D)
    model.enable_online_fix = False
    with torch.no_grad():
        outH = model.forward_high(H, M, E, Cc, outL)

    X_flat = model.flatten_u(
        outH["u_alpha"], outH["u_omega"], outH["u_rho"],
        outH["u_sin_theta"], outH["u_cos_theta"],
        outH.get("u_pres", None)
    )   # shape = (N, D)= (N, K*nout)

    # 同样 flatten target u
    Y_flat = model.flatten_u(
        U_target[:,:,0], U_target[:,:,1], U_target[:,:,2],
        U_target[:,:,3], U_target[:,:,4], U_target[:,:,5]
    )    # (N, D)

    if args.mode == "bias":
        # ---------- 偏置快修 ----------
        delta_vec = (Y_flat - X_flat).mean(dim=0)
        delta = delta_vec.view(1, K, nout)
        model.quick_fix_biasgain({"delta": delta})
        model.enable_online_fix = True
        print("[OK] 已应用偏置快修。")

    elif args.mode == "affine":
        # ---------- 仿射快修 ----------
        D_total = K * nout
        a = torch.zeros(D_total, device=device)
        b = torch.zeros(D_total, device=device)
        lam = args.lambda_ridge

        for d in range(D_total):
            z = X_flat[:, d]
            y = Y_flat[:, d]
            Xm = torch.stack([z, torch.ones_like(z)], dim=1)  # (N,2)
            G = Xm.T @ Xm
            Lambda = torch.zeros_like(G)
            Lambda[0, 0] = lam
            theta = torch.linalg.solve(G + Lambda, Xm.T @ y)
            a[d], b[d] = theta[0], theta[1]

        model.quick_fix_biasgain({
            "a": a.view(1, K, nout),
            "b": b.view(1, K, nout)
        })
        model.enable_online_fix = True
        print("[OK] 已应用仿射快修。")


# --------------------------------------------------------
# 5) 保存模型
# --------------------------------------------------------
base = Path(args.ckpt).with_suffix("")
suffix = {"lhf": "_lhf", "bias": "_biasfix", "affine": "_affinefix"}[args.mode]
save_path = args.save_ckpt or str(base) + suffix + ".pth"

torch.save({"model_state_dict": model.state_dict()}, save_path)
print(f"[OK] 在线标定完成（mode={args.mode}），模型已保存到：{save_path}")


"""
python -m surrogate_model.cli.online_ridge \
  --ckpt "surrogate_model/outputs_pzk/best_stage2_fold0.pth" \
  --calib_csv "surrogate_model/data/new_real.csv" \
  --mode lhf \
  --lambda_ridge 1e-2
"""

# %%
