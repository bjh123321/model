# surrogate_model/cli/online_ridge.py
import argparse
import torch
import csv
from pathlib import Path

from surrogate_model.config import Config
from surrogate_model.data.io import parse_rows_to_struct
from surrogate_model.data.normalizer import Normalizer
from surrogate_model.models.blocks import MFDNN


# ===========================
# 反推：physical → u-space
# ===========================
def invert_to_u_space(alpha, omega, rho, sin_t, cos_t, r):
    neg_alpha = -alpha
    u_alpha = torch.log(torch.exp(neg_alpha) - 1.0 + 1e-12)
    omega = omega.clamp(1e-6, 1 - 1e-6)
    u_omega = torch.log(omega / (1 - omega))
    u_rho = rho
    u_sin = r * sin_t
    u_cos = r * cos_t

    return u_alpha, u_omega, u_rho, u_sin, u_cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",        type=str, required=True)
    ap.add_argument("--calib_csv",   type=str, required=True)
    ap.add_argument("--save_ckpt",   type=str, default="")
    ap.add_argument("--lambda_ridge",type=float, default=1e-2)
    ap.add_argument("--mode", type=str, choices=["lhf", "bias", "affine"], default="bias",
                    help="lhf=重标定LHF; bias=仅偏置快修; affine=仿射快修")

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()

    # -------------------------------
    # 加载模型
    # -------------------------------
    model = MFDNN(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    # -------------------------------
    # 读取 normalizer
    # -------------------------------
    input_norm_path = Path(args.ckpt).with_name("Normalizer_Input_Sim.pth")  # with_name表示替换文件名，相当于拆分路径后生成新路径
    output_norm_path = Path(args.ckpt).with_name("Normalizer_Output_Real_Fold0.pth")
    in_norm: Normalizer  = Normalizer.load(input_norm_path)
    out_norm: Normalizer = Normalizer.load(output_norm_path)


# -------------------------------
# 读取校准数据 → ParsedRow
# -------------------------------
    rows_float = []
    with open(args.calib_csv, "r", newline="", encoding="utf-8-sig") as f:  # utf-8-sig 自动去掉 BOM，newline="" 兼容各种换行符
        reader = csv.reader(f)
        for row in reader:
            # 去掉左右空白，过滤掉空单元格
            cleaned = []
            for x in row:
                x = x.strip()
                if not x:
                    continue
                # 去掉可能存在的 BOM
                x = x.lstrip("\ufeff")
                cleaned.append(float(x))
            if cleaned:
                rows_float.append(cleaned)

    # 这里 rows_float 的类型就是 List[List[float]]
    parsed, stats = parse_rows_to_struct(rows_float)

    # parsed 是 List[ParsedRow]
    N = len(parsed)
    if N == 0:
        raise RuntimeError("校准数据为空！")

    # -------------------------------
    # 构建批量输入 (H,M,E,C)
    # -------------------------------
    H = torch.tensor([p.H for p in parsed], dtype=torch.float32, device=device).unsqueeze(-1)
    M = torch.tensor([p.M for p in parsed], dtype=torch.float32, device=device).unsqueeze(-1)
    E = torch.tensor([p.exc_id for p in parsed], dtype=torch.long,   device=device)
    C = torch.tensor([p.ch_id for p in parsed], dtype=torch.long,   device=device)

    # 输入归一化（与 train_pzk 保持一致）
    H = (H - in_norm.mean[0]) / in_norm.std[0]
    M = (M - in_norm.mean[1]) / in_norm.std[1]

    # -------------------------------
    # 构建目标物理域 → u-space（标准化后）
    # -------------------------------
    # 模态结构：alpha, omega, rho, sinθ, cosθ  (K_max × 5)
    K = cfg.K_max
    use_pres = cfg.use_presence_head
    nout = 6 if use_pres else 5
    D = K * nout

    U_list = []
    for p in parsed:
        # 物理域参数（训练时就是用的这些）
        alpha = torch.from_numpy(p.alpha.copy()).to(device)       # (K,)
        omega = torch.from_numpy(p.omega.copy()).to(device)       # (K,)
        rho   = torch.from_numpy(p.rho.copy()).to(device)         # (K,)
        sin_t = torch.from_numpy(p.sin_theta.copy()).to(device)   # (K,)
        cos_t = torch.from_numpy(p.cos_theta.copy()).to(device)   # (K,)

        # ---- 只对 alpha/omega/rho 做 z-score，与 ModalDataset 保持一致 ----
        lab = torch.stack([alpha, omega, rho], dim=0)             # (3,K)
        m = torch.as_tensor(out_norm.mean, dtype=lab.dtype, device=device).view(3, 1)
        s = torch.as_tensor(out_norm.std,  dtype=lab.dtype, device=device).view(3, 1).clamp_min(1e-8)
        lab_norm = (lab - m) / s                                  # (3,K)

        # sin/cos 不做归一化，直接拼上
        sin_t = sin_t.unsqueeze(0)                                # (1,K)
        cos_t = cos_t.unsqueeze(0)                                # (1,K)
        U = torch.cat([lab_norm, sin_t, cos_t], dim=0)            # (5,K)

        # 若使用 presence head，为 u_pres 构造“存在/不存在”的 logit 目标
        if use_pres:
            mask = torch.from_numpy(p.mask.copy()).to(device)     # (K,)
            # 存在 -> 大正数，不存在 -> 大负数（相当于目标概率接近 1 / 0）
            u_pres = torch.where(
                mask > 0.5,
                torch.full_like(mask, 10.0),
                torch.full_like(mask, -10.0),
            ).unsqueeze(0)                                       # (1,K)
            U = torch.cat([U, u_pres], dim=0)                     # (6,K)

        U_list.append(U)                                          # 每个样本 (nout,K)

    # 堆成批量：先 (N, nout, K)，再展平成 (N, D)
    U_target = torch.stack(U_list, dim=0)                         # (N, nout, K)
    U_target_flat = U_target.permute(0, 2, 1).reshape(len(parsed), -1)  # (N, K*nout)

    # -------------------------------
    # 计算特征 X：根据模式不同
    # -------------------------------
    with torch.no_grad():
        if args.mode == "lhf":      # 如果修LF后的输出, 则网络输出低保真
            outL = model.forward_low(H, M, E, C)
            X = model.flatten_u(
                outL["u_alpha"], outL["u_omega"], outL["u_rho"],
                outL["u_sin_theta"], outL["u_cos_theta"],
                outL.get("u_pres", None) # get安全取出u_pres，如果不存在就返回None
            )   # (N, D)
        else:
            model.enable_online_fix = False     # 关闭在线快修，获取原始输出
            outL = model.forward_low(H, M, E, C)
            outH = model.forward_high(H, M, E, C, outL)
            X = model.flatten_u(
                outH["u_alpha"], outH["u_omega"], outH["u_rho"],
                outH["u_sin_theta"], outH["u_cos_theta"],
                outH.get("u_pres", None)
            )

    # -------------------------------
    # 执行岭回归
    # -------------------------------
    if args.mode == "lhf":
        # ===== 重标定 LHF =====
        model.ridge_recalibrate_linear(X, U_target_flat, reg=args.lambda_ridge)     # 岭回归直接修改 LHF 的线性层

    elif args.mode == "bias":
        # ===== 偏置快修 =====
        delta_vec = (U_target_flat - X).sum(dim=0) / (N + args.lambda_ridge)
        delta = delta_vec.view(1, K, nout)
        model.quick_fix_biasgain({"delta": delta})
        model.enable_online_fix = True

    elif args.mode == "affine":
        # ===== 仿射快修 =====
        a = torch.zeros(D, device=device)
        b = torch.zeros(D, device=device)
        lam = args.lambda_ridge

        for d in range(D):
            z = X[:, d]
            y = U_target_flat[:, d]
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

    # -------------------------------
    # 保存 ckpt
    # -------------------------------
    base = Path(args.ckpt).with_suffix("")
    suffix = {"lhf": "_lhf", "bias": "_biasfix", "affine": "_affinefix"}[args.mode]
    save_path = args.save_ckpt or str(base) + suffix + ".pth"

    torch.save({"model_state_dict": model.state_dict()}, save_path)
    print(f"[OK] 在线标定完成 mode={args.mode}, 保存到: {save_path}")


if __name__ == "__main__":
    main()


"""
重标定 LHF
python -m surrogate_model.cli.online_ridge \
    --ckpt "your_checkpoint.pth" \
    --calib_csv "./surrogate_model/data/your_online_data.csv" \
    --mode lhf \
    --lambda_ridge 1e-2
输出文件your_checkpoint_lhf.pth
"""
"""
偏置快修
python -m surrogate_model.cli.online_ridge \
    --ckpt "your_checkpoint.pth" \
    --calib_csv "./surrogate_model/data/your_online_data.csv" \
    --mode bias \
    --lambda_ridge 1e-2
输出文件your_checkpoint_biasfix.pth
"""
"""
仿射快修
python -m surrogate_model.cli.online_ridge \
    --ckpt "your_checkpoint.pth" \
    --calib_csv "./surrogate_model/data/your_online_data.csv" \
    --mode affine \
    --lambda_ridge 1e-2
输出文件your_checkpoint_affinefix.pth
"""