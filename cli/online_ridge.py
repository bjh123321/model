import argparse, csv, torch
from pathlib import Path

from ..config import Config
from ..models.blocks import MFDNN


def from_physical_to_u(alpha, omega, rho, sin_t, cos_t):
    """
    将物理量 (alpha, omega, rho, sinθ, cosθ)
    反算回网络的 u-空间 (u_alpha, u_omega, u_rho, u_sinθ, u_cosθ)
    与 ParamPZK.to_physical 完全对应。
    """
    neg_alpha = -alpha
    u_alpha = torch.log(torch.exp(neg_alpha) - 1 + 1e-12)
    omega = omega.clamp(1e-6, 1 - 1e-6)
    u_omega = torch.log(omega / (1 - omega))
    u_rho = rho
    u_sin_theta = sin_t
    u_cos_theta = cos_t

    return u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="原始模型 ckpt 路径")
    ap.add_argument("--calib_csv", type=str, required=True, help="在线校准数据 CSV")
    ap.add_argument("--save_ckpt", type=str, default="", help="保存的新 ckpt 路径（默认按 mode 加后缀）")
    ap.add_argument("--lambda_ridge", type=float, default=1e-2, help="岭回归正则 λ")
    ap.add_argument("--mode", type=str, choices=["lhf", "bias", "affine"], default="lhf",
                    help="lhf: 重标定线性高保真; bias: 仅在线偏置; affine: 在线仿射")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    model = MFDNN(cfg).to(device)

    # 载入 ckpt
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    # 读取 CSV
    Hs, Ms, Es, Cs = [], [], [], []
    A_star, W_star, R_star, S_star, C_star = [], [], [], [], []
    with open(args.calib_csv, "r", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            Hs.append(float(row["H"]));           Ms.append(float(row["M"]))
            Es.append(int(row["exc_id"]));        Cs.append(int(row["ch_id"]))
            A_star.append(float(row["alpha_star"]))
            W_star.append(float(row["omega_star"]))
            R_star.append(float(row["rho_star"]))
            S_star.append(float(row["sin_theta_star"]))
            C_star.append(float(row["cos_theta_star"]))

    N = len(Hs)
    if N == 0:
        raise RuntimeError("calib_csv 为空，没有可用于在线标定的数据")

    H = torch.tensor(Hs, dtype=torch.float32, device=device).unsqueeze(-1)
    M = torch.tensor(Ms, dtype=torch.float32, device=device).unsqueeze(-1)
    E = torch.tensor(Es, dtype=torch.long,    device=device)
    C = torch.tensor(Cs, dtype=torch.long,    device=device)

    with torch.no_grad():
        # 目标物理量 → u*（与你的 ParamPZK.to_physical 对偶）
        uA, uW, uR, uS, uC = from_physical_to_u(
            A_star, W_star, R_star, S_star, C_star, device=device
        )
        K = cfg.K_max

        if cfg.use_presence_head:
            # 目标“存在”logit，可以给一个较大的常数
            uP = torch.full((N, 1), 10.0, device=device)
        else:
            uP = None

        # 不同 mode 对 X 和 Y 的定义不同
        if args.mode == "lhf":
            # ========== 模式 1：重标定 LHF ==========
            outL = model.forward_low(H, M, E, C)
            X = model.flatten_u(
                outL['u_alpha'], outL['u_omega'], outL['u_rho'],
                outL['u_sin_theta'], outL['u_cos_theta'], outL.get('u_pres', None)
            )   # (N, D)

            # 把目标 u* 复制到 K 个槽位，展平成 (N,D)
            uA_ = uA.view(N, 1).repeat(1, K)
            uW_ = uW.view(N, 1).repeat(1, K)
            uR_ = uR.view(N, 1).repeat(1, K)
            uS_ = uS.view(N, 1).repeat(1, K)
            uC_ = uC.view(N, 1).repeat(1, K)
            if cfg.use_presence_head:
                uP_ = uP.view(N, 1).repeat(1, K)
                Y = torch.cat([uA_, uW_, uR_, uS_, uC_, uP_], dim=-1)
            else:
                Y = torch.cat([uA_, uW_, uR_, uS_, uC_], dim=-1)

            model.ridge_recalibrate_linear(X, Y, reg=args.lambda_ridge)

        else:
            # ========== 模式 2/3：在最终输出上 quick_fix ==========
            outL = model.forward_low(H, M, E, C)
            outH = model.forward_high(H, M, E, C, outL)

            Z = model.flatten_u(
                outH['u_alpha'], outH['u_omega'], outH['u_rho'],
                outH['u_sin_theta'], outH['u_cos_theta'], outH.get('u_pres', None)
            )   # (N, D)

            # 同样构造 Y = 目标 u* 的展平
            uA_ = uA.view(N, 1).repeat(1, K)
            uW_ = uW.view(N, 1).repeat(1, K)
            uR_ = uR.view(N, 1).repeat(1, K)
            uS_ = uS.view(N, 1).repeat(1, K)
            uC_ = uC.view(N, 1).repeat(1, K)
            if cfg.use_presence_head:
                uP_ = uP.view(N, 1).repeat(1, K)
                Y = torch.cat([uA_, uW_, uR_, uS_, uC_, uP_], dim=-1)
            else:
                Y = torch.cat([uA_, uW_, uR_, uS_, uC_], dim=-1)

            if args.mode == "bias":
                # ----- 仅偏置：delta = argmin ||(Z + delta) - Y||^2 + λ||delta||^2 -----
                # 闭式解：delta = (Σ_i (Y_i - Z_i)) / (N + λ)
                delta_vec = (Y - Z).sum(dim=0) / (N + args.lambda_ridge)
                D = Z.shape[1]
                nout = D // K
                delta = delta_vec.view(1, K, nout)
                model.quick_fix_biasgain({"delta": delta})

            elif args.mode == "affine":
                # ----- 仿射：对每一维做 y ≈ a*z + b 的 1D 岭回归 -----
                N_, D = Z.shape
                a = torch.empty(D, device=device)
                b = torch.empty(D, device=device)
                lam = args.lambda_ridge

                for d in range(D):
                    z = Z[:, d]  # (N,)
                    y = Y[:, d]  # (N,)
                    Xmat = torch.stack([z, torch.ones_like(z)], dim=1)  # (N,2)
                    G = Xmat.T @ Xmat                                  # (2,2)
                    Lambda = torch.zeros_like(G)
                    Lambda[0, 0] = lam  # 只正则化 a，不正则化 b
                    theta = torch.linalg.solve(G + Lambda, Xmat.T @ y)  # (2,)
                    a[d] = theta[0]
                    b[d] = theta[1]

                nout = D // K
                a_3d = a.view(1, K, nout)
                b_3d = b.view(1, K, nout)
                model.quick_fix_biasgain({"a": a_3d, "b": b_3d})
            else:
                raise ValueError(f"未知 mode: {args.mode}")

    # 保存新 ckpt
    base = str(Path(args.ckpt).with_suffix(""))
    suffix = {"lhf": "_ridge_lhf", "bias": "_ridge_bias", "affine": "_ridge_affine"}[args.mode]
    save_path = args.save_ckpt or (base + suffix + ".pth")
    torch.save({"model_state_dict": model.state_dict()}, save_path)
    print(f"[OK] mode={args.mode} 已写回，保存到: {save_path}")


if __name__ == "__main__":
    main()

"""
python -m surrogate_model.cli.online_ridge \
  --ckpt "outputs_pzk/checkpoint_stage2_best.pth" \
  --calib_csv "surrogate_model/data/calib_online.csv" \
  --lambda_ridge 1e-2
"""