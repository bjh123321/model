# surrogate_model/cli/online_ridge.py
import argparse, csv, torch
from pathlib import Path

from surrogate_model.config import Config
from surrogate_model.blocks import MFDNN  # 你的模型主类，含 ridge_recalibrate_linear / flatten_u
# 注：ParamPZK.from_physical 要求你在 blocks/ParamPZK 里提供一个“反函数”；
# 如果你还没写，我在下面提供了一个“直写 u*”的备选（见注释）。

def from_physical_to_u(param, alpha, omega, rho, sin_t, cos_t):
    """
    把物理量反算回 u-空间（与 ParamPZK.to_physical 完全对应）。
    如果你已有 param.from_physical，就直接用那个；否则按你 to_physical 的实现自己写。
    这里示范：假设 to_physical 就是恒等（仅示意），你请替换为你自己的反函数。
    """
    # !!! 请按你的 ParamPZK.to_physical 精确写反函数 !!!
    # 例如：alpha = -softplus(u_alpha) => u_alpha = softplus^{-1}(-alpha)
    #       omega = omega_max * sigmoid(u_omega) => u_omega = logit(omega/omega_max)
    #       rho   = exp(u_rho) => u_rho = log(rho)
    #       sin/cos 若用了 tanh => u = atanh(sin/cos)
    u_alpha    = torch.as_tensor(alpha).float()
    u_omega    = torch.as_tensor(omega).float()
    u_rho      = torch.as_tensor(rho).float()
    u_sin_theta= torch.as_tensor(sin_t).float()
    u_cos_theta= torch.as_tensor(cos_t).float()
    return u_alpha, u_omega, u_rho, u_sin_theta, u_cos_theta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="原始模型 ckpt 路径")
    ap.add_argument("--calib_csv", type=str, required=True, help="在线校准数据 CSV")
    ap.add_argument("--save_ckpt", type=str, default="", help="保存的新 ckpt 路径（默认自动加 _ridge）")
    ap.add_argument("--lambda_ridge", type=float, default=1e-2, help="岭回归正则 λ")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    model = MFDNN(cfg).to(device)
    # 载入 ckpt
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    # 读取 CSV，收集批量
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
    H = torch.tensor(Hs, dtype=torch.float32, device=device).unsqueeze(-1)
    M = torch.tensor(Ms, dtype=torch.float32, device=device).unsqueeze(-1)
    E = torch.tensor(Es, dtype=torch.long,    device=device)
    C = torch.tensor(Cs, dtype=torch.long,    device=device)

    with torch.no_grad():
        # 1) 跑 LF，拿 X=(N,D)
        outL = model.forward_low(H, M, E, C)
        X = model.flatten_u(outL['u_alpha'], outL['u_omega'], outL['u_rho'],
                            outL['u_sin_theta'], outL['u_cos_theta'], outL.get('u_pres', None))

        # 2) 目标物理量 → u*（与你的 ParamPZK.to_physical 对偶）
        uA, uW, uR, uS, uC = from_physical_to_u(model.param, A_star, W_star, R_star, S_star, C_star)
        # 展平成 (N,D)。注意这里需要按槽位维度复制或填充 u*：
        # 实际工程中，通常你会把每个样本目标按匈牙利匹配好的槽位顺序填到 (K_max, nout)；
        # 这里示例用“全槽位同一目标”的简单占位（你按实际替换！）
        K = cfg.K_max
        uA = uA.view(N,1).repeat(1,K); uW = uW.view(N,1).repeat(1,K); uR = uR.view(N,1).repeat(1,K)
        uS = uS.view(N,1).repeat(1,K); uC = uC.view(N,1).repeat(1,K)
        if cfg.use_presence_head:
            uP = torch.full((N,K), 10.0, device=device)  # 目标“确定存在”的 logit，按需调整
            Y = torch.cat([uA,uW,uR,uS,uC,uP], dim=-1)
        else:
            Y = torch.cat([uA,uW,uR,uS,uC], dim=-1)

        # 3) 岭回归：更新 LHF 的 W,b
        model.ridge_recalibrate_linear(X, Y, reg=args.lambda_ridge)

    # 保存新 ckpt
    save_path = args.save_ckpt or (str(Path(args.ckpt).with_suffix("")) + "_ridge.pth")
    torch.save({"model_state_dict": model.state_dict()}, save_path)
    print(f"[OK] Ridge recalibrated LHF saved to: {save_path}")

if __name__ == "__main__":
    main()

"""
python -m surrogate_model.cli.online_ridge \
  --ckpt "outputs_pzk/checkpoint_stage2_best.pth" \
  --calib_csv "surrogate_model/data/calib_online.csv" \
  --lambda_ridge 1e-2
"""