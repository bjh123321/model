import torch
from pathlib import Path
import numpy as np
from surrogate_model.config import Config
from surrogate_model.models.blocks import MFDNN
from surrogate_model.data.normalizer import Normalizer
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import pandas as pd
from residue_pred import build_BA_from_residues


def make_cases_sweep_mach(H_fix, M_list, exc_id_fix, ch_id_fix, name_prefix="sweepM"):
    cases = []
    for M in M_list:
        cases.append(dict(
            name=f"{name_prefix}_H{H_fix}_M{M:.3f}_exc{exc_id_fix}_ch{ch_id_fix}",
            H=float(H_fix),
            M=float(M),
            exc_id=int(exc_id_fix),
            ch_id=int(ch_id_fix),
        ))
    return cases


def pick_mode_by_slot(p_exist, omega, alpha, rho, theta, slot=0, exist_thresh=0.5):
    """最简单：固定网络输出槽位 slot（不做重排）。"""
    ok = (p_exist[slot] >= exist_thresh)
    return ok, alpha[slot], omega[slot], rho[slot], theta[slot]


def pick_mode_by_nearest_freq(p_exist, omega, alpha, rho, theta, omega_ref, exist_thresh=0.5):
    """
    推荐：在“存在的模态”里选 omega 最接近 omega_ref 的那支，实现跨工况跟踪。
    omega_ref: 参考角频率(rad/s)，可取某个基准工况输出的该模态 omega。
    """
    idx = np.where(p_exist >= exist_thresh)[0]
    if len(idx) == 0:
        return False, np.nan, np.nan, np.nan, np.nan

    j = idx[np.argmin(np.abs(omega[idx] - omega_ref))]
    return True, alpha[j], omega[j], rho[j], theta[j]


def build_mode_series(cases, p_exist_b, alpha_b, omega_b, rho_b, theta_b,
                      exist_thresh=0.5, track="nearest", slot=0, omega_ref=None):
    """
    track:
      - "slot"：固定槽位 slot
      - "nearest"：按频率最近跟踪（需要 omega_ref）
    """
    rows = []
    for i, c in enumerate(cases):
        p_exist = p_exist_b[i]
        alpha = alpha_b[i]
        omega = omega_b[i]
        rho   = rho_b[i]
        theta = theta_b[i]

        if track == "slot":
            ok, a, w, r, th = pick_mode_by_slot(p_exist, omega, alpha, rho, theta,
                                                slot=slot, exist_thresh=exist_thresh)
        else:
            ok, a, w, r, th = pick_mode_by_nearest_freq(p_exist, omega, alpha, rho, theta,
                                                        omega_ref=omega_ref, exist_thresh=exist_thresh)

        rows.append(dict(
            name=c["name"],
            H=c["H"], M=c["M"], exc_id=c["exc_id"], ch_id=c["ch_id"],
            ok=int(ok),
            alpha=a,
            omega=w,
            freq_hz=(w / (2*np.pi)) if np.isfinite(w) else np.nan,
            rho=r,
            theta=th
        ))
    return pd.DataFrame(rows)


def plot_mode_vs_M(df, save_path=None):
    # 只画 ok=1 的点
    d = df[df["ok"] == 1].sort_values("M")

    plt.figure()
    plt.plot(d["M"], d["freq_hz"], marker="o")
    plt.xlabel("Mach")
    plt.ylabel("Mode freq (Hz)")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    plt.figure()
    plt.plot(d["M"], d["alpha"], marker="o")
    plt.xlabel("Mach")
    plt.ylabel("alpha")
    plt.grid(True)
    if save_path:
        stem = str(save_path).replace(".png", "")
        plt.savefig(stem + "_alpha.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


real = False  # True 用 outH；False 用 outL

cur_dir = Path(r'D:\BaiduSyncdisk\Code\Python\traindata\35')
checkpoint_dir = "outputs_pzk_from_opt_dev_full_final_sim0.2real1lossweight_LHF6_pcgrad"
checkpoint_file = "checkpoint_stage2_fold0_epoch5000.pth"

ckpt_path = cur_dir / checkpoint_dir / checkpoint_file
in_norm_path = cur_dir / checkpoint_dir / "Normalizer_Input_Sim.pth"

# 输出目录（批量结果会自动分 case 存）
out_root = Path(r"C:\Users\Leo\Desktop\batch_pred_outputs")
out_root.mkdir(parents=True, exist_ok=True)

# 采样设置（如果每个文件 fs 不同，可放到 case 里单独配）
fs = 512
Ts = 1.0 / fs
omegaN = np.pi * fs

# 数据列设置
col_u = 0  # 激励列

# 1) 生成扫 Mach 的输入
H_fix = 11
exc_fix = 0
ch_fix = 14

M_list = np.linspace(0.8, 1.2, num=100)
cases = make_cases_sweep_mach(H_fix, M_list, exc_fix, ch_fix, name_prefix="rudder_sym")

cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("==> Loading ckpt...")
try:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
except TypeError:
    # 兼容老版本 torch
    ckpt = torch.load(ckpt_path, map_location="cpu")

state = ckpt.get("model_state_dict", ckpt)

model = MFDNN(cfg).to(device)
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"Missing keys: {missing}")
print(f"Unexpected keys: {unexpected}")
model.eval()

in_norm = Normalizer.load(in_norm_path)


# =========================
# 5) 批量推理（模型部分可一次性 batch）
# =========================
# 先把所有 case 的 (H,M,exc,ch) 组装成 batch
Hs = np.array([c["H"] for c in cases], dtype=np.float32)
Ms = np.array([c["M"] for c in cases], dtype=np.float32)
excs = np.array([c["exc_id"] for c in cases], dtype=np.int64)
chs = np.array([c["ch_id"] for c in cases], dtype=np.int64)

x = torch.tensor(np.stack([Hs, Ms], axis=1), dtype=torch.float32, device=device)
x_norm = in_norm.transform(x.cpu()).to(device)
Hn, Mn = x_norm[:, :1], x_norm[:, 1:]

exc_t = torch.tensor(excs, dtype=torch.long, device=device)
ch_t = torch.tensor(chs, dtype=torch.long, device=device)

with torch.no_grad():
    outL, outH = model(Hn, Mn, exc_t, ch_t)
    out = outH if real else outL

    p_exist_b = out["p_exist"].detach().cpu().numpy()
    alpha_b = out["alpha"].detach().cpu().numpy()
    omega_b = out["omega"].detach().cpu().numpy() * omegaN
    rho_b = np.exp(out["rho"].detach().cpu().numpy())
    theta_b = np.arctan2(out["sin_theta"].detach().cpu().numpy(),
                        out["cos_theta"].detach().cpu().numpy())

print(p_exist_b.shape)
freqs = omega_b[:, 0] / (2*np.pi)
plt.figure()
plt.plot(M_list, freqs)
plt.show()
print(freqs)

# 2) 批量推理（你原来的那段 batch 推理保持不变）
# 得到：p_exist_b, alpha_b, omega_b, rho_b, theta_b
# 注意 omega_b 已经是物理域 rad/s（你之前乘了 omegaN）——保持一致即可

exist_thresh = 0.5

# 3) 选“某支模态”的方式 A：固定槽位（最省事，但可能跳模态）
slot = 2
df_mode = build_mode_series(cases, p_exist_b, alpha_b, omega_b, rho_b, theta_b,
                            exist_thresh=exist_thresh, track="slot", slot=slot)

# 4) 选“某支模态”的方式 B：按频率最近跟踪（推荐）
# 先用一个基准 Mach（例如 M_list 的第一个）确定 omega_ref
i0 = 0
omega_ref = omega_b[i0][np.argmax(p_exist_b[i0] >= exist_thresh)]  # 或你指定槽位/指定频率
# 更稳：如果你知道目标模态大概频率，比如 12Hz
# omega_ref = 2*np.pi*12

df_mode = build_mode_series(cases, p_exist_b, alpha_b, omega_b, rho_b, theta_b,
                            exist_thresh=exist_thresh, track="nearest", omega_ref=omega_ref)



# 5) 保存与画图
# out_dir = Path(r"C:\Users\Leo\Desktop\batch_pred_outputs")
# out_dir.mkdir(parents=True, exist_ok=True)

# df_mode.to_csv(out_dir / "mode_vs_M.csv", index=False, encoding="utf-8-sig")
# plot_mode_vs_M(df_mode, save_path=out_dir / "mode_vs_M.png")
# print("Saved:", out_dir / "mode_vs_M.csv")
