# %%
import torch
from pathlib import Path
import numpy as np
from surrogate_model.config import Config
from surrogate_model.models.blocks import MFDNN
from surrogate_model.data.normalizer import Normalizer
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import pandas as pd
from zero_pred import build_BA_from_zeros
from residue_pred import build_BA_from_residues
import argparse


plt.rcParams.update({
    "figure.autolayout": False,
    "figure.subplot.left": 0.15,
    "figure.subplot.right": 0.98,
    "figure.subplot.bottom": 0.15,
    "figure.subplot.top": 0.92,
    # "figure.subplot.left": 0.12,
    # "figure.subplot.right": 0.98,
    # "figure.subplot.bottom": 0.12,
    # "figure.subplot.top": 0.98,
    "axes.formatter.use_mathtext": True,
    "axes.formatter.useoffset": True,
    # "axes.formatter.limits": (0, 0),
    "legend.loc": "upper center",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # ===== 字体 =====
    "font.family": ["Times New Roman", "SimSun"],
    "font.size": 8,                        # 主体字体大小
    "axes.labelsize": 7,
    "axes.titlesize": 10,
    "figure.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 5,
    # ===== 线条 =====
    "lines.linewidth": 0.5,
    "axes.linewidth": 0.5,

    # ===== 刻度线方向 =====
    "xtick.direction": "in",
    "ytick.direction": "in",

    # ===== 刻度线长度 =====
    "xtick.major.size": 3,
    "ytick.major.size": 3,

    # ===== 刻度线宽度 =====
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,

    "figure.figsize": (2.626, 1.969),  # 一般大小（英寸），对应6.67cm×5cm
    # "figure.figsize": (3.543, 2.657),  # 一般大小（英寸），对应9cm×6.75cm
    # "figure.figsize": (5.315, 3.543),  # 一般大小（英寸），对应13.5cm×9cm
    "figure.dpi": 300                  # 分辨率 dpi
})

def compute_spectrum(x, fs, window=False, mode="amplitude"):
    """
    计算单边频谱（Amplitude Spectrum / Power Spectrum / PSD）

    参数
    ----
    x : 1D array
        输入信号
    fs : float
        采样率（Hz）
    window : bool
        是否使用汉宁窗减轻泄漏
    mode : str, optional
        "amplitude" → 单边幅度谱 (默认)
        "power"     → 功率谱 |X|^2
        "psd"       → 功率谱密度 (Power Spectral Density)

    返回
    ----
    f : 1D array
        频率轴（Hz）
    X : 1D array
        对应模式的频谱值
    """

    x = np.asarray(x, float)
    N = len(x)

    # 加窗
    if window:
        w = np.hanning(N)
        xw = x * w
        U = (w**2).sum() / N   # PSD 归一化用
    else:
        xw = x
        U = 1

    # FFT
    Xf = np.fft.rfft(xw)
    f = np.fft.rfftfreq(N, d=1/fs)

    # 单边幅度谱
    if mode == "amplitude":
        X_amp = np.abs(Xf) / (N / 2)
        X_amp[0] /= 2  # DC
        return f, X_amp

    # 功率谱（|X|^2）
    elif mode == "power":
        X_pow = (np.abs(Xf)**2) / (N**2)
        X_pow[1:-1] *= 2  # 单边谱修正
        return f, X_pow

    # 功率谱密度 PSD（与 Welch 一致的定义）
    elif mode == "psd":
        X_psd = (np.abs(Xf)**2) / (fs * N * U)
        X_psd[1:-1] *= 2
        return f, X_psd

    else:
        raise ValueError(f"mode 必须是 amplitude / power / psd，但现在是: {mode}")

# ==== 配置区域 ====
real = True  # 是否使用真实响应网络
data_file = Path(r"D:\BaiduSyncdisk\项目\试飞数据\35A\试飞数据\H8d23\20250919_H8d23_M1d4\H8d23_M1d4_FXD_同_幅值0d6.txt")
# data_file = Path(r"D:\BaiduSyncdisk\项目\试飞数据\35A\南航仿真\20250523标准\H8.23\M1.05\1_H8.23_M1.05_方向舵对称_1-45Hz_幅值0.3.txt")

# data_file = Path(r"D:\BaiduSyncdisk\项目\试飞数据\35A\试飞数据35A\H8.23\M1.4\H8.23_M1.4_方向舵对称_幅值0.6.txt")
# data_file = Path(r"D:\BaiduSyncdisk\项目\试飞数据\35A\试飞数据35A\H11\M0.95\H11_M0.95_方向舵对称_幅值0.6.txt")

# cur_dir = Path(__file__).parents[2] / "surrogate_model"
# checkpoint_dir = "outputs_pzk"

# cur_dir = Path(r'D:\BaiduSyncdisk\Code\Python\traindata\35')
cur_dir = Path(r"D:\BaiduSyncdisk\项目\项目材料\代理模型二期\20260202沈阳联合办公\GeneralParameter\J35_001_none\MX_XGD")
checkpoint_dir = Path(r"XGD_mx_20260201_v2")

checkpoint_file = "XGD_mx_20260201_v2.pth"
# checkpoint_file = "checkpoint_stage1_fold0_epoch5000_lhf.pth"

# checkpoint_file = "checkpoint_stage2_fold0_epoch5000.pth"
# checkpoint_file = "checkpoint_stage2_fold0_epoch5000_inc.pth"


H = 8.23         # 高度
M = 1.4        # 马赫数
exc_id = 0      # 激励方式
ch_id  = 14    # 响应通道
col_u = 0      # 激励
col_y = ch_id + 2
# col_y =17
fs = 512
Ts = 1.0 / 512

df = pd.read_table(data_file, sep=r"[,\t ]+", engine="python")
Y = df.to_numpy(dtype=np.float64)
Y = Y - Y.mean(axis=0, keepdims=True)
u = Y[:, col_u]
y = Y[:, col_y]
t = np.arange(len(u)) / fs

cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = cur_dir / checkpoint_dir / checkpoint_file
in_norm_path  = cur_dir / checkpoint_dir / "Normalizer_Input_Sim.pth"
# out_norm_path = cur_dir / checkpoint_dir / "Normalizer_Real_Fold0.pth"
# ==== 加载 ====
print("==> Loading ckpt...")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
state = ckpt.get("model_state_dict", ckpt)
# loss = ckpt.get("loss_history_real", None)
# print(min(loss))

# region
# 自动推断 embed 维度
# if "low.emb_exc.weight" in state:
#     cfg.embed_excitation_num = state["low.emb_exc.weight"].shape[0]
#     cfg.embed_excitation_dim = state["low.emb_exc.weight"].shape[1]
# if "low.emb_chn.weight" in state:
#     cfg.embed_channel_num = state["low.emb_chn.weight"].shape[0]
#     cfg.embed_channel_dim = state["low.emb_chn.weight"].shape[1]

# K_max 同样自动推断
# if "lin.A" in state:
#     cfg.K_max = state["lin.A"].shape[1]

model = MFDNN(cfg).to(device)
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"Missing keys: {missing}")
print(f"Unexpected keys: {unexpected}")
model.eval()
# model.lin.enable_state_correction(True)

# endregion
# ==== 输入归一化 ====
in_norm = Normalizer.load(in_norm_path)
x = torch.tensor([[H, M]], dtype=torch.float32, device=device)
x_norm = in_norm.transform(x)
Hn, Mn = x_norm[:, :1], x_norm[:, 1:]

exc = torch.tensor([exc_id], dtype=torch.long, device=device)
ch  = torch.tensor([ch_id], dtype=torch.long, device=device)
omegaN = np.pi * 512
# ==== 推理 ====
with torch.no_grad():
    outL, outH = model(Hn, Mn, exc, ch)
    if real:
        out = outH
    else:
        out = outL
    u_alpha = out["u_alpha"][0].cpu().numpy()
    u_omega = out["u_omega"][0].cpu().numpy()
    u_rho   = out["u_rho"][0].cpu().numpy()
    u_sin_theta = out["u_sin_theta"][0].cpu().numpy()
    u_cos_theta = out["u_cos_theta"][0].cpu().numpy()
    u_pres = out["u_pres"][0].cpu().numpy()

    alpha = out["alpha"][0].cpu().numpy()
    omega = out["omega"][0].cpu().numpy()
    rho   = out["rho"][0].cpu().numpy()
    sin_theta = out["sin_theta"][0].cpu().numpy()
    cos_theta = out["cos_theta"][0].cpu().numpy()
    p_exist = out["p_exist"][0].cpu().numpy()
    omega = omega*omegaN
    rho = np.exp(rho)
    theta = np.atan2(sin_theta, cos_theta)

print("\n--- 网络输出 (网络域) ---")
print("u_pres:", np.array2string(u_pres, precision=6))
print("u_alpha  :", np.array2string(u_alpha, precision=6))
print("u_omega  :", np.array2string(u_omega, precision=6))
print("u_rho   :", np.array2string(u_rho, precision=6))
print("u_sin_theta   :", np.array2string(u_sin_theta, precision=6))
print("u_cos_theta   :", np.array2string(u_cos_theta, precision=6))

print("----------------------------")
print("\n--- 网络输出 (物理域) ---")
print("p_exist:", np.array2string(p_exist, precision=6))
print("alpha  :", np.array2string(alpha, precision=6))
print("omega  :", np.array2string(omega, precision=6))
print("rho   :", np.array2string(rho, precision=6))
print("theta   :", np.array2string(theta, precision=6))
print("----------------------------")

mask = p_exist > 0.5
freqs = omega[mask] / (2 * np.pi)
# freqs = sorted(omega[mask] / (2 * np.pi))
# print(freqs)
damping_ratios = -alpha[mask] / np.sqrt(alpha[mask]**2 + omega[mask]**2)
# damping_ratios = damping_ratios[idx]

print("----------------------------")
print("\n--- 网络输出 (模态参数) ---")
print("frequency:", np.array2string(freqs, precision=2))
print("damping ratio:", np.array2string(damping_ratios, precision=2))
print("----------------------------")


# data_to_save = [p_exist, alpha, omega, rho, theta]
B, A = build_BA_from_residues(alpha, omega, rho, theta, Ts, p_exist=p_exist, exist_thresh=0.5)


# ==== 保存结果到 txt 文件 ====
# save_path = r"C:\Users\Leo\Desktop\model_output.txt"
# with open(save_path, "w", encoding="utf-8") as f:
#     for arr in data_to_save:
#         np.savetxt(f, arr[None, :], fmt="%.6e", delimiter="\t")

# print(f"\n结果已保存至: {save_path}")
y_hat = lfilter(B, A, u)


# plt.figure()
# plt.plot(loss, label="Loss")
# plt.xlim(0, 5000)
# plt.ylim(0, 0.2)
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Training Loss History")
# plt.legend()
# plt.grid()

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
ax1.plot(t, y, label="仿真信号")
ax1.legend()

ax2.plot(t, y_hat, label='预测信号')
ax2.legend()
ax2.set_xlabel("时间/s")
plt.subplots_adjust(hspace=0)
bbox1 = ax1.get_position()
bbox2 = ax2.get_position()
y_center = (bbox1.y1 + bbox2.y0) / 2
fig.text(
    0.02, y_center, "加速度/g",
    rotation="vertical",
    va="center",
    fontsize=7
)
# plt.show()


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
label_names = ["仿真信号频谱", "预测信号频谱"]
signals = [y, y_hat]
for i, signal in enumerate(signals):
    f, S = compute_spectrum(signal, fs, mode="amplitude")
    ax = ax1 if i == 0 else ax2
    ax.plot(f, S, label=label_names[i])
    ax.legend()

plt.xlim(0, 60)
ax2.set_xlabel("频率/Hz")

plt.subplots_adjust(hspace=0)
fig.text(
    0.02, y_center, "加速度幅度/g",
    rotation="vertical",
    va="center",
    fontsize=7
)
plt.show()


# plt.figure()
# plt.subplot(2, 2, 1)
# plt.plot(t, y)
# plt.title('origin')

# # plt.figure()
# plt.subplot(2, 2, 3)
# plt.subplots_adjust(hspace=0)
# plt.plot(t, y_hat)
# plt.title('predict')


# plt.figure()
# signals = [y, y_hat]
# for i, signal in enumerate(signals):
#     signal_fft = np.fft.fft(signal)
#     N = signal.shape[-1]
#     window = np.hanning(N)
#     signal = signal * window
#     f = np.fft.fftfreq(N, d=1/fs)
#     signal_mag = 2.0  * np.abs(signal_fft[:N//2]) / N
#     signal_freq = f[:N//2]
#     plt.figure()
#     # plt.subplot(2,2,2*(i+1))
#     plt.plot(signal_freq, signal_mag)
#     plt.xlabel('频率/Hz')
#     plt.ylabel('幅值')
#     plt.xlim(0, 60)
# plt.tight_layout()
# plt.show()
# # ==== 验证归一化关系 ====
# out_norm = Normalizer.load(out_norm_path)
# mean = torch.tensor(out_norm.mean, dtype=torch.float32)
# std  = torch.tensor(out_norm.std,  dtype=torch.float32)

# print("----------------------------")
# print("\n--- 归一化参数 ---")
# print(f"对应 mean/std = {mean[0].item():.6f} ± {std[0].item():.6f}")
# print(f"对应 mean/std = {mean[1].item():.6f} ± {std[1].item():.6f}")
# print(f"对应 mean/std = {mean[2].item():.6f} ± {std[2].item():.6f}")
# print("-------------------")

# %%
