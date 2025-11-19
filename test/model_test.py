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


plt.rcParams.update({
    "font.family": "Times New Roman",  # 字体
    "font.size": 10,                   # 默认字体大小
    "axes.labelsize": 10,              # 坐标轴标签
    "axes.titlesize": 12,              # 图标题
    "xtick.labelsize": 8,              # X轴刻度
    "ytick.labelsize": 8,              # Y轴刻度
    "legend.fontsize": 8,              # 图例字体
    "lines.linewidth": 1.0,            # 曲线宽度
    "axes.linewidth": 1.0,             # 坐标轴线宽
    # "figure.figsize": (3.5, 2.6),          # 图尺寸（英寸）单栏
    "figure.figsize": (7.0, 4.5),          # 图尺寸（英寸）双栏
    # "figure.dpi": 300                  # 分辨率 dpi
})


# ==== 配置区域 ====
real = True  # 是否使用真实响应网络
# data_file = Path(r"D:\BaiduSyncdisk\项目\项目数据\南航数据\35\20250523标准\H11\M1.90\1_H11_M1.90_方向舵对称_1-45Hz_幅值0.3.txt")
data_file = Path(r"D:\BaiduSyncdisk\项目\试飞数据\35A\试飞数据35A\H5.5\M0.74\H5.5_M0.74_方向舵对称_幅值0.6.txt")
cur_dir = Path(r'D:\BaiduSyncdisk\Code\Python\traindata\35')
# cur_dir = Path(__file__).parents[2] / "surrogate_model"
checkpoint_dir = "outputs_pzk"
# checkpoint_file = "checkpoint_stage2_fold0_epoch5000_affinefix.pth"
# checkpoint_file = "checkpoint_stage1_fold0_epoch5000_lhf.pth"
checkpoint_file = "checkpoint_stage2_fold0_epoch5000.pth"
# checkpoint_file = "checkpoint_stage2_fold0_epoch5000.pth"


H = 5.5          # 高度
M = 0.74        # 马赫数
exc_id = 0      # 激励方式
ch_id  = 13    # 响应通道
col_u = 0      # 激励
col_y = ch_id + 2
fs = 512
Ts = 1.0 / fs

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
out_norm_path = cur_dir / checkpoint_dir / "Normalizer_Real_Fold0.pth"
# ==== 加载 ====
print("==> Loading ckpt...")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
state = ckpt.get("model_state_dict", ckpt)
# loss = ckpt.get("loss_real_history", None)

# region
# 自动推断 embed 维度
if "low.emb_exc.weight" in state:
    cfg.embed_excitation_num = state["low.emb_exc.weight"].shape[0]
    cfg.embed_excitation_dim = state["low.emb_exc.weight"].shape[1]
if "low.emb_chn.weight" in state:
    cfg.embed_channel_num = state["low.emb_chn.weight"].shape[0]
    cfg.embed_channel_dim = state["low.emb_chn.weight"].shape[1]

# K_max 同样自动推断
if "lin.A" in state:
    cfg.K_max = state["lin.A"].shape[1]

model = MFDNN(cfg).to(device)
model.load_state_dict(state, strict=False)
model.eval()

# endregion
# ==== 输入归一化 ====
in_norm = Normalizer.load(in_norm_path)
x = torch.tensor([[H, M]], dtype=torch.float32, device=device)
x_norm = in_norm.transform(x)
Hn, Mn = x_norm[:, :1], x_norm[:, 1:]

exc = torch.tensor([exc_id], dtype=torch.long, device=device)
ch  = torch.tensor([ch_id], dtype=torch.long, device=device)
omegaN = np.pi * fs
# ==== 推理 ====
with torch.no_grad():
    outL, outH = model(Hn, Mn, exc, ch)
    if real:
        out = outH
    else:
        out = outL
    alpha = out["alpha"][0].cpu().numpy()
    omega = out["omega"][0].cpu().numpy()
    rho   = out["rho"][0].cpu().numpy()
    sin_theta = out["sin_theta"][0].cpu().numpy()
    cos_theta = out["cos_theta"][0].cpu().numpy()
    p_exist = out["p_exist"][0].cpu().numpy()
    omega = omega*omegaN
    rho = np.exp(rho)
    theta = np.atan2(sin_theta, cos_theta)
print("\n--- 网络输出 (物理域) ---")
print("p_exist:", np.array2string(p_exist, precision=6))
print("alpha  :", np.array2string(alpha, precision=6))
print("omega  :", np.array2string(omega, precision=6))
print("rho   :", np.array2string(rho, precision=6))
print("theta   :", np.array2string(theta, precision=6))
print("----------------------------")
data_to_save = [p_exist, alpha, omega, rho, theta]
B, A = build_BA_from_residues(alpha, omega, rho, theta, Ts, p_exist=p_exist, exist_thresh=0.5)


# ==== 保存结果到 txt 文件 ====
save_path = r"C:\Users\Leo\Desktop\model_output.txt"
with open(save_path, "w", encoding="utf-8") as f:
    for arr in data_to_save:
        np.savetxt(f, arr[None, :], fmt="%.6e", delimiter="\t")

# print(f"\n结果已保存至: {save_path}")
y_hat = lfilter(B, A, u)


# plt.figure()
# plt.plot(loss, label="Loss Sim")
# # plt.xlim(0, 2000)
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.yscale("log")
# plt.title("Training Loss History")
# plt.legend()
# plt.grid()

plt.figure()
plt.subplot(2, 2, 1)
plt.plot(t, y)
plt.title('origin')

# plt.figure()
plt.subplot(2, 2, 3)
plt.plot(t, y_hat)
plt.title('predict')


signals = [y, y_hat]
for i, signal in enumerate(signals):
    signal_fft = np.fft.fft(signal)
    N = signal.shape[-1]
    window = np.hanning(N)
    signal = signal * window
    f = np.fft.fftfreq(N, d=1/fs)
    signal_mag = 2.0  * np.abs(signal_fft[:N//2]) / N
    signal_freq = f[:N//2]
    # plt.figure()
    plt.subplot(2,2,2*(i+1))
    plt.plot(signal_freq, signal_mag)
    plt.xlim(0, 60)
plt.tight_layout()
plt.show()
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
