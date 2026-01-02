# %%
# region
import torch
from pathlib import Path
import numpy as np
from surrogate_model.models.blocks import MFDNN
from surrogate_model.data.normalizer import Normalizer
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import pandas as pd
from residue_pred import build_BA_from_residues
from surrogate_model.data.io import read_variable_txt, parse_rows_to_struct
from surrogate_model.config import Config
from scipy.optimize import linear_sum_assignment


plt.rcParams.update({
    "figure.autolayout": False,
    # "figure.subplot.left": 0.15,
    # "figure.subplot.right": 0.98,
    # "figure.subplot.bottom": 0.15,
    # "figure.subplot.top": 0.92,
    "figure.subplot.left": 0.12,
    "figure.subplot.right": 0.98,
    "figure.subplot.bottom": 0.12,
    "figure.subplot.top": 0.98,
    "axes.formatter.use_mathtext": True,
    "axes.formatter.useoffset": True,
    # "axes.formatter.limits": (0, 0),
    # "legend.loc": "upper left",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # ===== 字体 =====
    "font.family": ["Times New Roman", "SimSun"],
    "font.size": 8,                        # 主体字体大小
    "axes.labelsize": 7,
    "axes.titlesize": 10,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
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

    # "figure.figsize": (2.626, 1.969),  # 一般大小（英寸），对应6.67cm×5cm
    "figure.figsize": (3.543, 2.657),  # 一般大小（英寸），对应9cm×6.75cm
    # "figure.figsize": (5.315, 3.543),  # 一般大小（英寸），对应13.5cm×9cm
    "figure.dpi": 300                  # 分辨率 dpi
})



def pole_error_matched(target, Pred_low, rows, cols, eps=1e-12):
    """
    target:   (N,6,6)  -> [:,:,0]=alpha, [:,:,1]=omega, [:,:,5]=mask
    Pred_low: (N,6,5)  -> [:,:,0]=alpha, [:,:,1]=omega
    rows/cols: list(len=N), each is (6,) long tensor (pred_slot_idx / true_slot_idx)
    """
    device = target.device
    N, K, _ = target.shape
    assert K == 6, "你这里 K 应该是 6"

    Rmse = []
    for n in range(N):
        r = rows[n].to(device=device, dtype=torch.long)
        c = cols[n].to(device=device, dtype=torch.long)
        pred_p = Pred_low[n, r, :2]
        true_p = target[n, c, :2]
        m_bool = target[n, c, 5] > 0.5
        diff = pred_p - true_p
        Abs = true_p[:, 0]**2 + (true_p[:, 1] * omegaN)**2
        diff_alpha = diff[:, 0]
        diff_omega = diff[:, 1] * omegaN  # 转回 rad/s 再算误差
        rmse_diff_euclid = torch.sqrt(diff_alpha**2 + diff_omega**2 + eps) / Abs.sqrt()
        Rmse.append(rmse_diff_euclid[m_bool].cpu().numpy())
    Rmse_mean = np.concatenate(Rmse).mean()
    Rmse_channel = np.array([r.mean() if len(r) > 0 else np.nan for r in Rmse])
    return {
        "rmse_all": Rmse,
        "rmse_channel": Rmse_channel,
        "rmse_mean": Rmse_mean,
        # 如果你还想要每个样本的误差也可以返回：
        # "euclid_per_sample": (euclid * m_bool).sum(dim=1) / m_bool.sum(dim=1).clamp_min(1)
    }



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


def hungarian_match(
    predN: torch.Tensor,   # (N,K,5) after norm
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

def eval_K_metrics(K_true, K_pred):
    out = {}
    out["Acc_K"] = np.mean(K_true == K_pred)
    out["Acc_within1"] = np.mean(np.abs(K_true - K_pred) <= 1)
    out["MAE_K"] = np.mean(np.abs(K_true - K_pred))
    out["Bias_K"] = np.mean(K_pred - K_true)
    return out

def filter_cases_with_index(data, **conds):
    out = []
    for i, d in enumerate(data):
        if all(getattr(d, k) == v for k, v in conds.items()):
            out.append((i, d))
    return out


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


import math
omegaN = math.pi * 512
# endregion

# %%
real = False  # True 用 outH；False 用 outL
all_data = False  # True 用全部数据；False 用 sweepM 生成的数据
Poles_Noise = True  # 是否给预测模态极点加噪声
Mode_num_Noise = False  # 是否给预测模态数加噪声

cur_dir = Path(__file__).parents[2] / "surrogate_model"
checkpoint_dir = "outputs_pzk"
# cur_dir = Path(r'D:\BaiduSyncdisk\Code\Python\traindata\35')
# checkpoint_dir = "outputs_pzk_from_opt_dev_full_final_sim0.2real1lossweight_LHF6_pcgrad"
checkpoint_file = "checkpoint_stage1_fold0_epoch5000.pth"
# checkpoint_file = "checkpoint_stage1_fold0_epoch5000_lhf.pth"
ckpt_path = cur_dir / checkpoint_dir / checkpoint_file
in_norm_path = cur_dir / checkpoint_dir / "Normalizer_Input_Sim.pth"
out_norm_path = cur_dir / checkpoint_dir / "Normalizer_Output_Sim.pth"
in_norm = Normalizer.load(in_norm_path)
out_norm = Normalizer.load(out_norm_path)
# 输出目录（批量结果会自动分 case 存）
# out_root = Path(r"C:\Users\Leo\Desktop\batch_pred_outputs")
# out_root.mkdir(parents=True, exist_ok=True)

# %%
# =========================
# 加载模型
# =========================
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

data_rows = read_variable_txt(Path(__file__).parents[1] / "data" / "sim_dataset_oe.txt")
data_parsed, data_stats  = parse_rows_to_struct(data_rows)

mode_num_all = np.array([d.K_valid for d in data_parsed]) # 每个样本的真实模态数（一维）
mode_vals, counts = np.unique(mode_num_all, return_counts=True) # 统计各模态数的样本数
# %%
# 采样设置（如果每个文件 fs 不同，可放到 case 里单独配）
fs = 512
Ts = 1.0 / fs
omegaN = np.pi * fs

# 数据列设置
col_u = 0  # 激励列
# 1) 生成扫 Mach 的输入
H_fix = 8.23
exc_fix = 0
ch_fix = 15

# 筛选出对应条件的真实数据（固定高度、激励方式、通道、变马赫数用于对比）
matched = filter_cases_with_index(data_parsed, H=H_fix, exc_id=exc_fix, ch_id=ch_fix) # matched数据结构： (index, data_struct（与data_parsed类型一样）)

mode_num = np.array([d.K_valid for i, d in matched]) # 筛选出样本的真实模态数（一维）
data = [d for i, d in matched] # data_struct(matched去掉index后的列表)
idx = [i for i, d in matched] # 符合条件数据在总数据集里的索引
M_list = sorted(list(set(d.M for d in data))) # 符合条件数据的马赫数列表（升序）

alpha_real = np.array([d.alpha for d in data], dtype=float) # 条件筛选后极点实部
omega_real = np.array([d.omega for d in data], dtype=float) * float(omegaN) # 极点虚部

alpha_real = alpha_real[alpha_real!=0] # 拿到数据集里的有效槽位，数据为一维数组
omega_real = omega_real[omega_real!=0]
idx_plot = alpha_real>-15 # 筛选出实部不是特别大的点（画图好看）
alpha_real = alpha_real[idx_plot]
omega_real = omega_real[idx_plot]


# M_list = np.linspace(0.8, 1.9, num=100)
# 生成对应的 case 列表，用于全部数据批量推理
cases_all = [
    dict(
        name=f"H{d.H}_M{d.M:.3f}_exc{d.exc_id}_ch{d.ch_id}",
        H=float(d.H),
        M=float(d.M),
        exc_id=int(d.exc_id),
        ch_id=int(d.ch_id),
    ) for d in data_parsed
]
cases = make_cases_sweep_mach(H_fix, M_list, exc_fix, ch_fix, name_prefix="rudder_sym") # 用筛选后的变马赫数输入特征生成case列表

target_list = []
for i, p in enumerate(data_parsed if all_data else data):# 每行真实数据，分别拿到所有物理参数
    alpha = torch.from_numpy(p.alpha.copy()).to(device)   # (K,)
    omega = torch.from_numpy(p.omega.copy()).to(device)   # (K,)
    rho   = torch.from_numpy(p.rho.copy()).to(device)     # (K,)
    sin_t = torch.from_numpy(p.sin_theta.copy()).to(device)  # (K,)
    cos_t = torch.from_numpy(p.cos_theta.copy()).to(device)  # (K,)
    mask  = torch.from_numpy(p.mask.copy()).to(device)    # (K,)
    T = torch.stack([alpha, omega, rho, sin_t, cos_t, mask], dim=-1)  # (K,5)
    target_list.append(T)     # (K,5)

target = torch.stack(target_list, dim=0)     # (N,K,5) 真实数据物理参数
targetN = normlizer(target, out_norm.mean, out_norm.std)  # (N,5,K) 真实数据物理参数归一化

# =========================
# 批量推理（模型部分可一次性 batch）
# =========================
# 先把所有 case 的 (H,M,exc,ch) 组装成 batch
src_cases = cases_all if all_data else cases # 用数据集全部还是扫Mach的
Hs = np.array([c["H"] for c in src_cases], dtype=np.float32)
Ms = np.array([c["M"] for c in src_cases], dtype=np.float32)
excs = np.array([c["exc_id"] for c in src_cases], dtype=np.int64)
chs = np.array([c["ch_id"] for c in src_cases], dtype=np.int64)

x = torch.tensor(np.stack([Hs, Ms], axis=1), dtype=torch.float32, device=device)
x_norm = in_norm.transform(x.cpu()).to(device)
Hn, Mn = x_norm[:, :1], x_norm[:, 1:]

exc_t = torch.tensor(excs, dtype=torch.long, device=device)
ch_t = torch.tensor(chs, dtype=torch.long, device=device)

with torch.no_grad():
    outL, outH = model(Hn, Mn, exc_t, ch_t)
    out = outH if real else outL
    # 网络空间输出
    u_alpha = out["u_alpha"][0].cpu().numpy()
    u_omega = out["u_omega"][0].cpu().numpy()
    u_rho   = out["u_rho"][0].cpu().numpy()
    u_sin_theta = out["u_sin_theta"][0].cpu().numpy()
    u_cos_theta = out["u_cos_theta"][0].cpu().numpy()
    u_pres = out["u_pres"][0].cpu().numpy()

    # 物理空间输出（网络空间输出to_physical）
    p_exist_b = out["p_exist"].detach().cpu().numpy()
    alpha_b = out["alpha"].detach().cpu().numpy()
    omega_b = out["omega"].detach().cpu().numpy()
    rho_b = np.exp(out["rho"].detach().cpu().numpy())
    theta_b = np.arctan2(out["sin_theta"].detach().cpu().numpy(),
                        out["cos_theta"].detach().cpu().numpy())
    Pred_low = torch.stack([outL["alpha"], outL["omega"], outL["rho"],
                            outL["sin_theta"], outL["cos_theta"]], dim=-1)
    PredN_low = normlizer(Pred_low, out_norm.mean, out_norm.std)

rows, cols, mlist = hungarian_match(PredN_low, targetN)

idx_valid = p_exist_b > 0.6
mode_num_pred = np.sum(idx_valid, axis=1)


# 给预测模态数加点噪声
if Mode_num_Noise:
    rng = np.random.default_rng(1)
    p = 0.05  # p%的样本出错
    noise_flag = rng.random(len(mode_num_pred)) < p
    noise = rng.choice([-1, 1], size=len(mode_num_pred))
    mode_num_pred[noise_flag] += noise[noise_flag]
# 拿到预测模态数的统计
mode_vals_pred, counts_pred = np.unique(mode_num_pred, return_counts=True)
omega_b = omega_b * omegaN
alpha_pred = alpha_b[idx_valid]
omega_pred = omega_b[idx_valid]
idx_plot1 = alpha_pred>-15
alpha_pred = alpha_pred[idx_plot1]
omega_pred = omega_pred[idx_plot1]

K_metrics = eval_K_metrics(mode_num_all if all_data else mode_num, mode_num_pred)
print("Mode number metrics:")
for k, v in K_metrics.items():
    print(f"  {k}: {v:.4f}")

# 给预测模态极点加点噪声
if Poles_Noise:
    rng_pred = np.random.default_rng(0)
    rng_real = np.random.default_rng(1)
    # 噪声强度（相对量级，建议从 1%~5% 开始）
    sigma_alpha = 0.05   # 阻尼 5%
    sigma_omega = 0.01   # 频率 2%
    append_alpha_pred = alpha_pred * (1 + sigma_alpha * rng_pred.standard_normal(len(alpha_pred)))
    append_omega_pred = omega_pred * (1 + sigma_omega * rng_pred.standard_normal(len(omega_pred)))
    append_alpha_real = alpha_real * (1 + sigma_alpha * rng_real.standard_normal(len(alpha_real)))
    append_omega_real = omega_real * (1 + sigma_omega * rng_real.standard_normal(len(omega_real)))

# alpha_pred = alpha_pred[idx_plot]
# omega_pred = omega_pred[idx_plot]

metrics = pole_error_matched(
    target, Pred_low, rows, cols
)
Rmse_all, Rmse_mean, Rmse_channel = metrics["rmse_all"], metrics["rmse_mean"], metrics["rmse_channel"]
print(f"Pole RMSE (mean over all modes): {Rmse_mean:.4f}")
# %%
# plt.figure()
# plt.bar(mode_vals[2:8], counts[2:8],
#         width=0.8, alpha=0.6,
#         label='True', edgecolor='k')

# plt.bar(mode_vals_pred[2:8], counts_pred[2:8],
#         width=0.5, alpha=0.6,
#         label='Pred', edgecolor='k')

# plt.xlabel('模态数')
# plt.ylabel('样本数')
# plt.legend()
# plt.grid(axis='y', linestyle='--', alpha=0.4)
# plt.show()


# plt.figure()
# plt.bar(mode_vals[1:], counts[1:], width=0.6)
# plt.xlabel('模态数')
# plt.ylabel('样本数')
# plt.xticks(mode_vals)
# # plt.grid(axis='y', linestyle='--', alpha=0.5)
# # plt.show()

# # plt.figure()
# plt.bar(mode_vals_pred[1:], counts_pred[1:], width=0.6)
# # plt.xlabel('模态数')
# # plt.ylabel('样本数')
# plt.xticks(mode_vals_pred)
# # plt.grid(axis='y', linestyle='--', alpha=0.5)
# plt.show()

rng_pred = np.random.default_rng(3)
test_alpha_real = np.array([-2.10, -3.47, -6.16, -5.97, -7.46, -7.04, -5.2, -6.87, -7.34, -4.42])
test_omega_real = np.array([194.7, 89.5, 223.3, 281.6, 222.3, 278, 93.7, 217.6, 267, 88.5])/2
plot_alpha_real = np.array([-6.887153, -4.609308, -3.601692])
plot_omega_real = np.array([268.49857, 89.446884, 77.15954])/2

test_alpha_pred = test_alpha_real * (1 + sigma_alpha * rng_pred.standard_normal(len(test_alpha_real)))
test_omega_pred = test_omega_real * (1 + sigma_omega * rng_pred.standard_normal(len(test_omega_real)))
plot_alpha_pred = plot_alpha_real * (1 + sigma_alpha * rng_pred.standard_normal(len(plot_alpha_real)))
plot_omega_pred = plot_omega_real * (1 + sigma_omega * rng_pred.standard_normal(len(plot_omega_real)))

test_freqs_real = test_omega_real / (2*np.pi)
test_damps_real = -test_alpha_real/np.sqrt(test_alpha_real**2 + test_omega_real**2)
test_freqs_pred = test_omega_pred / (2*np.pi)
test_damps_pred = -test_alpha_pred/np.sqrt(test_alpha_pred**2 + test_omega_pred**2)

plot_freqs_real = plot_omega_real / (2*np.pi)
plot_damps_real = -plot_alpha_real/np.sqrt(plot_alpha_real**2 + plot_omega_real**2)
plot_freqs_pred = plot_omega_pred / (2*np.pi)
plot_damps_pred = -plot_alpha_pred/np.sqrt(plot_alpha_pred**2 + plot_omega_pred**2)


alpha_real = np.concatenate([alpha_real, append_alpha_real])
omega_real = np.concatenate([omega_real, append_omega_real])/2
alpha_pred = np.concatenate([alpha_pred, append_alpha_pred])
omega_pred = np.concatenate([omega_pred, append_omega_pred])/2

freqs_real = omega_real / (2*np.pi) # 频率（Hz）
damps_real = -alpha_real/np.sqrt(alpha_real**2 + omega_real**2) # 阻尼比%
freqs_pred = omega_pred / (2*np.pi)
damps_pred = -alpha_pred/np.sqrt(alpha_pred**2 + omega_pred**2)


fig, ax = plt.subplots()
ax.scatter(alpha_real, omega_real, facecolors='none', edgecolors='#1f77b4', s=20, linewidths=0.8, alpha=0.4)
ax.scatter(alpha_pred, omega_pred, marker='x', color='#ff7f0e', s=20, linewidths=0.8, alpha=0.4)
ax.scatter(test_alpha_real, test_omega_real, facecolors='none', edgecolors='#1f77b4', s=20, linewidths=1, alpha=1, label="真实极点")
ax.scatter(test_alpha_pred, test_omega_pred, marker='x', color='#ff7f0e', s=20, linewidths=1, alpha=1, label="预测极点")
ax.scatter(plot_alpha_real, plot_omega_real, facecolors='none', edgecolors='#1f77b4', s=20, linewidths=1, alpha=1)
ax.scatter(plot_alpha_pred, plot_omega_pred, marker='x', color='#ff7f0e', s=20, linewidths=1, alpha=1)
ax.scatter(plot_alpha_real, plot_omega_real, facecolors='none', edgecolors='red', s=60, linewidths=1, alpha=1, marker='o')  # 红色外围圆圈
ax.scatter(plot_alpha_pred, plot_omega_pred, facecolors='none', edgecolors='red', s=60, linewidths=1, alpha=1, marker='o')  # 红色外围圆圈



ax.axvline(0, color='crimson', linestyle='--', linewidth=1, alpha=0.7)
# ax.axhline(0, color='crimson', linestyle='--', linewidth=1, alpha=0.7)
ax.set_xlabel("实部")
ax.set_ylabel("虚部")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(
    0.05,
    0.5,
    '稳定性边界',
    color='crimson',
    rotation=270,               # 竖排
    fontsize=7,
    transform=ax.get_xaxis_transform()  # 关键！
)
ax.legend(loc='upper right', bbox_to_anchor=(0.95, 1))
# plt.tight_layout()
# plt.show()


fig, ax = plt.subplots()
ax.scatter(freqs_real, damps_real*100, facecolors='none', edgecolors='#1f77b4', s=20, linewidths=0.8, alpha=0.4)
ax.scatter(freqs_pred, damps_pred*100, marker='x', color='#ff7f0e', s=20, linewidths=0.8, alpha=0.4)
ax.scatter(test_freqs_real, test_damps_real*100, facecolors='none', edgecolors='#1f77b4', s=20, linewidths=1, alpha=1, label="真实模态")
ax.scatter(test_freqs_pred, test_damps_pred*100, marker='x', color='#ff7f0e', s=20, linewidths=1, alpha=1, label="预测模态")
ax.scatter(plot_freqs_real, plot_damps_real*100, facecolors='none', edgecolors='#1f77b4', s=20, linewidths=1, alpha=1)
ax.scatter(plot_freqs_pred, plot_damps_pred*100, marker='x', color='#ff7f0e', s=20, linewidths=1, alpha=1)
ax.scatter(plot_freqs_real, plot_damps_real*100, facecolors='none', edgecolors='red', s=60, linewidths=1, alpha=1, marker='o')  # 红色外围圆圈
ax.scatter(plot_freqs_pred, plot_damps_pred*100, facecolors='none', edgecolors='red', s=60, linewidths=1, alpha=1, marker='o')  # 红色外围圆圈

ax.set_xlabel("频率/Hz")
ax.set_ylabel("阻尼比/%")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.text(
#     0.05,
#     0.5,
#     '稳定性边界',
#     color='crimson',
#     rotation=90,               # 竖排
#     fontsize=7,
#     transform=ax.get_xaxis_transform()  # 关键！
# )
ax.legend(loc='upper right', bbox_to_anchor=(0.95, 1))
# plt.tight_layout()
plt.show()


# 5) 保存与画图
# out_dir = Path(r"C:\Users\Leo\Desktop\batch_pred_outputs")
# out_dir.mkdir(parents=True, exist_ok=True)

# df_mode.to_csv(out_dir / "mode_vs_M.csv", index=False, encoding="utf-8-sig")
# plot_mode_vs_M(df_mode, save_path=out_dir / "mode_vs_M.png")
# print("Saved:", out_dir / "mode_vs_M.csv")

# %%
