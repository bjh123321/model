import torch
from pathlib import Path
import numpy as np
from ..config import Config
from ..models.blocks import MFDNN
from ..data.normalizer import Normalizer
import matplotlib.pyplot as plt

# ==== 配置区域 ====
cur_dir = Path(__file__).parents[1]
ckpt_path = cur_dir / "outputs_pzk/checkpoint_stage1_fold0_epoch7000.pth"
in_norm_path  = cur_dir / "outputs_pzk/Normalizer_Input_Sim.pth"
out_norm_path = cur_dir / "outputs_pzk/Normalizer_Output_Sim.pth"

H, M = 11, 1.9
exc_id, ch_id = 2, 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 加载 ====
print("==> Loading ckpt...")
ckpt = torch.load(ckpt_path, map_location="cpu")
state = ckpt.get("model_state_dict", ckpt)
loss_sim = ckpt.get("loss_history", None)
plt.figure()
plt.plot(loss_sim, label="Loss Sim")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("Training Loss History (Simulated Data)")
plt.legend()
plt.grid()
plt.show()

cfg = Config()

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

# ==== 输入归一化 ====
in_norm = Normalizer.load(in_norm_path)
x = torch.tensor([[H, M]], dtype=torch.float32, device=device)
x_norm = in_norm.transform(x)
Hn, Mn = x_norm[:, :1], x_norm[:, 1:]

exc = torch.tensor([exc_id], dtype=torch.long, device=device)
ch  = torch.tensor([ch_id], dtype=torch.long, device=device)

# ==== 推理 ====
with torch.no_grad():
    outL, outH = model(Hn, Mn, exc, ch)
    out = outL  # stage1 输出
    alpha = out["alpha"][0].cpu().numpy()
    omega = out["omega"][0].cpu().numpy()
    gain  = out["gain"][0].cpu().numpy()
    zero  = out["zero"][0].cpu().numpy()
    p_exist = out["p_exist"][0].cpu().numpy()

print("\n--- 网络输出 (物理域) ---")
print("p_exist:", np.array2string(p_exist, precision=6))
print("alpha  :", np.array2string(alpha, precision=6))
print("omega  :", np.array2string(omega, precision=6))
print("gain   :", np.array2string(gain,  precision=6))
print("zero   :", np.array2string(zero,  precision=6))
print("----------------------------")

# ==== 验证归一化关系 ====
out_norm = Normalizer.load(out_norm_path)
mean = torch.tensor(out_norm.mean, dtype=torch.float32)
std  = torch.tensor(out_norm.std,  dtype=torch.float32)

alpha_back = alpha * std[0].item() + mean[0].item()
omega_back = omega * std[1].item() + mean[1].item()
gain_back  = gain  * std[2].item() + mean[2].item()
zero_back  = zero  * std[3].item() + mean[3].item()

print("\n反归一化后（如果模型输出是标准化值）")
print("alpha", np.array2string(alpha_back, precision=6))
print("omega", np.array2string(omega_back, precision=6))
print("gain ", np.array2string(gain_back,  precision=6))
print("zero ", np.array2string(zero_back,  precision=6))
print("----------------------------")
print("\n--- 归一化参数 ---")
print(f"对应 mean/std = {mean[0].item():.6f} ± {std[0].item():.6f}")
print(f"对应 mean/std = {mean[1].item():.6f} ± {std[1].item():.6f}")
print(f"对应 mean/std = {mean[2].item():.6f} ± {std[2].item():.6f}")
print(f"对应 mean/std = {mean[3].item():.6f} ± {std[3].item():.6f}")
print("-------------------")
