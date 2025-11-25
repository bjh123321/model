# surrogate_model/cli/train_incremental.py
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..config import Config
from ..models.blocks import MFDNN, LoRALinear
from ..losses.masked_mse import MaskedMSE
from ..data.normalizer import Normalizer
from ..data.io import parse_rows_to_struct


# ===========================
# 只对 alpha, omega, rho 做 z-score，sin/cos 不变
# ===========================
def normlizer(pred: torch.Tensor, mean, std):
    """
    pred: (..., 5) = [alpha, omega, rho, sin, cos]
    mean/std: 长度为 3，只对前 3 维标准化
    """
    mean_t = torch.as_tensor(mean, dtype=pred.dtype, device=pred.device)
    std_t = torch.as_tensor(std, dtype=pred.dtype, device=pred.device)

    core = pred[..., :3]   # (...,3)
    others = pred[..., 3:]  # (...,2)

    assert mean_t.numel() == 3 and std_t.numel() == 3, "out_norm.mean/std 应该是长度 3"
    view_shape = [1] * (core.ndim - 1) + [3]
    mean_t = mean_t.view(*view_shape)
    std_t = std_t.view(*view_shape)

    core_n = (core - mean_t) / std_t
    return torch.cat([core_n, others], dim=-1)  # (...,5)


# ===========================
# 增量学习用数据集：直接从 txt 读 + parse_rows_to_struct
# ===========================
class RealIncrementalDataset(Dataset):
    def __init__(self, txt_path: str, in_norm: Normalizer, out_norm: Normalizer):
        super().__init__()
        self.txt_path = txt_path

        # ------- 读 txt 为 rows_float，类似 online_ridge 里读 CSV -------
        rows_float = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 兼容逗号 / 空格
                parts = line.replace(",", " ").split()
                vals = [float(x) for x in parts]
                rows_float.append(vals)

        if not rows_float:
            raise RuntimeError(f"[RealIncrementalDataset] 文件 {txt_path} 为空或无法解析")

        parsed, _ = parse_rows_to_struct(rows_float)
        N = len(parsed)
        if N == 0:
            raise RuntimeError(f"[RealIncrementalDataset] parse_rows_to_struct 结果为空：{txt_path}")

        # ------- 构造输入 H, M, exc, ch -------
        H = torch.tensor([p.H for p in parsed], dtype=torch.float32)   # (N,)
        M = torch.tensor([p.M for p in parsed], dtype=torch.float32)   # (N,)
        exc = torch.tensor([p.exc_id for p in parsed], dtype=torch.long)  # (N,)
        chn = torch.tensor([p.ch_id for p in parsed], dtype=torch.long)   # (N,)

        # 输入归一化：用 Sim 的输入 normalizer（和训练保持一致）
        H = ((H - in_norm.mean[0]) / in_norm.std[0]).unsqueeze(-1)  # (N,1)
        M = ((M - in_norm.mean[1]) / in_norm.std[1]).unsqueeze(-1)  # (N,1)

        # ------- 构造输出：alpha, omega, rho, sin, cos, mask -------
        alpha_list, omega_list, rho_list = [], [], []
        sin_list, cos_list, mask_list = [], [], []

        for p in parsed:
            alpha_list.append(torch.from_numpy(p.alpha.copy()).float())       # (K,)
            omega_list.append(torch.from_numpy(p.omega.copy()).float())
            rho_list.append(torch.from_numpy(p.rho.copy()).float())
            sin_list.append(torch.from_numpy(p.sin_theta.copy()).float())
            cos_list.append(torch.from_numpy(p.cos_theta.copy()).float())
            mask_list.append(torch.from_numpy(p.mask.copy()).float())

        alpha = torch.stack(alpha_list, dim=0)  # (N,K)
        omega = torch.stack(omega_list, dim=0)
        rho = torch.stack(rho_list, dim=0)
        sin_t = torch.stack(sin_list, dim=0)
        cos_t = torch.stack(cos_list, dim=0)
        mask = torch.stack(mask_list, dim=0)    # (N,K)

        # 物理域 -> 组合成 (N,K,5)
        y_phys = torch.stack([alpha, omega, rho, sin_t, cos_t], dim=-1)  # (N,K,5)

        # 输出归一化：和训练阶段一样的 z-score（只动前 3 维）
        yN = normlizer(y_phys, out_norm.mean, out_norm.std)  # (N,K,5)

        self.H = H
        self.M = M
        self.exc = exc
        self.chn = chn
        self.yN = yN              # (N,K,5)
        self.mask = mask          # (N,K)

    def __len__(self):
        return self.H.size(0)

    def __getitem__(self, idx):
        # 将 (K,5) 拆成 dict（与训练阶段一致）
        yNk = self.yN[idx]  # (K,5)
        return {
            "H": self.H[idx],      # (1,)
            "M": self.M[idx],      # (1,)
            "exc": self.exc[idx],  # ()
            "chn": self.chn[idx],  # ()
            "yN": {
                "alpha":     yNk[:, 0],   # (K,)
                "omega":     yNk[:, 1],
                "rho":       yNk[:, 2],
                "sin_theta": yNk[:, 3],
                "cos_theta": yNk[:, 4],
            },
            "mask": self.mask[idx] # (K,)
        }


# ===========================
# 冻结主干，只开 LoRA
# ===========================
@torch.no_grad()
def freeze_backbone_open_lora(model: MFDNN):
    # 先全部冻结
    for p in model.parameters():
        p.requires_grad_(False)
    # 再只打开 LoRA 层的 A/B
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad_(True)
            m.lora_B.requires_grad_(True)
            # 明确关掉 base
            for p in m.base.parameters():
                p.requires_grad_(False)


# ===========================
# KD loss（teacher-student 一致性约束），与 MaskedMSE 一样的 dict 结构
# ===========================
def kd_loss_fn(new: dict, old: dict, mask: torch.Tensor, w=1.0):
    keys = ["alpha", "omega", "rho", "sin_theta", "cos_theta"]
    m = mask
    loss = 0.0
    for k in keys:
        diff2 = (new[k] - old[k]) ** 2      # (B,K)
        loss += (diff2 * m).mean()
    return loss * w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", type=str, required=True, help="作为 teacher 的旧模型（也是 student 初始权重）")
    ap.add_argument("--real_new_txt", type=str, required=True, help="新真实数据 txt")
    ap.add_argument("--replay_txt", type=str, default="", help="旧样本(真实/仿真) txt（可选）")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta_kd", type=float, default=0.2)
    ap.add_argument("--gamma_replay", type=float, default=0.2)
    ap.add_argument("--save_ckpt", type=str, default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    cfg.use_lora = True  # 增量阶段只训 LoRA

    # ------- 构建 student / teacher -------
    student = MFDNN(cfg).to(device)
    teacher = MFDNN(cfg).to(device).eval()

    # 载入同一份 base ckpt
    ckpt = torch.load(args.base_ckpt, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    student.load_state_dict(state, strict=False)
    teacher.load_state_dict(state, strict=False)

    # 冻结主干，只开 LoRA
    freeze_backbone_open_lora(student)

    # ------- 载入 normalizer（和 online_ridge 一样的命名） -------
    ckpt_path = Path(args.base_ckpt)
    in_norm_path = ckpt_path.with_name("Normalizer_Input_Sim.pth")
    out_norm_path = ckpt_path.with_name("Normalizer_Output_Real_Fold0.pth")
    in_norm = Normalizer.load(in_norm_path)
    out_norm = Normalizer.load(out_norm_path)

    # ------- 数据集：直接用上面的 RealIncrementalDataset -------
    ds_new = RealIncrementalDataset(args.real_new_txt, in_norm, out_norm)

    ds_replay = None
    if args.replay_txt and args.replay_txt.strip():
        ds_replay = RealIncrementalDataset(args.replay_txt, in_norm, out_norm)

    dl_new = DataLoader(ds_new, batch_size=args.batch_size, shuffle=True, drop_last=False)
    dl_replay = DataLoader(ds_replay, batch_size=args.batch_size, shuffle=True, drop_last=False) if ds_replay is not None else None

    # ------- 损失：MaskedMSE（与你训练阶段的结构一致：dict） -------
    mean_t = torch.as_tensor(out_norm.mean, dtype=torch.float32, device=device)
    std_t  = torch.as_tensor(out_norm.std, dtype=torch.float32, device=device)
    crit = MaskedMSE(mean=mean_t, std=std_t)

    opt = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )

    for epoch in range(1, args.epochs + 1):
        student.train()
        it_replay = iter(dl_replay) if dl_replay is not None else None

        # 防止没有 batch 时 loss_* 没定义
        loss_new = torch.tensor(0.0, device=device)
        loss_kd  = torch.tensor(0.0, device=device)
        loss_rep = torch.tensor(0.0, device=device)

        for batch in dl_new:
            H = batch["H"].to(device)      # (B,1)
            M = batch["M"].to(device)      # (B,1)
            e = batch["exc"].to(device)    # (B,)
            c = batch["chn"].to(device)    # (B,)

            # yN 是 dict，需要逐项转 device
            yN = {
                "alpha":      batch["yN"]["alpha"].to(device),
                "omega":      batch["yN"]["omega"].to(device),
                "rho":        batch["yN"]["rho"].to(device),
                "sin_theta":  batch["yN"]["sin_theta"].to(device),
                "cos_theta":  batch["yN"]["cos_theta"].to(device),
            }
            mask = batch["mask"].to(device)  # (B,K)

            # --------- 新真实数据上的监督损失 ----------
            _, outH_new = student(H, M, e, c)    # dict
            loss_new = crit(outH_new, yN, mask)  # dict 版 MaskedMSE

            # --------- Replay + KD（可选） ----------
            loss_kd = torch.tensor(0.0, device=device)
            loss_rep = torch.tensor(0.0, device=device)

            if it_replay is not None:
                try:
                    batch_r = next(it_replay)
                except StopIteration:
                    it_replay = iter(dl_replay)
                    batch_r = next(it_replay)

                Hr = batch_r["H"].to(device)
                Mr = batch_r["M"].to(device)
                er = batch_r["exc"].to(device)
                cr = batch_r["chn"].to(device)
                yNr = {
                    "alpha":      batch_r["yN"]["alpha"].to(device),
                    "omega":      batch_r["yN"]["omega"].to(device),
                    "rho":        batch_r["yN"]["rho"].to(device),
                    "sin_theta":  batch_r["yN"]["sin_theta"].to(device),
                    "cos_theta":  batch_r["yN"]["cos_theta"].to(device),
                }
                maskr = batch_r["mask"].to(device)

                # teacher 的高保真输出（冻结）
                with torch.no_grad():
                    _, outH_old = teacher(Hr, Mr, er, cr)   # dict

                # student 在 replay 样本上的输出
                _, outH_new_r = student(Hr, Mr, er, cr)    # dict

                # KD loss（物理域）
                loss_kd = kd_loss_fn(outH_new_r, outH_old, maskr, w=args.beta_kd)

                # 在 replay 样本上的 supervised loss（防灾忘）
                loss_rep = crit(outH_new_r, yNr, maskr) * args.gamma_replay

            loss = loss_new + loss_kd + loss_rep

            opt.zero_grad()
            loss.backward()
            opt.step()

        if epoch % 10 == 0:
            print(
                f"[inc] epoch {epoch}/{args.epochs} "
                f"loss_new={loss_new.item():.4f} "
                f"kd={float(loss_kd):.4f} "
                f"rep={float(loss_rep):.4f}"
            )

    # ------- 保存 student 权重（带 LoRA 增量更新） -------
    save_path = args.save_ckpt or (str(Path(args.base_ckpt).with_suffix("")) + "_inc.pth")
    torch.save({"model_state_dict": student.state_dict()}, save_path)
    print(f"[OK] Incremental (LoRA+Replay+KD) saved to: {save_path}")


if __name__ == "__main__":
    main()




"""
GPU 环境下运行示例：
torchrun --standalone --nproc_per_node=1 -m surrogate_model.cli.train_incremental \
  --base_ckpt "surrogate_model/outputs_pzk/checkpoint_stage2_fold0_earlystop_epoch1146.pth" \
  --real_new_txt "surrogate_model/data/new_real.txt" \
  --replay_txt   "surrogate_model/data/real_dataset_oe.txt" \
  --epochs 2000 --beta_kd 0.2 --gamma_replay 0.2 --lr 1e-3
"""
"""
CPU 环境下运行示例：
python -m surrogate_model.cli.train_incremental \
    --base_ckpt "surrogate_model/outputs_pzk/checkpoint_stage2_fold0_earlystop_epoch1146.pth" \
    --real_new_txt "surrogate_model/data/new_real.txt" \
    --replay_txt   "surrogate_model/data/real_dataset_oe.txt" \
    --epochs 2000 --beta_kd 0.2 --gamma_replay 0.2 --lr 1e-3
"""