# surrogate_model/cli/train_incremental.py
import argparse, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

from surrogate_model.config import Config
from surrogate_model.blocks import MFDNN, LoRALinear
from surrogate_model.dataloaders import build_datasets_for_paths  # 你可用自己现有的构建函数
from surrogate_model.masked_mse import MaskedMSE  # 你的损失
from surrogate_model.normalizer import Normalizer  # 如需

@torch.no_grad()
def freeze_backbone_open_lora(model: MFDNN):
    for p in model.parameters(): p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad_(True)
            m.lora_B.requires_grad_(True)
            for p in m.base.parameters(): p.requires_grad_(False)

def kd_loss_fn(new_out: dict, old_out: dict, mask, w=1.0):
    # 在 z-score 空间做 MSE（按你的训练损失口味可改）
    # 这里直接用物理域也可以，但建议与你训练时保持一致的尺度
    # 简化：对 alpha/omega/rho/sin/cos 做 MSE；存在头做 BCE（可选）
    loss = 0.0
    keys = ['alpha','omega','rho','sin_theta','cos_theta']
    for k in keys:
        loss = loss + nn.functional.mse_loss(new_out[k], old_out[k], reduction='none') * mask.unsqueeze(-1)
    loss = loss.mean()
    return w*loss

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
    cfg.use_lora = True  # 增量只训 LoRA
    student = MFDNN(cfg).to(device)
    teacher = MFDNN(cfg).to(device).eval()

    # 载入同一份 base ckpt
    ckpt = torch.load(args.base_ckpt, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    student.load_state_dict(state, strict=False)
    teacher.load_state_dict(state, strict=False)

    # 冻结主干，只开 LoRA
    freeze_backbone_open_lora(student)

    # 数据集（你可以用自己的 dataloaders；这里示意“通过路径构建”）
    ds_new, ds_replay = build_datasets_for_paths(args.real_new_txt, args.replay_txt)
    dl_new     = DataLoader(ds_new, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dl_replay  = DataLoader(ds_replay, batch_size=args.batch_size, shuffle=True, drop_last=True) if ds_replay is not None else None

    # 损失
    crit = MaskedMSE(mean=torch.zeros(5), std=torch.ones(5))  # 你项目里应当从保存的 Normalizer 载入；这里简化示意
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs+1):
        student.train()
        it_replay = iter(dl_replay) if dl_replay is not None else None

        for batch in dl_new:
            # new real
            H,M,e,c, yN, mask = batch['H'].to(device), batch['M'].to(device), batch['exc'].to(device), batch['chn'].to(device), batch['yN'].to(device), batch['mask'].to(device)
            out_new = student(H,M,e,c)['high']
            loss_new = crit(out_new, yN, mask)

            # KD on replay（可选）
            loss_kd, loss_rep = 0.0, 0.0
            if it_replay is not None:
                try:
                    batch_r = next(it_replay)
                except StopIteration:
                    it_replay = iter(dl_replay); batch_r = next(it_replay)
                Hr,Mr,er,cr, yNr, maskr = batch_r['H'].to(device), batch_r['M'].to(device), batch_r['exc'].to(device), batch_r['chn'].to(device), batch_r['yN'].to(device), batch_r['mask'].to(device)
                with torch.no_grad():
                    out_old = teacher(Hr,Mr,er,cr)['high']
                out_new_r = student(Hr,Mr,er,cr)['high']
                loss_kd = kd_loss_fn(out_new_r, out_old, maskr, w=args.beta_kd)
                loss_rep = crit(out_new_r, yNr, maskr) * args.gamma_replay

            loss = loss_new + loss_kd + loss_rep

            opt.zero_grad()
            loss.backward()
            opt.step()

        if epoch % 10 == 0:
            print(f"[inc] epoch {epoch}/{args.epochs} loss_new={loss_new.item():.4f} kd={float(loss_kd):.4f} rep={float(loss_rep):.4f}")

    save_path = args.save_ckpt or (str(Path(args.base_ckpt).with_suffix("")) + "_inc.pth")
    torch.save({"model_state_dict": student.state_dict()}, save_path)
    print(f"[OK] Incremental (LoRA+Replay+KD) saved to: {save_path}")

if __name__ == "__main__":
    main()


"""
torchrun --standalone --nproc_per_node=1 -m surrogate_model.cli.train_incremental \
  --base_ckpt "outputs_pzk/checkpoint_stage2_best.pth" \
  --real_new_txt "surrogate_model/data/real_new_2025_11.txt" \
  --replay_txt   "surrogate_model/data/real_replay_small.txt" \
  --epochs 200 --beta_kd 0.2 --gamma_replay 0.2 --lr 1e-3
"""