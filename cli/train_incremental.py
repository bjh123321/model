# surrogate_model/cli/train_incremental.py
import argparse
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader, Dataset
import configparser
from typing import Dict

from surrogate_model.config import Config
from surrogate_model.models.blocks import MFDNN, LoRALinear
from surrogate_model.losses.masked_mse import MaskedHungarianMSE
from surrogate_model.data.normalizer import Normalizer
from surrogate_model.data.io import parse_rows_to_struct
from surrogate_model.data.OE_BATCH_PROCESS import batch_process
from surrogate_model.data.build_replay_data import build_replay_data


def load_paths_ini(paths_ini: str) -> Dict[str, str]:
    cp = configparser.ConfigParser()
    cp.read(paths_ini, encoding="utf-8")

    out: Dict[str, str] = {}
    for sec in cp.sections():
        kv = [(k, v) for k, v in cp[sec].items() if k.lower().startswith("path_")]
        if not kv:
            continue
        _, v = kv[0]
        out[sec.strip()] = str(v).strip()
    return out


def get_exe_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    else:
        return Path(__file__).resolve().parent


# ===========================
# 只对 alpha, omega, rho 做 z-score，sin/cos 不变
# ===========================
def normlizer(pred: torch.Tensor, mean, std):
    mean_t = torch.as_tensor(mean, dtype=pred.dtype, device=pred.device)
    std_t = torch.as_tensor(std, dtype=pred.dtype, device=pred.device)

    core = pred[..., :3]
    others = pred[..., 3:]

    assert mean_t.numel() == 3 and std_t.numel() == 3, "out_norm.mean/std 应该是长度 3"
    view_shape = [1] * (core.ndim - 1) + [3]
    mean_t = mean_t.view(*view_shape)
    std_t = std_t.view(*view_shape)

    core_n = (core - mean_t) / std_t
    return torch.cat([core_n, others], dim=-1)


# ===========================
# 增量学习用数据集：直接从 txt 读 + parse_rows_to_struct
# ===========================
class RealIncrementalDataset(Dataset):
    def __init__(self, txt_path: str, in_norm: Normalizer, out_norm: Normalizer):
        super().__init__()
        self.txt_path = txt_path

        rows_float = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.replace(",", " ").split()
                vals = [float(x) for x in parts]
                rows_float.append(vals)

        if not rows_float:
            raise RuntimeError(f"[RealIncrementalDataset] 文件 {txt_path} 为空或无法解析")

        parsed, _ = parse_rows_to_struct(rows_float)
        N = len(parsed)
        if N == 0:
            raise RuntimeError(f"[RealIncrementalDataset] parse_rows_to_struct 结果为空：{txt_path}")

        H = torch.tensor([p.H for p in parsed], dtype=torch.float32)
        M = torch.tensor([p.M for p in parsed], dtype=torch.float32)
        exc = torch.tensor([p.exc_id for p in parsed], dtype=torch.long)
        chn = torch.tensor([p.ch_id for p in parsed], dtype=torch.long)

        H = ((H - in_norm.mean[0]) / in_norm.std[0]).unsqueeze(-1)
        M = ((M - in_norm.mean[1]) / in_norm.std[1]).unsqueeze(-1)

        alpha_list, omega_list, rho_list = [], [], []
        sin_list, cos_list, mask_list = [], [], []

        for p in parsed:
            alpha_list.append(torch.from_numpy(p.alpha.copy()).float())
            omega_list.append(torch.from_numpy(p.omega.copy()).float())
            rho_list.append(torch.from_numpy(p.rho.copy()).float())
            sin_list.append(torch.from_numpy(p.sin_theta.copy()).float())
            cos_list.append(torch.from_numpy(p.cos_theta.copy()).float())
            mask_list.append(torch.from_numpy(p.mask.copy()).float())

        alpha = torch.stack(alpha_list, dim=0)
        omega = torch.stack(omega_list, dim=0)
        rho = torch.stack(rho_list, dim=0)
        sin_t = torch.stack(sin_list, dim=0)
        cos_t = torch.stack(cos_list, dim=0)
        mask = torch.stack(mask_list, dim=0)

        y_phys = torch.stack([alpha, omega, rho, sin_t, cos_t], dim=-1)
        yN = normlizer(y_phys, out_norm.mean, out_norm.std)

        self.H = H
        self.M = M
        self.exc = exc
        self.chn = chn
        self.yN = yN
        self.mask = mask

    def __len__(self):
        return self.H.size(0)

    def __getitem__(self, idx):
        yNk = self.yN[idx]
        return {
            "H": self.H[idx],
            "M": self.M[idx],
            "exc": self.exc[idx],
            "chn": self.chn[idx],
            "yN": {
                "alpha": yNk[:, 0],
                "omega": yNk[:, 1],
                "rho": yNk[:, 2],
                "sin_theta": yNk[:, 3],
                "cos_theta": yNk[:, 4],
            },
            "mask": self.mask[idx],
        }


# ===========================
# 冻结全部，只开放 NHF(AdaptationHead) 内的 LoRA 参数
# ===========================
@torch.no_grad()
def freeze_all_and_open_adaptation_lora(model: MFDNN):
    for p in model.parameters():
        p.requires_grad_(False)

    for m in model.adaptation.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad_(True)
            m.lora_B.requires_grad_(True)
            for p in m.base.parameters():
                p.requires_grad_(False)


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def list_trainable_param_names(model: torch.nn.Module):
    return [name for name, p in model.named_parameters() if p.requires_grad]


# ===========================
# KD loss（teacher-student 一致性约束）
# ===========================
def kd_loss_fn(new: dict, old: dict, w=1.0):
    keys = ["alpha", "omega", "rho", "sin_theta", "cos_theta"]
    loss = 0.0
    for k in keys:
        loss = loss + ((new[k] - old[k]) ** 2).mean()
    return loss * w


def main():
    base_dir = get_exe_dir()
    user_input_path = base_dir / "user_input.dat"
    with open(user_input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    paths_ini = Path(lines[0].strip()).expanduser()
    paths_map = load_paths_ini(paths_ini)

    model_version = lines[1].strip()
    new_model_version = lines[2].strip()

    model_dir = base_dir.parents[1] / "MX_XGD"

    base_ckpt_path = model_dir / model_version / (model_version + ".pth")
    new_txt_path = base_dir / "new_data.txt"
    replay_txt_path = base_dir / "replay_data.txt"
    save_ckpt_path = model_dir / new_model_version / (new_model_version + ".pth")
    save_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    data_dir = Path(paths_map["CG"])
    new_data_out = base_dir / "new_data.txt"
    selected_list = base_dir / "selected_CG_path.txt"

    build_replay_data(base_dir)
    batch_process(data_dir, new_data_out, selected_list)

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta_kd", type=float, default=0.2)
    ap.add_argument("--gamma_replay", type=float, default=0.2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    cfg.use_lora = True

    student = MFDNN(cfg).to(device)
    teacher = MFDNN(cfg).to(device)

    ckpt = torch.load(base_ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    student.load_state_dict(state, strict=False)
    teacher.load_state_dict(state, strict=False)

    freeze_all_and_open_adaptation_lora(student)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    trainable_names = list_trainable_param_names(student)
    print(f"[INFO] trainable params: {count_trainable_params(student)}")
    print("[INFO] 仅微调 NHF(AdaptationHead) 的 LoRA 参数:")
    for name in trainable_names:
        print(f"  - {name}")

    in_norm_path = model_dir / "Normalizer_Input.pth"
    out_norm_path = model_dir / "Normalizer_Output.pth"
    in_norm = Normalizer.load(in_norm_path)
    out_norm = Normalizer.load(out_norm_path)

    ds_new = RealIncrementalDataset(new_txt_path, in_norm, out_norm)

    ds_replay = None
    if replay_txt_path.exists() and replay_txt_path.stat().st_size > 0:
        ds_replay = RealIncrementalDataset(replay_txt_path, in_norm, out_norm)

    dl_new = DataLoader(ds_new, batch_size=args.batch_size, shuffle=True, drop_last=False)
    dl_replay = DataLoader(ds_replay, batch_size=args.batch_size, shuffle=True, drop_last=False) if ds_replay is not None else None

    mean_t = torch.as_tensor(out_norm.mean, dtype=torch.float32, device=device)
    std_t = torch.as_tensor(out_norm.std, dtype=torch.float32, device=device)
    crit = MaskedHungarianMSE(mean=mean_t, std=std_t)

    opt = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )

    for epoch in range(1, args.epochs + 1):
        student.train()
        it_replay = iter(dl_replay) if dl_replay is not None else None

        sum_new = 0.0
        sum_kd = 0.0
        sum_rep = 0.0
        sum_total = 0.0
        n_steps = 0

        for batch in dl_new:
            H = batch["H"].to(device)
            M = batch["M"].to(device)
            e = batch["exc"].to(device)
            c = batch["chn"].to(device)

            yN = {
                "alpha": batch["yN"]["alpha"].to(device),
                "omega": batch["yN"]["omega"].to(device),
                "rho": batch["yN"]["rho"].to(device),
                "sin_theta": batch["yN"]["sin_theta"].to(device),
                "cos_theta": batch["yN"]["cos_theta"].to(device),
            }
            mask = batch["mask"].to(device)

            _, outH_new = student(H, M, e, c)
            loss_new = crit(outH_new, yN, mask)

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
                    "alpha": batch_r["yN"]["alpha"].to(device),
                    "omega": batch_r["yN"]["omega"].to(device),
                    "rho": batch_r["yN"]["rho"].to(device),
                    "sin_theta": batch_r["yN"]["sin_theta"].to(device),
                    "cos_theta": batch_r["yN"]["cos_theta"].to(device),
                }
                maskr = batch_r["mask"].to(device)

                with torch.no_grad():
                    _, outH_old = teacher(Hr, Mr, er, cr)
                _, outH_new_r = student(Hr, Mr, er, cr)
                loss_kd = kd_loss_fn(outH_new_r, outH_old, w=args.beta_kd)
                loss_rep = crit(outH_new_r, yNr, maskr) * args.gamma_replay

            loss = loss_new + loss_kd + loss_rep

            opt.zero_grad()
            loss.backward()
            opt.step()

            sum_new += float(loss_new.detach().item())
            sum_kd += float(loss_kd.detach().item())
            sum_rep += float(loss_rep.detach().item())
            sum_total += float(loss.detach().item())
            n_steps += 1

        avg_new = sum_new / max(n_steps, 1)
        avg_kd = sum_kd / max(n_steps, 1)
        avg_rep = sum_rep / max(n_steps, 1)
        avg_total = sum_total / max(n_steps, 1)

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"[inc] epoch {epoch}/{args.epochs} "
                f"loss_total={avg_total:.4f} "
                f"loss_new={avg_new:.4f} "
                f"kd={avg_kd:.4f} "
                f"rep={avg_rep:.4f}"
            )

    save_path = save_ckpt_path or (str(Path(base_ckpt_path).with_suffix("")) + "_inc.pth")
    torch.save({"model_state_dict": student.state_dict()}, save_path)
    print(f"[OK] Incremental (NHF-LoRA+Replay+KD) saved to: {save_path}")

    import socket

    def udp_sender():
        HOST = 'localhost'
        PORT = 12345
        MESSAGE = "XGD_CORRECTION_EXE_FINISHED"

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        try:
            print("UDP发送端已启动")
            print(f"目标地址: {HOST}:{PORT}")
            print(f"发送内容: {MESSAGE}")
            print("\n正在发送...")
            sock.sendto(MESSAGE.encode('utf-8'), (HOST, PORT))
            print("消息发送成功！")
        except Exception as e:
            print(f"发送失败: {e}")
        finally:
            sock.close()
            print("发送端已关闭")

    udp_sender()


if __name__ == "__main__":
    main()
