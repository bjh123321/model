from __future__ import annotations

import argparse
import configparser
import socket
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from surrogate_model.config import Config
from surrogate_model.data.OE_BATCH_PROCESS import batch_process
from surrogate_model.data.build_replay_data import build_replay_data
from surrogate_model.data.datasets import _fixed_len_tensor, build_label_tensor
from surrogate_model.data.io import parse_rows_to_struct
from surrogate_model.data.normalizer import Normalizer
from surrogate_model.losses.masked_mse import MaskedMSE
from surrogate_model.models.blocks import MFDNN


def get_exe_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


def _path_from_ini(cp: configparser.ConfigParser, paths_ini_path: Path, section: str) -> Path:
    key = f"path_{section}".lower()
    try:
        return Path(cp[section][key])
    except KeyError:
        raise KeyError(f"{paths_ini_path} is missing [{section}] / {key}") from None


class RealIncrementalDataset(Dataset):
    def __init__(self, txt_path: Path, in_norm: Normalizer, out_norm: Normalizer, K_max: int = 3):
        super().__init__()
        self.txt_path = Path(txt_path)
        rows_float = []
        with open(self.txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows_float.append([float(x) for x in line.replace(",", " ").split()])
        if not rows_float:
            raise RuntimeError(f"[RealIncrementalDataset] {self.txt_path} is empty or cannot be parsed")

        parsed, _ = parse_rows_to_struct(rows_float)
        H = torch.tensor([p.H for p in parsed], dtype=torch.float32)
        M = torch.tensor([p.M for p in parsed], dtype=torch.float32)
        exc = torch.tensor([p.exc_id for p in parsed], dtype=torch.long)
        chn = torch.tensor([p.ch_id for p in parsed], dtype=torch.long)
        H = ((H - float(in_norm.mean[0])) / max(float(in_norm.std[0]), 1e-12)).unsqueeze(-1)
        M = ((M - float(in_norm.mean[1])) / max(float(in_norm.std[1]), 1e-12)).unsqueeze(-1)

        label_list, mask_list = [], []
        for p in parsed:
            label_list.append(build_label_tensor(p, K_max, out_norm).float())
            mask_list.append(_fixed_len_tensor(p.mask, K_max))

        self.H = H
        self.M = M
        self.exc = exc
        self.chn = chn
        self.yN = torch.stack(label_list, dim=0)
        self.mask = torch.stack(mask_list, dim=0)

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
                "freq": yNk[0],
                "damping": yNk[1],
                "residue_re": yNk[2],
                "residue_im": yNk[3],
            },
            "mask": self.mask[idx],
        }


def _batch_to_device(batch, device):
    yN = {key: value.to(device) for key, value in batch["yN"].items()}
    return (
        batch["H"].to(device),
        batch["M"].to(device),
        batch["exc"].to(device),
        batch["chn"].to(device),
        yN,
        batch["mask"].to(device),
    )


def kd_loss_fn(new: dict, old: dict, w: float = 1.0):
    loss = 0.0
    for key in ["freq", "damping", "residue_re", "residue_im"]:
        loss = loss + ((new[key] - old[key]) ** 2).mean()
    return loss * w


def _send_finished_udp():
    host, port, message = "localhost", 12345, "XGD_CORRECTION_EXE_FINISHED"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(message.encode("utf-8"), (host, port))
    except Exception as exc:
        print(f"[WARN] UDP notification failed: {exc}")
    finally:
        sock.close()


def main():
    base_dir = get_exe_dir()
    user_input_path = base_dir / "user_input.dat"
    with open(user_input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) < 3:
        raise ValueError("user_input.dat needs at least 3 lines: paths.ini, base_model_version, new_model_version")

    paths_ini_path = Path(lines[0].strip()).expanduser()
    cp = configparser.ConfigParser()
    cp.read(paths_ini_path, encoding="utf-8")
    data_dir = _path_from_ini(cp, paths_ini_path, "CG")
    model_dir = _path_from_ini(cp, paths_ini_path, "MX_XGD")

    model_version = lines[1].strip()
    new_model_version = lines[2].strip()
    base_ckpt_path = model_dir / model_version / (model_version + ".pth")
    replay_txt_path = base_dir / "replay_data.txt"
    new_data_out = base_dir / "new_data.txt"
    selected_list = base_dir / "selected_CG_path.txt"
    save_ckpt_path = model_dir / new_model_version / (new_model_version + ".pth")
    save_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    build_replay_data(base_dir, model_dir)
    batch_process(data_dir, new_data_out, selected_list)

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--beta_kd", type=float, default=0.2)
    ap.add_argument("--gamma_replay", type=float, default=0.2)
    ap.add_argument("--warmup_epochs", type=int, default=10)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    cfg.K_max = 3
    cfg.residue_target = "fixed"

    in_norm = Normalizer.load(model_dir / "Normalizer_Input.pth")
    out_norm = Normalizer.load(model_dir / "Normalizer_Output.pth")
    if len(in_norm.mean) != 2 or len(in_norm.std) != 2:
        raise RuntimeError(f"Normalizer_Input.pth must be 2D [H, M], got mean={in_norm.mean}, std={in_norm.std}")

    student = MFDNN(cfg).to(device)
    teacher = MFDNN(cfg).to(device)
    ckpt = torch.load(base_ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    student.load_state_dict(state, strict=False)
    teacher.load_state_dict(state, strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    ds_new = RealIncrementalDataset(new_data_out, in_norm, out_norm, K_max=cfg.K_max)
    ds_replay = None
    if replay_txt_path.exists() and replay_txt_path.stat().st_size > 0:
        ds_replay = RealIncrementalDataset(replay_txt_path, in_norm, out_norm, K_max=cfg.K_max)

    dl_new = DataLoader(ds_new, batch_size=args.batch_size, shuffle=True, drop_last=False)
    dl_replay = DataLoader(ds_replay, batch_size=args.batch_size, shuffle=True, drop_last=False) if ds_replay is not None else None

    crit = MaskedMSE(
        mean=torch.as_tensor(out_norm.mean, dtype=torch.float32, device=device),
        std=torch.as_tensor(out_norm.std, dtype=torch.float32, device=device),
        residue_target="fixed",
    )
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        student.train()
        it_replay = iter(dl_replay) if dl_replay is not None else None
        warm = min(float(epoch) / float(max(args.warmup_epochs, 1)), 1.0)
        sum_new = sum_kd = sum_rep = sum_total = 0.0
        n_steps = 0

        for batch in dl_new:
            H, M, e, c, yN, mask = _batch_to_device(batch, device)
            out_new = student(H, M, e, c)
            loss_new = crit(out_new, yN, mask)
            loss_kd = torch.tensor(0.0, device=device)
            loss_rep = torch.tensor(0.0, device=device)

            if it_replay is not None:
                try:
                    batch_r = next(it_replay)
                except StopIteration:
                    it_replay = iter(dl_replay)
                    batch_r = next(it_replay)
                Hr, Mr, er, cr, yNr, maskr = _batch_to_device(batch_r, device)
                with torch.no_grad():
                    out_old = teacher(Hr, Mr, er, cr)
                out_new_r = student(Hr, Mr, er, cr)
                loss_kd = kd_loss_fn(out_new_r, out_old, w=args.beta_kd)
                loss_rep = crit(out_new_r, yNr, maskr) * args.gamma_replay

            loss = loss_new + warm * loss_kd + warm * loss_rep
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            opt.step()

            sum_new += float(loss_new.detach().item())
            sum_kd += float(loss_kd.detach().item())
            sum_rep += float(loss_rep.detach().item())
            sum_total += float(loss.detach().item())
            n_steps += 1

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            denom = max(n_steps, 1)
            print(
                f"[fixed-sf inc] epoch {epoch}/{args.epochs} "
                f"loss_total={sum_total / denom:.4f} "
                f"loss_new={sum_new / denom:.4f} "
                f"kd={sum_kd / denom:.4f} rep={sum_rep / denom:.4f}"
            )

    torch.save({"model_state_dict": student.state_dict(), "stage": "fixed_single_fidelity_incremental"}, save_ckpt_path)
    print(f"[OK] Incremental checkpoint saved to: {save_ckpt_path}")
    _send_finished_udp()


if __name__ == "__main__":
    main()
