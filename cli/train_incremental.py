from __future__ import annotations

import argparse
import configparser
import socket
import sys
import warnings
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from surrogate_model.config import Config
from surrogate_model.data.OE_BATCH_PROCESS import batch_process
from surrogate_model.data.build_replay_data import build_replay_data
from surrogate_model.data.datasets import _fixed_len_tensor, build_label_tensor
from surrogate_model.data.io import parse_rows_to_struct
from surrogate_model.data.normalizer import Normalizer
from surrogate_model.losses.masked_mse import MaskedHungarianMSE
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
    def __init__(self, txt_path: Path, in_norm: Normalizer, out_norm: Normalizer):
        super().__init__()
        self.txt_path = Path(txt_path)

        rows_float = []
        with open(self.txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows_float.append([float(x) for x in line.replace(",", " ").split()])

        if not rows_float:
            raise RuntimeError(f"[RealIncrementalDataset] {self.txt_path} is empty or cannot be parsed")

        parsed, _ = parse_rows_to_struct(rows_float)
        if not parsed:
            raise RuntimeError(f"[RealIncrementalDataset] parse_rows_to_struct returned no rows: {self.txt_path}")

        H = torch.tensor([p.H for p in parsed], dtype=torch.float32)
        M = torch.tensor([p.M for p in parsed], dtype=torch.float32)
        exc = torch.tensor([p.exc_id for p in parsed], dtype=torch.long)
        chn = torch.tensor([p.ch_id for p in parsed], dtype=torch.long)

        H = ((H - float(in_norm.mean[0])) / max(float(in_norm.std[0]), 1e-12)).unsqueeze(-1)
        M = ((M - float(in_norm.mean[1])) / max(float(in_norm.std[1]), 1e-12)).unsqueeze(-1)

        K_max = (
            int(out_norm.residue_mean.shape[2])
            if getattr(out_norm, "residue_mean", None) is not None
            else max(len(p.mask) for p in parsed)
        )
        label_list, res_mean_list, res_std_list, mask_list = [], [], [], []
        for p in parsed:
            y_norm, res_mean, res_std = build_label_tensor(p, K_max, out_norm)
            label_list.append(y_norm.float())
            res_mean_list.append(res_mean.float())
            res_std_list.append(res_std.float())
            mask_list.append(_fixed_len_tensor(p.mask, K_max))

        mask = torch.stack(mask_list, dim=0)
        yN = torch.stack(label_list, dim=0)
        res_mean = torch.stack(res_mean_list, dim=0)
        res_std = torch.stack(res_std_list, dim=0)

        self.H = H
        self.M = M
        self.exc = exc
        self.chn = chn
        self.yN = yN
        self.res_mean = res_mean
        self.res_std = res_std
        self.mask = mask

    def __len__(self):
        return self.H.size(0)

    def __getitem__(self, idx):
        yNk = self.yN[idx]
        y_dict = {
            "alpha": yNk[0],
            "omega": yNk[1],
            "residue_mean": self.res_mean[idx],
            "residue_std": self.res_std[idx],
        }
        if yNk.size(0) == 5:
            y_dict.update({
                "rho": yNk[2],
                "sin_theta": yNk[3],
                "cos_theta": yNk[4],
            })
        elif yNk.size(0) == 4:
            y_dict.update({
                "residue_re": yNk[2],
                "residue_im": yNk[3],
            })
        else:
            raise RuntimeError(f"Unexpected label shape: {tuple(yNk.shape)}")
        return {
            "H": self.H[idx],
            "M": self.M[idx],
            "exc": self.exc[idx],
            "chn": self.chn[idx],
            "yN": y_dict,
            "mask": self.mask[idx],
        }


@torch.no_grad()
def freeze_all_and_open_adaptation(model: MFDNN):
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.adaptation.parameters():
        p.requires_grad_(True)


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def list_trainable_param_names(model: torch.nn.Module):
    return [name for name, p in model.named_parameters() if p.requires_grad]


@torch.no_grad()
def snapshot_adaptation_params(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in module.named_parameters()}


def adaptation_anchor_loss(module: nn.Module, ref_state: Dict[str, torch.Tensor]) -> torch.Tensor:
    loss = None
    for name, param in module.named_parameters():
        ref = ref_state[name].to(device=param.device, dtype=param.dtype)
        term = (param - ref).pow(2).mean()
        loss = term if loss is None else (loss + term)
    if loss is None:
        return torch.zeros((), device=next(module.parameters()).device)
    return loss


def build_adaptation_param_groups(model: MFDNN, lr_base: float, lr_lora: float, lr_embed: float):
    base_params = []
    lora_params = []
    embed_params = []
    unmatched = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if not name.startswith("adaptation."):
            raise RuntimeError(f"Non-adaptation parameter unexpectedly trainable: {name}")
        if ".mlp." in name and ".base." in name:
            base_params.append(param)
        elif ".mlp." in name and ".lora_" in name:
            lora_params.append(param)
        elif name.startswith("adaptation.emb_") or name.startswith("adaptation.mod."):
            embed_params.append(param)
        else:
            unmatched.append(name)

    if unmatched:
        raise RuntimeError(f"Unmatched trainable adaptation parameters: {unmatched}")
    if not base_params:
        raise RuntimeError("Optimizer base group is empty")
    if not lora_params:
        raise RuntimeError("Optimizer lora group is empty")
    if not embed_params:
        warnings.warn("Optimizer embed group is empty", RuntimeWarning)

    groups = [
        {"params": base_params, "lr": lr_base, "name": "base"},
        {"params": lora_params, "lr": lr_lora, "name": "lora"},
    ]
    if embed_params:
        groups.append({"params": embed_params, "lr": lr_embed, "name": "embed"})
    return groups


def kd_loss_fn(new: dict, old: dict, w: float = 1.0):
    keys = ["alpha", "omega", "rho", "sin_theta", "cos_theta"]
    loss = 0.0
    for key in keys:
        loss = loss + ((new[key] - old[key]) ** 2).mean()
    return loss * w


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
    ap.add_argument("--lr_base", type=float, default=3e-5)
    ap.add_argument("--lr_lora", type=float, default=1e-4)
    ap.add_argument("--lr_embed", type=float, default=2e-4)
    ap.add_argument("--beta_kd", type=float, default=0.2)
    ap.add_argument("--gamma_replay", type=float, default=0.2)
    ap.add_argument("--lambda_anchor", type=float, default=1e-3)
    ap.add_argument("--warmup_epochs", type=int, default=10)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    cfg.use_lora = True

    in_norm = Normalizer.load(model_dir / "Normalizer_Input.pth")
    out_norm = Normalizer.load(model_dir / "Normalizer_Output.pth")
    cfg.residue_target = getattr(out_norm, "residue_target", getattr(cfg, "residue_target", "polar"))

    student = MFDNN(cfg).to(device)
    teacher = MFDNN(cfg).to(device)

    ckpt = torch.load(base_ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    student.load_state_dict(state, strict=False)
    teacher.load_state_dict(state, strict=False)

    freeze_all_and_open_adaptation(student)
    adapt_ref_state = snapshot_adaptation_params(student.adaptation)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    trainable_names = list_trainable_param_names(student)
    bad_trainables = [name for name in trainable_names if not name.startswith("adaptation.")]
    if bad_trainables:
        raise RuntimeError(f"Trainable parameters outside adaptation: {bad_trainables}")
    print(f"[INFO] trainable params: {count_trainable_params(student)}")
    print("[INFO] Fine-tuning adaptation parameters:")
    for name in trainable_names:
        print(f"  - {name}")

    if len(in_norm.mean) != 2 or len(in_norm.std) != 2:
        raise RuntimeError(f"Normalizer_Input.pth must be 2D [H, M], got mean={in_norm.mean}, std={in_norm.std}")

    ds_new = RealIncrementalDataset(new_data_out, in_norm, out_norm)
    ds_replay = None
    if replay_txt_path.exists() and replay_txt_path.stat().st_size > 0:
        ds_replay = RealIncrementalDataset(replay_txt_path, in_norm, out_norm)

    dl_new = DataLoader(ds_new, batch_size=args.batch_size, shuffle=True, drop_last=False)
    dl_replay = DataLoader(ds_replay, batch_size=args.batch_size, shuffle=True, drop_last=False) if ds_replay is not None else None

    mean_t = torch.as_tensor(out_norm.mean, dtype=torch.float32, device=device)
    std_t = torch.as_tensor(out_norm.std, dtype=torch.float32, device=device)
    crit = MaskedHungarianMSE(
        mean=mean_t,
        std=std_t,
        residue_target=getattr(out_norm, "residue_target", getattr(cfg, "residue_target", "polar")),
    )

    opt = torch.optim.AdamW(
        build_adaptation_param_groups(
            student,
            lr_base=args.lr_base,
            lr_lora=args.lr_lora,
            lr_embed=args.lr_embed,
        ),
        weight_decay=1e-4,
    )

    for epoch in range(1, args.epochs + 1):
        student.train()
        it_replay = iter(dl_replay) if dl_replay is not None else None
        warm = min(float(epoch) / float(max(args.warmup_epochs, 1)), 1.0)
        sum_new = sum_kd = sum_rep = sum_anchor = sum_total = 0.0
        n_steps = 0

        for batch in dl_new:
            H, M, e, c, yN, mask = _batch_to_device(batch, device)
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

                Hr, Mr, er, cr, yNr, maskr = _batch_to_device(batch_r, device)
                with torch.no_grad():
                    _, outH_old = teacher(Hr, Mr, er, cr)
                _, outH_new_r = student(Hr, Mr, er, cr)
                loss_kd = kd_loss_fn(outH_new_r, outH_old, w=args.beta_kd)
                loss_rep = crit(outH_new_r, yNr, maskr) * args.gamma_replay

            loss_anchor = adaptation_anchor_loss(student.adaptation, adapt_ref_state) * args.lambda_anchor
            loss = loss_new + warm * loss_kd + warm * loss_rep + loss_anchor

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            sum_new += float(loss_new.detach().item())
            sum_kd += float(loss_kd.detach().item())
            sum_rep += float(loss_rep.detach().item())
            sum_anchor += float(loss_anchor.detach().item())
            sum_total += float(loss.detach().item())
            n_steps += 1

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            denom = max(n_steps, 1)
            print(
                f"[inc] epoch {epoch}/{args.epochs} "
                f"loss_total={sum_total / denom:.4f} "
                f"loss_new={sum_new / denom:.4f} "
                f"kd={sum_kd / denom:.4f} "
                f"rep={sum_rep / denom:.4f} "
                f"anchor={sum_anchor / denom:.4f}"
            )

    torch.save({"model_state_dict": student.state_dict()}, save_ckpt_path)
    print(f"[OK] Incremental checkpoint saved to: {save_ckpt_path}")
    _send_finished_udp()


if __name__ == "__main__":
    main()
