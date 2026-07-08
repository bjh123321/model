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
from surrogate_model.data.io import OMEGA_N, parse_rows_to_struct
from surrogate_model.data.normalizer import Normalizer
from surrogate_model.models.blocks import (
    MFDNN,
    count_parameters,
    freeze_base_train_lora_only,
    get_lora_parameters,
    load_base_weights_into_lora_model,
)


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
        self.target_names = out_norm.target_names()

    def __len__(self):
        return self.H.size(0)

    def __getitem__(self, idx):
        yNk = self.yN[idx]
        return {
            "H": self.H[idx],
            "M": self.M[idx],
            "exc": self.exc[idx],
            "chn": self.chn[idx],
            "yN": {name: yNk[i] for i, name in enumerate(self.target_names)},
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


def _model_output_as_targets(pred, target_names):
    alpha = pred["alpha"]
    omega_norm = pred["omega"]
    omega_phys = omega_norm * float(OMEGA_N)
    rho_log = pred["rho"]
    sin_theta = pred["sin_theta"]
    cos_theta = pred["cos_theta"]
    denom = torch.sqrt(alpha.pow(2) + omega_phys.pow(2) + 1e-12)
    converted = {
        "alpha": alpha,
        "omega": omega_norm,
        "freq": omega_phys / (2.0 * torch.pi),
        "damping": -alpha / denom,
        "rho": rho_log,
        "sin_theta": sin_theta,
        "cos_theta": cos_theta,
        "residue_re": torch.exp(torch.clamp(rho_log, min=-80.0, max=80.0)) * cos_theta,
        "residue_im": torch.exp(torch.clamp(rho_log, min=-80.0, max=80.0)) * sin_theta,
    }
    return {name: converted[name] for name in target_names}


def representation_loss(pred, tgt, mask, mean, std, target_names):
    width = min(mask.size(1), pred["alpha"].size(1), min(tgt[name].size(1) for name in target_names))
    w = mask[:, :width].to(dtype=pred["alpha"].dtype)
    den = w.sum().clamp_min(1.0)
    pred_raw = _model_output_as_targets(pred, target_names)
    loss = 0.0
    for i, name in enumerate(target_names):
        pred_cmp = (pred_raw[name][:, :width] - mean[i]) / std[i].clamp_min(1e-8)
        loss = loss + (w * (pred_cmp - tgt[name][:, :width]).pow(2)).sum() / den
    return loss


def kd_loss_fn(new: dict, old: dict, target_names, w: float = 1.0):
    new_targets = _model_output_as_targets(new, target_names)
    old_targets = _model_output_as_targets(old, target_names)
    loss = 0.0
    for key in target_names:
        loss = loss + ((new_targets[key] - old_targets[key]) ** 2).mean()
    return loss * w


def _restore_cfg_from_checkpoint(ckpt: dict, use_lora: bool) -> Config:
    cfg = Config()
    for attr in (
        "K_max",
        "embed_excitation_num",
        "embed_channel_num",
        "embed_excitation_dim",
        "embed_channel_dim",
        "low_hidden",
        "mod_hidden",
        "dropout_prob",
        "use_modulation",
        "mod_gamma",
        "pole_target",
        "residue_target",
    ):
        if attr in ckpt:
            setattr(cfg, attr, ckpt[attr])
    cfg.use_lora = bool(use_lora)
    return cfg


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
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument("--lora_rank", type=int, default=None)
    ap.add_argument("--lora_alpha", type=float, default=None)
    ap.add_argument("--lora_dropout", type=float, default=None)
    ap.add_argument("--beta_kd", type=float, default=None)
    ap.add_argument("--gamma_replay", type=float, default=None)
    ap.add_argument("--warmup_epochs", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_cfg = Config()
    epochs = int(args.epochs if args.epochs is not None else default_cfg.incremental_epochs)
    batch_size = int(args.batch_size if args.batch_size is not None else default_cfg.incremental_batch_size)
    lr = float(args.lr if args.lr is not None else default_cfg.incremental_lr)
    weight_decay = float(args.weight_decay if args.weight_decay is not None else default_cfg.incremental_weight_decay)
    beta_kd = float(args.beta_kd if args.beta_kd is not None else default_cfg.incremental_beta_kd)
    gamma_replay = float(args.gamma_replay if args.gamma_replay is not None else default_cfg.incremental_gamma_replay)
    warmup_epochs = int(args.warmup_epochs if args.warmup_epochs is not None else default_cfg.incremental_warmup_epochs)

    in_norm = Normalizer.load(model_dir / "Normalizer_Input.pth")
    out_norm = Normalizer.load(model_dir / "Normalizer_Output.pth")
    if len(in_norm.mean) != 2 or len(in_norm.std) != 2:
        raise RuntimeError(f"Normalizer_Input.pth must be 2D [H, M], got mean={in_norm.mean}, std={in_norm.std}")

    ckpt = torch.load(base_ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    teacher_cfg = _restore_cfg_from_checkpoint(ckpt if isinstance(ckpt, dict) else {}, use_lora=False)
    teacher_cfg.use_lora = False
    student_cfg = _restore_cfg_from_checkpoint(ckpt if isinstance(ckpt, dict) else {}, use_lora=True)
    student_cfg.lora_rank = int(args.lora_rank if args.lora_rank is not None else default_cfg.lora_rank)
    student_cfg.lora_alpha = float(args.lora_alpha if args.lora_alpha is not None else default_cfg.lora_alpha)
    student_cfg.lora_dropout = float(args.lora_dropout if args.lora_dropout is not None else default_cfg.lora_dropout)

    student = MFDNN(student_cfg).to(device)
    teacher = MFDNN(teacher_cfg).to(device)
    load_base_weights_into_lora_model(student, state)
    teacher.load_state_dict(state, strict=True)
    freeze_base_train_lora_only(student)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    lora_params = get_lora_parameters(student)
    if not lora_params:
        raise RuntimeError("No trainable LoRA parameters found. Check cfg.use_lora/lora_rank.")
    trainable, total = count_parameters(student)
    print(f"[LoRA] trainable={trainable}, total={total}, ratio={trainable / max(total, 1):.6f}", flush=True)

    ds_new = RealIncrementalDataset(new_data_out, in_norm, out_norm, K_max=student_cfg.K_max)
    ds_replay = None
    if replay_txt_path.exists() and replay_txt_path.stat().st_size > 0:
        ds_replay = RealIncrementalDataset(replay_txt_path, in_norm, out_norm, K_max=student_cfg.K_max)

    dl_new = DataLoader(ds_new, batch_size=batch_size, shuffle=True, drop_last=False)
    dl_replay = DataLoader(ds_replay, batch_size=batch_size, shuffle=True, drop_last=False) if ds_replay is not None else None

    target_names = out_norm.target_names()
    mean_t = torch.as_tensor(out_norm.mean, dtype=torch.float32, device=device)
    std_t = torch.as_tensor(out_norm.std, dtype=torch.float32, device=device).clamp_min(1e-8)
    opt = torch.optim.AdamW(lora_params, lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        student.train()
        it_replay = iter(dl_replay) if dl_replay is not None else None
        warm = min(float(epoch) / float(max(warmup_epochs, 1)), 1.0)
        sum_new = sum_kd = sum_rep = sum_total = 0.0
        n_steps = 0

        for batch in dl_new:
            H, M, e, c, yN, mask = _batch_to_device(batch, device)
            out_new = student(H, M, e, c)
            loss_new = representation_loss(out_new, yN, mask, mean_t, std_t, target_names)
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
                loss_kd = kd_loss_fn(out_new_r, out_old, target_names, w=beta_kd)
                loss_rep = representation_loss(out_new_r, yNr, maskr, mean_t, std_t, target_names) * gamma_replay

            loss = loss_new + warm * loss_kd + warm * loss_rep
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=5.0)
            opt.step()

            sum_new += float(loss_new.detach().item())
            sum_kd += float(loss_kd.detach().item())
            sum_rep += float(loss_rep.detach().item())
            sum_total += float(loss.detach().item())
            n_steps += 1

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            denom = max(n_steps, 1)
            print(
                f"[fixed-sf LoRA inc] epoch {epoch}/{epochs} "
                f"loss_total={sum_total / denom:.4f} "
                f"loss_new={sum_new / denom:.4f} "
                f"kd={sum_kd / denom:.4f} rep={sum_rep / denom:.4f}"
            )

    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "stage": "fixed_single_fidelity_lora_incremental",
            "peft_type": "lora",
            "lora_rank": student_cfg.lora_rank,
            "lora_alpha": student_cfg.lora_alpha,
            "lora_dropout": student_cfg.lora_dropout,
            "base_model_version": model_version,
            "K_max": student_cfg.K_max,
            "pole_target": getattr(out_norm, "pole_target", getattr(student_cfg, "pole_target", "alpha_omega")),
            "residue_target": getattr(out_norm, "residue_target", getattr(student_cfg, "residue_target", "polar")),
            "target_names": out_norm.target_names(),
            "low_hidden": student_cfg.low_hidden,
            "mod_hidden": student_cfg.mod_hidden,
            "dropout_prob": student_cfg.dropout_prob,
            "use_lora": True,
            "use_modulation": student_cfg.use_modulation,
            "mod_gamma": student_cfg.mod_gamma,
            "embed_excitation_num": student_cfg.embed_excitation_num,
            "embed_excitation_dim": student_cfg.embed_excitation_dim,
            "embed_channel_num": student_cfg.embed_channel_num,
            "embed_channel_dim": student_cfg.embed_channel_dim,
        },
        save_ckpt_path,
    )
    print(f"[OK] Incremental checkpoint saved to: {save_ckpt_path}")
    _send_finished_udp()


if __name__ == "__main__":
    main()
