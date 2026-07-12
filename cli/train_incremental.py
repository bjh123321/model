from __future__ import annotations

import argparse
import configparser
import socket
import sys
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from surrogate_model.config import Config
from surrogate_model.data.OE_BATCH_PROCESS import batch_process_split
from surrogate_model.data.build_replay_data import build_replay_data
from surrogate_model.data.datasets import _fixed_len_tensor, build_label_tensor
from surrogate_model.data.io import OMEGA_N, parse_rows_to_struct, read_real_dataset
from surrogate_model.data.normalizer import Normalizer
from surrogate_model.models.blocks import (
    MFDNN,
    count_parameters,
    freeze_base_train_lora_only,
    get_lora_parameters,
    load_base_weights_into_lora_model,
)
from surrogate_model.models.checkpoint_utils import (
    checkpoint_uses_lora,
    restore_model_config_from_checkpoint,
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
    def __init__(
        self,
        txt_path: Path | list[Path] | None,
        in_norm: Normalizer,
        out_norm: Normalizer,
        K_max: int = 3,
        rows_float: list[list[float]] | None = None,
    ):
        super().__init__()
        self.txt_paths = [] if txt_path is None else (
            [Path(p) for p in txt_path] if isinstance(txt_path, (list, tuple)) else [Path(txt_path)]
        )
        if rows_float is None:
            rows_float = []
            for path in self.txt_paths:
                rows_float.extend(read_real_dataset(path))
        else:
            rows_float = [list(row) for row in rows_float]
        if not rows_float:
            raise RuntimeError(f"[RealIncrementalDataset] inputs are empty or cannot be parsed: {self.txt_paths}")

        parsed, _ = parse_rows_to_struct(rows_float)
        self.group_keys = sorted({(float(p.H), float(p.M), int(p.exc_id)) for p in parsed})
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


def evaluate_representation_loss(model, data_loader, device, mean, std, target_names) -> float:
    """Evaluate one deterministic, valid-mode-weighted loss over the complete dataset."""
    model.eval()
    weighted_sum = 0.0
    valid_count = 0.0
    with torch.no_grad():
        for batch in data_loader:
            H, M, e, c, yN, mask = _batch_to_device(batch, device)
            pred = model(H, M, e, c)
            loss = representation_loss(pred, yN, mask, mean, std, target_names)
            width = min(mask.size(1), pred["alpha"].size(1), min(yN[name].size(1) for name in target_names))
            batch_valid = float(mask[:, :width].sum().item())
            weighted_sum += float(loss.item()) * batch_valid
            valid_count += batch_valid
    if valid_count <= 0:
        raise RuntimeError("Full new-data evaluation found no valid modal slots")
    return weighted_sum / valid_count


def kd_loss_fn(student_output, teacher_output, mask, std, target_names, kd_weight: float):
    """Masked KD in the same normalized target representation as supervision."""
    new_targets = _model_output_as_targets(student_output, target_names)
    old_targets = _model_output_as_targets(teacher_output, target_names)
    width = min(mask.size(1), min(new_targets[n].size(1) for n in target_names), min(old_targets[n].size(1) for n in target_names))
    valid = mask[:, :width].to(dtype=next(iter(new_targets.values())).dtype)
    denominator = valid.sum().clamp_min(1.0)
    loss = 0.0
    std = torch.as_tensor(std, device=valid.device, dtype=valid.dtype).clamp_min(1e-8)
    for index, key in enumerate(target_names):
        delta = (new_targets[key][:, :width] - old_targets[key][:, :width]) / std[index]
        loss = loss + (valid * delta.pow(2)).sum() / denominator
    return loss * float(kd_weight)


def guard_metrics(loss_new, loss_guard, baseline_guard_loss, tolerance, lambda_guard):
    denominator = max(float(baseline_guard_loss), 1e-12)
    ratio = float(loss_guard) / denominator
    safe = float(loss_guard) <= float(baseline_guard_loss) * (1.0 + float(tolerance))
    score = float(loss_new) + float(lambda_guard) * max(0.0, ratio - 1.0)
    return ratio, safe, score


def choose_safe_checkpoint(current_new, current_guard, baseline_guard, tolerance, best_new, min_delta=0.0):
    """Pure selection helper used by training and regression tests."""
    _, safe, _ = guard_metrics(current_new, current_guard, baseline_guard, tolerance, 0.0)
    improved = best_new is None or float(current_new) < float(best_new) - float(min_delta)
    return safe and improved


def restore_safe_or_initial(model, best_state, initial_state):
    found = best_state is not None
    model.load_state_dict(best_state if found else initial_state, strict=True)
    return found


def _restore_cfg_from_checkpoint(ckpt: dict, use_lora: bool) -> Config:
    return restore_model_config_from_checkpoint(ckpt, use_lora=use_lora)


def _checkpoint_uses_lora(ckpt: dict, state: dict) -> bool:
    return checkpoint_uses_lora(ckpt, state)


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
    selected_list = base_dir / "selected_CG_path.txt"
    save_ckpt_path = model_dir / new_model_version / (new_model_version + ".pth")
    save_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

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
    ap.add_argument("--old_loss_tolerance", type=float, default=None)
    ap.add_argument("--lambda_guard", type=float, default=None)
    ap.add_argument("--early_stop_patience", type=int, default=None)
    ap.add_argument("--early_stop_min_delta", type=float, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_cfg = Config()

    def resolve_user_path(raw: str) -> Path:
        path = Path(raw.strip()).expanduser()
        return path if path.is_absolute() else (base_dir / path).resolve()

    direct_data_paths: list[Path] = []
    fourth_raw = lines[3].strip() if len(lines) >= 4 else ""
    fourth_path = resolve_user_path(fourth_raw) if fourth_raw else None
    if fourth_path is not None and fourth_path.is_file():
        correction_mode = "direct_dataset"
        for line_no, raw in enumerate(lines[3:], start=4):
            value = raw.strip()
            if not value:
                continue
            path = resolve_user_path(value)
            if not path.is_file():
                raise FileNotFoundError(f"user_input.dat line {line_no} is not an existing dataset file: {path}")
            direct_data_paths.append(path)

        parent_dirs = {path.resolve().parent for path in direct_data_paths}
        if len(parent_dirs) != 1:
            raise ValueError(f"Direct correction dataset files must share one parent directory, got: {sorted(map(str, parent_dirs))}")
        replay_dataset_path = next(iter(parent_dirs))
        replay_result = build_replay_data(
            base_dir,
            model_dir,
            replay_dataset_path=replay_dataset_path,
            selected_names=[path.name for path in direct_data_paths],
            structured=True,
            guard_max_groups=default_cfg.incremental_guard_max_groups,
            guard_min_groups_per_excitation=default_cfg.incremental_guard_min_groups_per_excitation,
            selection_seed=default_cfg.incremental_selection_seed,
            primary_h_thr=default_cfg.incremental_replay_primary_h_thr,
            primary_m_thr=default_cfg.incremental_replay_primary_m_thr,
            fallback_h_thr=default_cfg.incremental_replay_fallback_h_thr,
            fallback_m_thr=default_cfg.incremental_replay_fallback_m_thr,
            max_groups_per_selected=default_cfg.incremental_replay_max_groups_per_selected,
            min_groups_per_selected=default_cfg.incremental_replay_min_groups_per_selected,
        )
        new_data_paths = direct_data_paths
        print(
            f"[CORRECTION] mode=direct_dataset files={len(new_data_paths)} "
            f"replay_pool={replay_dataset_path}",
            flush=True,
        )
    else:
        correction_mode = "cg_oe"
        if fourth_path is not None:
            print(
                f"[WARN] user_input.dat line 4 is not an existing file: {fourth_path}; "
                "falling back to selected_CG_path + OE mode.",
                flush=True,
            )
        replay_dataset_path = Path(default_cfg.incremental_replay_dataset_dir).expanduser()
        if not replay_dataset_path.is_absolute():
            replay_dataset_path = (base_dir / replay_dataset_path).resolve()
        if not replay_dataset_path.is_dir():
            raise FileNotFoundError(f"Configured incremental replay dataset directory does not exist: {replay_dataset_path}")

        # Snapshot replay before generating this run's new split files.
        replay_result = build_replay_data(
            base_dir, model_dir, replay_dataset_path=replay_dataset_path, structured=True,
            guard_max_groups=default_cfg.incremental_guard_max_groups,
            guard_min_groups_per_excitation=default_cfg.incremental_guard_min_groups_per_excitation,
            selection_seed=default_cfg.incremental_selection_seed,
            primary_h_thr=default_cfg.incremental_replay_primary_h_thr,
            primary_m_thr=default_cfg.incremental_replay_primary_m_thr,
            fallback_h_thr=default_cfg.incremental_replay_fallback_h_thr,
            fallback_m_thr=default_cfg.incremental_replay_fallback_m_thr,
            max_groups_per_selected=default_cfg.incremental_replay_max_groups_per_selected,
            min_groups_per_selected=default_cfg.incremental_replay_min_groups_per_selected,
        )
        new_data_paths = batch_process_split(
            data_dir,
            base_dir,
            selected_list,
            reserved_names_dir=replay_dataset_path,
        )
        print(
            f"[CORRECTION] mode=cg_oe files={len(new_data_paths)} "
            f"output_dir={base_dir} replay_pool={replay_dataset_path}",
            flush=True,
        )

    epochs = int(args.epochs if args.epochs is not None else default_cfg.incremental_epochs)
    batch_size = int(args.batch_size if args.batch_size is not None else default_cfg.incremental_batch_size)
    lr = float(args.lr if args.lr is not None else default_cfg.incremental_lr)
    weight_decay = float(args.weight_decay if args.weight_decay is not None else default_cfg.incremental_weight_decay)
    beta_kd = float(args.beta_kd if args.beta_kd is not None else default_cfg.incremental_beta_kd)
    gamma_replay = float(args.gamma_replay if args.gamma_replay is not None else default_cfg.incremental_gamma_replay)
    warmup_epochs = int(args.warmup_epochs if args.warmup_epochs is not None else default_cfg.incremental_warmup_epochs)
    old_loss_tolerance = float(args.old_loss_tolerance if args.old_loss_tolerance is not None else default_cfg.incremental_old_loss_tolerance)
    lambda_guard = float(args.lambda_guard if args.lambda_guard is not None else default_cfg.incremental_lambda_guard)
    early_stop_patience = int(args.early_stop_patience if args.early_stop_patience is not None else default_cfg.incremental_early_stop_patience)
    early_stop_min_delta = float(args.early_stop_min_delta if args.early_stop_min_delta is not None else default_cfg.incremental_early_stop_min_delta)
    replay_rows = replay_result["local_replay_rows"]
    guard_rows = replay_result["global_guard_rows"]
    local_group_keys = replay_result["local_group_keys"]
    guard_group_keys = replay_result["guard_group_keys"]

    in_norm = Normalizer.load(model_dir / "Normalizer_Input.pth")
    out_norm = Normalizer.load(model_dir / "Normalizer_Output.pth")
    if len(in_norm.mean) != 2 or len(in_norm.std) != 2:
        raise RuntimeError(f"Normalizer_Input.pth must be 2D [H, M], got mean={in_norm.mean}, std={in_norm.std}")

    ckpt = torch.load(base_ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    ckpt_meta = ckpt if isinstance(ckpt, dict) else {}
    base_uses_lora = _checkpoint_uses_lora(ckpt_meta, state)
    teacher_cfg = _restore_cfg_from_checkpoint(ckpt_meta, use_lora=base_uses_lora)
    student_cfg = _restore_cfg_from_checkpoint(ckpt_meta, use_lora=True)

    checkpoint_rank = int(student_cfg.lora_rank)
    requested_rank = int(args.lora_rank if args.lora_rank is not None else checkpoint_rank)
    if base_uses_lora and requested_rank != checkpoint_rank:
        raise ValueError(
            f"LoRA base checkpoint rank={checkpoint_rank} cannot be loaded into requested rank={requested_rank}. "
            "Omit --lora_rank or use the checkpoint rank."
        )
    student_cfg.lora_rank = requested_rank
    student_cfg.lora_alpha = float(args.lora_alpha if args.lora_alpha is not None else student_cfg.lora_alpha)
    student_cfg.lora_dropout = float(args.lora_dropout if args.lora_dropout is not None else student_cfg.lora_dropout)

    student = MFDNN(student_cfg).to(device)
    teacher = MFDNN(teacher_cfg).to(device)
    if base_uses_lora:
        student.load_state_dict(state, strict=True)
        teacher.load_state_dict(state, strict=True)
        print(
            f"[LoRA load] continuing from LoRA base checkpoint: rank={checkpoint_rank}",
            flush=True,
        )
    else:
        load_base_weights_into_lora_model(student, state)
        teacher.load_state_dict(state, strict=True)
        print("[LoRA load] initialized LoRA student from plain base checkpoint", flush=True)
    freeze_base_train_lora_only(student)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    lora_params = get_lora_parameters(student)
    if not lora_params:
        raise RuntimeError("No trainable LoRA parameters found. Check cfg.use_lora/lora_rank.")
    trainable, total = count_parameters(student)
    trainable_names = [name for name, param in student.named_parameters() if param.requires_grad]
    invalid_trainable = [name for name in trainable_names if ".A." not in name and ".B." not in name]
    if invalid_trainable:
        raise RuntimeError(f"Non-LoRA parameters are trainable: {invalid_trainable}")
    print(f"[LoRA] trainable parameter names: {trainable_names}", flush=True)
    print(f"[LoRA] trainable={trainable}, total={total}, ratio={trainable / max(total, 1):.6f}", flush=True)

    ds_new = RealIncrementalDataset(new_data_paths, in_norm, out_norm, K_max=student_cfg.K_max)
    ds_replay = RealIncrementalDataset(
        None,
        in_norm,
        out_norm,
        K_max=student_cfg.K_max,
        rows_float=replay_rows,
    )
    ds_guard = RealIncrementalDataset(None, in_norm, out_norm, K_max=student_cfg.K_max, rows_float=guard_rows)

    dl_new = DataLoader(ds_new, batch_size=batch_size, shuffle=True, drop_last=False)
    dl_new_eval = DataLoader(ds_new, batch_size=batch_size, shuffle=False, drop_last=False)
    dl_replay = DataLoader(ds_replay, batch_size=batch_size, shuffle=True, drop_last=False) if ds_replay is not None else None
    dl_guard = DataLoader(ds_guard, batch_size=batch_size, shuffle=False, drop_last=False)

    target_names = out_norm.target_names()
    mean_t = torch.as_tensor(out_norm.mean, dtype=torch.float32, device=device)
    std_t = torch.as_tensor(out_norm.std, dtype=torch.float32, device=device).clamp_min(1e-8)
    opt = torch.optim.AdamW(lora_params, lr=lr, weight_decay=weight_decay)

    initial_state = deepcopy(student.state_dict())
    baseline_guard_loss = evaluate_representation_loss(teacher, dl_guard, device, mean_t, std_t, target_names)
    initial_guard_loss = evaluate_representation_loss(student, dl_guard, device, mean_t, std_t, target_names)
    if abs(initial_guard_loss - baseline_guard_loss) > float(default_cfg.incremental_initial_guard_tolerance):
        raise RuntimeError(f"LoRA initialization changed guard loss: teacher={baseline_guard_loss:.9g}, student={initial_guard_loss:.9g}")
    initial_new_loss = evaluate_representation_loss(student, dl_new_eval, device, mean_t, std_t, target_names)
    best_new_full = None
    best_guard_loss = None
    best_epoch = 0
    best_state = None
    no_improve_epochs = 0
    print(f"[DATA] new rows/groups={len(ds_new)}/{len(ds_new.group_keys)} local replay rows/groups={len(ds_replay)}/{len(local_group_keys)} global guard rows/groups={len(ds_guard)}/{len(guard_group_keys)}", flush=True)
    print(f"[CONFIG] target_names={target_names} beta_kd={beta_kd} gamma_replay={gamma_replay} old_loss_tolerance={old_loss_tolerance} lambda_guard={lambda_guard} learning_rate={lr}", flush=True)
    print(f"[FULL-EVAL] epoch=0 loss_new_full={initial_new_loss:.6f} baseline_guard_loss={baseline_guard_loss:.6f} initial_guard_loss={initial_guard_loss:.6f}", flush=True)

    for epoch in range(1, epochs + 1):
        student.train()
        it_replay = iter(dl_replay) if dl_replay is not None else None
        warm = 1.0 if warmup_epochs <= 0 else min(float(epoch) / float(warmup_epochs), 1.0)
        sum_new = sum_kd = sum_rep = sum_total = 0.0
        count_new = count_kd = count_rep = 0.0
        n_steps = 0

        for batch in dl_new:
            H, M, e, c, yN, mask = _batch_to_device(batch, device)
            out_new = student(H, M, e, c)
            loss_new = representation_loss(out_new, yN, mask, mean_t, std_t, target_names)
            new_width = min(mask.size(1), out_new["alpha"].size(1), min(yN[name].size(1) for name in target_names))
            new_valid = float(mask[:, :new_width].sum().item())
            loss_kd = torch.tensor(0.0, device=device)
            loss_rep = torch.tensor(0.0, device=device)
            replay_valid = 0.0
            replay_samples = 0.0

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
                loss_kd = kd_loss_fn(out_new_r, out_old, maskr, std_t, target_names, beta_kd)
                loss_rep = representation_loss(out_new_r, yNr, maskr, mean_t, std_t, target_names) * gamma_replay
                replay_width = min(maskr.size(1), out_new_r["alpha"].size(1), min(yNr[name].size(1) for name in target_names))
                replay_valid = float(maskr[:, :replay_width].sum().item())
                replay_samples = float(Hr.size(0))

            loss = loss_new + warm * loss_kd + warm * loss_rep
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=5.0)
            opt.step()

            sum_new += float(loss_new.detach().item()) * new_valid
            count_new += new_valid
            sum_kd += float(loss_kd.detach().item()) * replay_samples
            count_kd += replay_samples
            sum_rep += float(loss_rep.detach().item()) * replay_valid
            count_rep += replay_valid
            sum_total += float(loss.detach().item())
            n_steps += 1

        loss_new_full = evaluate_representation_loss(student, dl_new_eval, device, mean_t, std_t, target_names)
        loss_guard_full = evaluate_representation_loss(student, dl_guard, device, mean_t, std_t, target_names)
        guard_ratio, guard_safe, combined_score = guard_metrics(loss_new_full, loss_guard_full, baseline_guard_loss, old_loss_tolerance, lambda_guard)
        if choose_safe_checkpoint(loss_new_full, loss_guard_full, baseline_guard_loss, old_loss_tolerance, best_new_full, early_stop_min_delta):
            best_new_full = float(loss_new_full)
            best_guard_loss = float(loss_guard_full)
            best_epoch = int(epoch)
            best_state = deepcopy(student.state_dict())
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        print(f"[fixed-sf LoRA inc] epoch={epoch}/{epochs} lr={opt.param_groups[0]['lr']:.6g} warm={warm:.4f} loss_new_batch={sum_new/max(count_new,1.0):.6f} loss_new_full={loss_new_full:.6f} loss_local_replay={sum_rep/max(count_rep,1.0):.6f} loss_kd={sum_kd/max(count_kd,1.0):.6f} loss_guard_full={loss_guard_full:.6f} baseline_guard_loss={baseline_guard_loss:.6f} guard_ratio={guard_ratio:.6f} guard_safe={guard_safe} combined_score={combined_score:.6f} best_safe_epoch={best_epoch}", flush=True)
        if early_stop_patience > 0 and no_improve_epochs >= early_stop_patience:
            print(f"[EARLY-STOP] no safe improvement for {no_improve_epochs} epochs", flush=True)
            break

    guard_safe_checkpoint_found = best_state is not None
    restore_safe_or_initial(student, best_state, initial_state)
    if not guard_safe_checkpoint_found:
        best_epoch, best_new_full, best_guard_loss = 0, initial_new_loss, initial_guard_loss
        print("No guard-safe checkpoint found; restored epoch 0.", flush=True)
    best_guard_ratio = float(best_guard_loss) / max(float(baseline_guard_loss), 1e-12)
    print(f"[FINAL] restored_epoch={best_epoch} best_new_loss={best_new_full:.6f} best_guard_loss={best_guard_loss:.6f} best_guard_ratio={best_guard_ratio:.6f} whether_guard_safe_checkpoint_found={guard_safe_checkpoint_found}", flush=True)

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
            "modulation_mode": student_cfg.modulation_mode,
            "mod_dropout": student_cfg.mod_dropout,
            "mod_exc_pole_scale": student_cfg.mod_exc_pole_scale,
            "mod_exc_residue_scale": student_cfg.mod_exc_residue_scale,
            "mod_channel_residue_scale": student_cfg.mod_channel_residue_scale,
            "mod_channel_affects_poles": student_cfg.mod_channel_affects_poles,
            "dropout_prob": student_cfg.dropout_prob,
            "use_lora": True,
            "use_modulation": student_cfg.use_modulation,
            "mod_gamma": student_cfg.mod_gamma,
            "embed_excitation_num": student_cfg.embed_excitation_num,
            "embed_excitation_dim": student_cfg.embed_excitation_dim,
            "embed_channel_num": student_cfg.embed_channel_num,
            "embed_channel_dim": student_cfg.embed_channel_dim,
            "incremental_data_files": [path.name for path in new_data_paths],
            "replay_dataset_path": str(replay_dataset_path),
            "correction_mode": correction_mode,
            "best_epoch": best_epoch,
            "best_new_full": best_new_full,
            "baseline_guard_loss": baseline_guard_loss,
            "best_guard_loss": best_guard_loss,
            "best_guard_ratio": best_guard_ratio,
            "guard_safe_checkpoint_found": guard_safe_checkpoint_found,
            "old_loss_tolerance": old_loss_tolerance,
            "lambda_guard": lambda_guard,
            "beta_kd": beta_kd,
            "gamma_replay": gamma_replay,
            "incremental_lr": lr,
            "local_replay_group_keys": [list(k) for k in local_group_keys],
            "global_guard_group_keys": [list(k) for k in guard_group_keys],
        },
        save_ckpt_path,
    )
    print(f"[OK] Incremental checkpoint saved to: {save_ckpt_path}")
    _send_finished_udp()


if __name__ == "__main__":
    main()
