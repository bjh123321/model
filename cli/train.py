from __future__ import annotations

import argparse
import configparser
import multiprocessing as mp
import os
import socket
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from surrogate_model.config import Config
from surrogate_model.data.io import parse_rows_to_struct, read_variable_txt
from surrogate_model.train.loops import train_one_fold


def get_exe_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


def _ddp_is_requested() -> bool:
    return ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)


def _ddp_backend() -> str:
    if torch.cuda.is_available() and sys.platform != "win32":
        return "nccl"
    return "gloo"


def _ddp_setup() -> tuple[bool, int, int]:
    if not _ddp_is_requested():
        return False, 0, 1
    dist.init_process_group(backend=_ddp_backend(), init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    return True, rank, world_size


def _path_from_ini(paths_ini_path: Path, section: str) -> Path | None:
    cp = configparser.ConfigParser()
    if not paths_ini_path.exists():
        return None
    cp.read(paths_ini_path, encoding="utf-8")
    key = f"path_{section}".lower()
    if not cp.has_section(section) or key not in cp[section]:
        return None
    return Path(cp[section][key])


def _default_model_dir(base_dir: Path, paths_ini_path: Path) -> Path:
    model_dir = _path_from_ini(paths_ini_path, "MX_XGD")
    if model_dir is not None:
        return model_dir
    fallback = base_dir.parents[1] / "MX_XGD"
    print(
        f"[WARN] {paths_ini_path} does not define [MX_XGD]/path_mx_xgd; "
        f"using legacy default: {fallback}",
        flush=True,
    )
    return fallback


def _resolve_real_dataset_path(base_dir: Path, raw_value: str) -> Path:
    candidate = Path(raw_value).expanduser()
    if candidate.exists():
        return candidate
    local_real = base_dir / "real_dataset.txt"
    if local_real.exists():
        print(
            f"[WARN] user_input.dat line 3 is not an existing dataset path: {raw_value!r}; "
            f"using local dataset: {local_real}",
            flush=True,
        )
        return local_real
    raise FileNotFoundError(
        "Real dataset file was not found. For base training, user_input.dat line 3 "
        f"must be a real_dataset.txt path, or {local_real} must exist. "
        "If this user_input.dat is for model correction, run "
        "`python -m surrogate_model.cli.train_incremental` instead."
    )


def _enable_fast_cuda(args) -> torch.device:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Device was set to cuda, but torch.cuda.is_available() is False.")
    use_cuda = (args.device != "cpu") and torch.cuda.is_available()
    if not use_cuda:
        return torch.device("cpu")

    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    return device


def _send_finished_udp():
    host, port, message = "localhost", 12345, "XGD_GENERATION_EXE_FINISHED"
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
        raise ValueError("user_input.dat needs at least 3 lines: paths.ini, model_version, real_dataset_path")

    paths_ini_path = Path(lines[0].strip()).expanduser()
    model_version = lines[1].strip()
    real_dataset_path = _resolve_real_dataset_path(base_dir, lines[2].strip())
    model_dir = _default_model_dir(base_dir, paths_ini_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=str(model_dir / model_version))
    parser.add_argument("--kfold", type=int, default=0)
    parser.add_argument("--embed_excitation_dim", type=int, default=4)
    parser.add_argument("--embed_channel_dim", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_real", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--residue_target", type=str, default="polar", choices=["polar"])
    parser.add_argument("--val_split_strategy", type=str, default=None, choices=["height_max_mach", "height_mach_kfold", ""])
    parser.add_argument("--debug_overfit_n", type=int, default=0)
    parser.add_argument("--debug_overfit_offset", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args, _unknown = parser.parse_known_args()

    _ddp_setup()
    _enable_fast_cuda(args)

    real_rows = read_variable_txt(real_dataset_path)
    real_parsed, real_stats = parse_rows_to_struct(real_rows)

    cfg = Config()
    cfg.embed_excitation_dim = args.embed_excitation_dim
    cfg.embed_channel_dim = args.embed_channel_dim
    cfg.epochs_stage1 = args.epochs
    cfg.batch_real = args.batch_real
    cfg.num_workers = args.num_workers
    cfg.prefetch = args.prefetch
    cfg.residue_target = "polar"
    cfg.debug_direct_output = False
    if args.debug_overfit_n > 0:
        cfg.use_lora = False
        cfg.use_modulation = False
        cfg.weight_decay = 0.0
        cfg.debug_direct_output = True
        cfg.lr_low_adam = 3e-3
        cfg.num_workers = 0
    if args.val_split_strategy is not None:
        cfg.val_split_strategy = args.val_split_strategy
    cfg.K_max = 3
    cfg.embed_channel_num = real_stats["channel_num"]
    cfg.embed_excitation_num = real_stats["excitation_num"]

    num_folds = args.kfold if (cfg.val_split_strategy == "height_mach_kfold" and args.kfold > 1) else 1

    for fold in range(num_folds):
        if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
            print(f"========== Fold {fold + 1}/{num_folds} ==========")
        train_one_fold(fold, cfg, real_parsed, args)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    _send_finished_udp()


if __name__ == "__main__":
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
