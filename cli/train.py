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


def _path_from_ini(paths_ini_path: Path, section: str) -> Path:
    cp = configparser.ConfigParser()
    cp.read(paths_ini_path, encoding="utf-8")
    key = f"path_{section}".lower()
    try:
        return Path(cp[section][key])
    except KeyError:
        raise KeyError(f"{paths_ini_path} is missing [{section}] / {key}") from None


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
    real_dataset_path = Path(lines[2].strip()).expanduser()
    model_dir = _path_from_ini(paths_ini_path, "MX_XGD")

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=str(model_dir / model_version))
    parser.add_argument("--kfold", type=int, default=0)
    parser.add_argument("--stage2_kfold", type=int, default=0, choices=[0, 1])
    parser.add_argument("--embed_excitation_dim", type=int, default=4)
    parser.add_argument("--embed_channel_dim", type=int, default=12)
    parser.add_argument("--epochs_s1", type=int, default=2000)
    parser.add_argument("--epochs_s2", type=int, default=1000)
    parser.add_argument("--batch_sim", type=int, default=256)
    parser.add_argument("--batch_real", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--stage2_unfreeze_low_tail_layers", type=int, default=2)
    parser.add_argument("--lr_low_stage2", type=float, default=5e-5)
    parser.add_argument("--freeze_low_stage2", type=int, default=1, choices=[0, 1])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args, _unknown = parser.parse_known_args()

    _ddp_setup()
    _enable_fast_cuda(args)

    sim_dataset_path = base_dir / "sim_dataset.txt"
    sim_rows = read_variable_txt(sim_dataset_path)
    real_rows = read_variable_txt(real_dataset_path)
    sim_parsed, sim_stats = parse_rows_to_struct(sim_rows)
    real_parsed, real_stats = parse_rows_to_struct(real_rows)

    cfg = Config()
    cfg.embed_excitation_dim = args.embed_excitation_dim
    cfg.embed_channel_dim = args.embed_channel_dim
    cfg.epochs_stage1 = args.epochs_s1
    cfg.epochs_stage2 = args.epochs_s2
    cfg.batch_sim = args.batch_sim
    cfg.batch_real = args.batch_real
    cfg.num_workers = args.num_workers
    cfg.prefetch = args.prefetch
    cfg.stage2_unfreeze_low_tail_layers = args.stage2_unfreeze_low_tail_layers
    cfg.lr_low_stage2 = args.lr_low_stage2
    cfg.freeze_low_stage2 = bool(args.freeze_low_stage2)
    cfg.K_max = max(sim_stats["K_max"], real_stats["K_max"])
    cfg.embed_channel_num = max(sim_stats["channel_num"], real_stats["channel_num"])
    cfg.embed_excitation_num = max(sim_stats["excitation_num"], real_stats["excitation_num"])

    num_folds = args.kfold if bool(args.stage2_kfold) else 1
    setattr(args, "stage1_once", 1)

    for fold in range(num_folds):
        if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
            print(f"========== Fold {fold + 1}/{num_folds} ==========", flush=True)
        train_one_fold(fold, cfg, sim_parsed, real_parsed, args)
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
