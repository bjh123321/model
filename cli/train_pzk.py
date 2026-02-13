from __future__ import annotations
from __future__ import annotations
import multiprocessing as mp
import os
import sys
import argparse
from pathlib import Path

# ------------------------------------------------------------
# Make "python train_pzk_user.py" runnable from any working dir
# This file is: .../surrogate_model/cli/train_pzk_user.py
# Project root should be: .../ (two levels up from surrogate_model)
# ------------------------------------------------------------

import torch
import torch.distributed as dist

from surrogate_model.config import Config
from surrogate_model.data.io import read_variable_txt, parse_rows_to_struct
from surrogate_model.train.loops import train_one_fold


def get_exe_dir():
    """
    返回 exe 所在目录：
    - 打包后（exe）：exe 文件所在目录
    - 未打包（python 运行）：当前脚本所在目录
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    else:
        return Path(__file__).resolve().parent


def _ddp_is_requested() -> bool:
    # torchrun 会注入这些环境变量；普通 python 运行不会有
    return ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)


def _ddp_backend() -> str:
    # Linux + CUDA 用 nccl；Windows 或 CPU 用 gloo
    if torch.cuda.is_available() and sys.platform != "win32":
        return "nccl"
    return "gloo"


def _ddp_setup() -> tuple[bool, int, int]:
    """
    Returns: (is_ddp, rank, world_size)
    Only initializes process group when launched by torchrun.
    """
    if not _ddp_is_requested():
        return False, 0, 1

    dist.init_process_group(backend=_ddp_backend(), init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

    return True, rank, world_size


def main():
    base_dir = get_exe_dir()
    user_input_path = base_dir / "user_input.dat"
    with open(user_input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    model_version = lines[1].strip()

    base_dir_parent = base_dir.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default=base_dir_parent / 'MX_XGD' / model_version)
    parser.add_argument('--kfold', type=int, default=0)
    # parser.add_argument('--stage1_kfold', type=int, default=0, choices=[0, 1],
    #                     help='Set to 1 to train stage 1 on every fold, 0 to train once and reuse')
    parser.add_argument('--stage2_kfold', type=int, default=0, choices=[0, 1],
                        help='Set to 1 to train stage 2 on every fold, 0 to train once')
    parser.add_argument('--embed_excitation_dim', type=int, default=4)
    parser.add_argument('--embed_channel_dim', type=int, default=12)
    parser.add_argument('--epochs_s1', type=int, default=2000)
    parser.add_argument('--epochs_s2', type=int, default=2000)
    parser.add_argument('--batch_sim', type=int, default=256)
    parser.add_argument('--batch_real', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--prefetch', type=int, default=4)
    args, _unknown = parser.parse_known_args()

    # DDP / device init
    is_ddp, rank, world_size = _ddp_setup()

    if torch.cuda.is_available():
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    else:
        device = torch.device("cpu")


    sim_dataset_path = base_dir / "sim_dataset.txt"
    real_dataset_path = base_dir / "real_dataset.txt"
    
    # Read data
    sim_rows  = read_variable_txt(sim_dataset_path)
    real_rows = read_variable_txt(real_dataset_path)

    sim_parsed, sim_stats   = parse_rows_to_struct(sim_rows)
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
    cfg.K_max          = max(sim_stats["K_max"], real_stats["K_max"])
    cfg.embed_channel_num = max(sim_stats["channel_num"], real_stats["channel_num"])
    cfg.embed_excitation_num = max(sim_stats["excitation_num"], real_stats["excitation_num"])

    num_folds = args.kfold if bool(args.stage2_kfold) else 1 # 如果开fold训练则为stage2_kfold，否则为1
    setattr(args, 'stage1_once', 1)

    for fold in range(num_folds):
        if (not dist.is_initialized()) or (dist.get_rank() == 0):
            print(f'========== Fold {fold+1}/{num_folds} ==========', flush=True)
        train_one_fold(fold, cfg, sim_parsed, real_parsed, args)
        if dist.is_initialized():
            torch.cuda.empty_cache()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    import socket
    def udp_sender():
        # 配置参数
        HOST = 'localhost'
        PORT = 12345
        MESSAGE = "XGD_CONSTRUCTION_EXE_FINISHED"

        # 创建UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        try:
            print("UDP发送端已启动")
            print(f"目标地址: {HOST}:{PORT}")
            print(f"发送内容: {MESSAGE}")
            print("\n正在发送...")

            # 发送消息
            sock.sendto(MESSAGE.encode('utf-8'), (HOST, PORT))

            print("消息发送成功！")
            
        except Exception as e:
            print(f"发送失败: {e}")
        finally:
            sock.close()
            print("发送端已关闭")

    udp_sender()


if __name__ == "__main__":
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
