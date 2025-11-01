from __future__ import annotations
import os, argparse
from pathlib import Path
import torch
import torch.distributed as dist
from ..config import Config
from ..data.io import read_variable_txt, parse_rows_to_struct
from ..train.loops import train_one_fold

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_txt', type=str, required=True)
    parser.add_argument('--real_txt', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='./surrogate_model/outputs_pzk')
    parser.add_argument('--kfold', type=int, default=5)
    parser.add_argument('--stage1_kfold', type=int, default=0, choices=[0, 1],
                        help='Set to 1 to train stage 1 on every fold, 0 to train once and reuse')
    parser.add_argument('--stage2_kfold', type=int, default=1, choices=[0, 1],
                        help='Set to 1 to train stage 2 on every fold, 0 to train once')
    parser.add_argument('--embed_excitation_dim', type=int, default=4)
    parser.add_argument('--embed_channel_dim', type=int, default=12)
    parser.add_argument('--epochs_s1', type=int, default=10000)
    parser.add_argument('--epochs_s2', type=int, default=10000)
    parser.add_argument('--batch_sim', type=int, default=256)
    parser.add_argument('--batch_real', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--prefetch', type=int, default=4)
    args = parser.parse_args()

    # DDP init
    if torch.cuda.is_available():
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0))) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        # 1) 允许 TF32 (Ampere+ GPU)：对 matmul 和 cuDNN 都打开
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # 2) cuDNN 算法自动搜索最优实现（卷积更快；输入尺寸稳定时收益更大）
        torch.backends.cudnn.benchmark = True
        # 3) PyTorch 2.x：提高 float32 matmul 的内部精度/性能权衡
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    # Read data
    sim_rows  = read_variable_txt(Path(args.sim_txt))
    real_rows = read_variable_txt(Path(args.real_txt))

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
    setattr(args, 'stage1_once', int(not bool(args.stage1_kfold)))

    for fold in range(num_folds):
        if (not dist.is_initialized()) or (dist.get_rank() == 0):
            print(f'========== Fold {fold+1}/{num_folds} ==========', flush=True)
        train_one_fold(fold, cfg, sim_parsed, real_parsed, args)
        if dist.is_initialized():
            torch.cuda.empty_cache()

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
