from __future__ import annotations

import argparse
import configparser
import copy
import multiprocessing as mp
import os
import socket
import sys
import traceback
from pathlib import Path

import torch
import torch.distributed as dist

from surrogate_model.config import Config
from surrogate_model.data.io import parse_rows_to_struct, read_real_dataset
from surrogate_model.data.normalizer import canonical_pole_target, canonical_residue_target
from surrogate_model.train.dataloaders import split_single_fidelity_rows
from surrogate_model.train.full_training_runner import run_full_training
from surrogate_model.utils.model_artifacts import (
    atomic_write_json,
    build_dataset_manifest,
    build_input_dataset_snapshot,
    group_key_from_parsed,
    load_json,
)
from surrogate_model.utils.excitation_channel_models import (
    channel_name,
    combination_dir,
    combinations_in_rows,
    excitation_name,
    model_entry,
    write_model_inventory,
)


def _parse_int_list(raw: str) -> list[int]:
    values = []
    for part in str(raw).replace(";", ",").split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


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


def _resume_model_ready(output_dir: Path, excitation_id: int, channel_id: int) -> tuple[bool, str]:
    required = (
        "model.pth",
        "Normalizer_Input.pth",
        "Normalizer_Output.pth",
        "training_config.json",
        "train_log.csv",
    )
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        return False, f"missing {', '.join(missing)}"
    try:
        try:
            checkpoint = torch.load(output_dir / "model.pth", map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(output_dir / "model.pth", map_location="cpu")
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            return False, "checkpoint has no model_state_dict"
        for key, expected in (("excitation_id", excitation_id), ("channel_id", channel_id)):
            if key in checkpoint and int(checkpoint[key]) != int(expected):
                return False, f"checkpoint {key}={checkpoint[key]} does not match {expected}"
        for filename in ("Normalizer_Input.pth", "Normalizer_Output.pth"):
            normalizer = torch.load(output_dir / filename, map_location="cpu", weights_only=True)
            if not isinstance(normalizer, dict) or "mean" not in normalizer or "std" not in normalizer:
                return False, f"invalid {filename}"
        training_config = load_json(output_dir / "training_config.json")
        if int(training_config.get("excitation_id", excitation_id)) != int(excitation_id):
            return False, "training_config excitation_id mismatch"
        if int(training_config.get("channel_id", channel_id)) != int(channel_id):
            return False, "training_config channel_id mismatch"
        if (output_dir / "train_log.csv").stat().st_size <= 0:
            return False, "empty train_log.csv"
    except Exception as exc:
        return False, f"artifact validation failed: {type(exc).__name__}: {exc}"
    return True, "complete"


def main():
    base_dir = get_exe_dir()
    user_input_path = base_dir / "user_input.dat"
    with open(user_input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) < 2:
        raise ValueError("user_input.dat needs at least 2 lines: paths.ini, model_version")

    paths_ini_path = Path(lines[0].strip()).expanduser()
    model_version = lines[1].strip()
    real_dataset_path = base_dir / "real_dataset"
    if not real_dataset_path.is_dir():
        raise FileNotFoundError(
            f"Base training requires a real_dataset directory next to the CLI entry: "
            f"{real_dataset_path}"
        )
    model_dir = _default_model_dir(base_dir, paths_ini_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=str(model_dir / model_version))
    parser.add_argument("--kfold", type=int, default=0)
    parser.add_argument("--embed_excitation_dim", type=int, default=None)
    parser.add_argument("--embed_channel_dim", type=int, default=None)
    parser.add_argument("--low_hidden", type=_parse_int_list, default=None)
    parser.add_argument("--use_modulation", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dropout_prob", type=float, default=None)
    parser.add_argument("--mod_gamma", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--early_stop_min_delta", type=float, default=None)
    parser.add_argument(
        "--checkpoint_selection",
        choices=["validation", "training"],
        default=None,
        help="Select the best checkpoint by validation group_FXD loss (default) or total training loss.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip channel models whose checkpoint, normalizers, config, and log are complete.",
    )
    parser.add_argument("--batch_real", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--prefetch", type=int, default=None)
    parser.add_argument(
        "--pole_target",
        type=str,
        default=None,
        choices=["alpha_omega", "freq_damping", "pole", "modal"],
        help="alpha_omega keeps the old pole-real/imag path; freq_damping trains on converted frequency/damping.",
    )
    parser.add_argument(
        "--residue_target",
        type=str,
        default=None,
        choices=["polar", "cartesian", "fixed"],
        help="polar keeps the old log-magnitude/sin/cos path; cartesian trains on residue real/imag.",
    )
    parser.add_argument(
        "--val_split_strategy",
        type=str,
        default=None,
        choices=["height_max_mach", "height_mach_points", "height_mach_kfold", ""],
    )
    parser.add_argument("--debug_overfit_n", type=int, default=0)
    parser.add_argument("--debug_overfit_offset", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args, _unknown = parser.parse_known_args()

    ddp_enabled, rank, _world_size = _ddp_setup()
    _enable_fast_cuda(args)

    real_rows = read_real_dataset(real_dataset_path)
    real_parsed, real_stats = parse_rows_to_struct(real_rows)

    cfg = Config()
    if args.embed_excitation_dim is not None:
        cfg.embed_excitation_dim = args.embed_excitation_dim
    if args.embed_channel_dim is not None:
        cfg.embed_channel_dim = args.embed_channel_dim
    if args.low_hidden is not None:
        cfg.low_hidden = list(args.low_hidden)
    if args.use_modulation is not None:
        cfg.use_modulation = bool(args.use_modulation)
    if args.dropout_prob is not None:
        cfg.dropout_prob = float(args.dropout_prob)
    if args.mod_gamma is not None:
        cfg.mod_gamma = float(args.mod_gamma)
    if args.lr is not None:
        cfg.lr_low_adam = float(args.lr)
    if args.weight_decay is not None:
        cfg.weight_decay = float(args.weight_decay)
    if args.early_stop_patience is not None:
        cfg.early_stop_patience_s1 = int(args.early_stop_patience)
    if args.early_stop_min_delta is not None:
        cfg.early_stop_min_delta = float(args.early_stop_min_delta)
    if args.checkpoint_selection is not None:
        cfg.checkpoint_selection = str(args.checkpoint_selection)
    if args.epochs is not None:
        cfg.epochs_stage1 = int(args.epochs)
    if args.batch_real is not None:
        cfg.batch_real = int(args.batch_real)
    if args.num_workers is not None:
        cfg.num_workers = int(args.num_workers)
    if args.prefetch is not None:
        cfg.prefetch = int(args.prefetch)
    if args.pole_target is not None:
        cfg.pole_target = canonical_pole_target(args.pole_target)
    if args.residue_target is not None:
        cfg.residue_target = canonical_residue_target(args.residue_target)
    cfg.debug_direct_output = False
    if args.debug_overfit_n > 0:
        offset = int(args.debug_overfit_offset or 0)
        real_parsed = list(real_parsed)[offset:offset + int(args.debug_overfit_n)]
        print(
            f"[DEBUG-CLI] pre-sliced samples={len(real_parsed)} offset={offset}; "
            f"pole_target={cfg.pole_target}, residue_target={cfg.residue_target}",
            flush=True,
        )
        args.debug_overfit_n = 0
        args.debug_overfit_offset = 0
        cfg.use_val_split = False
        cfg.early_stop_patience_s1 = 0
        cfg.num_workers = 0
    if args.val_split_strategy is not None:
        cfg.val_split_strategy = args.val_split_strategy
    cfg.K_max = 3
    cfg.embed_channel_num = real_stats["channel_num"]
    cfg.embed_excitation_num = real_stats["excitation_num"]

    model_root = Path(args.out_dir)
    preview_train, preview_val, split_strategy = split_single_fidelity_rows(real_parsed, cfg, 0, 0)
    train_group_keys = sorted({group_key_from_parsed(row) for row in preview_train})
    validation_group_keys = sorted({group_key_from_parsed(row) for row in preview_val})
    if not ddp_enabled or rank == 0:
        relative_files_by_group = build_input_dataset_snapshot(
            real_dataset_path, model_root / "dataset_snapshot"
        )
        manifest = build_dataset_manifest(
            rows=real_rows,
            model_version=model_version,
            snapshot_dir=model_root / "dataset_snapshot",
            train_group_keys=train_group_keys,
            validation_group_keys=validation_group_keys,
            split_strategy=split_strategy,
            relative_files_by_group=relative_files_by_group,
        )
        atomic_write_json(model_root / "dataset_manifest.json", manifest)
    if ddp_enabled:
        dist.barrier()
    if ddp_enabled and rank != 0:
        manifest = load_json(model_root / "dataset_manifest.json")

    if args.kfold > 1:
        raise ValueError("Per-excitation/channel training currently requires --kfold 0 or 1")

    target_names = (
        (["freq", "damping"] if cfg.pole_target == "freq_damping" else ["alpha", "omega"])
        + (["rho", "sin_theta", "cos_theta"] if cfg.residue_target == "polar" else ["residue_re", "residue_im"])
    )
    trained_models = []
    failures = []
    combinations = combinations_in_rows(real_parsed)
    for index, (excitation_id, channel_id) in enumerate(combinations, 1):
        combo_rows = [
            row for row in real_parsed
            if int(row.exc_id) == excitation_id and int(row.ch_id) == channel_id
        ]
        combo_cfg = copy.deepcopy(cfg)
        available_keys = {group_key_from_parsed(row) for row in combo_rows}
        combo_cfg.explicit_train_group_keys = sorted(set(train_group_keys) & available_keys)
        combo_cfg.explicit_validation_group_keys = sorted(set(validation_group_keys) & available_keys)
        combo_cfg.require_validation = combo_cfg.checkpoint_selection == "validation"
        combo_cfg.validation_metric_scope = "all"
        output_dir = combination_dir(model_root, excitation_id, channel_id)
        label = f"{excitation_name(excitation_id)}/{channel_name(channel_id)}"
        print(f"========== Model {index}/{len(combinations)}: {label}, rows={len(combo_rows)} ==========", flush=True)
        try:
            if args.resume:
                ready, reason = _resume_model_ready(output_dir, excitation_id, channel_id)
                if ready:
                    trained_models.append(
                        model_entry(
                            excitation_id=excitation_id,
                            channel_id=channel_id,
                            row_count=len(combo_rows),
                            status="resumed",
                        )
                    )
                    print(f"[RESUME] complete model retained: {output_dir}", flush=True)
                    continue
                if output_dir.exists():
                    print(f"[RESUME] {label} will be retrained: {reason}", flush=True)
            if not combo_cfg.explicit_train_group_keys:
                raise RuntimeError("No training operating points are available for this combination")
            if combo_cfg.require_validation and not combo_cfg.explicit_validation_group_keys:
                raise RuntimeError("No validation operating points are available for this combination")
            run_full_training(
                cfg=combo_cfg,
                parsed_rows=combo_rows,
                model_version=model_version,
                version_dir=output_dir,
                args=args,
                dataset_metadata={
                    "manifest": manifest,
                    "dataset_manifest_path": "../../dataset_manifest.json",
                    "training_mode": "base_full_training",
                    "model_filename": "model.pth",
                    "excitation_id": excitation_id,
                    "excitation_name": excitation_name(excitation_id),
                    "channel_id": channel_id,
                    "target_names": target_names,
                },
            )
            trained_models.append(
                model_entry(
                    excitation_id=excitation_id,
                    channel_id=channel_id,
                    row_count=len(combo_rows),
                    status="trained",
                )
            )
        except Exception as exc:
            failure = {
                "excitation_id": excitation_id,
                "excitation_name": excitation_name(excitation_id),
                "channel_id": channel_id,
                "row_count": len(combo_rows),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            print(f"[SKIP] {label}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if (not ddp_enabled) or rank == 0:
        write_model_inventory(
            model_root,
            model_version=model_version,
            models=trained_models,
            failures=failures,
        )
        print(
            f"[MODEL-INVENTORY] trained={len(trained_models)} skipped={len(failures)} "
            f"path={model_root / 'models_manifest.json'}",
            flush=True,
        )

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
