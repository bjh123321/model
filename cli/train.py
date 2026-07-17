from __future__ import annotations

import argparse
import configparser
import copy
import multiprocessing as mp
import os
import socket
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
import torch.distributed as dist

from surrogate_model.config import Config
from surrogate_model.data.io import parse_rows_to_struct, read_real_dataset
from surrogate_model.data.normalizer import canonical_pole_target, canonical_residue_target
from surrogate_model.train.dataloaders import split_single_fidelity_rows
from surrogate_model.train.full_training_runner import run_component_training
from surrogate_model.utils.model_artifacts import (
    atomic_write_json,
    build_dataset_manifest,
    build_input_dataset_snapshot,
    group_key_from_parsed,
    load_json,
)
from surrogate_model.utils.excitation_channel_models import (
    CHANNEL_NAMES,
    EXCITATION_NAMES,
    FDR_COMPONENT_TARGETS,
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


def _resolve_id_selection(raw_values, id_to_name: dict[int, str], label: str):
    """Resolve repeatable comma/semicolon-separated names or numeric IDs."""
    if not raw_values:
        return None
    tokens = []
    for raw in raw_values:
        tokens.extend(
            part.strip()
            for part in str(raw).replace(";", ",").split(",")
            if part.strip()
        )
    name_to_id = {str(name).upper(): int(identifier) for identifier, name in id_to_name.items()}
    selected = set()
    invalid = []
    for token in tokens:
        try:
            identifier = int(token)
        except ValueError:
            identifier = name_to_id.get(token.upper())
        if identifier is None or int(identifier) not in id_to_name:
            invalid.append(token)
        else:
            selected.add(int(identifier))
    if invalid:
        available = ", ".join(f"{identifier}:{name}" for identifier, name in id_to_name.items())
        raise ValueError(f"Unknown {label} selector(s) {invalid}; available: {available}")
    if not selected:
        raise ValueError(f"At least one {label} selector is required")
    return selected


def _filter_combinations(combinations, selected_excitations=None, selected_channels=None):
    return [
        (excitation_id, channel_id)
        for excitation_id, channel_id in combinations
        if (selected_excitations is None or excitation_id in selected_excitations)
        and (selected_channels is None or channel_id in selected_channels)
    ]


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


def _resume_component_ready(
    output_dir: Path,
    excitation_id: int,
    channel_id: int,
    component: str,
    required_profile_version: int | None = None,
) -> tuple[bool, str]:
    try:
        required = (
            "model.pth", "Normalizer_Input.pth", "Normalizer_Output.pth",
            "training_config.json", "train_log.csv", "slot_support.json",
        )
        target_names = FDR_COMPONENT_TARGETS[component]
        component_dir = output_dir / component
        missing = [name for name in required if not (component_dir / name).is_file()]
        if missing:
            return False, f"missing {', '.join(missing)}"
        try:
            checkpoint = torch.load(component_dir / "model.pth", map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(component_dir / "model.pth", map_location="cpu")
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            return False, "checkpoint has no model_state_dict"
        if checkpoint.get("model_type") != "TargetMFDNN":
            return False, "checkpoint is not TargetMFDNN"
        if tuple(checkpoint.get("target_names", ())) != tuple(target_names):
            return False, "checkpoint target_names mismatch"
        for key, expected in (("excitation_id", excitation_id), ("channel_id", channel_id)):
            if key in checkpoint and int(checkpoint[key]) != int(expected):
                return False, f"checkpoint {key}={checkpoint[key]} does not match {expected}"
        for filename in ("Normalizer_Input.pth", "Normalizer_Output.pth"):
            normalizer = torch.load(component_dir / filename, map_location="cpu", weights_only=True)
            if not isinstance(normalizer, dict) or "mean" not in normalizer or "std" not in normalizer:
                return False, f"invalid {filename}"
        output_normalizer = torch.load(
            component_dir / "Normalizer_Output.pth", map_location="cpu", weights_only=True
        )
        transforms = dict(output_normalizer.get("target_transforms", {}) or {})
        if any(str(transforms.get(name, {}).get("kind", "identity")) == "identity" for name in target_names if name in {"freq", "magnitude"}):
            return False, "output normalizer lacks physical target transforms"
        training_config = load_json(component_dir / "training_config.json")
        if int(training_config.get("excitation_id", excitation_id)) != int(excitation_id):
            return False, "training_config excitation_id mismatch"
        if int(training_config.get("channel_id", channel_id)) != int(channel_id):
            return False, "training_config channel_id mismatch"
        if required_profile_version is not None:
            actual_profile_version = int(
                dict(training_config.get("training", {}) or {}).get(
                    "fdr_profile_version", 1
                )
            )
            if actual_profile_version < int(required_profile_version):
                return False, (
                    f"obsolete FDR training profile {actual_profile_version}; "
                    f"requires {required_profile_version}"
                )
        if (component_dir / "train_log.csv").stat().st_size <= 0:
            return False, "empty train_log.csv"
    except Exception as exc:
        return False, f"artifact validation failed: {type(exc).__name__}: {exc}"
    return True, "complete"


def _required_component_profile_version(cfg, component: str) -> int:
    versions = dict(
        getattr(cfg, "fdr_component_training_profile_version", {}) or {}
    )
    return int(
        versions.get(component, getattr(cfg, "fdr_training_profile_version", 1))
    )


def _resume_model_ready(output_dir: Path, excitation_id: int, channel_id: int) -> tuple[bool, str]:
    for component in FDR_COMPONENT_TARGETS:
        ready, reason = _resume_component_ready(
            output_dir, excitation_id, channel_id, component
        )
        if not ready:
            return False, f"{component}: {reason}"
    return True, "complete"


def _train_component_worker(task: dict) -> dict:
    """Process-pool entry point; each task owns one unique component directory."""
    torch.set_num_threads(1)
    try:
        run_component_training(
            cfg=task["cfg"],
            parsed_rows=task["parsed_rows"],
            model_version=task["model_version"],
            version_dir=Path(task["version_dir"]),
            args=task["args"],
            dataset_metadata=task["dataset_metadata"],
            component=task["component"],
        )
        return {
            "ok": True,
            "excitation_id": task["excitation_id"],
            "channel_id": task["channel_id"],
            "component": task["component"],
        }
    except Exception as exc:
        return {
            "ok": False,
            "excitation_id": task["excitation_id"],
            "channel_id": task["channel_id"],
            "component": task["component"],
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if (
            str(getattr(task["args"], "device", "auto")) != "cpu"
            and torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()


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
    parser.add_argument("--checkpoint_train_loss_ceiling", type=float, default=None)
    parser.add_argument("--damping_slope_weight", type=float, default=None)
    parser.add_argument("--damping_curvature_weight", type=float, default=None)
    parser.add_argument("--damping_slope_limit_multiplier", type=float, default=None)
    parser.add_argument("--min_slot_train_samples", type=int, default=None)
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
        help="Skip each FDR component whose checkpoint, normalizers, config, and log are complete.",
    )
    parser.add_argument(
        "--parallel_workers",
        type=int,
        default=1,
        help="Number of independent FDR component networks to train concurrently.",
    )
    parser.add_argument(
        "--excitation",
        "--excitations",
        action="append",
        default=None,
        help="Train only selected excitation names or IDs; repeat or separate with commas.",
    )
    parser.add_argument(
        "--channel",
        "--channels",
        action="append",
        default=None,
        help="Train only selected response-channel names or IDs; repeat or separate with commas.",
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
    if int(args.parallel_workers) <= 0:
        raise ValueError("--parallel_workers must be a positive integer")
    if (
        args.checkpoint_train_loss_ceiling is not None
        and float(args.checkpoint_train_loss_ceiling) <= 0.0
    ):
        raise ValueError("--checkpoint_train_loss_ceiling must be positive")
    if args.damping_slope_weight is not None and float(args.damping_slope_weight) < 0.0:
        raise ValueError("--damping_slope_weight must be non-negative")
    if args.damping_curvature_weight is not None and float(args.damping_curvature_weight) < 0.0:
        raise ValueError("--damping_curvature_weight must be non-negative")
    if args.damping_slope_limit_multiplier is not None and float(args.damping_slope_limit_multiplier) <= 0.0:
        raise ValueError("--damping_slope_limit_multiplier must be positive")
    if args.min_slot_train_samples is not None and int(args.min_slot_train_samples) <= 0:
        raise ValueError("--min_slot_train_samples must be positive")

    ddp_enabled, rank, _world_size = _ddp_setup()
    if ddp_enabled:
        dist.destroy_process_group()
        raise RuntimeError(
            "torchrun/DDP is not supported for independent FDR component tasks; "
            "run ordinary python with --parallel_workers instead."
        )
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
    if args.checkpoint_train_loss_ceiling is not None:
        cfg.checkpoint_train_loss_ceiling = float(
            args.checkpoint_train_loss_ceiling
        )
    if args.damping_slope_weight is not None:
        cfg.damping_slope_weight = float(args.damping_slope_weight)
    if args.damping_curvature_weight is not None:
        cfg.damping_curvature_weight = float(args.damping_curvature_weight)
    if args.damping_slope_limit_multiplier is not None:
        cfg.damping_slope_limit_multiplier = float(args.damping_slope_limit_multiplier)
    if args.min_slot_train_samples is not None:
        cfg.min_slot_train_samples = int(args.min_slot_train_samples)
    if args.checkpoint_selection is not None:
        cfg.checkpoint_selection = str(args.checkpoint_selection)
    if args.epochs is not None:
        cfg.epochs_stage1 = int(args.epochs)
    if args.batch_real is not None:
        cfg.batch_real = int(args.batch_real)
    if args.num_workers is not None:
        cfg.num_workers = int(args.num_workers)
    elif int(args.parallel_workers) > 1:
        cfg.num_workers = 0
        print(
            "[PARALLEL] --num_workers was not specified; using 0 per training process "
            "to avoid CPU oversubscription.",
            flush=True,
        )
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
    if cfg.pole_target != "freq_damping" or cfg.residue_target != "polar":
        raise ValueError(
            "FDR split training requires pole_target='freq_damping' and residue_target='polar'"
        )

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

    target_names = [name for names in FDR_COMPONENT_TARGETS.values() for name in names]
    trained_models = {}
    failures = []
    tasks = []
    component_states = {}
    combination_info = {}
    all_combinations = combinations_in_rows(real_parsed)
    selected_excitations = _resolve_id_selection(
        args.excitation, EXCITATION_NAMES, "excitation"
    )
    selected_channels = _resolve_id_selection(
        args.channel,
        {index: name for index, name in enumerate(CHANNEL_NAMES)},
        "channel",
    )
    combinations = _filter_combinations(
        all_combinations, selected_excitations, selected_channels
    )
    if not combinations:
        raise RuntimeError(
            "The excitation/channel selectors matched no combinations in the dataset"
        )
    selected_keys = set(combinations)
    if selected_excitations is not None or selected_channels is not None:
        print(
            f"[SELECTION] selected_combinations={len(combinations)}/{len(all_combinations)} "
            f"excitations={sorted(selected_excitations) if selected_excitations is not None else 'all'} "
            f"channels={sorted(selected_channels) if selected_channels is not None else 'all'}",
            flush=True,
        )
        rows_by_combination = {
            key: [
                row for row in real_parsed
                if (int(row.exc_id), int(row.ch_id)) == key
            ]
            for key in all_combinations
        }
        for excitation_id, channel_id in all_combinations:
            key = (excitation_id, channel_id)
            if key in selected_keys:
                continue
            output_dir = combination_dir(model_root, excitation_id, channel_id)
            ready, _reason = _resume_model_ready(
                output_dir, excitation_id, channel_id
            )
            if ready:
                trained_models[key] = model_entry(
                    excitation_id=excitation_id,
                    channel_id=channel_id,
                    row_count=len(rows_by_combination[key]),
                    status="retained",
                )
        print(
            f"[SELECTION] retained_unselected_complete_models={len(trained_models)}",
            flush=True,
        )
    for index, (excitation_id, channel_id) in enumerate(combinations, 1):
        key = (excitation_id, channel_id)
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
        print(
            f"========== Queue model {index}/{len(combinations)}: "
            f"{label}, rows={len(combo_rows)} ==========",
            flush=True,
        )
        combination_info[key] = {"row_count": len(combo_rows), "label": label}
        component_states[key] = {}
        try:
            if not combo_cfg.explicit_train_group_keys:
                raise RuntimeError("No training operating points are available for this combination")
            if combo_cfg.require_validation and not combo_cfg.explicit_validation_group_keys:
                raise RuntimeError("No validation operating points are available for this combination")
        except Exception as exc:
            failures.append(
                {
                    "excitation_id": excitation_id,
                    "excitation_name": excitation_name(excitation_id),
                    "channel_id": channel_id,
                    "row_count": len(combo_rows),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"[SKIP] {label}: {type(exc).__name__}: {exc}", flush=True)
            for component in FDR_COMPONENT_TARGETS:
                component_states[key][component] = "failed"
            continue

        dataset_metadata = {
            "manifest": manifest,
            "dataset_manifest_path": "../../../dataset_manifest.json",
            "training_mode": "base_full_training",
            "model_filename": "model.pth",
            "excitation_id": excitation_id,
            "excitation_name": excitation_name(excitation_id),
            "channel_id": channel_id,
            "target_names": target_names,
        }
        for component in FDR_COMPONENT_TARGETS:
            if args.resume:
                ready, reason = _resume_component_ready(
                    output_dir,
                    excitation_id,
                    channel_id,
                    component,
                    required_profile_version=_required_component_profile_version(
                        combo_cfg, component
                    ),
                )
                if ready:
                    component_states[key][component] = "resumed"
                    print(f"[RESUME] retained {label}/{component}", flush=True)
                    continue
                if (output_dir / component).exists():
                    print(f"[RESUME] retrain {label}/{component}: {reason}", flush=True)
            component_states[key][component] = "pending"
            tasks.append(
                {
                    "cfg": combo_cfg,
                    "parsed_rows": combo_rows,
                    "model_version": model_version,
                    "version_dir": str(output_dir),
                    "args": args,
                    "dataset_metadata": dataset_metadata,
                    "excitation_id": excitation_id,
                    "channel_id": channel_id,
                    "component": component,
                }
            )

        if all(state == "resumed" for state in component_states[key].values()):
            trained_models[key] = model_entry(
                excitation_id=excitation_id,
                channel_id=channel_id,
                row_count=len(combo_rows),
                status="resumed",
            )

    def write_progress_inventory():
        write_model_inventory(
            model_root,
            model_version=model_version,
            models=trained_models.values(),
            failures=failures,
        )

    def handle_task_result(result: dict):
        key = (int(result["excitation_id"]), int(result["channel_id"]))
        component = str(result["component"])
        info = combination_info[key]
        if result.get("ok"):
            ready, reason = _resume_component_ready(
                combination_dir(model_root, *key),
                *key,
                component,
                required_profile_version=_required_component_profile_version(
                    cfg, component
                ),
            )
            if ready:
                component_states[key][component] = "trained"
                print(f"[TASK-OK] {info['label']}/{component}", flush=True)
            else:
                result = {
                    **result,
                    "ok": False,
                    "error_type": "ArtifactValidationError",
                    "error": reason,
                    "traceback": "",
                }
        if not result.get("ok"):
            component_states[key][component] = "failed"
            failure = {
                "excitation_id": key[0],
                "excitation_name": excitation_name(key[0]),
                "channel_id": key[1],
                "component": component,
                "row_count": info["row_count"],
                "error_type": result.get("error_type", "WorkerError"),
                "error": result.get("error", "Unknown component worker failure"),
                "traceback": result.get("traceback", ""),
            }
            failures.append(failure)
            print(
                f"[TASK-FAIL] {info['label']}/{component}: "
                f"{failure['error_type']}: {failure['error']}",
                flush=True,
            )

        if all(
            state in {"resumed", "trained"}
            for state in component_states[key].values()
        ):
            status = (
                "resumed"
                if all(state == "resumed" for state in component_states[key].values())
                else "trained"
            )
            trained_models[key] = model_entry(
                excitation_id=key[0],
                channel_id=key[1],
                row_count=info["row_count"],
                status=status,
            )
            print(f"[MODEL-COMPLETE] {info['label']} status={status}", flush=True)
        write_progress_inventory()

    write_progress_inventory()
    print(
        f"[PARALLEL] workers={args.parallel_workers} pending_components={len(tasks)} "
        f"resumed_models={len(trained_models)}",
        flush=True,
    )
    if int(args.parallel_workers) == 1:
        for task in tasks:
            handle_task_result(_train_component_worker(task))
    elif tasks:
        context = mp.get_context("spawn")
        future_tasks = {}
        with ProcessPoolExecutor(
            max_workers=min(int(args.parallel_workers), len(tasks)),
            mp_context=context,
        ) as executor:
            for task in tasks:
                future_tasks[executor.submit(_train_component_worker, task)] = task
            for future in as_completed(future_tasks):
                task = future_tasks[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "ok": False,
                        "excitation_id": task["excitation_id"],
                        "channel_id": task["channel_id"],
                        "component": task["component"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                handle_task_result(result)

    if (not ddp_enabled) or rank == 0:
        write_progress_inventory()
        print(
            f"[MODEL-INVENTORY] complete={len(trained_models)} failures={len(failures)} "
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
