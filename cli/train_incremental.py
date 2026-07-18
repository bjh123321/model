from __future__ import annotations

import argparse
import copy
import configparser
import math
import multiprocessing as mp
import os
import shutil
import socket
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from surrogate_model.config import Config
from surrogate_model.data.OE_BATCH_PROCESS import batch_process_split
from surrogate_model.data.io import parse_rows_to_struct, read_real_dataset
from surrogate_model.data.normalizer import canonical_pole_target, canonical_residue_target
from surrogate_model.models.checkpoint_utils import restore_model_config_from_checkpoint
from surrogate_model.cli.train import (
    _apply_component_training_cli,
    _filter_combinations,
    _resolve_component_selection,
    _resolve_id_selection,
    _train_component_worker,
)
from surrogate_model.train.dataloaders import split_single_fidelity_rows
from surrogate_model.utils.model_artifacts import (
    atomic_write_json,
    build_dataset_manifest,
    build_dataset_snapshot,
    build_input_dataset_snapshot,
    group_key_from_parsed,
    group_rows,
    load_json,
    merge_dataset_rows,
    resolve_parent_dataset,
    validate_new_version_directory,
)
from surrogate_model.utils.excitation_channel_models import (
    CHANNEL_NAMES,
    EXCITATION_NAMES,
    FDR_COMPONENT_TARGETS,
    channel_name,
    combination_dir,
    combinations_in_rows,
    copy_parent_model_directories,
    excitation_name,
    inventory_index,
    load_model_inventory,
    model_entry,
    write_model_inventory,
)


def get_exe_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


def _path_from_ini(cp: configparser.ConfigParser, paths_ini_path: Path, section: str) -> Path:
    key = f"path_{section}".lower()
    try:
        return Path(cp[section][key]).expanduser()
    except KeyError:
        raise KeyError(f"{paths_ini_path} is missing [{section}] / {key}") from None


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in str(raw).replace(";", ",").split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def _send_finished_udp():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(b"XGD_CORRECTION_EXE_FINISHED", ("localhost", 12345))
    except Exception as exc:
        print(f"[WARN] UDP notification failed: {exc}", flush=True)
    finally:
        sock.close()


def _run_incremental_component_tasks(
    tasks: list[dict], parallel_workers: int, *, worker=None
):
    """Yield component-worker results; each task owns one staged component directory."""
    worker = _train_component_worker if worker is None else worker
    worker_count = int(parallel_workers)
    if worker_count <= 0:
        raise ValueError("parallel_workers must be positive")
    if worker_count == 1:
        for task in tasks:
            yield worker(task)
        return
    if not tasks:
        return

    context = mp.get_context("spawn")
    future_tasks = {}
    with ProcessPoolExecutor(
        max_workers=min(worker_count, len(tasks)),
        mp_context=context,
    ) as executor:
        for task in tasks:
            future_tasks[executor.submit(worker, task)] = task
        for future in as_completed(future_tasks):
            task = future_tasks[future]
            try:
                yield future.result()
            except Exception as exc:
                yield {
                    "ok": False,
                    "excitation_id": task["excitation_id"],
                    "channel_id": task["channel_id"],
                    "component": task["component"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }


def _replace_selected_component_directories(
    staging_dir: Path, final_dir: Path, components
) -> None:
    """Replace only selected component directories, rolling back commit failures."""
    staging_dir = Path(staging_dir)
    final_dir = Path(final_dir)
    selected = tuple(str(component) for component in components)
    missing = [component for component in selected if not (staging_dir / component).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Staged component directories are missing: {missing} in {staging_dir}"
        )

    final_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = final_dir.parent / f".{final_dir.name}.component_backup"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    records = []
    try:
        for component in selected:
            staged = staging_dir / component
            target = final_dir / component
            backup = backup_dir / component
            had_previous = target.exists()
            if had_previous:
                os.replace(target, backup)
            records.append((target, backup, had_previous))
            os.replace(staged, target)
    except Exception:
        for target, backup, had_previous in reversed(records):
            if target.exists():
                shutil.rmtree(target)
            if had_previous and backup.exists():
                os.replace(backup, target)
        raise
    finally:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)


def _restore_parent_configuration(
    base_version_dir: Path, parent_inventory: dict | None = None
) -> tuple[Config, dict | None]:
    cfg = Config()
    training_path = base_version_dir / "training_config.json"
    checkpoint_path = base_version_dir / f"{base_version_dir.name}.pth"
    if parent_inventory is not None and parent_inventory.get("models"):
        representative = parent_inventory["models"][0]
        frequency = representative.get("components", {}).get("frequency")
        if not frequency:
            raise RuntimeError("Parent model inventory has no FDR frequency component")
        training_path = base_version_dir / frequency["training_config"]
        checkpoint_path = base_version_dir / frequency["checkpoint"]
    if training_path.is_file():
        parent_training = load_json(training_path)
        model = parent_training.get("model", {})
        training = parent_training.get("training", {})
        targets = parent_training.get("targets", {})
        for name in (
            "K_max", "low_hidden", "adapt_hidden", "dropout_prob", "embed_excitation_num",
            "embed_excitation_dim", "embed_channel_num", "embed_channel_dim",
            "use_modulation", "mod_gamma",
        ):
            if name in model:
                setattr(cfg, name, model[name])
        cfg.pole_target = canonical_pole_target(targets.get("pole_target", cfg.pole_target))
        cfg.residue_target = canonical_residue_target(targets.get("residue_target", cfg.residue_target))
        mapping = {
            "learning_rate": "lr_low_adam",
            "learning_rate_high": "lr_high_adam",
            "weight_decay": "weight_decay",
            "batch_size": "batch_real",
            "batch_sim": "batch_sim",
            "epochs": "epochs_stage1",
            "epochs_stage2": "epochs_stage2",
            "num_workers": "num_workers",
            "prefetch": "prefetch",
            "early_stop_patience": "early_stop_patience_s1",
            "early_stop_patience_stage2": "early_stop_patience_s2",
            "early_stop_min_delta": "early_stop_min_delta",
        }
        for source, destination in mapping.items():
            if source in training:
                setattr(cfg, destination, training[source])
        validation = training.get("validation_split", {})
        if "strategy" in validation:
            cfg.val_split_strategy = validation["strategy"]
        if "ratio" in validation:
            cfg.val_ratio = validation["ratio"]
        if "seed" in validation:
            cfg.val_seed = validation["seed"]
        if parent_inventory is not None and parent_inventory.get("models"):
            representative = parent_inventory["models"][0]
            for component, component_entry in representative.get("components", {}).items():
                component_training_path = base_version_dir / component_entry.get(
                    "training_config", ""
                )
                if not component_training_path.is_file():
                    continue
                component_training = load_json(component_training_path).get("training", {})
                multifidelity = dict(component_training.get("multifidelity", {}) or {})
                if "real_loss_weight_stage2" in multifidelity:
                    cfg.fdr_component_real_loss_weight[component] = float(
                        multifidelity["real_loss_weight_stage2"]
                    )
                if "sim_loss_weight_stage2" in multifidelity:
                    cfg.fdr_component_sim_loss_weight[component] = float(
                        multifidelity["sim_loss_weight_stage2"]
                    )
        return cfg, parent_training

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Parent checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        cfg = restore_model_config_from_checkpoint(checkpoint, use_lora=False)
    cfg.use_lora = False
    return cfg, None


def _parent_component_train_losses(base_version_dir: Path, parent_entry: dict) -> dict[str, float]:
    losses = {}
    components = parent_entry.get("components", {})
    for component in FDR_COMPONENT_TARGETS:
        component_entry = components.get(component)
        if not component_entry:
            raise RuntimeError(f"Parent inventory is missing FDR component {component!r}")
        training = load_json(base_version_dir / component_entry["training_config"])
        value = training.get("selection", {}).get("final_train_loss")
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise RuntimeError(
                f"Parent FDR component {component!r} has no numeric selection.final_train_loss"
            ) from None
        if not math.isfinite(value) or value < 0:
            raise RuntimeError(
                f"Parent FDR component {component!r} final_train_loss must be finite and "
                f"non-negative, got {value!r}"
            )
        losses[component] = value
    return losses


def _apply_cli_overrides(cfg: Config, args) -> None:
    mapping = {
        "epochs": "epochs_stage1",
        "batch_real": "batch_real",
        "lr": "lr_low_adam",
        "weight_decay": "weight_decay",
        "early_stop_patience": "early_stop_patience_s1",
        "early_stop_min_delta": "early_stop_min_delta",
        "checkpoint_selection": "checkpoint_selection",
        "low_hidden": "low_hidden",
        "dropout_prob": "dropout_prob",
        "use_modulation": "use_modulation",
        "mod_gamma": "mod_gamma",
        "num_workers": "num_workers",
        "prefetch": "prefetch",
    }
    for source, destination in mapping.items():
        value = getattr(args, source, None)
        if value is not None:
            setattr(cfg, destination, value)
    if args.epochs is not None:
        cfg.epochs_stage2 = int(args.epochs)
    if getattr(args, "epochs_s2", None) is not None:
        cfg.epochs_stage2 = int(args.epochs_s2)
    if getattr(args, "lr_high", None) is not None:
        cfg.lr_high_adam = float(args.lr_high)
    if getattr(args, "batch_sim", None) is not None:
        cfg.batch_sim = int(args.batch_sim)
    if args.pole_target is not None:
        cfg.pole_target = canonical_pole_target(args.pole_target)
    if args.residue_target is not None:
        cfg.residue_target = canonical_residue_target(args.residue_target)
    _apply_component_training_cli(cfg, args)
    cfg.use_lora = False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--epochs_s2", type=int, default=None)
    parser.add_argument("--lr_high", type=float, default=None)
    parser.add_argument("--batch_sim", type=int, default=None)
    parser.add_argument("--batch_real", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--early_stop_min_delta", type=float, default=None)
    parser.add_argument("--epochs_after_parent_loss", type=int, default=50)
    parser.add_argument("--checkpoint_selection", choices=["validation", "training"], default=None)
    parser.add_argument("--component_checkpoint_selection", action="append", default=None)
    parser.add_argument("--real_loss_weight", type=float, default=None)
    parser.add_argument("--sim_loss_weight", type=float, default=None)
    parser.add_argument("--component_real_loss_weight", action="append", default=None)
    parser.add_argument("--component_sim_loss_weight", action="append", default=None)
    parser.add_argument("--low_hidden", type=_parse_int_list, default=None)
    parser.add_argument("--dropout_prob", type=float, default=None)
    parser.add_argument("--use_modulation", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--mod_gamma", type=float, default=None)
    parser.add_argument("--pole_target", choices=["alpha_omega", "freq_damping", "pole", "modal"], default=None)
    parser.add_argument("--residue_target", choices=["polar", "cartesian", "fixed"], default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--prefetch", type=int, default=None)
    parser.add_argument(
        "--parallel_workers",
        type=int,
        default=1,
        help="Number of independent correction component networks to retrain concurrently.",
    )
    parser.add_argument("--history_dataset", type=str, default=None)
    parser.add_argument(
        "--excitation",
        "--excitations",
        action="append",
        default=None,
        help=(
            "Retrain only correction-data networks for selected excitation names or IDs; "
            "repeat or separate with commas."
        ),
    )
    parser.add_argument(
        "--channel",
        "--channels",
        action="append",
        default=None,
        help=(
            "Retrain only correction-data networks for selected channel names or IDs; "
            "repeat or separate with commas."
        ),
    )
    parser.add_argument(
        "--component",
        "--components",
        action="append",
        default=None,
        help=(
            "Retrain only selected FDR components in place in the new version. "
            "Available: frequency, damping, residue_magnitude, residue_phase."
        ),
    )
    return parser


def main():
    args = _parser().parse_args()
    if int(args.parallel_workers) <= 0:
        raise ValueError("--parallel_workers must be positive")
    base_dir = get_exe_dir()
    with (base_dir / "user_input.dat").open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle]
    if len(lines) < 3:
        raise ValueError("user_input.dat needs at least 3 lines: paths.ini, base_model_version, new_model_version")

    paths_ini_path = Path(lines[0]).expanduser()
    cp = configparser.ConfigParser()
    cp.read(paths_ini_path, encoding="utf-8")
    data_dir = _path_from_ini(cp, paths_ini_path, "CG")
    model_dir = _path_from_ini(cp, paths_ini_path, "MX_XGD")
    base_model_version, new_model_version = lines[1], lines[2]
    base_version_dir = model_dir / base_model_version
    new_version_dir = model_dir / new_model_version
    if not base_version_dir.is_dir():
        raise FileNotFoundError(f"Parent model version does not exist: {base_version_dir}")
    validate_new_version_directory(base_version_dir, new_version_dir)
    parent_inventory = load_model_inventory(base_version_dir)
    parent_models = inventory_index(parent_inventory)
    if args.device is None:
        args.device = "auto"
    args.kfold = 0
    args.debug_overfit_n = 0
    args.debug_overfit_offset = 0

    sim_dataset_path = base_dir / "sim_dataset"
    if not sim_dataset_path.is_dir():
        raise FileNotFoundError(
            f"Multifidelity correction requires sim dataset: {sim_dataset_path}"
        )
    sim_rows = read_real_dataset(sim_dataset_path, allow_duplicate_keys=True)
    sim_parsed, sim_stats = parse_rows_to_struct(sim_rows)

    cfg, _representative_parent_training = _restore_parent_configuration(
        base_version_dir, parent_inventory
    )
    _apply_cli_overrides(cfg, args)
    if cfg.pole_target != "freq_damping" or cfg.residue_target != "polar":
        raise ValueError(
            "FDR split training requires pole_target='freq_damping' and residue_target='polar'"
        )

    def resolve_user_path(raw: str) -> Path:
        candidate = Path(raw).expanduser()
        return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()

    direct_paths = [resolve_user_path(raw) for raw in lines[3:] if raw]
    if direct_paths:
        missing = [path for path in direct_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Direct correction data files do not exist: {missing}")
        correction_mode = "direct_dataset"
        correction_paths = direct_paths
    else:
        correction_mode = "cg_oe"
        correction_paths = list(
            batch_process_split(
                data_dir,
                base_dir,
                base_dir / "selected_CG_path.txt",
                reserved_names_dir=(base_version_dir / "dataset_snapshot"),
            )
        )
    print(f"[CORRECTION] mode={correction_mode} files={len(correction_paths)}", flush=True)

    history_dataset = None
    if args.history_dataset:
        history_dataset = resolve_user_path(args.history_dataset)
    elif not (base_version_dir / "dataset_snapshot").is_dir():
        configured = str(getattr(cfg, "incremental_history_dataset_dir", "") or "").strip()
        history_dataset = resolve_user_path(configured) if configured else None
    parent_source, parent_manifest = resolve_parent_dataset(base_version_dir, history_dataset)
    parent_rows = read_real_dataset(parent_source)
    correction_rows = []
    correction_group_owner: dict[tuple[int, int, int], Path] = {}
    for path in correction_paths:
        file_rows = read_real_dataset(path)
        file_groups = group_rows(file_rows, source=str(path))
        for key in file_groups:
            previous = correction_group_owner.get(key)
            if previous is not None:
                raise ValueError(
                    f"Correction group {list(key)} is repeated across files: {previous} and {path}"
                )
            correction_group_owner[key] = path
        correction_rows.extend(file_rows)
    group_rows(correction_rows, source="selected correction files")
    merged_rows, merge_info = merge_dataset_rows(parent_rows, correction_rows)
    parsed_rows, stats = parse_rows_to_struct(merged_rows)
    cfg.embed_channel_num = max(
        int(cfg.embed_channel_num), int(stats["channel_num"]), int(sim_stats["channel_num"])
    )
    cfg.embed_excitation_num = max(
        int(cfg.embed_excitation_num), int(stats["excitation_num"]), int(sim_stats["excitation_num"])
    )

    correction_keys = set(merge_info["correction_group_keys"])
    if parent_manifest is not None:
        parent_train = {tuple(key) for key in parent_manifest["splits"]["train_group_keys"]}
        parent_validation = {tuple(key) for key in parent_manifest["splits"]["validation_group_keys"]}
        train_keys = parent_train | correction_keys
        removed_from_validation = parent_validation & correction_keys
        validation_keys = parent_validation - correction_keys
    else:
        parent_parsed, _ = parse_rows_to_struct(parent_rows)
        legacy_train, legacy_validation, _ = split_single_fidelity_rows(parent_parsed, cfg, 0, 0)
        parent_train = {group_key_from_parsed(row) for row in legacy_train}
        parent_validation = {group_key_from_parsed(row) for row in legacy_validation}
        train_keys = parent_train | correction_keys
        removed_from_validation = parent_validation & correction_keys
        validation_keys = parent_validation - correction_keys
    all_keys = set(group_rows(merged_rows, source="merged dataset"))
    train_keys = (train_keys & all_keys) | correction_keys | (all_keys - train_keys - validation_keys)
    validation_keys &= all_keys
    if args.epochs_after_parent_loss < 0:
        raise ValueError("--epochs_after_parent_loss must be non-negative")

    build_dataset_snapshot(merged_rows, new_version_dir / "dataset_snapshot")
    replaced_records = [
        record for record in merge_info["change_records"] if "old_sha256" in record
    ]
    added_records = [
        record for record in merge_info["change_records"] if "old_sha256" not in record
    ]
    source_types = {
        key: ("corrected" if key in set(merge_info["replaced_group_keys"]) else "added")
        for key in correction_keys
    }
    manifest = build_dataset_manifest(
        rows=merged_rows,
        model_version=new_model_version,
        snapshot_dir=new_version_dir / "dataset_snapshot",
        train_group_keys=train_keys,
        validation_group_keys=validation_keys,
        correction_group_keys=correction_keys,
        parent_manifest=parent_manifest,
        parent_model_version=base_model_version,
        source_types=source_types,
        changes={
            "replaced_groups": replaced_records,
            "added_groups": added_records,
            "removed_groups": [],
        },
        split_strategy=(
            "all_data_training_parent_loss_early_stop"
            if not validation_keys
            else "inherited_parent_manifest"
        ),
    )
    manifest["split_changes"]["removed_from_validation"] = [
        list(key) for key in sorted(removed_from_validation)
    ]
    atomic_write_json(new_version_dir / "dataset_manifest.json", manifest)
    sim_relative_files = build_input_dataset_snapshot(
        sim_dataset_path,
        new_version_dir / "sim_dataset_snapshot",
        allow_duplicate_keys=True,
    )
    sim_keys = sorted({group_key_from_parsed(row) for row in sim_parsed})
    sim_manifest = build_dataset_manifest(
        rows=sim_rows,
        model_version=f"{new_model_version}_sim",
        snapshot_dir=new_version_dir / "sim_dataset_snapshot",
        train_group_keys=sim_keys,
        validation_group_keys=[],
        split_strategy="sim_all_rows",
        relative_files_by_group=sim_relative_files,
        allow_duplicate_keys=True,
    )
    atomic_write_json(new_version_dir / "sim_dataset_manifest.json", sim_manifest)

    target_names = [name for names in FDR_COMPONENT_TARGETS.values() for name in names]
    inherited_models = copy_parent_model_directories(
        base_version_dir, new_version_dir, parent_inventory
    )
    result_models = {
        (int(item["excitation_id"]), int(item["channel_id"])): item
        for item in inherited_models
    }
    failures = []
    correction_parsed, _ = parse_rows_to_struct(correction_rows)
    correction_combinations = combinations_in_rows(correction_parsed)
    selected_excitations = _resolve_id_selection(
        args.excitation, EXCITATION_NAMES, "excitation"
    )
    selected_channels = _resolve_id_selection(
        args.channel,
        {index: name for index, name in enumerate(CHANNEL_NAMES)},
        "channel",
    )
    selected_components = _resolve_component_selection(args.component)
    components_to_train = tuple(
        component
        for component in FDR_COMPONENT_TARGETS
        if selected_components is None or component in selected_components
    )
    affected_combinations = _filter_combinations(
        correction_combinations, selected_excitations, selected_channels
    )
    if not affected_combinations:
        raise RuntimeError(
            "The excitation/channel selectors matched no combinations in the correction "
            "data listed on user_input.dat line 4 and later"
        )
    print(
        f"[SELECTION] correction_combinations={len(affected_combinations)}/"
        f"{len(correction_combinations)} "
        f"excitations={sorted(selected_excitations) if selected_excitations is not None else 'all'} "
        f"channels={sorted(selected_channels) if selected_channels is not None else 'all'} "
        f"components={list(components_to_train)}",
        flush=True,
    )
    tasks: list[dict] = []
    combination_states: dict[tuple[int, int], dict] = {}
    for index, (excitation_id, channel_id) in enumerate(affected_combinations, 1):
        key = (int(excitation_id), int(channel_id))
        combo_sim_rows = [
            row for row in sim_parsed
            if int(row.exc_id) == excitation_id and int(row.ch_id) == channel_id
        ]
        combo_rows = [
            row for row in parsed_rows
            if int(row.exc_id) == excitation_id and int(row.ch_id) == channel_id
        ]
        label = f"{excitation_name(excitation_id)}/{channel_name(channel_id)}"
        final_dir = combination_dir(new_version_dir, excitation_id, channel_id)
        training_dir = final_dir.parent / f".{final_dir.name}.training"
        if training_dir.exists():
            shutil.rmtree(training_dir)
        print(
            f"========== Correction model {index}/{len(affected_combinations)}: "
            f"{label}, merged_rows={len(combo_rows)} ==========",
            flush=True,
        )
        try:
            parent_entry = parent_models.get(key)
            if parent_entry is not None:
                combo_cfg, _parent_training = _restore_parent_configuration(
                    base_version_dir, {"models": [parent_entry]}
                )
            else:
                combo_cfg, _parent_training = copy.deepcopy(cfg), None
            _apply_cli_overrides(combo_cfg, args)
            combo_cfg.validation_metric_scope = "all"
            available_keys = {group_key_from_parsed(row) for row in combo_rows}
            combo_train_keys = set(train_keys) & available_keys
            combo_validation_keys = set(validation_keys) & available_keys
            if not combo_sim_rows:
                raise RuntimeError("No simulation data is available for this combination")
            if not combo_rows or not combo_train_keys:
                raise RuntimeError("No merged training data is available for this combination")

            if combo_validation_keys:
                combo_cfg.explicit_train_group_keys = sorted(combo_train_keys)
                combo_cfg.explicit_validation_group_keys = sorted(combo_validation_keys)
                combo_cfg.require_validation = True
            else:
                combo_cfg.checkpoint_selection = "training"
                combo_cfg.explicit_train_group_keys = sorted(available_keys)
                combo_cfg.explicit_validation_group_keys = []
                combo_cfg.require_validation = False
                parent_fidelity = (
                    None
                    if parent_entry is None
                    else str(parent_entry.get("fidelity_mode", "multifidelity"))
                )
                if parent_entry is not None and parent_fidelity == "multifidelity":
                    parent_loss_targets = _parent_component_train_losses(
                        base_version_dir, parent_entry
                    )
                    combo_cfg.parent_train_loss_targets = parent_loss_targets
                    combo_cfg.epochs_after_parent_loss = int(args.epochs_after_parent_loss)
                    print(
                        f"[CORRECTION] {label}: all rows train; component train_loss targets="
                        f"{parent_loss_targets}, then {combo_cfg.epochs_after_parent_loss} epochs.",
                        flush=True,
                    )
                else:
                    # A simulation-only parent has no comparable real-data loss.
                    # When real samples first arrive, train the new correction head
                    # for the configured stage-2 epochs and select by real train loss.
                    if hasattr(combo_cfg, "parent_train_loss_targets"):
                        delattr(combo_cfg, "parent_train_loss_targets")
                    print(
                        f"[CORRECTION] {label}: upgrading "
                        f"{parent_fidelity or 'new'} model to multifidelity; "
                        "no real validation or comparable parent real-loss target, "
                        "so stage 2 selects by real training loss.",
                        flush=True,
                    )

            combo_correction_keys = {
                group_key_from_parsed(row) for row in correction_parsed
                if int(row.exc_id) == excitation_id and int(row.ch_id) == channel_id
            }
            dataset_metadata = {
                "manifest": manifest,
                "dataset_manifest_path": "../../../dataset_manifest.json",
                "training_mode": "correction_full_retrain",
                "sim_dataset_fingerprint": sim_manifest["dataset_fingerprint"],
                "fidelity_mode": "multifidelity",
                "prediction_output": "high",
                "model_filename": "model.pth",
                "parent_model_version": base_model_version,
                "parent_dataset_fingerprint": None if parent_manifest is None else parent_manifest.get("dataset_fingerprint"),
                "excitation_id": excitation_id,
                "excitation_name": excitation_name(excitation_id),
                "channel_id": channel_id,
                "corrected_group_keys": [list(key) for key in sorted(combo_correction_keys)],
                "replaced_group_keys": [list(key) for key in merge_info["replaced_group_keys"]],
                "added_group_keys": [list(key) for key in merge_info["added_group_keys"]],
                "selected_components": list(components_to_train),
                "target_names": target_names,
            }
            combination_states[key] = {
                "label": label,
                "final_dir": final_dir,
                "training_dir": training_dir,
                "has_parent": parent_entry is not None,
                "row_count": len(combo_rows),
                "sim_row_count": len(combo_sim_rows),
                "real_row_count": len(combo_rows),
                "components": tuple(components_to_train),
                "results": {},
                "finalized": False,
            }
            for component in components_to_train:
                print(
                    f"[CORRECTION-QUEUE] full retrain {label}/{component} from merged data",
                    flush=True,
                )
                tasks.append(
                    {
                        "cfg": combo_cfg,
                        "sim_parsed": combo_sim_rows,
                        "real_parsed": combo_rows,
                        "model_version": new_model_version,
                        "version_dir": str(training_dir),
                        "args": args,
                        "dataset_metadata": dataset_metadata,
                        "excitation_id": excitation_id,
                        "channel_id": channel_id,
                        "component": component,
                    }
                )
        except Exception as exc:
            if training_dir.exists():
                shutil.rmtree(training_dir)
            failures.append(
                {
                    "excitation_id": excitation_id,
                    "excitation_name": excitation_name(excitation_id),
                    "channel_id": channel_id,
                    "row_count": len(combo_rows),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "parent_model_retained": key in result_models,
                }
            )
            print(f"[SKIP] {label}: {type(exc).__name__}: {exc}", flush=True)

    def write_incremental_inventory():
        write_model_inventory(
            new_version_dir,
            model_version=new_model_version,
            parent_model_version=base_model_version,
            models=result_models.values(),
            failures=failures,
        )

    def finalize_combination(key: tuple[int, int]):
        state = combination_states[key]
        if state["finalized"]:
            return
        state["finalized"] = True
        failed_results = [
            result
            for result in state["results"].values()
            if not result.get("ok")
        ]
        training_dir = Path(state["training_dir"])
        if failed_results:
            if training_dir.exists():
                shutil.rmtree(training_dir)
            details = "; ".join(
                f"{result['component']}: {result.get('error_type', 'WorkerError')}: "
                f"{result.get('error', 'unknown worker failure')}"
                for result in failed_results
            )
            failures.append(
                {
                    "excitation_id": key[0],
                    "excitation_name": excitation_name(key[0]),
                    "channel_id": key[1],
                    "components": [result["component"] for result in failed_results],
                    "row_count": state["row_count"],
                    "error_type": "ComponentTrainingError",
                    "error": details,
                    "traceback": "\n".join(
                        result.get("traceback", "") for result in failed_results
                    ),
                    "parent_model_retained": key in result_models,
                }
            )
            print(
                f"[CORRECTION-FAIL] {state['label']}: {details}; "
                f"parent_retained={key in result_models}",
                flush=True,
            )
            return

        try:
            _replace_selected_component_directories(
                training_dir, state["final_dir"], state["components"]
            )
            if training_dir.exists():
                shutil.rmtree(training_dir)

            # A copied parent bundle remains complete after partial replacement.
            # A brand-new combination is deployable only after all four components.
            if state["has_parent"] or set(state["components"]) == set(FDR_COMPONENT_TARGETS):
                result_models[key] = model_entry(
                    excitation_id=key[0],
                    channel_id=key[1],
                    row_count=state["row_count"],
                    status="corrected",
                    source_model_version=base_model_version,
                    fidelity_mode="multifidelity",
                    sim_row_count=state["sim_row_count"],
                    real_row_count=state["real_row_count"],
                )
                print(f"[CORRECTION-OK] {state['label']}", flush=True)
            else:
                result_models.pop(key, None)
                print(
                    f"[MODEL-INCOMPLETE] {state['label']}: trained "
                    f"{list(state['components'])} but no parent bundle supplies the "
                    "unselected components; omitted from models_manifest.json",
                    flush=True,
                )
        except Exception as exc:
            if training_dir.exists():
                shutil.rmtree(training_dir)
            failures.append(
                {
                    "excitation_id": key[0],
                    "excitation_name": excitation_name(key[0]),
                    "channel_id": key[1],
                    "components": list(state["components"]),
                    "row_count": state["row_count"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "parent_model_retained": key in result_models,
                }
            )
            print(
                f"[CORRECTION-COMMIT-FAIL] {state['label']}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

    write_incremental_inventory()
    print(
        f"[PARALLEL] workers={args.parallel_workers} "
        f"pending_components={len(tasks)} combinations={len(combination_states)}",
        flush=True,
    )
    completed_tasks = 0
    for result in _run_incremental_component_tasks(tasks, args.parallel_workers):
        key = (int(result["excitation_id"]), int(result["channel_id"]))
        component = str(result["component"])
        state = combination_states[key]
        state["results"][component] = result
        completed_tasks += 1
        if result.get("ok"):
            print(f"[TASK-OK] {state['label']}/{component}", flush=True)
        else:
            print(
                f"[TASK-FAIL] {state['label']}/{component}: "
                f"{result.get('error_type', 'WorkerError')}: "
                f"{result.get('error', 'unknown worker failure')}",
                flush=True,
            )
        if len(state["results"]) == len(state["components"]):
            finalize_combination(key)
            write_incremental_inventory()
        print(
            f"[PROGRESS] components={completed_tasks}/{len(tasks)} "
            f"combinations={sum(item['finalized'] for item in combination_states.values())}/"
            f"{len(combination_states)}",
            flush=True,
        )

    write_incremental_inventory()
    print(
        f"[MODEL-INVENTORY] total={len(result_models)} corrected="
        f"{sum(item.get('status') == 'corrected' for item in result_models.values())} "
        f"skipped={len(failures)} path={new_version_dir / 'models_manifest.json'}",
        flush=True,
    )
    _send_finished_udp()


if __name__ == "__main__":
    main()
