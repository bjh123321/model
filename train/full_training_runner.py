from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.distributed as dist

from surrogate_model.train.loops import train_one_fold
from surrogate_model.utils.dist_utils import is_main_process
from surrogate_model.utils.excitation_channel_models import FDR_COMPONENT_TARGETS
from surrogate_model.utils.model_artifacts import atomic_write_json, load_json, read_code_metadata, sha256_file


def _model_config(cfg) -> dict:
    return {
        "model_type": "TargetMFDNN",
        "target_component": str(getattr(cfg, "target_component", "")),
        "model_target_names": list(getattr(cfg, "model_target_names", [])),
        "K_max": int(cfg.K_max),
        "low_hidden": list(cfg.low_hidden),
        "dropout_prob": float(cfg.dropout_prob),
        "embed_excitation_num": int(cfg.embed_excitation_num),
        "embed_excitation_dim": int(cfg.embed_excitation_dim),
        "embed_channel_num": int(cfg.embed_channel_num),
        "embed_channel_dim": int(cfg.embed_channel_dim),
        "use_modulation": bool(cfg.use_modulation),
        "modulation_type": "single_linear_hm",
        "mod_gamma": float(cfg.mod_gamma),
        "use_lora": False,
    }


def _training_values(cfg, args) -> dict:
    requested_device = str(getattr(args, "device", "auto"))
    actual_device = "cuda" if requested_device != "cpu" and torch.cuda.is_available() else "cpu"
    values = {
        "optimizer": "AdamW",
        "learning_rate": float(cfg.lr_low_adam),
        "weight_decay": float(cfg.weight_decay),
        "batch_size": int(cfg.batch_real),
        "epochs": int(cfg.epochs_stage1),
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "monitor": "training_loss",
            "mode": "min",
            "factor": 0.5,
            "patience": 20,
            "min_lr": 1e-6,
        },
        "gradient_clip_norm": 5.0,
        "seed": int(getattr(cfg, "val_seed", 42)),
        "num_workers": int(cfg.num_workers),
        "prefetch": int(cfg.prefetch),
        "device": actual_device,
        "checkpoint_selection": str(cfg.checkpoint_selection),
        "early_stop_patience": int(cfg.early_stop_patience_s1),
        "early_stop_min_delta": float(cfg.early_stop_min_delta),
        "validation_split": {
            "strategy": str(getattr(cfg, "val_split_strategy", "")),
            "ratio": float(getattr(cfg, "val_ratio", 0.0)),
            "seed": int(getattr(cfg, "val_seed", 42)),
        },
    }
    parent_loss_target = getattr(cfg, "parent_train_loss_target", None)
    if parent_loss_target is not None:
        values["parent_loss_early_stop"] = {
            "target_loss": float(parent_loss_target),
            "epochs_after_target": int(getattr(cfg, "epochs_after_parent_loss", 0)),
            "target_reached_epoch": getattr(cfg, "_last_training_result", {}).get(
                "parent_loss_reached_epoch"
            ),
        }
    return values


def build_checkpoint_metadata(*, model_version: str, dataset_metadata: Mapping) -> dict:
    manifest = dataset_metadata["manifest"]
    training_mode = str(dataset_metadata.get("training_mode", "base_full_training"))
    correction = training_mode == "correction_full_retrain"
    metadata = {
        "stage": (
            "fixed_single_fidelity_correction_full_retrain"
            if correction
            else "fixed_single_fidelity"
        ),
        "training_mode": training_mode,
        "initialization_mode": "random",
        "model_version": model_version,
        "parent_model_version": dataset_metadata.get("parent_model_version"),
        "parent_checkpoint_used_for_weights": False,
        "use_lora": False,
        "model_type": "TargetMFDNN",
        "target_component": str(dataset_metadata.get("target_component", "")),
        "dataset_manifest": str(dataset_metadata.get("dataset_manifest_path", "dataset_manifest.json")),
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "parent_dataset_fingerprint": dataset_metadata.get("parent_dataset_fingerprint"),
        "dataset_snapshot_root": "dataset_snapshot",
        "training_config": "training_config.json",
        "normalizer_input": "Normalizer_Input.pth",
        "normalizer_output": "Normalizer_Output.pth",
        "corrected_group_keys": dataset_metadata.get("corrected_group_keys", []),
        "replaced_group_keys": dataset_metadata.get("replaced_group_keys", []),
        "added_group_keys": dataset_metadata.get("added_group_keys", []),
    }
    if "excitation_id" in dataset_metadata:
        metadata["excitation_id"] = int(dataset_metadata["excitation_id"])
        metadata["excitation_name"] = str(dataset_metadata.get("excitation_name", ""))
    if "channel_id" in dataset_metadata:
        metadata["channel_id"] = int(dataset_metadata["channel_id"])
    return metadata


def write_training_config(
    *,
    cfg,
    args,
    model_version: str,
    version_dir: Path,
    dataset_metadata: Mapping,
) -> dict:
    """Write the standard provenance/config artifact from effective training values."""
    version_dir = Path(version_dir)
    manifest = dataset_metadata["manifest"]
    training_mode = str(dataset_metadata.get("training_mode", "base_full_training"))
    input_path = version_dir / "Normalizer_Input.pth"
    output_path = version_dir / "Normalizer_Output.pth"
    training_config = {
        "schema_version": 1,
        "model_version": model_version,
        "parent_model_version": dataset_metadata.get("parent_model_version"),
        "training_mode": training_mode,
        "excitation_id": dataset_metadata.get("excitation_id"),
        "excitation_name": dataset_metadata.get("excitation_name"),
        "channel_id": dataset_metadata.get("channel_id"),
        "initialization": {
            "mode": "random",
            "parent_checkpoint_used_for_weights": False,
        },
        "dataset": {
            "manifest": str(dataset_metadata.get("dataset_manifest_path", "dataset_manifest.json")),
            "dataset_fingerprint": manifest["dataset_fingerprint"],
            "parent_dataset_fingerprint": dataset_metadata.get("parent_dataset_fingerprint"),
            "snapshot_root": "dataset_snapshot",
        },
        "model": _model_config(cfg),
        "targets": {
            "pole_target": str(cfg.pole_target),
            "residue_target": str(cfg.residue_target),
            "target_names": list(dataset_metadata.get("target_names", [])),
        },
        "normalizers": {
            "input_file": "Normalizer_Input.pth",
            "output_file": "Normalizer_Output.pth",
            "fit_on": "training_split_only",
            "input_sha256": sha256_file(input_path),
            "output_sha256": sha256_file(output_path),
        },
        "training": _training_values(cfg, args),
        "selection": dict(getattr(cfg, "_last_training_result", {})),
        "code": read_code_metadata(Path(__file__).resolve().parents[1]),
    }
    atomic_write_json(version_dir / "training_config.json", training_config)
    return training_config


def run_full_training(
    *,
    cfg,
    parsed_rows,
    model_version: str,
    version_dir: Path,
    args,
    dataset_metadata: Mapping,
):
    """Train the four independent FDR component networks for one excitation/channel."""
    version_dir = Path(version_dir)
    version_dir.mkdir(parents=True, exist_ok=True)
    parent_targets = dict(getattr(cfg, "parent_train_loss_targets", {}) or {})
    results = {}
    for component_index, (component, target_names) in enumerate(FDR_COMPONENT_TARGETS.items()):
        component_cfg = copy.deepcopy(cfg)
        component_cfg.use_lora = False
        component_cfg.model_type = "TargetMFDNN"
        component_cfg.target_component = component
        component_cfg.model_target_names = list(target_names)
        if parent_targets:
            if component not in parent_targets:
                raise RuntimeError(f"Missing parent train-loss target for FDR component {component!r}")
            component_cfg.parent_train_loss_target = float(parent_targets[component])
        elif hasattr(component_cfg, "parent_train_loss_target"):
            delattr(component_cfg, "parent_train_loss_target")

        seed = int(getattr(component_cfg, "val_seed", 42)) + component_index
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        component_dir = version_dir / component
        component_dir.mkdir(parents=True, exist_ok=True)
        component_metadata = dict(dataset_metadata)
        component_metadata.update(
            {
                "model_filename": "model.pth",
                "target_component": component,
                "target_names": list(target_names),
            }
        )

        run_args = copy.copy(args)
        run_args.out_dir = str(component_dir)
        run_args.normalizer_dir = str(component_dir)
        run_args.model_save_path = str(component_dir / "model.pth")
        run_args.train_log_path = str(component_dir / "train_log.csv")
        run_args.kfold = int(getattr(run_args, "kfold", 0) or 0)
        run_args.fold_index = 0
        run_args.num_folds = 1
        run_args.checkpoint_metadata = build_checkpoint_metadata(
            model_version=model_version, dataset_metadata=component_metadata
        )
        if is_main_process():
            print(
                f"========== FDR component {component_index + 1}/{len(FDR_COMPONENT_TARGETS)}: "
                f"{component}, targets={list(target_names)} ==========",
                flush=True,
            )
        best, train_hist, val_hist = train_one_fold(
            0, component_cfg, parsed_rows, run_args
        )

        training_config_path = component_dir / "training_config.json"
        if is_main_process():
            training_config = write_training_config(
                cfg=component_cfg,
                args=args,
                model_version=model_version,
                version_dir=component_dir,
                dataset_metadata=component_metadata,
            )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if not is_main_process():
            training_config = load_json(training_config_path)
        results[component] = {
            "best_metric": best,
            "train_history": train_hist,
            "validation_history": val_hist,
            "training_config": training_config,
        }
    return {"components": results}
