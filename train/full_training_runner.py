from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.distributed as dist

from surrogate_model.train.loops import train_multifidelity_component
from surrogate_model.utils.dist_utils import is_main_process
from surrogate_model.utils.excitation_channel_models import FDR_COMPONENT_TARGETS
from surrogate_model.utils.model_artifacts import atomic_write_json, load_json, read_code_metadata, sha256_file


def _model_config(cfg) -> dict:
    return {
        "model_type": "TargetMultifidelityMFDNN",
        "target_component": str(getattr(cfg, "target_component", "")),
        "model_target_names": list(getattr(cfg, "model_target_names", [])),
        "K_max": int(cfg.K_max),
        "low_hidden": list(cfg.low_hidden),
        "adapt_hidden": list(getattr(cfg, "adapt_hidden", [32, 32])),
        "dropout_prob": float(cfg.dropout_prob),
        "embed_excitation_num": int(cfg.embed_excitation_num),
        "embed_excitation_dim": int(cfg.embed_excitation_dim),
        "embed_channel_num": int(cfg.embed_channel_num),
        "embed_channel_dim": int(cfg.embed_channel_dim),
        "use_modulation": bool(cfg.use_modulation),
        "modulation_type": "single_linear_hm",
        "mod_gamma": float(cfg.mod_gamma),
        "use_lora": False,
        "physical_parameterization": bool(getattr(cfg, "fdr_physical_parameterization", False)),
        "multifidelity": True,
    }


def _training_values(cfg, args) -> dict:
    requested_device = str(getattr(args, "device", "auto"))
    actual_device = "cuda" if requested_device != "cpu" and torch.cuda.is_available() else "cpu"
    values = {
        "fdr_profile_version": int(
            getattr(cfg, "fdr_training_profile_version", 1)
        ),
        "optimizer": "AdamW",
        "learning_rate": float(cfg.lr_low_adam),
        "learning_rate_high": float(cfg.lr_high_adam),
        "weight_decay": float(cfg.weight_decay),
        "batch_size": int(cfg.batch_real),
        "batch_sim": int(cfg.batch_sim),
        "epochs": int(cfg.epochs_stage1),
        "epochs_stage2": int(cfg.epochs_stage2),
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
        "early_stop_patience_stage2": int(cfg.early_stop_patience_s2),
        "early_stop_min_delta": float(cfg.early_stop_min_delta),
        "checkpoint_train_loss_ceiling": getattr(
            cfg, "checkpoint_train_loss_ceiling", None
        ),
        "validation_split": {
            "strategy": str(getattr(cfg, "val_split_strategy", "")),
            "ratio": float(getattr(cfg, "val_ratio", 0.0)),
            "seed": int(getattr(cfg, "val_seed", 42)),
        },
        "damping_smoothness": {
            "slope_weight": float(getattr(cfg, "damping_slope_weight", 0.0)),
            "curvature_weight": float(getattr(cfg, "damping_curvature_weight", 0.0)),
            "slope_limit_multiplier": float(getattr(cfg, "damping_slope_limit_multiplier", 1.0)),
            "slope_limit_floor": float(getattr(cfg, "damping_slope_limit_floor", 0.0)),
        },
        "prediction_support": {
            "coordinate_scope": "real_training_rows_per_height",
            "slope_fit_scope": "real_training_rows_only",
            "default_extrapolation_policy": "allow",
            "damping_extrapolation": {
                "boundary_join": "c1_cubic_hermite",
                "tail": "straight_line_fit_on_damping_x200",
                "height_outside_training_range": "network_prediction",
                "transition_mach": 0.05,
                "slope_fit_mach": 0.05,
            },
        },
        "sim_slot_outlier_filter": {
            "enabled": bool(
                getattr(cfg, "sim_slot_outlier_filter_enabled", True)
            ),
            "fit_scope": "simulation_training_split_per_excitation_channel_slot",
            "min_samples": int(
                getattr(cfg, "sim_slot_outlier_min_samples", 8)
            ),
            "mad_threshold": float(
                getattr(cfg, "sim_slot_outlier_mad_threshold", 12.0)
            ),
            "scale_floor_ratio": float(
                getattr(cfg, "sim_slot_outlier_scale_floor_ratio", 0.02)
            ),
            "report_file": "sim_slot_outlier_filter.json",
        },
        "real_damping_slot_filter": {
            "enabled": bool(
                getattr(cfg, "real_damping_upper_filter_enabled", True)
            ),
            "comparison": "damping > excitation_channel_threshold",
            "thresholds_by_excitation": dict(
                getattr(cfg, "real_damping_upper_thresholds", {}) or {}
            ),
            "channel_ids_by_excitation": {
                int(excitation_id): list(channel_ids)
                for excitation_id, channel_ids in dict(
                    getattr(cfg, "real_damping_upper_channel_ids", {}) or {}
                ).items()
            },
            "applies_to": ["real_training_split", "real_validation_split"],
            "action": "mask_entire_slot_for_all_fdr_components",
            "log_file": "damping_slot_filter.log",
        },
        "multifidelity": {
            "real_loss_weight_stage2": float(
                getattr(cfg, "multifidelity_real_loss_weight", 1.0)
            ),
            "sim_loss_weight_stage2": float(
                getattr(cfg, "multifidelity_sim_loss_weight", 0.2)
            ),
            "freeze_low_stage2": bool(getattr(cfg, "freeze_low_stage2", False)),
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
    fidelity_mode = str(dataset_metadata.get("fidelity_mode", "multifidelity"))
    prediction_output = str(
        dataset_metadata.get(
            "prediction_output", "low" if fidelity_mode == "sim_only" else "high"
        )
    )
    metadata = {
        "stage": (
            "fixed_fdr_sim_only"
            if fidelity_mode == "sim_only"
            else "fixed_fdr_multifidelity"
        ),
        "training_mode": training_mode,
        "initialization_mode": "random",
        "model_version": model_version,
        "parent_model_version": dataset_metadata.get("parent_model_version"),
        "parent_checkpoint_used_for_weights": False,
        "use_lora": False,
        "model_type": "TargetMultifidelityMFDNN",
        "target_component": str(dataset_metadata.get("target_component", "")),
        "fidelity_mode": fidelity_mode,
        "prediction_output": prediction_output,
        "multifidelity": fidelity_mode == "multifidelity",
        "dataset_manifest": str(dataset_metadata.get("dataset_manifest_path", "dataset_manifest.json")),
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "parent_dataset_fingerprint": dataset_metadata.get("parent_dataset_fingerprint"),
        "dataset_snapshot_root": str(
            dataset_metadata.get("dataset_snapshot_root", "dataset_snapshot")
        ),
        "data_selection": dict(dataset_metadata.get("data_selection", {}) or {}),
        "training_config": "training_config.json",
        "normalizer_input": (
            "Normalizer_Input_Sim.pth"
            if fidelity_mode == "sim_only"
            else "Normalizer_Input_Real.pth"
        ),
        "normalizer_output": (
            "Normalizer_Output_Sim.pth"
            if fidelity_mode == "sim_only"
            else "Normalizer_Output_Real.pth"
        ),
        "normalizer_input_sim": "Normalizer_Input_Sim.pth",
        "normalizer_output_sim": "Normalizer_Output_Sim.pth",
        "normalizer_input_real": (
            None if fidelity_mode == "sim_only" else "Normalizer_Input_Real.pth"
        ),
        "normalizer_output_real": (
            None if fidelity_mode == "sim_only" else "Normalizer_Output_Real.pth"
        ),
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
    fidelity_mode = str(dataset_metadata.get("fidelity_mode", "multifidelity"))
    input_file = (
        "Normalizer_Input_Sim.pth"
        if fidelity_mode == "sim_only"
        else "Normalizer_Input_Real.pth"
    )
    output_file = (
        "Normalizer_Output_Sim.pth"
        if fidelity_mode == "sim_only"
        else "Normalizer_Output_Real.pth"
    )
    input_path = version_dir / input_file
    output_path = version_dir / output_file
    sim_input_path = version_dir / "Normalizer_Input_Sim.pth"
    sim_output_path = version_dir / "Normalizer_Output_Sim.pth"
    real_input_path = version_dir / "Normalizer_Input_Real.pth"
    real_output_path = version_dir / "Normalizer_Output_Real.pth"
    training_config = {
        "schema_version": 2,
        "model_version": model_version,
        "parent_model_version": dataset_metadata.get("parent_model_version"),
        "training_mode": training_mode,
        "excitation_id": dataset_metadata.get("excitation_id"),
        "excitation_name": dataset_metadata.get("excitation_name"),
        "channel_id": dataset_metadata.get("channel_id"),
        "fidelity_mode": fidelity_mode,
        "prediction_output": str(
            dataset_metadata.get(
                "prediction_output", "low" if fidelity_mode == "sim_only" else "high"
            )
        ),
        "initialization": {
            "mode": "random",
            "parent_checkpoint_used_for_weights": False,
        },
        "dataset": {
            "manifest": str(dataset_metadata.get("dataset_manifest_path", "dataset_manifest.json")),
            "dataset_fingerprint": manifest["dataset_fingerprint"],
            "parent_dataset_fingerprint": dataset_metadata.get("parent_dataset_fingerprint"),
            "sim_dataset_fingerprint": dataset_metadata.get("sim_dataset_fingerprint"),
            "snapshot_root": str(
                dataset_metadata.get("dataset_snapshot_root", "dataset_snapshot")
            ),
            "data_selection": dict(dataset_metadata.get("data_selection", {}) or {}),
        },
        "model": _model_config(cfg),
        "targets": {
            "pole_target": str(cfg.pole_target),
            "residue_target": str(cfg.residue_target),
            "target_names": list(dataset_metadata.get("target_names", [])),
            "target_transforms": dict(getattr(cfg, "_target_transforms", {}) or {}),
        },
        "slot_support": dict(getattr(cfg, "_slot_support_metadata", {}) or {}),
        "normalizers": {
            "input_file": input_file,
            "output_file": output_file,
            "fit_on": "training_split_only",
            "input_sha256": sha256_file(input_path),
            "output_sha256": sha256_file(output_path),
            "sim_input_file": "Normalizer_Input_Sim.pth",
            "sim_output_file": "Normalizer_Output_Sim.pth",
            "sim_input_sha256": sha256_file(sim_input_path),
            "sim_output_sha256": sha256_file(sim_output_path),
            "real_input_file": (
                None if fidelity_mode == "sim_only" else "Normalizer_Input_Real.pth"
            ),
            "real_output_file": (
                None if fidelity_mode == "sim_only" else "Normalizer_Output_Real.pth"
            ),
            "real_input_sha256": (
                None if fidelity_mode == "sim_only" else sha256_file(real_input_path)
            ),
            "real_output_sha256": (
                None if fidelity_mode == "sim_only" else sha256_file(real_output_path)
            ),
            "input_coordinate_policy": "shared_simulation_coordinates",
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
    parsed_rows=None,
    sim_parsed=None,
    real_parsed=None,
    model_version: str,
    version_dir: Path,
    args,
    dataset_metadata: Mapping,
):
    """Train the four independent FDR component networks for one excitation/channel."""
    version_dir = Path(version_dir)
    version_dir.mkdir(parents=True, exist_ok=True)
    if sim_parsed is None:
        raise RuntimeError("FDR multifidelity training requires sim_parsed")
    if real_parsed is None:
        real_parsed = parsed_rows or []
    results = {}
    for component in FDR_COMPONENT_TARGETS:
        results[component] = run_component_training(
            cfg=cfg,
            sim_parsed=sim_parsed,
            real_parsed=real_parsed,
            model_version=model_version,
            version_dir=version_dir,
            args=args,
            dataset_metadata=dataset_metadata,
            component=component,
        )
    return {"components": results}


def run_component_training(
    *,
    cfg,
    parsed_rows=None,
    sim_parsed=None,
    real_parsed=None,
    model_version: str,
    version_dir: Path,
    args,
    dataset_metadata: Mapping,
    component: str,
):
    """Train one independently resumable FDR component network."""
    if component not in FDR_COMPONENT_TARGETS:
        raise ValueError(
            f"Unsupported FDR component {component!r}; expected {list(FDR_COMPONENT_TARGETS)}"
        )
    component_index = list(FDR_COMPONENT_TARGETS).index(component)
    target_names = FDR_COMPONENT_TARGETS[component]
    version_dir = Path(version_dir)
    component_dir = version_dir / component
    component_dir.mkdir(parents=True, exist_ok=True)

    component_cfg = copy.deepcopy(cfg)
    component_cfg.use_lora = False
    component_cfg.model_type = "TargetMultifidelityMFDNN"
    component_cfg.target_component = component
    component_cfg.model_target_names = list(target_names)
    component_cfg.fdr_physical_parameterization = True
    profile_versions = dict(
        getattr(component_cfg, "fdr_component_training_profile_version", {}) or {}
    )
    component_cfg.fdr_training_profile_version = int(
        profile_versions.get(component, component_cfg.fdr_training_profile_version)
    )
    if hasattr(args, "epochs"):
        if (
            getattr(args, "epochs", None) is None
            and getattr(args, "epochs_s1", None) is None
        ):
            profiles = dict(
                getattr(component_cfg, "fdr_component_epochs_stage1", {}) or {}
            )
            component_cfg.epochs_stage1 = int(
                profiles.get(component, component_cfg.epochs_stage1)
            )
        if (
            getattr(args, "epochs", None) is None
            and getattr(args, "epochs_s2", None) is None
        ):
            profiles = dict(
                getattr(component_cfg, "fdr_component_epochs_stage2", {}) or {}
            )
            component_cfg.epochs_stage2 = int(
                profiles.get(component, component_cfg.epochs_stage2)
            )
    if getattr(args, "low_hidden", None) is None:
        profiles = dict(getattr(component_cfg, "fdr_component_low_hidden", {}) or {})
        component_cfg.low_hidden = list(
            profiles.get(component, component_cfg.fdr_low_hidden)
        )
    if getattr(args, "dropout_prob", None) is None:
        profiles = dict(getattr(component_cfg, "fdr_component_dropout_prob", {}) or {})
        component_cfg.dropout_prob = float(
            profiles.get(component, component_cfg.fdr_dropout_prob)
        )
    if getattr(args, "weight_decay", None) is None:
        profiles = dict(getattr(component_cfg, "fdr_component_weight_decay", {}) or {})
        component_cfg.weight_decay = float(
            profiles.get(component, component_cfg.fdr_weight_decay)
        )
    if getattr(args, "early_stop_patience", None) is None:
        profiles = dict(
            getattr(component_cfg, "fdr_component_early_stop_patience", {}) or {}
        )
        component_cfg.early_stop_patience_s1 = int(
            profiles.get(component, component_cfg.fdr_early_stop_patience)
        )
    if getattr(args, "checkpoint_train_loss_ceiling", None) is None:
        profiles = dict(
            getattr(
                component_cfg,
                "fdr_component_checkpoint_train_loss_ceiling",
                {},
            )
            or {}
        )
        ceiling = profiles.get(
            component,
            component_cfg.fdr_checkpoint_train_loss_ceiling,
        )
        component_cfg.checkpoint_train_loss_ceiling = (
            None if ceiling is None else float(ceiling)
        )
    if getattr(args, "checkpoint_selection", None) is None:
        profiles = dict(
            getattr(component_cfg, "fdr_component_checkpoint_selection", {}) or {}
        )
        component_cfg.checkpoint_selection = str(
            profiles.get(component, component_cfg.checkpoint_selection)
        ).lower()
    if getattr(args, "real_loss_weight", None) is None:
        profiles = dict(
            getattr(component_cfg, "fdr_component_real_loss_weight", {}) or {}
        )
        component_cfg.multifidelity_real_loss_weight = float(
            profiles.get(component, component_cfg.multifidelity_real_loss_weight)
        )
    if getattr(args, "sim_loss_weight", None) is None:
        profiles = dict(
            getattr(component_cfg, "fdr_component_sim_loss_weight", {}) or {}
        )
        component_cfg.multifidelity_sim_loss_weight = float(
            profiles.get(component, component_cfg.multifidelity_sim_loss_weight)
        )
    if str(component_cfg.checkpoint_selection).lower() not in {"training", "validation"}:
        raise ValueError(
            f"Unsupported checkpoint selection for {component}: "
            f"{component_cfg.checkpoint_selection!r}"
        )
    if (
        not np.isfinite(component_cfg.multifidelity_real_loss_weight)
        or component_cfg.multifidelity_real_loss_weight < 0.0
    ):
        raise ValueError(f"Real loss weight for {component} must be finite and non-negative")
    if (
        not np.isfinite(component_cfg.multifidelity_sim_loss_weight)
        or component_cfg.multifidelity_sim_loss_weight < 0.0
    ):
        raise ValueError(
            f"Simulation loss weight for {component} must be finite and non-negative"
        )
    if not bool(getattr(component_cfg, "require_validation", True)):
        component_cfg.checkpoint_selection = "training"
    parent_targets = dict(getattr(component_cfg, "parent_train_loss_targets", {}) or {})
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

    if sim_parsed is None:
        raise RuntimeError("FDR multifidelity component training requires sim_parsed")
    if real_parsed is None:
        real_parsed = parsed_rows or []
    sim_parsed = list(sim_parsed)
    real_parsed = list(real_parsed)
    fidelity_mode = "multifidelity" if real_parsed else "sim_only"

    component_metadata = dict(dataset_metadata)
    component_metadata.update(
        {
            "model_filename": "model.pth",
            "target_component": component,
            "target_names": list(target_names),
            "fidelity_mode": fidelity_mode,
            "prediction_output": "high" if real_parsed else "low",
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
    run_args.fidelity_normalizer_names = True
    run_args.checkpoint_metadata = build_checkpoint_metadata(
        model_version=model_version, dataset_metadata=component_metadata
    )
    if is_main_process():
        print(
            f"========== FDR component {component_index + 1}/{len(FDR_COMPONENT_TARGETS)}: "
            f"{component}, targets={list(target_names)} ==========",
            flush=True,
        )
    best, train_hist, val_hist = train_multifidelity_component(
        0, component_cfg, sim_parsed, real_parsed, run_args
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
    return {
        "best_metric": best,
        "train_history": train_hist,
        "validation_history": val_hist,
        "training_config": training_config,
    }
