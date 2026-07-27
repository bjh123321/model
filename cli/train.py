from __future__ import annotations

import argparse
import configparser
import copy
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.distributed as dist

from surrogate_model.config import Config
from surrogate_model.data.io import parse_rows_to_struct, read_real_dataset
from surrogate_model.data.normalizer import canonical_pole_target, canonical_residue_target
from surrogate_model.train.dataloaders import split_single_fidelity_rows
from surrogate_model.train.full_training_runner import run_component_training
from surrogate_model.utils.argparse_compat import BooleanOptionalAction
from surrogate_model.utils.exe_runtime import run_executable, split_user_input_arg
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


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--:--"
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _parse_int_list(raw: str) -> List[int]:
    values = []
    for part in str(raw).replace(";", ",").split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def _resolve_id_selection(
    raw_values, id_to_name: Dict[int, str], label: str
):
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


def _resolve_component_selection(raw_values) -> Optional[Set[str]]:
    """Resolve repeatable comma/semicolon-separated FDR component names."""
    if not raw_values:
        return None
    tokens = [
        part.strip().lower().replace("-", "_")
        for raw in raw_values
        for part in str(raw).replace(";", ",").split(",")
        if part.strip()
    ]
    available = set(FDR_COMPONENT_TARGETS)
    invalid = sorted({token for token in tokens if token not in available})
    if invalid:
        raise ValueError(
            f"Unknown component selector(s) {invalid}; available: "
            + ", ".join(FDR_COMPONENT_TARGETS)
        )
    selected = set(tokens)
    if not selected:
        raise ValueError("At least one component selector is required")
    return selected


def _resolve_component_value_mapping(
    raw_values,
    *,
    label: str,
    value_type,
    choices=None,
    minimum: Optional[float] = None,
) -> Dict[str, object]:
    """Resolve repeatable ``component=value`` assignments."""
    if not raw_values:
        return {}
    available = set(FDR_COMPONENT_TARGETS)
    allowed = None if choices is None else {str(value).lower() for value in choices}
    resolved = {}
    for raw in raw_values:
        for assignment in str(raw).replace(";", ",").split(","):
            assignment = assignment.strip()
            if not assignment:
                continue
            if "=" not in assignment:
                raise ValueError(
                    f"Invalid {label} assignment {assignment!r}; expected component=value"
                )
            component_raw, value_raw = assignment.split("=", 1)
            component = component_raw.strip().lower().replace("-", "_")
            if component not in available:
                raise ValueError(
                    f"Unknown component in {label} assignment {component_raw!r}; available: "
                    + ", ".join(FDR_COMPONENT_TARGETS)
                )
            try:
                value = value_type(value_raw.strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid {label} value for {component}: {value_raw!r}"
                ) from exc
            if allowed is not None:
                normalized = str(value).lower()
                if normalized not in allowed:
                    raise ValueError(
                        f"Invalid {label} value for {component}: {value!r}; "
                        f"expected one of {sorted(allowed)}"
                    )
                value = normalized
            if minimum is not None:
                numeric_value = float(value)
                if not math.isfinite(numeric_value) or numeric_value < float(minimum):
                    raise ValueError(
                        f"Invalid {label} value for {component}: {value!r}; "
                        f"must be finite and >= {minimum:g}"
                    )
            resolved[component] = value
    if not resolved:
        raise ValueError(f"At least one {label} component=value assignment is required")
    return resolved


def _apply_component_training_cli(cfg: Config, args) -> None:
    checkpoint_values = _resolve_component_value_mapping(
        getattr(args, "component_checkpoint_selection", None),
        label="component checkpoint selection",
        value_type=str,
        choices=("training", "validation"),
    )
    real_values = _resolve_component_value_mapping(
        getattr(args, "component_real_loss_weight", None),
        label="component real loss weight",
        value_type=float,
        minimum=0.0,
    )
    sim_values = _resolve_component_value_mapping(
        getattr(args, "component_sim_loss_weight", None),
        label="component simulation loss weight",
        value_type=float,
        minimum=0.0,
    )
    cfg.fdr_component_checkpoint_selection.update(checkpoint_values)
    cfg.fdr_component_real_loss_weight.update(real_values)
    cfg.fdr_component_sim_loss_weight.update(sim_values)

    scalar_overrides = {
        "real_loss_weight": "multifidelity_real_loss_weight",
        "sim_loss_weight": "multifidelity_sim_loss_weight",
    }
    for source, destination in scalar_overrides.items():
        value = getattr(args, source, None)
        if value is not None:
            setattr(cfg, destination, value)
    for name in (
        "multifidelity_real_loss_weight",
        "multifidelity_sim_loss_weight",
    ):
        value = float(getattr(cfg, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")


def _filter_combinations(combinations, selected_excitations=None, selected_channels=None):
    return [
        (excitation_id, channel_id)
        for excitation_id, channel_id in combinations
        if (selected_excitations is None or excitation_id in selected_excitations)
        and (selected_channels is None or channel_id in selected_channels)
    ]


def _resolve_height_selection(raw_values) -> Optional[List[float]]:
    """Resolve repeatable comma/semicolon-separated finite height values."""
    if not raw_values:
        return None
    selected = []
    for raw in raw_values:
        for token in str(raw).replace(";", ",").split(","):
            token = token.strip()
            if not token:
                continue
            try:
                value = float(token)
            except ValueError as exc:
                raise ValueError(f"Invalid height selector {token!r}; expected a number") from exc
            if not math.isfinite(value):
                raise ValueError(f"Invalid height selector {token!r}; height must be finite")
            selected.append(value)
    if not selected:
        raise ValueError("At least one height selector is required")
    return sorted(set(selected))


def _filter_rows_by_heights(rows, selected_heights, height_tolerance: float):
    if selected_heights is None:
        return list(rows)
    tolerance = float(height_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Height-selection tolerance must be a finite non-negative value")
    return [
        row
        for row in rows
        if any(abs(float(row.H) - height) <= tolerance for height in selected_heights)
    ]


def _configure_validation_for_height_selection(cfg, selected_heights, real_parsed) -> None:
    """Keep validation usable after restricting the dataset to selected heights."""
    if selected_heights is None or not real_parsed or not bool(cfg.use_val_split):
        return
    tolerance = float(cfg.val_height_tol)
    real_heights = [
        height
        for height in selected_heights
        if any(abs(float(row.H) - height) <= tolerance for row in real_parsed)
    ]
    strategy = str(getattr(cfg, "val_split_strategy", "")).lower()
    if strategy == "height_max_mach":
        cfg.val_heights = real_heights
        return
    if strategy != "height_mach_points":
        return

    configured_points = [
        (float(point[0]), float(point[1]))
        for point in getattr(cfg, "val_height_mach_points", [])
    ]
    matching_points = [
        point
        for point in configured_points
        if any(abs(point[0] - height) <= tolerance for height in real_heights)
    ]
    covered_heights = {
        height
        for height in real_heights
        if any(abs(point[0] - height) <= tolerance for point in matching_points)
    }
    if len(covered_heights) == len(real_heights):
        cfg.val_height_mach_points = matching_points
        return

    cfg.val_split_strategy = "height_max_mach"
    cfg.val_heights = real_heights
    missing = [height for height in real_heights if height not in covered_heights]
    print(
        "[SELECTION] No configured validation (H, M) point covers selected "
        f"height(s) {missing}; using the maximum Mach number at each selected "
        "height for validation.",
        flush=True,
    )


def _height_selection_matches(
    training_config: dict,
    expected_heights: Optional[List[float]],
    expected_tolerance: float,
) -> Tuple[bool, str]:
    dataset_config = dict(training_config.get("dataset", {}) or {})
    actual_selection = dict(dataset_config.get("data_selection", {}) or {})
    actual_heights = actual_selection.get("heights")
    if actual_heights is not None:
        actual_heights = sorted(float(value) for value in actual_heights)
    expected = None if expected_heights is None else sorted(float(value) for value in expected_heights)
    if actual_heights is None or expected is None:
        if actual_heights != expected:
            return False, f"height selection {actual_heights!r} does not match {expected!r}"
        return True, "complete"
    if len(actual_heights) != len(expected):
        return False, f"height selection {actual_heights!r} does not match {expected!r}"
    tolerance = float(expected_tolerance)
    if any(abs(actual - wanted) > tolerance for actual, wanted in zip(actual_heights, expected)):
        return False, f"height selection {actual_heights!r} does not match {expected!r}"
    actual_tolerance = float(actual_selection.get("height_tolerance", tolerance))
    if not math.isclose(actual_tolerance, tolerance, rel_tol=0.0, abs_tol=1e-12):
        return False, (
            f"height-selection tolerance {actual_tolerance!r} does not match {tolerance!r}"
        )
    return True, "complete"


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


def _ddp_setup() -> Tuple[bool, int, int]:
    if not _ddp_is_requested():
        return False, 0, 1
    dist.init_process_group(backend=_ddp_backend(), init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    return True, rank, world_size


def _path_from_ini(paths_ini_path: Path, section: str) -> Optional[Path]:
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


def _resume_component_ready(
    output_dir: Path,
    excitation_id: int,
    channel_id: int,
    component: str,
    required_profile_version: Optional[int] = None,
    expected_fidelity_mode: Optional[str] = None,
    expected_heights: Optional[List[float]] = None,
    expected_height_tolerance: float = 1e-6,
    check_height_selection: bool = False,
) -> Tuple[bool, str]:
    try:
        required = [
            "model.pth", "Normalizer_Input_Sim.pth",
            "Normalizer_Output_Sim.pth", "training_config.json",
            "train_log.csv", "slot_support.json",
        ]
        if required_profile_version is not None and int(required_profile_version) >= 6:
            required.append("sim_slot_outlier_filter.json")
        if expected_fidelity_mode == "multifidelity":
            required.extend(
                ["Normalizer_Input_Real.pth", "Normalizer_Output_Real.pth"]
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
        if checkpoint.get("model_type") != "TargetMultifidelityMFDNN":
            return False, "checkpoint is not TargetMultifidelityMFDNN"
        checkpoint_fidelity_mode = str(checkpoint.get("fidelity_mode", ""))
        if (
            (expected_fidelity_mode == "multifidelity" or checkpoint_fidelity_mode == "multifidelity")
            and not (component_dir / "Normalizer_Output_Real.pth").is_file()
        ):
            return False, "missing Normalizer_Output_Real.pth"
        if tuple(checkpoint.get("target_names", ())) != tuple(target_names):
            return False, "checkpoint target_names mismatch"
        for key, expected in (("excitation_id", excitation_id), ("channel_id", channel_id)):
            if key in checkpoint and int(checkpoint[key]) != int(expected):
                return False, f"checkpoint {key}={checkpoint[key]} does not match {expected}"
        normalizer_filenames = [
            "Normalizer_Input_Sim.pth", "Normalizer_Output_Sim.pth"
        ]
        if expected_fidelity_mode == "multifidelity" or checkpoint_fidelity_mode == "multifidelity":
            normalizer_filenames.extend(
                ["Normalizer_Input_Real.pth", "Normalizer_Output_Real.pth"]
            )
        for filename in normalizer_filenames:
            normalizer = torch.load(component_dir / filename, map_location="cpu", weights_only=True)
            if not isinstance(normalizer, dict) or "mean" not in normalizer or "std" not in normalizer:
                return False, f"invalid {filename}"
        output_normalizer = torch.load(
            component_dir
            / (
                "Normalizer_Output_Real.pth"
                if checkpoint_fidelity_mode == "multifidelity"
                else "Normalizer_Output_Sim.pth"
            ),
            map_location="cpu",
            weights_only=True,
        )
        transforms = dict(output_normalizer.get("target_transforms", {}) or {})
        if any(str(transforms.get(name, {}).get("kind", "identity")) == "identity" for name in target_names if name in {"freq", "magnitude"}):
            return False, "output normalizer lacks physical target transforms"
        training_config = load_json(component_dir / "training_config.json")
        if int(training_config.get("excitation_id", excitation_id)) != int(excitation_id):
            return False, "training_config excitation_id mismatch"
        if int(training_config.get("channel_id", channel_id)) != int(channel_id):
            return False, "training_config channel_id mismatch"
        if check_height_selection:
            selection_matches, selection_reason = _height_selection_matches(
                training_config,
                expected_heights,
                expected_height_tolerance,
            )
            if not selection_matches:
                return False, selection_reason
        actual_fidelity_mode = str(
            training_config.get(
                "fidelity_mode", checkpoint.get("fidelity_mode", "")
            )
        )
        if (
            expected_fidelity_mode is not None
            and actual_fidelity_mode != str(expected_fidelity_mode)
        ):
            return False, (
                f"fidelity mode {actual_fidelity_mode!r} does not match "
                f"{expected_fidelity_mode!r}"
            )
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


def _resume_model_ready(
    output_dir: Path, excitation_id: int, channel_id: int
) -> Tuple[bool, str]:
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
            sim_parsed=task.get("sim_parsed", task.get("parsed_rows")),
            real_parsed=task.get("real_parsed", task.get("parsed_rows", [])),
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


def main(argv: Optional[List[str]] = None) -> int:
    run_started_at = time.perf_counter()
    base_dir = get_exe_dir()
    user_input_path, runtime_args = split_user_input_arg(argv, base_dir)
    with open(user_input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) < 2:
        raise ValueError("user_input.dat needs at least 2 lines: paths.ini, model_version")

    paths_ini_path = Path(lines[0].strip()).expanduser()
    model_version = lines[1].strip()
    sim_dataset_path = base_dir / "sim_dataset"
    real_dataset_path = base_dir / "real_dataset"
    if not sim_dataset_path.is_dir() or not real_dataset_path.is_dir():
        raise FileNotFoundError(
            "Multifidelity training requires both dataset directories next to the CLI entry: "
            f"sim={sim_dataset_path}, real={real_dataset_path}"
        )
    model_dir = _default_model_dir(base_dir, paths_ini_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=str(model_dir / model_version))
    parser.add_argument("--kfold", type=int, default=0)
    parser.add_argument("--embed_excitation_dim", type=int, default=None)
    parser.add_argument("--embed_channel_dim", type=int, default=None)
    parser.add_argument("--low_hidden", type=_parse_int_list, default=None)
    parser.add_argument("--use_modulation", action=BooleanOptionalAction, default=None)
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
        help="Select the best checkpoint by total training loss (default) or validation group_FXD loss.",
    )
    parser.add_argument(
        "--component_checkpoint_selection",
        action="append",
        default=None,
        help=(
            "Per-component early-stop/checkpoint metric as component=training|validation; "
            "repeat or separate assignments with commas."
        ),
    )
    parser.add_argument("--real_loss_weight", type=float, default=None)
    parser.add_argument("--sim_loss_weight", type=float, default=None)
    parser.add_argument(
        "--component_real_loss_weight",
        action="append",
        default=None,
        help="Per-component stage-2 real-data weights as component=value.",
    )
    parser.add_argument(
        "--component_sim_loss_weight",
        action="append",
        default=None,
        help="Per-component stage-2 simulation weights as component=value.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--epochs_s1", type=int, default=None)
    parser.add_argument("--epochs_s2", type=int, default=None)
    parser.add_argument("--lr_high", type=float, default=None)
    parser.add_argument("--batch_sim", type=int, default=None)
    parser.add_argument(
        "--freeze_low_stage2",
        action=BooleanOptionalAction,
        default=None,
        help="Freeze the simulation low head during real-data correction.",
    )
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
    parser.add_argument(
        "--component",
        "--components",
        action="append",
        default=None,
        help=(
            "Train only selected FDR component networks; repeat or separate with commas. "
            "Available: frequency, damping, residue_magnitude, residue_phase. Selected "
            "components are retrained in place even when --resume is used."
        ),
    )
    parser.add_argument(
        "--height",
        "--heights",
        action="append",
        default=None,
        help=(
            "Train using only selected height values in both simulation and real data; "
            "repeat or separate with commas."
        ),
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
    args, _unknown = parser.parse_known_args(runtime_args)
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
    for name in (
        "real_loss_weight",
        "sim_loss_weight",
    ):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
            raise ValueError(f"--{name} must be finite and non-negative")

    ddp_enabled, rank, _world_size = _ddp_setup()
    if ddp_enabled:
        dist.destroy_process_group()
        raise RuntimeError(
            "torchrun/DDP is not supported for independent FDR component tasks; "
            "run ordinary python with --parallel_workers instead."
        )
    _enable_fast_cuda(args)

    sim_rows = read_real_dataset(sim_dataset_path, allow_duplicate_keys=True)
    real_rows = read_real_dataset(real_dataset_path)
    sim_parsed, sim_stats = parse_rows_to_struct(sim_rows)
    real_parsed, real_stats = parse_rows_to_struct(real_rows)

    cfg = Config()
    _apply_component_training_cli(cfg, args)
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
        cfg.epochs_stage2 = int(args.epochs)
    if args.epochs_s1 is not None:
        cfg.epochs_stage1 = int(args.epochs_s1)
    if args.epochs_s2 is not None:
        cfg.epochs_stage2 = int(args.epochs_s2)
    if args.lr_high is not None:
        cfg.lr_high_adam = float(args.lr_high)
    if args.batch_sim is not None:
        cfg.batch_sim = int(args.batch_sim)
    if args.freeze_low_stage2 is not None:
        cfg.freeze_low_stage2 = bool(args.freeze_low_stage2)
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
    selected_heights = _resolve_height_selection(args.height)
    height_tolerance = float(cfg.val_height_tol)
    if selected_heights is not None:
        original_sim_count = len(sim_parsed)
        original_real_count = len(real_parsed)
        available_sim_heights = sorted({float(row.H) for row in sim_parsed})
        sim_parsed = _filter_rows_by_heights(
            sim_parsed, selected_heights, height_tolerance
        )
        real_parsed = _filter_rows_by_heights(
            real_parsed, selected_heights, height_tolerance
        )
        if not sim_parsed:
            raise RuntimeError(
                f"The height selector(s) {selected_heights} matched no simulation rows; "
                f"available simulation heights: {available_sim_heights}"
            )
        _configure_validation_for_height_selection(cfg, selected_heights, real_parsed)
        print(
            f"[SELECTION] heights={selected_heights} tolerance={height_tolerance:g} "
            f"sim_rows={len(sim_parsed)}/{original_sim_count} "
            f"real_rows={len(real_parsed)}/{original_real_count}",
            flush=True,
        )
    cfg.K_max = 3
    cfg.embed_channel_num = max(sim_stats["channel_num"], real_stats["channel_num"])
    cfg.embed_excitation_num = max(sim_stats["excitation_num"], real_stats["excitation_num"])
    if cfg.pole_target != "freq_damping" or cfg.residue_target != "polar":
        raise ValueError(
            "FDR split training requires pole_target='freq_damping' and residue_target='polar'"
        )

    model_root = Path(args.out_dir)
    if real_parsed:
        preview_train, preview_val, split_strategy = split_single_fidelity_rows(
            real_parsed, cfg, 0, 0
        )
    else:
        preview_train, preview_val, split_strategy = [], [], "sim_only_no_real_rows"
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
        sim_relative_files = build_input_dataset_snapshot(
            sim_dataset_path,
            model_root / "sim_dataset_snapshot",
            allow_duplicate_keys=True,
        )
        sim_keys = sorted({group_key_from_parsed(row) for row in sim_parsed})
        sim_manifest = build_dataset_manifest(
            rows=sim_rows,
            model_version=f"{model_version}_sim",
            snapshot_dir=model_root / "sim_dataset_snapshot",
            train_group_keys=sim_keys,
            validation_group_keys=[],
            split_strategy="sim_all_rows",
            relative_files_by_group=sim_relative_files,
            allow_duplicate_keys=True,
        )
        atomic_write_json(model_root / "sim_dataset_manifest.json", sim_manifest)
    if ddp_enabled:
        dist.barrier()
    if ddp_enabled and rank != 0:
        manifest = load_json(model_root / "dataset_manifest.json")
        sim_manifest = load_json(model_root / "sim_dataset_manifest.json")

    if args.kfold > 1:
        raise ValueError("Per-excitation/channel training currently requires --kfold 0 or 1")

    target_names = [name for names in FDR_COMPONENT_TARGETS.values() for name in names]
    trained_models = {}
    failures = []
    tasks = []
    component_states = {}
    combination_info = {}
    all_combinations = sorted(
        set(combinations_in_rows(sim_parsed)) | set(combinations_in_rows(real_parsed))
    )
    selected_excitations = _resolve_id_selection(
        args.excitation, EXCITATION_NAMES, "excitation"
    )
    selected_channels = _resolve_id_selection(
        args.channel,
        {index: name for index, name in enumerate(CHANNEL_NAMES)},
        "channel",
    )
    selected_components = _resolve_component_selection(args.component)
    combinations = _filter_combinations(
        all_combinations, selected_excitations, selected_channels
    )
    if not combinations:
        raise RuntimeError(
            "The excitation/channel selectors matched no combinations in the dataset"
        )
    selected_keys = set(combinations)
    if (
        selected_excitations is not None
        or selected_channels is not None
        or selected_components is not None
    ):
        print(
            f"[SELECTION] selected_combinations={len(combinations)}/{len(all_combinations)} "
            f"excitations={sorted(selected_excitations) if selected_excitations is not None else 'all'} "
            f"channels={sorted(selected_channels) if selected_channels is not None else 'all'} "
            f"components={sorted(selected_components) if selected_components is not None else 'all'}",
            flush=True,
        )
        rows_by_combination = {
            key: (
                [
                    row for row in real_parsed
                    if (int(row.exc_id), int(row.ch_id)) == key
                ]
                or [
                    row for row in sim_parsed
                    if (int(row.exc_id), int(row.ch_id)) == key
                ]
            )
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
                sim_count = sum(
                    (int(row.exc_id), int(row.ch_id)) == key for row in sim_parsed
                )
                real_count = sum(
                    (int(row.exc_id), int(row.ch_id)) == key for row in real_parsed
                )
                fidelity_mode = "multifidelity" if real_count else "sim_only"
                trained_models[key] = model_entry(
                    excitation_id=excitation_id,
                    channel_id=channel_id,
                    row_count=len(rows_by_combination[key]),
                    status="retained",
                    fidelity_mode=fidelity_mode,
                    sim_row_count=sim_count,
                    real_row_count=real_count,
                )
        print(
            f"[SELECTION] retained_unselected_complete_models={len(trained_models)}",
            flush=True,
        )
    for index, (excitation_id, channel_id) in enumerate(combinations, 1):
        key = (excitation_id, channel_id)
        combo_sim_rows = [
            row for row in sim_parsed
            if int(row.exc_id) == excitation_id and int(row.ch_id) == channel_id
        ]
        combo_rows = [
            row for row in real_parsed
            if int(row.exc_id) == excitation_id and int(row.ch_id) == channel_id
        ]
        fidelity_mode = "multifidelity" if combo_rows else "sim_only"
        combo_cfg = copy.deepcopy(cfg)
        available_keys = {group_key_from_parsed(row) for row in combo_rows}
        combo_cfg.explicit_train_group_keys = sorted(set(train_group_keys) & available_keys)
        combo_cfg.explicit_validation_group_keys = sorted(set(validation_group_keys) & available_keys)
        combo_cfg.require_validation = bool(
            combo_rows
            and combo_cfg.checkpoint_selection == "validation"
            and combo_cfg.explicit_validation_group_keys
        )
        if not combo_rows or (
            combo_cfg.checkpoint_selection == "validation"
            and not combo_cfg.explicit_validation_group_keys
        ):
            combo_cfg.checkpoint_selection = "training"
        combo_cfg.validation_metric_scope = "all"
        output_dir = combination_dir(model_root, excitation_id, channel_id)
        label = f"{excitation_name(excitation_id)}/{channel_name(channel_id)}"
        print(
            f"========== Queue model {index}/{len(combinations)}: "
            f"{label}, sim_rows={len(combo_sim_rows)}, real_rows={len(combo_rows)}, "
            f"mode={fidelity_mode} ==========",
            flush=True,
        )
        combination_info[key] = {
            "row_count": len(combo_rows) or len(combo_sim_rows),
            "sim_row_count": len(combo_sim_rows),
            "real_row_count": len(combo_rows),
            "fidelity_mode": fidelity_mode,
            "label": label,
            "selected_heights": selected_heights,
            "height_tolerance": height_tolerance,
        }
        component_states[key] = {}
        try:
            if not combo_sim_rows:
                raise RuntimeError("No simulation data is available for this combination")
            if combo_rows and not combo_cfg.explicit_train_group_keys:
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
                if selected_components is None or component in selected_components:
                    component_states[key][component] = "failed"
                else:
                    ready, _reason = _resume_component_ready(
                        output_dir, excitation_id, channel_id, component
                    )
                    component_states[key][component] = (
                        "resumed" if ready else "unselected"
                    )
            continue

        dataset_metadata = {
            "manifest": manifest if combo_rows else sim_manifest,
            "dataset_manifest_path": (
                "../../../dataset_manifest.json"
                if combo_rows
                else "../../../sim_dataset_manifest.json"
            ),
            "training_mode": (
                "base_multifidelity_training"
                if combo_rows
                else "base_sim_only_training"
            ),
            "sim_dataset_fingerprint": sim_manifest["dataset_fingerprint"],
            "dataset_snapshot_root": (
                "dataset_snapshot" if combo_rows else "sim_dataset_snapshot"
            ),
            "fidelity_mode": fidelity_mode,
            "prediction_output": "high" if combo_rows else "low",
            "model_filename": "model.pth",
            "excitation_id": excitation_id,
            "excitation_name": excitation_name(excitation_id),
            "channel_id": channel_id,
            "target_names": target_names,
            "data_selection": {
                "heights": selected_heights,
                "height_tolerance": height_tolerance,
            },
        }
        for component in FDR_COMPONENT_TARGETS:
            if selected_components is not None and component not in selected_components:
                ready, reason = _resume_component_ready(
                    output_dir, excitation_id, channel_id, component
                )
                component_states[key][component] = "resumed" if ready else "unselected"
                if ready:
                    print(f"[SELECTION] retained {label}/{component}", flush=True)
                else:
                    print(
                        f"[SELECTION] skipped unavailable {label}/{component}: {reason}",
                        flush=True,
                    )
                continue

            # An explicit component selector means "retrain this exact network".
            # It intentionally takes precedence over --resume and writes back to
            # the existing component directory.
            if selected_components is not None:
                if (output_dir / component).exists():
                    print(f"[SELECTION] overwrite {label}/{component}", flush=True)
            elif args.resume:
                ready, reason = _resume_component_ready(
                    output_dir,
                    excitation_id,
                    channel_id,
                    component,
                    required_profile_version=_required_component_profile_version(
                        combo_cfg, component
                    ),
                    expected_fidelity_mode=fidelity_mode,
                    expected_heights=selected_heights,
                    expected_height_tolerance=height_tolerance,
                    check_height_selection=True,
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
                    "sim_parsed": combo_sim_rows,
                    "real_parsed": combo_rows,
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
                row_count=len(combo_rows) or len(combo_sim_rows),
                status="resumed",
                fidelity_mode=fidelity_mode,
                sim_row_count=len(combo_sim_rows),
                real_row_count=len(combo_rows),
            )

    def write_progress_inventory():
        write_model_inventory(
            model_root,
            model_version=model_version,
            models=trained_models.values(),
            failures=failures,
        )

    scheduled_total = len(tasks)
    scheduled_completed = 0
    training_started_at = time.perf_counter()

    def print_training_progress():
        state_values = [
            state
            for states in component_states.values()
            for state in states.values()
            if state != "unselected"
        ]
        total_components = len(state_values)
        trained_components = sum(state == "trained" for state in state_values)
        resumed_components = sum(state == "resumed" for state in state_values)
        failed_components = sum(state == "failed" for state in state_values)
        done_components = trained_components + resumed_components + failed_components
        remaining_components = max(total_components - done_components, 0)
        complete_models = sum(
            bool(states)
            and all(state in {"trained", "resumed"} for state in states.values())
            for states in component_states.values()
        )
        failed_models = sum(
            any(state == "failed" for state in states.values())
            for states in component_states.values()
        )
        elapsed = time.perf_counter() - training_started_at
        eta = None
        if scheduled_total == 0 or scheduled_completed >= scheduled_total:
            eta = 0.0
        elif scheduled_completed > 0:
            eta = (
                elapsed
                / scheduled_completed
                * max(scheduled_total - scheduled_completed, 0)
            )
        percent = 100.0 if total_components == 0 else 100.0 * done_components / total_components
        print(
            f"[PROGRESS] components={done_components}/{total_components} "
            f"({percent:.1f}%) remaining={remaining_components} "
            f"trained={trained_components} resumed={resumed_components} "
            f"failed={failed_components} models={complete_models}/{len(component_states)} "
            f"model_failures={failed_models} elapsed={_format_duration(elapsed)} "
            f"ETA={_format_duration(eta)}",
            flush=True,
        )

    def handle_task_result(result: dict):
        nonlocal scheduled_completed
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
                expected_fidelity_mode=info["fidelity_mode"],
                expected_heights=info["selected_heights"],
                expected_height_tolerance=info["height_tolerance"],
                check_height_selection=True,
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
                fidelity_mode=info["fidelity_mode"],
                sim_row_count=info["sim_row_count"],
                real_row_count=info["real_row_count"],
            )
            print(f"[MODEL-COMPLETE] {info['label']} status={status}", flush=True)
        scheduled_completed += 1
        write_progress_inventory()
        print_training_progress()

    write_progress_inventory()
    print(
        f"[PARALLEL] workers={args.parallel_workers} pending_components={len(tasks)} "
        f"resumed_models={len(trained_models)}",
        flush=True,
    )
    print_training_progress()
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
        training_elapsed = time.perf_counter() - training_started_at
        total_elapsed = time.perf_counter() - run_started_at
        print(
            f"[TIMING] preparation={_format_duration(total_elapsed - training_elapsed)} "
            f"training={_format_duration(training_elapsed)} "
            f"total={_format_duration(total_elapsed)}",
            flush=True,
        )

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    mp.freeze_support()
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    raise SystemExit(run_executable(main, "train", get_exe_dir()))
