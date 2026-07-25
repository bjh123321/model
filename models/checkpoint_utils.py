from __future__ import annotations

import warnings
import re
from pathlib import Path
from typing import Mapping, Optional, Tuple, Union

import torch

from surrogate_model.config import Config
from surrogate_model.models.blocks import (
    MFDNN,
    TargetMFDNN,
    TargetMultifidelityMFDNN,
)


STRUCTURE_FIELDS = (
    "K_max", "embed_excitation_num", "embed_excitation_dim", "embed_channel_num",
    "embed_channel_dim", "low_hidden", "dropout_prob", "use_lora", "lora_rank",
    "lora_alpha", "lora_dropout", "lora_target_hidden_indices", "use_modulation",
    "mod_gamma", "pole_target", "residue_target", "adapt_hidden",
    "model_type", "model_target_names", "target_component",
    "multifidelity_real_loss_weight", "multifidelity_sim_loss_weight",
)


def checkpoint_state_dict(checkpoint: Mapping) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint must be a mapping, got {type(checkpoint).__name__}")
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("Checkpoint does not contain a non-empty model state dictionary")
    return state


def checkpoint_uses_lora(checkpoint: Mapping, state: Optional[Mapping] = None) -> bool:
    state = checkpoint_state_dict(checkpoint) if state is None else state
    state_uses_lora = any(".A.weight" in key or ".B.weight" in key or ".base.weight" in key for key in state)
    if "use_lora" in checkpoint and bool(checkpoint["use_lora"]) != state_uses_lora:
        raise RuntimeError(
            f"Checkpoint LoRA metadata conflict: use_lora={checkpoint['use_lora']!r}, "
            f"but state_uses_lora={state_uses_lora}"
        )
    return state_uses_lora


def infer_lora_target_hidden_indices(state: Mapping) -> Tuple[int, ...]:
    """Infer logical hidden-layer indices from sequential module positions in a LoRA state."""
    linear_pattern = re.compile(r"^low\.mlp\.net\.(\d+)\.(?:base\.)?weight$")
    lora_pattern = re.compile(r"^low\.mlp\.net\.(\d+)\.A\.weight$")
    linear_modules = sorted(
        {int(match.group(1)) for key in state for match in [linear_pattern.match(key)] if match}
    )
    lora_modules = sorted(
        {int(match.group(1)) for key in state for match in [lora_pattern.match(key)] if match}
    )
    if not lora_modules:
        return ()
    logical_index = {module_index: index for index, module_index in enumerate(linear_modules[:-1])}
    unknown = [module_index for module_index in lora_modules if module_index not in logical_index]
    if unknown:
        raise RuntimeError(f"Cannot map checkpoint LoRA module positions to hidden layers: {unknown}")
    return tuple(logical_index[module_index] for module_index in lora_modules)


def _restore_lora_structure(cfg: Config, checkpoint: Mapping, state: Mapping) -> None:
    detected_targets = infer_lora_target_hidden_indices(state)
    metadata_targets = checkpoint.get("lora_target_hidden_indices")
    if metadata_targets is not None:
        metadata_targets = tuple(int(index) for index in metadata_targets)
        if metadata_targets != detected_targets:
            raise RuntimeError(
                "Checkpoint LoRA target metadata conflicts with state dictionary: "
                f"metadata={metadata_targets}, detected={detected_targets}"
            )
    cfg.lora_target_hidden_indices = detected_targets

    ranks = {
        int(tensor.shape[0])
        for key, tensor in state.items()
        if key.endswith(".A.weight") and isinstance(tensor, torch.Tensor) and tensor.ndim == 2
    }
    if len(ranks) != 1:
        raise RuntimeError(f"Checkpoint LoRA layers do not share one rank: {sorted(ranks)}")
    detected_rank = next(iter(ranks))
    if "lora_rank" in checkpoint and int(checkpoint["lora_rank"]) != detected_rank:
        raise RuntimeError(
            f"Checkpoint lora_rank={checkpoint['lora_rank']} conflicts with state rank={detected_rank}"
        )
    cfg.lora_rank = detected_rank


def _state_modulation_kind(state: Mapping) -> bool:
    legacy = any(key.startswith("low.mod.") for key in state)
    conditional_parts = {
        "hm_mod": any(key.startswith("low.hm_mod.") for key in state),
        "exc_gate": any(key.startswith("low.exc_gate.") for key in state),
        "ch_gate": any(key.startswith("low.ch_gate.") for key in state),
    }
    if any(conditional_parts.values()):
        raise RuntimeError(
            "This checkpoint uses the deprecated conditional modulation architecture "
            "and cannot be loaded by the restored single-layer model."
        )
    return legacy


def _restore_embedding_shape(cfg: Config, checkpoint: Mapping, state: Mapping, key: str, num_attr: str, dim_attr: str):
    tensor = state.get(key)
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
        raise RuntimeError(f"Cannot restore embedding shape: missing or invalid {key}")
    actual = (int(tensor.shape[0]), int(tensor.shape[1]))
    metadata = (checkpoint.get(num_attr), checkpoint.get(dim_attr))
    if all(value is not None for value in metadata) and tuple(map(int, metadata)) != actual:
        warnings.warn(
            f"Checkpoint metadata {num_attr}/{dim_attr}={metadata} conflicts with {key} shape={actual}; using state shape",
            RuntimeWarning,
        )
    setattr(cfg, num_attr, actual[0])
    setattr(cfg, dim_attr, actual[1])


def restore_model_config_from_checkpoint(
    checkpoint: Mapping, *, use_lora: Optional[bool] = None
) -> Config:
    state = checkpoint_state_dict(checkpoint)
    cfg = Config()
    for field in STRUCTURE_FIELDS:
        if field in checkpoint:
            setattr(cfg, field, checkpoint[field])
    if "slot_support" in checkpoint:
        cfg._slot_support_metadata = dict(checkpoint.get("slot_support") or {})

    legacy = _state_modulation_kind(state)
    metadata_mode = checkpoint.get("modulation_mode")
    metadata_use_modulation = checkpoint.get("use_modulation")
    if metadata_mode is not None and str(metadata_mode) != "legacy":
        raise RuntimeError(
            "This checkpoint uses the deprecated conditional modulation architecture "
            "and cannot be loaded by the restored single-layer model."
        )
    if metadata_use_modulation is False and legacy:
        raise RuntimeError("use_modulation=False conflicts with modulation parameters present in the state dictionary")
    if legacy:
        cfg.modulation_mode, cfg.use_modulation = "legacy", True
    elif metadata_use_modulation is False:
        cfg.modulation_mode = "legacy"
        cfg.use_modulation = False
    else:
        raise RuntimeError("Cannot infer modulation structure: no legacy/conditional state keys and use_modulation is not False")

    if "K_max" not in checkpoint:
        gain = state.get("online_gain")
        if not isinstance(gain, torch.Tensor) or gain.ndim != 3:
            raise RuntimeError("Cannot restore K_max from metadata or online_gain buffer")
        cfg.K_max = int(gain.shape[1])
    _restore_embedding_shape(cfg, checkpoint, state, "emb_exc.weight", "embed_excitation_num", "embed_excitation_dim")
    _restore_embedding_shape(cfg, checkpoint, state, "emb_ch.weight", "embed_channel_num", "embed_channel_dim")

    detected_lora = checkpoint_uses_lora(checkpoint, state)
    if detected_lora:
        _restore_lora_structure(cfg, checkpoint, state)
    if use_lora is None:
        cfg.use_lora = detected_lora
    else:
        cfg.use_lora = bool(use_lora)
    return cfg


def build_model_from_checkpoint(
    checkpoint: Mapping,
    *,
    device=None,
    checkpoint_path: Union[str, Path] = "<memory>",
) -> Union[MFDNN, TargetMFDNN, TargetMultifidelityMFDNN]:
    state = checkpoint_state_dict(checkpoint)
    cfg = restore_model_config_from_checkpoint(checkpoint)
    model_type = str(checkpoint.get("model_type", getattr(cfg, "model_type", "MFDNN")))
    if model_type in {"TargetMFDNN", "TargetMultifidelityMFDNN"}:
        target_names = checkpoint.get("target_names", getattr(cfg, "model_target_names", None))
        if not target_names:
            raise RuntimeError(f"Target checkpoint {checkpoint_path} has no target_names metadata")
        cfg.model_target_names = [str(name) for name in target_names]
        cfg.target_component = str(checkpoint.get("target_component", getattr(cfg, "target_component", "")))
        model = (
            TargetMultifidelityMFDNN(cfg)
            if model_type == "TargetMultifidelityMFDNN"
            else TargetMFDNN(cfg)
        )
    elif model_type in {"MFDNN", "MFDNN_fixed_single_fidelity"}:
        model = MFDNN(cfg)
    else:
        raise RuntimeError(f"Unsupported model_type={model_type!r} in checkpoint {checkpoint_path}")
    if device is not None:
        model = model.to(device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load checkpoint {checkpoint_path} with modulation_mode={cfg.modulation_mode!r}; "
            f"strict loading reported missing/unexpected or shape-mismatched keys: {exc}. "
            "Possible deprecated conditional or LoRA metadata conflict."
        ) from exc
    model.eval()
    return model
