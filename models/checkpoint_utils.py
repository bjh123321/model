from __future__ import annotations

import warnings
from pathlib import Path
from typing import Mapping

import torch

from surrogate_model.config import Config
from surrogate_model.models.blocks import MFDNN


STRUCTURE_FIELDS = (
    "K_max", "embed_excitation_num", "embed_excitation_dim", "embed_channel_num",
    "embed_channel_dim", "low_hidden", "dropout_prob", "use_lora", "lora_rank",
    "lora_alpha", "lora_dropout", "use_modulation", "mod_hidden", "mod_dropout",
    "mod_gamma", "mod_exc_pole_scale", "mod_exc_residue_scale",
    "mod_channel_residue_scale", "mod_channel_affects_poles", "pole_target",
    "residue_target",
)


def checkpoint_state_dict(checkpoint: Mapping) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint must be a mapping, got {type(checkpoint).__name__}")
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("Checkpoint does not contain a non-empty model state dictionary")
    return state


def checkpoint_uses_lora(checkpoint: Mapping, state: Mapping | None = None) -> bool:
    state = checkpoint_state_dict(checkpoint) if state is None else state
    state_uses_lora = any(".A.weight" in key or ".B.weight" in key or ".base.weight" in key for key in state)
    if "use_lora" in checkpoint and bool(checkpoint["use_lora"]) != state_uses_lora:
        raise RuntimeError(
            f"Checkpoint LoRA metadata conflict: use_lora={checkpoint['use_lora']!r}, "
            f"but state_uses_lora={state_uses_lora}"
        )
    return state_uses_lora


def _state_modulation_kind(state: Mapping) -> tuple[bool, bool]:
    legacy = any(key.startswith("low.mod.") for key in state)
    conditional_parts = {
        "hm_mod": any(key.startswith("low.hm_mod.") for key in state),
        "exc_gate": any(key.startswith("low.exc_gate.") for key in state),
        "ch_gate": any(key.startswith("low.ch_gate.") for key in state),
    }
    conditional = any(conditional_parts.values())
    if conditional and not all(conditional_parts.values()):
        missing = [name for name, present in conditional_parts.items() if not present]
        raise RuntimeError(f"Conditional checkpoint is incomplete; missing parameter groups: {missing}")
    if legacy and conditional:
        raise RuntimeError("Checkpoint contains both legacy low.mod.* and conditional modulation parameters")
    return legacy, conditional


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


def restore_model_config_from_checkpoint(checkpoint: Mapping, *, use_lora: bool | None = None) -> Config:
    state = checkpoint_state_dict(checkpoint)
    cfg = Config()
    for field in STRUCTURE_FIELDS:
        if field in checkpoint:
            setattr(cfg, field, checkpoint[field])

    legacy, conditional = _state_modulation_kind(state)
    metadata_mode = checkpoint.get("modulation_mode")
    metadata_use_modulation = checkpoint.get("use_modulation")
    if metadata_mode is not None and str(metadata_mode) not in ("legacy", "conditional"):
        raise RuntimeError(f"Invalid modulation_mode={metadata_mode!r}")
    if metadata_mode == "legacy" and conditional:
        raise RuntimeError("modulation_mode='legacy' conflicts with conditional state parameters")
    if metadata_mode == "conditional" and legacy:
        raise RuntimeError("modulation_mode='conditional' conflicts with legacy state parameters")
    if metadata_use_modulation is False and (legacy or conditional):
        raise RuntimeError("use_modulation=False conflicts with modulation parameters present in the state dictionary")
    if legacy:
        cfg.modulation_mode, cfg.use_modulation = "legacy", True
    elif conditional:
        cfg.modulation_mode, cfg.use_modulation = "conditional", True
    elif metadata_use_modulation is False:
        cfg.modulation_mode = str(metadata_mode or "legacy")
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
    if use_lora is None:
        cfg.use_lora = detected_lora
    else:
        cfg.use_lora = bool(use_lora)
    return cfg


def build_model_from_checkpoint(checkpoint: Mapping, *, device=None, checkpoint_path: str | Path = "<memory>") -> MFDNN:
    state = checkpoint_state_dict(checkpoint)
    cfg = restore_model_config_from_checkpoint(checkpoint)
    model = MFDNN(cfg)
    if device is not None:
        model = model.to(device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load checkpoint {checkpoint_path} with modulation_mode={cfg.modulation_mode!r}; "
            f"strict loading reported missing/unexpected or shape-mismatched keys: {exc}. "
            "Possible legacy/conditional or LoRA metadata conflict."
        ) from exc
    model.eval()
    return model
