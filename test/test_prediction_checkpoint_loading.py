from copy import deepcopy
import warnings

import torch

from surrogate_model.config import Config
from surrogate_model.models.blocks import MFDNN
from surrogate_model.models.checkpoint_utils import (
    build_model_from_checkpoint,
    restore_model_config_from_checkpoint,
)


FIELDS = (
    "K_max", "embed_excitation_num", "embed_excitation_dim", "embed_channel_num",
    "embed_channel_dim", "low_hidden", "dropout_prob", "use_lora", "lora_rank",
    "lora_alpha", "lora_dropout", "use_modulation", "modulation_mode", "mod_hidden",
    "mod_dropout", "mod_gamma", "mod_exc_pole_scale", "mod_exc_residue_scale",
    "mod_channel_residue_scale", "mod_channel_affects_poles", "pole_target", "residue_target",
)


def _checkpoint(cfg, *, omit=()):
    model = MFDNN(cfg).eval()
    payload = {name: deepcopy(getattr(cfg, name)) for name in FIELDS if name not in omit}
    payload["model_state_dict"] = deepcopy(model.state_dict())
    return payload, model


def _assert_same_output(left, right, cfg):
    inputs = (torch.randn(3, 1), torch.randn(3, 1), torch.randint(0, cfg.embed_excitation_num, (3,)), torch.randint(0, cfg.embed_channel_num, (3,)))
    with torch.no_grad():
        old, new = left(*inputs), right(*inputs)
    for key in ("alpha", "omega", "rho", "sin_theta", "cos_theta"):
        assert torch.allclose(old[key], new[key], atol=1e-6, rtol=1e-6)


def test_legacy_without_mode_is_inferred_and_strictly_loaded():
    cfg = Config(); cfg.modulation_mode = "legacy"
    checkpoint, _ = _checkpoint(cfg, omit=("modulation_mode",))
    restored_cfg = restore_model_config_from_checkpoint(checkpoint)
    model = build_model_from_checkpoint(checkpoint)
    assert restored_cfg.modulation_mode == "legacy"
    assert model.low.mod is not None and model.low.hm_mod is None


def test_conditional_nondefault_configuration_round_trip():
    cfg = Config(); cfg.mod_hidden = [16, 24]; cfg.mod_dropout = 0.1; cfg.mod_gamma = 0.15
    cfg.mod_exc_pole_scale = 0.03; cfg.mod_exc_residue_scale = 0.17
    cfg.mod_channel_residue_scale = 0.08; cfg.mod_channel_affects_poles = False
    checkpoint, original = _checkpoint(cfg)
    restored = build_model_from_checkpoint(checkpoint)
    for field in FIELDS:
        assert getattr(restored.cfg, field) == getattr(cfg, field)
    original.eval(); restored.eval()
    _assert_same_output(original, restored, cfg)


def test_legacy_state_conflicts_with_conditional_metadata():
    cfg = Config(); cfg.modulation_mode = "legacy"
    checkpoint, _ = _checkpoint(cfg); checkpoint["modulation_mode"] = "conditional"
    try:
        restore_model_config_from_checkpoint(checkpoint)
    except RuntimeError as exc:
        assert "conflicts" in str(exc)
    else:
        raise AssertionError("expected modulation conflict")


def test_conditional_state_conflicts_with_legacy_metadata():
    checkpoint, _ = _checkpoint(Config()); checkpoint["modulation_mode"] = "legacy"
    try:
        restore_model_config_from_checkpoint(checkpoint)
    except RuntimeError as exc:
        assert "conflicts" in str(exc)
    else:
        raise AssertionError("expected modulation conflict")


def test_mixed_modulation_state_is_rejected():
    checkpoint, _ = _checkpoint(Config())
    checkpoint["model_state_dict"]["low.mod.0.weight"] = torch.randn(15, 2)
    try:
        restore_model_config_from_checkpoint(checkpoint)
    except RuntimeError as exc:
        assert "both legacy" in str(exc)
    else:
        raise AssertionError("expected mixed-state rejection")


def test_disabled_modulation_strict_loads_without_heads():
    cfg = Config(); cfg.use_modulation = False
    checkpoint, _ = _checkpoint(cfg, omit=("modulation_mode",))
    model = build_model_from_checkpoint(checkpoint)
    assert not model.cfg.use_modulation
    assert model.low.mod is None and model.low.hm_mod is None


def test_plain_and_lora_checkpoints_are_detected_and_strictly_loaded():
    plain_checkpoint, _ = _checkpoint(Config())
    assert build_model_from_checkpoint(plain_checkpoint).cfg.use_lora is False
    lora_cfg = Config(); lora_cfg.use_lora = True
    lora_checkpoint, _ = _checkpoint(lora_cfg)
    assert build_model_from_checkpoint(lora_checkpoint).cfg.use_lora is True


def test_lora_metadata_conflict_is_rejected():
    cfg = Config(); cfg.use_lora = True
    checkpoint, _ = _checkpoint(cfg); checkpoint["use_lora"] = False
    try:
        restore_model_config_from_checkpoint(checkpoint)
    except RuntimeError as exc:
        assert "LoRA metadata conflict" in str(exc)
    else:
        raise AssertionError("expected LoRA conflict")


def test_embedding_dimensions_use_state_dictionary_shapes():
    cfg = Config(); cfg.embed_excitation_num = 7; cfg.embed_excitation_dim = 5
    cfg.embed_channel_num = 11; cfg.embed_channel_dim = 6
    checkpoint, _ = _checkpoint(cfg)
    checkpoint.update(embed_excitation_num=99, embed_excitation_dim=98, embed_channel_num=97, embed_channel_dim=96)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restored = restore_model_config_from_checkpoint(checkpoint)
    assert (restored.embed_excitation_num, restored.embed_excitation_dim) == (7, 5)
    assert (restored.embed_channel_num, restored.embed_channel_dim) == (11, 6)
    assert len(caught) == 2


def test_prediction_fixed_sf_constructor_uses_complete_checkpoint():
    import test.Predict_user_fixed_sf as prediction

    cfg = Config(); cfg.mod_hidden = [16, 24]; cfg.mod_gamma = 0.15
    checkpoint, original = _checkpoint(cfg)
    prediction._LAST_CHECKPOINT = checkpoint
    prediction._LAST_STATE = checkpoint["model_state_dict"]
    model = prediction._MFDNNFromCheckpoint(Config()).eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert model.cfg.mod_hidden == [16, 24]
    _assert_same_output(original, model, cfg)
