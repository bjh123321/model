from copy import deepcopy

import torch

from surrogate_model.cli.train_incremental import _restore_cfg_from_checkpoint
from surrogate_model.config import Config
from surrogate_model.models.blocks import (
    MFDNN,
    SLOT_PARAM_COUNT,
    count_parameter_groups,
    freeze_base_train_lora_only,
    load_base_weights_into_lora_model,
)


def _inputs(cfg, batch):
    return (
        torch.randn(batch, 1), torch.randn(batch, 1),
        torch.randint(0, cfg.embed_excitation_num, (batch,)),
        torch.randint(0, cfg.embed_channel_num, (batch,)),
    )


def _stack_raw(output):
    return torch.stack([output[name] for name in ("u_alpha", "u_omega", "u_rho", "u_sin_theta", "u_cos_theta")], dim=-1)


def test_main_mlp_input_dimension_is_unchanged():
    cfg = Config()
    model = MFDNN(cfg)
    first_linear = next(layer for layer in model.low.mlp.net if isinstance(layer, torch.nn.Linear))
    assert first_linear.in_features == 2 + cfg.embed_excitation_dim + cfg.embed_channel_dim


def test_gates_are_identity_at_initialization():
    cfg = Config()
    model = MFDNN(cfg)
    exc_gate, ch_gate = model.low.conditional_gates(torch.randn(3, cfg.embed_excitation_dim), torch.randn(3, cfg.embed_channel_dim))
    assert torch.equal(exc_gate, torch.ones_like(exc_gate))
    assert torch.equal(ch_gate, torch.ones_like(ch_gate))


def test_channel_gate_never_affects_poles_by_default():
    cfg = Config()
    model = MFDNN(cfg)
    with torch.no_grad():
        model.low.ch_gate.weight.fill_(1.0)
        model.low.ch_gate.bias.fill_(1.0)
    _, ch_gate = model.low.conditional_gates(torch.randn(2, cfg.embed_excitation_dim), torch.ones(2, cfg.embed_channel_dim))
    assert torch.equal(ch_gate[..., :2], torch.ones_like(ch_gate[..., :2]))
    assert not torch.equal(ch_gate[..., 2:], torch.ones_like(ch_gate[..., 2:]))


def test_excitation_gate_affects_poles_and_residues_at_distinct_scales():
    cfg = Config()
    model = MFDNN(cfg)
    with torch.no_grad():
        model.low.exc_gate.weight.fill_(10.0)
        model.low.exc_gate.bias.fill_(10.0)
    exc_gate, _ = model.low.conditional_gates(torch.ones(1, cfg.embed_excitation_dim), torch.zeros(1, cfg.embed_channel_dim))
    assert torch.all(exc_gate[..., :2] > 1.0)
    assert torch.all(exc_gate[..., 2:] > 1.0)
    assert torch.allclose(exc_gate[..., 0] - 1.0, torch.full_like(exc_gate[..., 0], cfg.mod_exc_pole_scale), atol=1e-6)
    assert torch.allclose(exc_gate[..., 2] - 1.0, torch.full_like(exc_gate[..., 2], cfg.mod_exc_residue_scale), atol=1e-6)


def test_output_shapes_for_batch_and_kmax():
    for batch, kmax in ((1, 1), (4, 3), (2, 5)):
        cfg = Config(K_max=kmax)
        model = MFDNN(cfg)
        raw = _stack_raw(model.low(*_low_inputs(model, cfg, batch)))
        assert raw.shape == (batch, kmax, SLOT_PARAM_COUNT)


def _low_inputs(model, cfg, batch):
    H, M, exc, ch = _inputs(cfg, batch)
    return H, M, model.emb_exc(exc), model.emb_ch(ch)


def test_disabling_modulation_returns_exact_main_mlp_output():
    cfg = Config()
    cfg.use_modulation = False
    model = MFDNN(cfg)
    H, M, exc_emb, ch_emb = _low_inputs(model, cfg, 3)
    actual = _stack_raw(model.low(H, M, exc_emb, ch_emb))
    expected = model.low.mlp(torch.cat([H, M, exc_emb, ch_emb], dim=1)).view(-1, cfg.K_max, SLOT_PARAM_COUNT)
    assert torch.equal(actual, expected)


def test_old_checkpoint_without_mode_restores_legacy_strictly():
    cfg = Config()
    cfg.modulation_mode = "legacy"
    old_model = MFDNN(cfg)
    metadata = {name: getattr(cfg, name) for name in ("K_max", "embed_excitation_num", "embed_channel_num", "embed_excitation_dim", "embed_channel_dim", "low_hidden", "mod_hidden", "use_modulation", "mod_gamma")}
    restored_cfg = _restore_cfg_from_checkpoint(metadata, use_lora=False)
    restored = MFDNN(restored_cfg)
    restored.load_state_dict(old_model.state_dict(), strict=True)
    assert restored_cfg.modulation_mode == "legacy"
    assert restored.low.mod is not None and restored.low.hm_mod is None


def test_conditional_checkpoint_round_trip_predictions_match():
    cfg = Config()
    original = MFDNN(cfg).eval()
    metadata = {name: getattr(cfg, name) for name in ("K_max", "embed_excitation_num", "embed_channel_num", "embed_excitation_dim", "embed_channel_dim", "low_hidden", "mod_hidden", "mod_dropout", "modulation_mode", "mod_exc_pole_scale", "mod_exc_residue_scale", "mod_channel_residue_scale", "mod_channel_affects_poles", "use_modulation", "mod_gamma")}
    restored = MFDNN(_restore_cfg_from_checkpoint(metadata, use_lora=False)).eval()
    restored.load_state_dict(original.state_dict(), strict=True)
    inputs = _inputs(cfg, 3)
    with torch.no_grad():
        old, new = original(*inputs), restored(*inputs)
    for name in ("alpha", "omega", "rho", "sin_theta", "cos_theta"):
        assert torch.equal(old[name], new[name])


def test_plain_conditional_model_to_lora_preserves_predictions():
    cfg = Config()
    teacher = MFDNN(cfg).eval()
    student_cfg = deepcopy(cfg)
    student_cfg.use_lora = True
    student = MFDNN(student_cfg).eval()
    load_base_weights_into_lora_model(student, teacher.state_dict())
    inputs = _inputs(cfg, 3)
    with torch.no_grad():
        old, new = teacher(*inputs), student(*inputs)
    for name in ("alpha", "omega", "rho", "sin_theta", "cos_theta"):
        assert torch.allclose(old[name], new[name], atol=1e-6, rtol=1e-6)


def test_lora_freezing_excludes_all_modulation_parameters():
    cfg = Config()
    cfg.use_lora = True
    model = MFDNN(cfg)
    freeze_base_train_lora_only(model)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable and all(".A." in name or ".B." in name for name in trainable)
    assert not any(name.startswith(("low.hm_mod", "low.exc_gate", "low.ch_gate")) for name in trainable)


def test_parameter_group_counts_cover_the_model():
    counts = count_parameter_groups(MFDNN(Config()))
    assert counts["hm_mod_parameters"] > 0
    assert counts["exc_gate_parameters"] > 0
    assert counts["ch_gate_parameters"] > 0
    assert counts["total_parameters"] >= sum(value for key, value in counts.items() if key != "total_parameters")
