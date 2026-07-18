import inspect
from types import SimpleNamespace

import numpy as np
import torch

from surrogate_model.config import Config
from surrogate_model.data.normalizer import Normalizer
from surrogate_model.models.blocks import TargetMFDNN, TargetMultifidelityMFDNN
from surrogate_model.models.checkpoint_utils import build_model_from_checkpoint
from surrogate_model.test.Predict_user import (
    _bounded_damping_tanh_pull,
    _c1_smooth_linear_tail,
    _height_conditioned_mach_bounds,
    _parse_optional_float,
    _prediction_damping_slot_mask,
    _run_infer_fdr_bundle_batched,
    run_infer_batched,
)
from surrogate_model.test.Predict_user import _build_yhat_and_slot_strings
from surrogate_model.train.dataloaders import (
    _build_slot_support,
    _filter_damping_slots,
    _filter_sim_outlier_slots,
)
from surrogate_model.utils.fdr_target_transforms import (
    build_target_transform_specs,
    decode_numpy,
    encode_numpy,
)
from surrogate_model.utils.excitation_channel_models import (
    FDR_COMPONENT_TARGETS,
    model_entry,
)


def _target_config(target_names):
    cfg = Config()
    cfg.K_max = 3
    cfg.embed_excitation_num = 6
    cfg.embed_channel_num = 34
    cfg.model_type = "TargetMFDNN"
    cfg.target_component = "test"
    cfg.model_target_names = list(target_names)
    return cfg


def test_target_model_checkpoint_roundtrip():
    torch.manual_seed(0)
    cfg = _target_config(("sin_theta", "cos_theta"))
    model = TargetMFDNN(cfg)
    checkpoint = {
        "model_type": "TargetMFDNN",
        "target_component": "residue_phase",
        "target_names": ["sin_theta", "cos_theta"],
        "model_target_names": ["sin_theta", "cos_theta"],
        "model_state_dict": model.state_dict(),
        "K_max": cfg.K_max,
        "low_hidden": list(cfg.low_hidden),
        "dropout_prob": cfg.dropout_prob,
        "use_lora": False,
        "use_modulation": cfg.use_modulation,
        "modulation_mode": "legacy",
        "mod_gamma": cfg.mod_gamma,
        "embed_excitation_num": cfg.embed_excitation_num,
        "embed_excitation_dim": cfg.embed_excitation_dim,
        "embed_channel_num": cfg.embed_channel_num,
        "embed_channel_dim": cfg.embed_channel_dim,
        "pole_target": "freq_damping",
        "residue_target": "polar",
    }
    restored = build_model_from_checkpoint(checkpoint)
    output = restored(
        torch.zeros(2, 1),
        torch.zeros(2, 1),
        torch.tensor([0, 1]),
        torch.tensor([8, 9]),
    )
    assert set(output) == {"sin_theta", "cos_theta"}
    assert output["sin_theta"].shape == (2, 3)
    norm = torch.sqrt(output["sin_theta"].square() + output["cos_theta"].square())
    torch.testing.assert_close(norm, torch.ones_like(norm), atol=1e-5, rtol=1e-5)


def test_multifidelity_target_checkpoint_roundtrip_returns_low_and_high():
    torch.manual_seed(0)
    cfg = _target_config(("freq",))
    cfg.model_type = "TargetMultifidelityMFDNN"
    model = TargetMultifidelityMFDNN(cfg)
    checkpoint = {
        "model_type": "TargetMultifidelityMFDNN",
        "target_component": "frequency",
        "target_names": ["freq"],
        "model_target_names": ["freq"],
        "model_state_dict": model.state_dict(),
        "K_max": cfg.K_max,
        "low_hidden": list(cfg.low_hidden),
        "adapt_hidden": list(cfg.adapt_hidden),
        "dropout_prob": cfg.dropout_prob,
        "use_lora": False,
        "use_modulation": cfg.use_modulation,
        "modulation_mode": "legacy",
        "mod_gamma": cfg.mod_gamma,
        "embed_excitation_num": cfg.embed_excitation_num,
        "embed_excitation_dim": cfg.embed_excitation_dim,
        "embed_channel_num": cfg.embed_channel_num,
        "embed_channel_dim": cfg.embed_channel_dim,
        "pole_target": "freq_damping",
        "residue_target": "polar",
    }
    restored = build_model_from_checkpoint(checkpoint)
    low, high = restored(
        torch.zeros(2, 1),
        torch.zeros(2, 1),
        torch.tensor([0, 1]),
        torch.tensor([8, 9]),
    )
    assert low["freq"].shape == (2, 3)
    assert high["freq"].shape == (2, 3)


def test_fdr_defaults_restore_capacity_without_dropout():
    cfg = Config()
    assert cfg.fdr_component_low_hidden["frequency"] == [64, 128, 64]
    assert cfg.fdr_component_low_hidden["damping"] == [64, 128, 64]
    assert cfg.fdr_component_dropout_prob["frequency"] == 0.0
    assert cfg.fdr_component_dropout_prob["damping"] == 0.0
    assert cfg.fdr_component_weight_decay["damping"] == 5e-5
    assert cfg.fdr_component_early_stop_patience["damping"] == 50
    assert cfg.fdr_component_epochs_stage1["damping"] == 200
    assert cfg.fdr_component_epochs_stage2["damping"] == 200
    assert cfg.fdr_component_epochs_stage1["frequency"] == 500
    assert cfg.fdr_component_checkpoint_train_loss_ceiling["frequency"] == 0.05
    assert cfg.fdr_component_checkpoint_train_loss_ceiling["damping"] is None
    assert cfg.fdr_training_profile_version == 8
    assert cfg.fdr_component_training_profile_version["damping"] == 9
    assert cfg.fdr_component_training_profile_version["frequency"] == 8
    assert cfg.multifidelity_sim_loss_weight == 0.2
    assert cfg.multifidelity_real_loss_weight == 1.0
    assert cfg.checkpoint_selection == "training"
    assert set(cfg.fdr_component_checkpoint_selection.values()) == {"training"}
    assert cfg.fdr_component_real_loss_weight["damping"] == 1.0
    assert cfg.fdr_component_sim_loss_weight["damping"] == 0.2
    assert cfg.damping_slope_weight == 0.0
    assert cfg.damping_curvature_weight == 0.0
    assert cfg.sim_slot_outlier_filter_enabled is True
    assert cfg.sim_slot_outlier_min_samples == 8
    assert cfg.sim_slot_outlier_mad_threshold == 12.0
    assert cfg.sim_slot_outlier_scale_floor_ratio == 0.02
    assert cfg.real_damping_upper_filter_enabled is True
    assert cfg.real_damping_upper_thresholds == {
        0: 0.15, 1: 0.15, 2: 0.30, 3: 0.30, 4: 0.30, 5: 0.30,
    }
    assert cfg.real_damping_upper_channel_ids[0] == tuple(range(8, 20))
    assert cfg.real_damping_upper_channel_ids[2] == tuple(range(20, 28))
    assert cfg.real_damping_upper_channel_ids[4] == tuple(range(0, 8))


class _ConstantTargetModel(torch.nn.Module):
    outputs_normalized = True

    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs
        self.cfg = SimpleNamespace(K_max=3)

    def forward(self, H, M, excitation_id, channel_id):
        batch = H.shape[0]
        return {
            name: torch.as_tensor(values, dtype=H.dtype, device=H.device)
            .reshape(1, 3)
            .expand(batch, -1)
            for name, values in self.outputs.items()
        }


class _MachDampingModel(torch.nn.Module):
    outputs_normalized = True

    def __init__(self, intercept=0.05, slope=0.10):
        super().__init__()
        self.intercept = float(intercept)
        self.slope = float(slope)
        self.cfg = SimpleNamespace(K_max=3)

    def forward(self, H, M, excitation_id, channel_id):
        value = self.intercept + self.slope * M[:, :1]
        return {"damping": value.expand(-1, 3)}


class _MachTargetModel(torch.nn.Module):
    outputs_normalized = True

    def __init__(self, name, intercept, slope):
        super().__init__()
        self.name = str(name)
        self.intercept = float(intercept)
        self.slope = float(slope)
        self.cfg = SimpleNamespace(K_max=3)

    def forward(self, H, M, excitation_id, channel_id):
        value = self.intercept + self.slope * M[:, :1]
        return {self.name: value.expand(-1, 3)}


def _normalizer(names, mean, std):
    return Normalizer(
        mean=np.asarray(mean, dtype=np.float32),
        std=np.asarray(std, dtype=np.float32),
        pole_target="freq_damping",
        residue_target="polar",
        names=list(names),
    )


def _support_normalizer(names, mean, std, slots):
    normalizer = _normalizer(names, mean, std)
    normalizer.support = {
        "schema_version": 2,
        "coordinate_support_scope": "training_and_validation_rows",
        "slope_fit_scope": "training_rows_only",
        "slots": slots,
    }
    return normalizer


def _component_bundle(model, input_norm, output_norm, *, fidelity_mode="multifidelity"):
    return {
        "model": model,
        "input_normalizer": input_norm,
        "output_normalizer_sim": output_norm,
        "output_normalizer_real": output_norm if fidelity_mode == "multifidelity" else None,
        "fidelity_mode": fidelity_mode,
        "prediction_output": "high" if fidelity_mode == "multifidelity" else "low",
    }


def test_fdr_inference_merges_components_and_reconstructs_physical_values():
    input_norm = _normalizer(("H", "M"), (0.0, 0.0), (1.0, 1.0))
    bundle = {
        "frequency": _component_bundle(
            _ConstantTargetModel({"freq": [10.0, 20.0, 30.0]}),
            input_norm,
            _normalizer(("freq",), (0.0,), (1.0,)),
        ),
        "damping": _component_bundle(
            _ConstantTargetModel({"damping": [0.1, 0.2, 0.3]}),
            input_norm,
            _normalizer(("damping",), (0.0,), (1.0,)),
        ),
        "residue_magnitude": _component_bundle(
            _ConstantTargetModel({"magnitude": [1.0, 2.0, 3.0]}),
            input_norm,
            _normalizer(("magnitude",), (0.0,), (1.0,)),
        ),
        "residue_phase": _component_bundle(
            _ConstantTargetModel(
                {"sin_theta": [0.0, 0.6, 1.0], "cos_theta": [1.0, 0.8, 0.0]}
            ),
            input_norm,
            _normalizer(("sin_theta", "cos_theta"), (0.0, 0.0), (1.0, 1.0)),
        ),
    }
    output = _run_infer_fdr_bundle_batched(
        bundle,
        torch.device("cpu"),
        np.asarray([3.0], dtype=np.float32),
        np.asarray([0.98], dtype=np.float32),
        np.asarray([1], dtype=np.int64),
        np.asarray([8], dtype=np.int64),
    )
    np.testing.assert_allclose(output["omega"][0], 2.0 * np.pi * np.asarray([10.0, 20.0, 30.0]))
    np.testing.assert_allclose(output["rho"][0], np.asarray([1.0, 2.0, 3.0]), rtol=1e-6)
    np.testing.assert_allclose(output["sin_theta"][0], np.asarray([0.0, 0.6, 1.0]), atol=1e-6)
    np.testing.assert_allclose(output["cos_theta"][0], np.asarray([1.0, 0.8, 0.0]), atol=1e-6)
    assert np.all(output["alpha"] < 0.0)
    assert np.all(output["valid_mask"] == 1.0)


def test_prediction_damping_bounds_mask_the_whole_slot_after_x200():
    input_norm = _normalizer(("H", "M"), (0.0, 0.0), (1.0, 1.0))
    bundle = {
        "frequency": _component_bundle(
            _ConstantTargetModel({"freq": [10.0, 20.0, 30.0]}),
            input_norm,
            _normalizer(("freq",), (0.0,), (1.0,)),
        ),
        "damping": _component_bundle(
            _ConstantTargetModel({"damping": [0.009, 0.01, 0.60]}),
            input_norm,
            _normalizer(("damping",), (0.0,), (1.0,)),
        ),
        "residue_magnitude": _component_bundle(
            _ConstantTargetModel({"magnitude": [1.0, 2.0, 3.0]}),
            input_norm,
            _normalizer(("magnitude",), (0.0,), (1.0,)),
        ),
        "residue_phase": _component_bundle(
            _ConstantTargetModel(
                {"sin_theta": [0.0, 0.6, 1.0], "cos_theta": [1.0, 0.8, 0.0]}
            ),
            input_norm,
            _normalizer(("sin_theta", "cos_theta"), (0.0, 0.0), (1.0, 1.0)),
        ),
    }
    arguments = (
        bundle,
        torch.device("cpu"),
        np.asarray([3.0], dtype=np.float32),
        np.asarray([0.98], dtype=np.float32),
        np.asarray([1], dtype=np.int64),
        np.asarray([8], dtype=np.int64),
    )

    default_output = _run_infer_fdr_bundle_batched(*arguments)
    expected_default = np.asarray([[0.0, 1.0, 1.0]], dtype=np.float32)
    for mask_name in ("valid_mask", "damping_valid_mask"):
        np.testing.assert_array_equal(default_output[mask_name], expected_default)
    expected_scan = np.ones((1, 3), dtype=np.float32)
    for mask_name in (
        "scan_valid_mask",
        "frequency_valid_mask",
        "magnitude_valid_mask",
        "phase_valid_mask",
    ):
        np.testing.assert_array_equal(default_output[mask_name], expected_scan)

    bounded_output = _run_infer_fdr_bundle_batched(
        *arguments, prediction_damping_max=60.0
    )
    expected_bounded = np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32)
    for mask_name in ("valid_mask", "damping_valid_mask"):
        np.testing.assert_array_equal(bounded_output[mask_name], expected_bounded)
    for mask_name in (
        "scan_valid_mask",
        "frequency_valid_mask",
        "magnitude_valid_mask",
        "phase_valid_mask",
    ):
        np.testing.assert_array_equal(bounded_output[mask_name], expected_scan)

    y_hat, freq_slots, damping_slots = _build_yhat_and_slot_strings(
        bounded_output["alpha"][0],
        bounded_output["omega"][0],
        bounded_output["rho"][0],
        bounded_output["sin_theta"][0],
        bounded_output["cos_theta"][0],
        bounded_output["scan_valid_mask"][0],
        "RRY",
        None,
        np.ones(64),
        1.0 / 512.0,
        frequency_valid_mask_row=bounded_output["frequency_valid_mask"][0],
        damping_valid_mask_row=bounded_output["damping_valid_mask"][0],
    )
    assert all(freq_slots)
    assert damping_slots[0] == ""
    assert damping_slots[1] != ""
    assert damping_slots[2] == ""
    assert np.any(np.abs(y_hat) > 0.0)

    _, pr_freq_slots, pr_damping_slots = _build_yhat_and_slot_strings(
        bounded_output["alpha"][0],
        bounded_output["omega"][0],
        bounded_output["rho"][0],
        bounded_output["sin_theta"][0],
        bounded_output["cos_theta"][0],
        bounded_output["valid_mask"][0],
        "RRY",
        None,
        np.ones(64),
        1.0 / 512.0,
        frequency_valid_mask_row=bounded_output["valid_mask"][0],
        damping_valid_mask_row=bounded_output["valid_mask"][0],
    )
    assert pr_freq_slots[0] == "" and pr_damping_slots[0] == ""
    assert pr_freq_slots[1] != "" and pr_damping_slots[1] != ""
    assert pr_freq_slots[2] == "" and pr_damping_slots[2] == ""

    np.testing.assert_array_equal(
        _prediction_damping_slot_mask(
            np.asarray([0.009, 0.01, 0.60]), damping_min=None, damping_max=None
        ),
        np.ones(3, dtype=bool),
    )
    assert _parse_optional_float("none") is None
    try:
        _prediction_damping_slot_mask(
            np.asarray([0.10]), damping_min=20.0, damping_max=10.0
        )
    except ValueError:
        pass
    else:
        raise AssertionError("An inverted prediction damping range was accepted")


def test_damping_tanh_pull_has_bounded_slope_and_positive_floor():
    boundary = np.asarray([0.15, -0.01], dtype=np.float64)
    center = np.asarray([0.09, 0.04], dtype=np.float64)
    slope_limit = 0.03
    epsilon = 1e-6
    at_boundary = _bounded_damping_tanh_pull(
        boundary, center, np.zeros(2), slope_limit
    )
    just_outside = _bounded_damping_tanh_pull(
        boundary, center, np.full(2, epsilon), slope_limit
    )
    far_outside = _bounded_damping_tanh_pull(
        boundary, center, np.full(2, 100.0), slope_limit
    )

    assert at_boundary[0] == boundary[0]
    assert at_boundary[1] == 1e-6
    assert np.all(np.abs((just_outside - at_boundary) / epsilon) <= slope_limit + 1e-6)
    np.testing.assert_allclose(far_outside, center, atol=1e-8)
    assert np.all(far_outside > 0.0)


def test_height_conditioned_mach_bounds_interpolate_only_supported_heights():
    support = {
        "training_height_mach_envelope": [
            {"height": 3.0, "mach_min": 0.5, "mach_max": 1.0, "count": 3},
            {"height": 8.0, "mach_min": 0.7, "mach_max": 1.2, "count": 3},
        ]
    }
    lower, upper, supported = _height_conditioned_mach_bounds(
        support, np.asarray([2.0, 3.0, 5.5, 8.0, 9.0])
    )
    np.testing.assert_allclose(lower, [0.5, 0.5, 0.6, 0.7, 0.7])
    np.testing.assert_allclose(upper, [1.0, 1.0, 1.1, 1.2, 1.2])
    np.testing.assert_array_equal(supported, [False, True, True, True, False])


def test_c1_smooth_linear_tail_matches_boundary_and_linear_tail_derivatives():
    transition = 0.05
    epsilon = 1e-6
    boundary = np.asarray([1.0])
    value = np.asarray([0.15])
    boundary_slope = np.asarray([0.10])
    tail_slope = np.asarray([0.02])

    def evaluate(mach):
        return _c1_smooth_linear_tail(
            np.asarray([mach]),
            boundary,
            value,
            boundary_slope,
            tail_slope,
            transition,
            side=1,
        )[0]

    assert evaluate(1.0) == value[0]
    boundary_derivative = (evaluate(1.0 + epsilon) - evaluate(1.0)) / epsilon
    assert abs(boundary_derivative - boundary_slope[0]) < 1e-5
    derivative_before_join = (
        evaluate(1.0 + transition) - evaluate(1.0 + transition - epsilon)
    ) / epsilon
    derivative_after_join = (
        evaluate(1.0 + transition + epsilon) - evaluate(1.0 + transition)
    ) / epsilon
    assert abs(derivative_before_join - tail_slope[0]) < 1e-5
    assert abs(derivative_after_join - tail_slope[0]) < 1e-8
    assert abs((evaluate(1.20) - evaluate(1.10)) / 0.10 - tail_slope[0]) < 1e-10

    def evaluate_lower(mach):
        return _c1_smooth_linear_tail(
            np.asarray([mach]),
            boundary,
            value,
            boundary_slope,
            tail_slope,
            transition,
            side=-1,
        )[0]

    lower_boundary_derivative = (
        evaluate_lower(1.0) - evaluate_lower(1.0 - epsilon)
    ) / epsilon
    assert evaluate_lower(1.0) == value[0]
    assert abs(lower_boundary_derivative - boundary_slope[0]) < 1e-5
    assert (
        abs((evaluate_lower(0.90) - evaluate_lower(0.80)) / 0.10 - tail_slope[0])
        < 1e-10
    )


def test_fdr_inference_defaults_to_allow_and_smooth_linear_preserves_network_boundary():
    parameter = inspect.signature(_run_infer_fdr_bundle_batched).parameters[
        "extrapolation_policy"
    ]
    assert parameter.default == "allow"
    assert inspect.signature(run_infer_batched).parameters["extrapolation_policy"].default == "allow"

    envelope = [
        {"height": 3.0, "mach_min": 0.5, "mach_max": 1.0, "count": 4},
        {"height": 8.0, "mach_min": 0.7, "mach_max": 1.2, "count": 4},
    ]
    slots = [
        {
            "slot_index": slot,
            "slot_number": slot + 1,
            "height_min": 3.0,
            "height_max": 8.0,
            "mach_min": 0.5,
            "mach_max": 1.0,
            "slope_limit": 0.02,
            "train_count": 10,
            "training_height_mach_envelope": envelope,
        }
        for slot in range(3)
    ]
    input_norm = _normalizer(("H", "M"), (0.0, 0.0), (1.0, 1.0))
    bundle = {
        "frequency": _component_bundle(
            _MachTargetModel("freq", 10.0, 1.0),
            input_norm,
            _support_normalizer(("freq",), (0.0,), (1.0,), slots),
        ),
        "damping": _component_bundle(
            _MachDampingModel(),
            input_norm,
            _support_normalizer(("damping",), (0.0,), (1.0,), slots),
        ),
        "residue_magnitude": _component_bundle(
            _MachTargetModel("magnitude", 1.0, 1.0),
            input_norm,
            _support_normalizer(("magnitude",), (0.0,), (1.0,), slots),
        ),
        "residue_phase": _component_bundle(
            _ConstantTargetModel(
                {"sin_theta": [0.0, 0.6, 1.0], "cos_theta": [1.0, 0.8, 0.0]}
            ),
            input_norm,
            _support_normalizer(
                ("sin_theta", "cos_theta"), (0.0, 0.0), (1.0, 1.0), slots
            ),
        ),
    }
    mach = np.asarray(
        [0.95, 0.999, 1.0, 1.001, 1.05, 1.06, 1.10], dtype=np.float32
    )
    output = _run_infer_fdr_bundle_batched(
        bundle,
        torch.device("cpu"),
        np.full_like(mach, 3.0),
        mach,
        np.ones(len(mach), dtype=np.int64),
        np.full(len(mach), 8, dtype=np.int64),
        extrapolation_policy="smooth_linear",
        prediction_damping_min=None,
    )
    damping = -output["alpha"][:, 0] / np.hypot(
        output["alpha"][:, 0], output["omega"][:, 0]
    )

    np.testing.assert_allclose(damping[:3], 0.05 + 0.10 * mach[:3], atol=1e-6)
    assert abs(damping[2] - 0.15) < 1e-6
    boundary_slope = (damping[3] - damping[2]) / (mach[3] - mach[2])
    assert abs(boundary_slope - 0.10) < 2e-3
    tail_slopes_x200 = 200.0 * np.diff(damping[4:]) / np.diff(mach[4:])
    np.testing.assert_allclose(tail_slopes_x200, 20.0, atol=2e-4)
    np.testing.assert_allclose(output["omega"][-1, 0], 2.0 * np.pi * 11.1, atol=1e-5)
    np.testing.assert_allclose(output["rho"][-1, 0], 2.1, atol=1e-6)
    assert np.all(output["damping_valid_mask"] == 1.0)

    outside_height = _run_infer_fdr_bundle_batched(
        bundle,
        torch.device("cpu"),
        np.asarray([2.0], dtype=np.float32),
        np.asarray([1.5], dtype=np.float32),
        np.asarray([1], dtype=np.int64),
        np.asarray([8], dtype=np.int64),
        prediction_damping_min=None,
    )
    outside_height_damping = -outside_height["alpha"][0, 0] / np.hypot(
        outside_height["alpha"][0, 0], outside_height["omega"][0, 0]
    )
    np.testing.assert_allclose(outside_height_damping, 0.20, atol=1e-6)


def test_slot_support_uses_train_and_validation_coordinates_but_train_only_slopes():
    cfg = Config()
    cfg.K_max = 1

    def row(height, mach, damping):
        return SimpleNamespace(
            H=height,
            M=mach,
            mask=np.ones(1, dtype=np.float32),
            damping=np.asarray([damping], dtype=np.float32),
        )

    support = _build_slot_support(
        [row(3.0, 0.7, 0.05), row(3.0, 0.8, 0.06)],
        [row(8.0, 1.2, 0.40)],
        cfg,
    )
    slot = support["slots"][0]
    assert support["schema_version"] == 3
    assert support["coordinate_support_scope"] == "training_and_validation_rows"
    assert support["slope_fit_scope"] == "training_rows_only"
    assert slot["height_max"] == 8.0
    assert slot["mach_max"] == 1.2
    assert slot["support_count"] == 3
    assert slot["training_height_mach_envelope"] == [
        {"height": 3.0, "mach_min": 0.7, "mach_max": 0.8, "count": 2}
    ]
    assert slot["damping_max"] < 0.40
    np.testing.assert_allclose(slot["slope_limit"], 0.15, rtol=1e-5)


def test_inventory_entry_lists_all_four_component_artifacts():
    entry = model_entry(
        excitation_id=1,
        channel_id=8,
        row_count=31,
        status="trained",
    )
    assert tuple(entry["components"]) == tuple(FDR_COMPONENT_TARGETS)
    assert entry["components"]["frequency"]["checkpoint"] == "FXD_DC/LVT1/frequency/model.pth"
    assert entry["components"]["residue_magnitude"]["target_names"] == ["magnitude"]
    assert entry["components"]["residue_phase"]["target_names"] == [
        "sin_theta",
        "cos_theta",
    ]
    assert entry["components"]["damping"]["slot_support"] == "FXD_DC/LVT1/damping/slot_support.json"


def test_target_transforms_roundtrip_without_damping_upper_bound():
    specs = build_target_transform_specs(
        ("freq", "damping", "magnitude"), (1,), enabled=True
    )
    assert specs["damping"]["kind"] == "identity"
    physical = {
        "freq": np.asarray([1.0, 30.0, 100.0]),
        "damping": np.asarray([0.01, 0.075, 0.149]),
        "magnitude": np.asarray([1e-4, 2.0, 100.0]),
    }
    for name, values in physical.items():
        recovered = decode_numpy(encode_numpy(values, specs[name]), specs[name])
        np.testing.assert_allclose(recovered, values, rtol=1e-6, atol=1e-8)
    assert np.all(decode_numpy(np.asarray([-1e6, 1e6]), specs["magnitude"]) > 0.0)


def test_sim_damping_filter_keeps_high_positive_values_and_drops_non_positive_values():
    row = SimpleNamespace(
        H=3.0,
        M=0.9,
        exc_id=1,
        ch_id=12,
        mask=np.ones(3, dtype=np.float32),
        damping=np.asarray([0.10, 0.40, -0.01], dtype=np.float32),
        K_valid=3,
    )
    filtered, dropped = _filter_damping_slots([row], "sim_train")
    np.testing.assert_array_equal(filtered[0].mask, [1.0, 1.0, 0.0])
    assert filtered[0].K_valid == 2
    assert len(dropped) == 1
    assert dropped[0]["reason"] == "non_positive"


def test_real_damping_filter_applies_excitation_channel_thresholds_to_whole_slots():
    cfg = Config()
    cases = (
        (1, 8, 0.15, 0.151),
        (2, 20, 0.30, 0.301),
        (4, 0, 0.30, 0.301),
    )
    for excitation_id, channel_id, boundary, above in cases:
        row = SimpleNamespace(
            H=3.0,
            M=0.9,
            exc_id=excitation_id,
            ch_id=channel_id,
            mask=np.ones(3, dtype=np.float32),
            damping=np.asarray([boundary, above, 0.10], dtype=np.float32),
            K_valid=3,
        )
        filtered, dropped = _filter_damping_slots(
            [row], "train", cfg, apply_real_upper_limits=True
        )
        np.testing.assert_array_equal(filtered[0].mask, [1.0, 0.0, 1.0])
        assert filtered[0].K_valid == 2
        assert len(dropped) == 1
        assert dropped[0]["reason"] == "above_real_upper_limit"
        assert dropped[0]["upper_threshold"] == boundary


def test_real_damping_filter_does_not_apply_upper_limit_outside_mapped_channels():
    cfg = Config()
    row = SimpleNamespace(
        H=3.0,
        M=0.9,
        exc_id=1,
        ch_id=7,
        mask=np.ones(1, dtype=np.float32),
        damping=np.asarray([0.90], dtype=np.float32),
        K_valid=1,
    )
    filtered, dropped = _filter_damping_slots(
        [row], "train", cfg, apply_real_upper_limits=True
    )
    np.testing.assert_array_equal(filtered[0].mask, [1.0])
    assert not dropped


def test_sim_frequency_or_damping_outlier_masks_the_whole_slot():
    cfg = Config()
    rows = []
    for index in range(12):
        frequency = np.asarray(
            [10.0 + 0.05 * ((index % 3) - 1), 20.0, 30.0],
            dtype=np.float32,
        )
        damping = np.asarray([0.05, 0.08, 0.10], dtype=np.float32)
        if index == 10:
            frequency[0] = 100.0
        if index == 11:
            damping[1] = 0.80
        rows.append(
            SimpleNamespace(
                H=3.0,
                M=0.7 + 0.01 * index,
                exc_id=1,
                ch_id=12,
                mask=np.ones(3, dtype=np.float32),
                freq=frequency,
                damping=damping,
                K_valid=3,
            )
        )

    filtered, validation, report = _filter_sim_outlier_slots(rows, [], cfg)

    assert not validation
    np.testing.assert_array_equal(filtered[10].mask, [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(filtered[11].mask, [1.0, 0.0, 1.0])
    assert all(np.all(row.mask == 1.0) for row in rows)
    assert report["masked_slot_count"] == 2
    assert {tuple(item["reasons"]) for item in report["masked_slots"]} == {
        ("frequency_outlier",),
        ("damping_outlier",),
    }


def test_sim_outlier_filter_is_disabled_for_small_groups():
    cfg = Config()
    rows = []
    for index in range(cfg.sim_slot_outlier_min_samples - 1):
        frequency = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
        if index == 0:
            frequency[0] = 1000.0
        rows.append(
            SimpleNamespace(
                H=3.0,
                M=0.7 + 0.01 * index,
                exc_id=1,
                ch_id=12,
                mask=np.ones(3, dtype=np.float32),
                freq=frequency,
                damping=np.asarray([0.05, 0.08, 0.10], dtype=np.float32),
                K_valid=3,
            )
        )

    filtered, _validation, report = _filter_sim_outlier_slots(rows, [], cfg)

    assert all(np.all(row.mask == 1.0) for row in filtered)
    assert report["masked_slot_count"] == 0
    assert all(not group["filter_active"] for group in report["groups"])


def test_new_normalizer_metadata_decodes_fdr_outputs_to_physical_domain():
    input_norm = _normalizer(("H", "M"), (0.0, 0.0), (1.0, 1.0))
    specs = build_target_transform_specs(
        ("freq", "damping", "magnitude"), (1,), enabled=True
    )

    def transformed_normalizer(name):
        return Normalizer(
            mean=np.asarray([0.0], dtype=np.float32),
            std=np.asarray([1.0], dtype=np.float32),
            pole_target="freq_damping",
            residue_target="polar",
            names=[name],
            target_transforms={name: specs[name]},
        )

    frequency = np.log(np.asarray([10.0, 20.0, 30.0]))
    damping = encode_numpy(np.asarray([0.01, 0.05, 0.149]), specs["damping"])
    magnitude = np.log(np.asarray([1e-3, 2.0, 5.0]))
    bundle = {
        "frequency": _component_bundle(
            _ConstantTargetModel({"freq": frequency}), input_norm, transformed_normalizer("freq")
        ),
        "damping": _component_bundle(
            _ConstantTargetModel({"damping": damping}), input_norm, transformed_normalizer("damping")
        ),
        "residue_magnitude": _component_bundle(
            _ConstantTargetModel({"magnitude": magnitude}), input_norm, transformed_normalizer("magnitude")
        ),
        "residue_phase": _component_bundle(
            _ConstantTargetModel({"sin_theta": [0.0, 0.6, 1.0], "cos_theta": [1.0, 0.8, 0.0]}),
            input_norm,
            _normalizer(("sin_theta", "cos_theta"), (0.0, 0.0), (1.0, 1.0)),
        ),
    }
    output = _run_infer_fdr_bundle_batched(
        bundle,
        torch.device("cpu"),
        np.asarray([3.0], dtype=np.float32),
        np.asarray([0.98], dtype=np.float32),
        np.asarray([1], dtype=np.int64),
        np.asarray([8], dtype=np.int64),
    )
    np.testing.assert_allclose(output["rho"][0], [1e-3, 2.0, 5.0], rtol=1e-6)
    recovered_damping = -output["alpha"][0] / np.hypot(output["alpha"][0], output["omega"][0])
    np.testing.assert_allclose(recovered_damping, [0.01, 0.05, 0.149], rtol=1e-5)
    np.testing.assert_array_equal(
        output["damping_valid_mask"], np.asarray([[1.0, 1.0, 1.0]])
    )
    np.testing.assert_array_equal(
        output["valid_mask"], np.asarray([[1.0, 1.0, 1.0]])
    )


def test_damping_plot_mask_is_independent_from_invalid_magnitude():
    alpha = np.asarray([-1.0, -2.0, -3.0])
    omega = np.asarray([10.0, 20.0, 30.0])
    _, freq_slots, damping_slots = _build_yhat_and_slot_strings(
        alpha,
        omega,
        np.asarray([-1.0, -1.0, -1.0]),
        np.zeros(3),
        np.ones(3),
        np.zeros(3),
        "RRY",
        None,
        np.ones(8),
        1.0 / 512.0,
        frequency_valid_mask_row=np.ones(3),
        damping_valid_mask_row=np.ones(3),
    )
    assert all(freq_slots)
    assert all(damping_slots)
