from types import SimpleNamespace

import numpy as np
import torch

from surrogate_model.config import Config
from surrogate_model.data.normalizer import Normalizer
from surrogate_model.models.blocks import TargetMFDNN
from surrogate_model.models.checkpoint_utils import build_model_from_checkpoint
from surrogate_model.test.Predict_user import _run_infer_fdr_bundle_batched
from surrogate_model.test.Predict_user import _build_yhat_and_slot_strings
from surrogate_model.train.dataloaders import _filter_damping_slots
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


def test_fdr_defaults_restore_capacity_without_dropout():
    cfg = Config()
    assert cfg.fdr_component_low_hidden["frequency"] == [64, 128, 64]
    assert cfg.fdr_component_low_hidden["damping"] == [64, 128, 64]
    assert cfg.fdr_component_dropout_prob["frequency"] == 0.0
    assert cfg.fdr_component_dropout_prob["damping"] == 0.0
    assert cfg.fdr_component_weight_decay["damping"] == 5e-5
    assert cfg.fdr_component_early_stop_patience["damping"] == 500
    assert cfg.fdr_component_checkpoint_train_loss_ceiling["frequency"] == 0.05
    assert cfg.fdr_component_checkpoint_train_loss_ceiling["damping"] is None
    assert cfg.fdr_training_profile_version == 4
    assert cfg.fdr_component_training_profile_version["frequency"] == 3
    assert cfg.fdr_component_training_profile_version["damping"] == 4
    assert cfg.damping_slope_weight == 0.0
    assert cfg.damping_curvature_weight == 0.0


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


def _normalizer(names, mean, std):
    return Normalizer(
        mean=np.asarray(mean, dtype=np.float32),
        std=np.asarray(std, dtype=np.float32),
        pole_target="freq_damping",
        residue_target="polar",
        names=list(names),
    )


def test_fdr_inference_merges_components_and_reconstructs_physical_values():
    input_norm = _normalizer(("H", "M"), (0.0, 0.0), (1.0, 1.0))
    bundle = {
        "frequency": (
            _ConstantTargetModel({"freq": [10.0, 20.0, 30.0]}),
            input_norm,
            _normalizer(("freq",), (0.0,), (1.0,)),
        ),
        "damping": (
            _ConstantTargetModel({"damping": [0.1, 0.2, 0.3]}),
            input_norm,
            _normalizer(("damping",), (0.0,), (1.0,)),
        ),
        "residue_magnitude": (
            _ConstantTargetModel({"magnitude": [1.0, 2.0, 3.0]}),
            input_norm,
            _normalizer(("magnitude",), (0.0,), (1.0,)),
        ),
        "residue_phase": (
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


def test_damping_filter_keeps_high_positive_values_and_drops_non_positive_values():
    row = SimpleNamespace(
        H=3.0,
        M=0.9,
        exc_id=1,
        ch_id=12,
        mask=np.ones(3, dtype=np.float32),
        damping=np.asarray([0.10, 0.40, -0.01], dtype=np.float32),
        K_valid=3,
    )
    filtered, dropped = _filter_damping_slots([row], "train")
    np.testing.assert_array_equal(filtered[0].mask, [1.0, 1.0, 0.0])
    assert filtered[0].K_valid == 2
    assert len(dropped) == 1
    assert dropped[0]["reason"] == "non_positive"


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
        "frequency": (
            _ConstantTargetModel({"freq": frequency}), input_norm, transformed_normalizer("freq")
        ),
        "damping": (
            _ConstantTargetModel({"damping": damping}), input_norm, transformed_normalizer("damping")
        ),
        "residue_magnitude": (
            _ConstantTargetModel({"magnitude": magnitude}), input_norm, transformed_normalizer("magnitude")
        ),
        "residue_phase": (
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
    assert np.all(output["damping_valid_mask"] == 1.0)


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
