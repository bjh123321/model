from types import SimpleNamespace

import numpy as np
import torch

from surrogate_model.config import Config
from surrogate_model.data.normalizer import Normalizer
from surrogate_model.models.blocks import TargetMFDNN
from surrogate_model.models.checkpoint_utils import build_model_from_checkpoint
from surrogate_model.test.Predict_user import _run_infer_fdr_bundle_batched
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
