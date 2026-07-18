import csv
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from surrogate_model.config import Config
from surrogate_model.data.io import parse_rows_to_struct
from surrogate_model.train.loops import train_multifidelity_component


def _mode(real_damping: float, frequency_hz: float, magnitude: float, phase: float):
    omega = 2.0 * math.pi * float(frequency_hz)
    damping = float(real_damping)
    real_part = -damping * omega / math.sqrt(max(1.0 - damping * damping, 1e-8))
    return [real_part, omega, magnitude, phase]


def _row(height: float, mach: float, *, real: bool):
    damping_offset = 0.01 if real else 0.0
    values = [height, mach, 1, 8]
    for slot, frequency in enumerate((8.0, 14.0, 22.0)):
        values.extend(
            _mode(
                0.06 + damping_offset + 0.005 * slot + 0.01 * (mach - 0.7),
                frequency + 0.3 * mach,
                1.0 + slot,
                0.2 * slot,
            )
        )
    return values


def test_damping_multifidelity_training_smoke():
    sim_rows = [
        _row(height, mach, real=False)
        for height in (3.0, 8.0)
        for mach in (0.4, 0.7, 0.9, 1.2)
    ]
    real_rows = [
        _row(height, mach, real=True)
        for height in (3.0, 8.0)
        for mach in (0.7, 0.8, 0.9)
    ]
    sim_parsed, _ = parse_rows_to_struct(sim_rows)
    real_parsed, _ = parse_rows_to_struct(real_rows)

    cfg = Config()
    cfg.target_component = "damping"
    cfg.model_target_names = ["damping"]
    cfg.low_hidden = [8]
    cfg.adapt_hidden = [8]
    cfg.dropout_prob = 0.0
    cfg.epochs_stage1 = 2
    cfg.epochs_stage2 = 2
    cfg.early_stop_patience_s1 = 0
    cfg.early_stop_patience_s2 = 0
    cfg.checkpoint_selection = "training"
    cfg.use_val_split = False
    cfg.batch_sim = 4
    cfg.batch_real = 3
    cfg.num_workers = 0
    cfg.multifidelity_real_loss_weight = 1.0
    cfg.multifidelity_sim_loss_weight = 0.2

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        args = SimpleNamespace(
            device="cpu",
            out_dir=str(out_dir),
            kfold=0,
            model_save_path=str(out_dir / "model.pth"),
            normalizer_dir=str(out_dir),
            train_log_path=str(out_dir / "train_log.csv"),
            checkpoint_metadata={},
            fold_index=0,
            num_folds=1,
            deploy_model_save_path="",
            deploy_normalizer_dir="",
        )
        best, train_history, _validation_history = train_multifidelity_component(
            0, cfg, sim_parsed, real_parsed, args
        )
        assert math.isfinite(best)
        assert len(train_history) == 2
        checkpoint = torch.load(out_dir / "model.pth", map_location="cpu", weights_only=True)
        assert (
            checkpoint["slot_support"]["height_mach_envelope_scope"]
            == "training_rows_only"
        )
        assert checkpoint["slot_support"]["slots"][0][
            "training_height_mach_envelope"
        ]
        assert checkpoint["checkpoint_selection"] == "training"
        with (out_dir / "train_log.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        stage2 = [row for row in rows if row["stage"] == "2"]
        assert stage2
        assert all(row["loss_sim"] != "" for row in stage2)


if __name__ == "__main__":
    test_damping_multifidelity_training_smoke()
    print("damping multifidelity training smoke: PASS")
