from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from surrogate_model.cli import train, train_incremental
from surrogate_model.config import Config
from surrogate_model.data.io import parse_rows_to_struct, read_real_dataset
from surrogate_model.train.dataloaders import build_single_fidelity_dataloaders
from surrogate_model.train.full_training_runner import run_full_training
from surrogate_model.utils.model_artifacts import (
    build_dataset_manifest,
    build_dataset_snapshot,
    build_input_dataset_snapshot,
    dataset_fingerprint,
    group_rows,
    merge_dataset_rows,
    sha256_file,
)


def row(h: float, m: float, excitation: int, channel: int, alpha: float = -1.0):
    return [h, m, excitation, channel, alpha, 20.0, 1.0, 0.25]


class DatasetVersioningTests(unittest.TestCase):
    def test_replace_is_whole_group_and_append_adds_new_group(self):
        parent = [row(3.0, 0.98, 1, 0), row(3.0, 0.98, 1, 1), row(5.5, 1.05, 0, 0)]
        correction = [row(3.0, 0.98, 1, 0, -2.0), row(3.0, 0.98, 1, 1, -2.0), row(8.0, 1.2, 2, 0)]
        merged, info = merge_dataset_rows(parent, correction)
        grouped = group_rows(merged, source="test")
        self.assertEqual([value[4] for value in grouped[(300, 98, 1)]], [-2.0, -2.0])
        self.assertIn((800, 120, 2), grouped)
        self.assertEqual(info["replaced_group_keys"], [(300, 98, 1)])
        self.assertEqual(info["added_group_keys"], [(800, 120, 2)])

    def test_partial_or_duplicate_correction_is_rejected(self):
        parent = [row(3.0, 0.98, 1, 0), row(3.0, 0.98, 1, 1)]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            merge_dataset_rows(parent, [row(3.0, 0.98, 1, 0, -2.0)])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            merge_dataset_rows(parent, [row(4.0, 1.0, 0, 0), row(4.0, 1.0, 0, 0)])

    def test_parent_snapshot_is_immutable_and_fingerprint_is_content_based(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent = [row(3.0, 0.98, 1, 0), row(4.0, 1.0, 0, 0)]
            build_dataset_snapshot(parent, root / "parent")
            before = {path.name: sha256_file(path) for path in (root / "parent").iterdir()}
            merged, _ = merge_dataset_rows(parent, [row(4.0, 1.0, 0, 0, -2.0)])
            build_dataset_snapshot(merged, root / "child")
            after = {path.name: sha256_file(path) for path in (root / "parent").iterdir()}
            self.assertEqual(before, after)

            groups = [{"group_key": [300, 98, 1], "sha256": "sha256:a", "rows": 1}]
            splits = {"train_group_keys": [[300, 98, 1]], "validation_group_keys": []}
            self.assertEqual(dataset_fingerprint(groups, splits), dataset_fingerprint(groups, splits))
            changed = [{"group_key": [300, 98, 1], "sha256": "sha256:b", "rows": 1}]
            self.assertNotEqual(dataset_fingerprint(groups, splits), dataset_fingerprint(changed, splits))

    def test_base_single_file_snapshot_is_an_exact_reusable_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.txt"
            source.write_text("3\t0.9\t0\t0\t-1\t20\t1\t0.25\n", encoding="utf-8")
            mapping = build_input_dataset_snapshot(source, root / "snapshot")
            copied = root / "snapshot" / "real_dataset.txt"
            self.assertEqual(source.read_bytes(), copied.read_bytes())
            self.assertEqual(mapping[(300, 90, 0)], ["dataset_snapshot/real_dataset.txt"])
            self.assertEqual(read_real_dataset(root / "snapshot"), [row(3.0, 0.9, 0, 0)])


class FullRetrainTests(unittest.TestCase):
    def test_incremental_entry_has_no_lora_replay_teacher_or_parent_weight_load(self):
        source = inspect.getsource(train_incremental)
        forbidden = (
            "freeze_base_train_lora_only", "get_lora_parameters", "load_base_weights_into_lora_model",
            "build_replay_data", "kd_loss_fn", "teacher =", "model.load_state_dict(parent_state)",
        )
        for token in forbidden:
            self.assertNotIn(token, source)
        self.assertIn("run_full_training", inspect.getsource(train))
        self.assertIn("run_full_training", source)

    def test_prediction_prefers_version_local_normalizers(self):
        prediction_source = (Path(__file__).parent / "Predict_user.py").read_text(encoding="utf-8")
        self.assertIn('version_dir / "Normalizer_Input.pth"', prediction_source)
        self.assertIn('version_dir / "Normalizer_Output.pth"', prediction_source)

    def test_dataloader_has_no_normalizer_write_side_effect(self):
        parsed, _ = parse_rows_to_struct([row(3.0, 0.9, 0, 0), row(4.0, 1.0, 0, 0)])
        cfg = Config()
        cfg.num_workers = 0
        cfg.batch_real = 1
        cfg.explicit_train_group_keys = [(300, 90, 0)]
        cfg.explicit_validation_group_keys = [(400, 100, 0)]
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            build_single_fidelity_dataloaders(parsed, cfg, 0, 0, torch.device("cpu"), out)
            names = {path.name for path in out.iterdir()}
            self.assertNotIn("Normalizer_Input_Real.pth", names)
            self.assertFalse(any(name.startswith("Normalizer_Output_Real_Fold") for name in names))

    def test_one_epoch_dry_run_writes_standard_non_lora_artifacts(self):
        rows = [
            row(3.0, 0.9, 0, 0), row(3.0, 0.9, 0, 1),
            row(4.0, 1.0, 0, 0), row(4.0, 1.0, 0, 1),
        ]
        parsed, stats = parse_rows_to_struct(rows)
        cfg = Config()
        cfg.epochs_stage1 = 1
        cfg.early_stop_patience_s1 = 0
        cfg.batch_real = 2
        cfg.num_workers = 0
        cfg.prefetch = 1
        cfg.embed_excitation_num = stats["excitation_num"]
        cfg.embed_channel_num = stats["channel_num"]
        cfg.explicit_train_group_keys = [(300, 90, 0)]
        cfg.explicit_validation_group_keys = [(400, 100, 0)]
        cfg.require_validation = True
        args = SimpleNamespace(device="cpu", kfold=0, debug_overfit_n=0, debug_overfit_offset=0)
        with tempfile.TemporaryDirectory() as temp:
            version_dir = Path(temp) / "v2"
            build_dataset_snapshot(rows, version_dir / "dataset_snapshot")
            manifest = build_dataset_manifest(
                rows=rows,
                model_version="v2",
                snapshot_dir=version_dir / "dataset_snapshot",
                train_group_keys=[(300, 90, 0)],
                validation_group_keys=[(400, 100, 0)],
                correction_group_keys=[(300, 90, 0)],
                parent_model_version="v1",
            )
            run_full_training(
                cfg=cfg,
                parsed_rows=parsed,
                model_version="v2",
                version_dir=version_dir,
                args=args,
                dataset_metadata={
                    "manifest": manifest,
                    "training_mode": "correction_full_retrain",
                    "parent_model_version": "v1",
                    "corrected_group_keys": [[300, 90, 0]],
                    "replaced_group_keys": [[300, 90, 0]],
                    "added_group_keys": [],
                    "target_names": ["freq", "damping", "rho", "sin_theta", "cos_theta"],
                },
            )
            expected = {
                "v2.pth", "Normalizer_Input.pth", "Normalizer_Output.pth",
                "train_log.csv", "training_config.json", "damping_slot_filter.log",
            }
            self.assertTrue(expected.issubset({path.name for path in version_dir.iterdir()}))
            forbidden = {
                "Normalizer_Input_Real.pth", "Normalizer_Output_Real_Fold0.pth",
                "replay_guard_split.json", ".incremental_replay_guard_split.json",
            }
            self.assertTrue(forbidden.isdisjoint({path.name for path in version_dir.iterdir()}))
            checkpoint = torch.load(version_dir / "v2.pth", map_location="cpu")
            self.assertFalse(checkpoint["use_lora"])
            self.assertEqual(checkpoint["training_mode"], "correction_full_retrain")
            self.assertFalse(checkpoint["parent_checkpoint_used_for_weights"])
            for key in ("peft_type", "beta_kd", "gamma_replay", "guard_ratio", "lora_rank"):
                self.assertNotIn(key, checkpoint)


if __name__ == "__main__":
    unittest.main()
