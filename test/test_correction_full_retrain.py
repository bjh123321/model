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
from surrogate_model.models.blocks import MFDNN
from surrogate_model.train.dataloaders import build_single_fidelity_dataloaders
from surrogate_model.train.full_training_runner import (
    build_checkpoint_metadata,
    run_full_training,
    write_training_config,
)
from surrogate_model.utils.model_artifacts import (
    build_dataset_manifest,
    build_dataset_snapshot,
    build_input_dataset_snapshot,
    dataset_fingerprint,
    group_rows,
    merge_dataset_rows,
    resolve_parent_dataset,
    sha256_file,
    validate_new_version_directory,
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

    def test_same_parent_and_child_version_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            version = Path(temp) / "v1"
            version.mkdir()
            with self.assertRaisesRegex(ValueError, "must differ"):
                validate_new_version_directory(version, version)

    def test_existing_child_version_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent = root / "v1"
            child = root / "v2"
            parent.mkdir()
            child.mkdir()
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                validate_new_version_directory(parent, child)

    def test_consecutive_correction_reads_immediate_parent_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            v1_rows = [row(3.0, 0.9, 0, 0), row(4.0, 1.0, 0, 0)]
            build_dataset_snapshot(v1_rows, root / "v1" / "dataset_snapshot")
            v2_rows, _ = merge_dataset_rows(v1_rows, [row(3.0, 0.9, 0, 0, -2.0)])
            build_dataset_snapshot(v2_rows, root / "v2" / "dataset_snapshot")
            v3_rows, _ = merge_dataset_rows(
                read_real_dataset(root / "v2" / "dataset_snapshot"),
                [row(5.0, 1.1, 0, 0, -3.0)],
            )
            grouped = group_rows(v3_rows, source="v3")
            self.assertEqual(grouped[(300, 90, 0)][0][4], -2.0)
            self.assertIn((500, 110, 0), grouped)

    def test_manifest_records_lineage_changes_and_correction_split(self):
        rows = [row(3.0, 0.9, 0, 0), row(4.0, 1.0, 0, 0)]
        with tempfile.TemporaryDirectory() as temp:
            snapshot = Path(temp) / "dataset_snapshot"
            build_dataset_snapshot(rows, snapshot)
            parent = {"dataset_version": "v1_dataset", "dataset_fingerprint": "sha256:parent"}
            replacement = {"group_key": [300, 90, 0], "old_sha256": "a", "new_sha256": "b"}
            manifest = build_dataset_manifest(
                rows=rows,
                model_version="v2",
                snapshot_dir=snapshot,
                train_group_keys=[(300, 90, 0)],
                validation_group_keys=[(400, 100, 0)],
                correction_group_keys=[(300, 90, 0)],
                parent_manifest=parent,
                parent_model_version="v1",
                changes={"replaced_groups": [replacement], "added_groups": [], "removed_groups": []},
            )
            self.assertEqual(manifest["parent_dataset_version"], "v1_dataset")
            self.assertEqual(manifest["created_from_model"], "v1")
            self.assertIn([300, 90, 0], manifest["splits"]["train_group_keys"])
            self.assertNotIn([300, 90, 0], manifest["splits"]["validation_group_keys"])
            self.assertEqual(manifest["changes"]["replaced_groups"], [replacement])

    def test_fingerprint_changes_when_split_identity_changes(self):
        groups = [{"group_key": [300, 98, 1], "sha256": "sha256:a", "rows": 1}]
        train = {"train_group_keys": [[300, 98, 1]], "validation_group_keys": []}
        validation = {"train_group_keys": [], "validation_group_keys": [[300, 98, 1]]}
        self.assertNotEqual(dataset_fingerprint(groups, train), dataset_fingerprint(groups, validation))

    def test_legacy_parent_uses_explicit_history_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent = root / "v1"
            parent.mkdir()
            history = root / "history.txt"
            history.write_text("3\t0.9\t0\t0\t-1\t20\t1\t0.25\n", encoding="utf-8")
            source, manifest = resolve_parent_dataset(parent, history)
            self.assertEqual(source, history)
            self.assertIsNone(manifest)

    def test_legacy_parent_without_history_fails_clearly(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp) / "v1"
            parent.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "--history_dataset"):
                resolve_parent_dataset(parent, None)


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

    def test_incremental_cli_contains_full_training_overrides_only(self):
        option_strings = {
            option
            for action in train_incremental._parser()._actions
            for option in action.option_strings
        }
        for required in ("--epochs", "--batch_real", "--lr", "--checkpoint_selection", "--history_dataset"):
            self.assertIn(required, option_strings)
        for removed in ("--lora_rank", "--beta_kd", "--gamma_replay", "--replay_fraction"):
            self.assertNotIn(removed, option_strings)

    def test_shared_runner_is_the_only_single_fold_training_bridge(self):
        runner_source = inspect.getsource(run_full_training)
        self.assertIn("train_one_fold(0, cfg, parsed_rows, run_args)", runner_source)
        self.assertIn("run_full_training", inspect.getsource(train.main))
        self.assertIn("run_full_training", inspect.getsource(train_incremental.main))

    def test_fresh_model_defaults_to_all_parameters_trainable(self):
        cfg = Config()
        cfg.use_lora = False
        model = MFDNN(cfg)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_checkpoint_metadata_is_plain_randomly_initialized_model(self):
        manifest = {"dataset_fingerprint": "sha256:dataset"}
        metadata = build_checkpoint_metadata(
            model_version="v2",
            dataset_metadata={
                "manifest": manifest,
                "training_mode": "correction_full_retrain",
                "parent_model_version": "v1",
            },
        )
        self.assertEqual(metadata["initialization_mode"], "random")
        self.assertFalse(metadata["parent_checkpoint_used_for_weights"])
        self.assertFalse(metadata["use_lora"])
        self.assertNotIn("lora_rank", metadata)

    def test_training_config_records_effective_values_and_hashes(self):
        cfg = Config()
        cfg.use_lora = False
        cfg.lr_low_adam = 0.0123
        cfg.batch_real = 7
        cfg._last_training_result = {"best_epoch": 3, "best_metric": 1.25}
        args = SimpleNamespace(device="cpu")
        with tempfile.TemporaryDirectory() as temp:
            version_dir = Path(temp)
            (version_dir / "Normalizer_Input.pth").write_bytes(b"input-normalizer")
            (version_dir / "Normalizer_Output.pth").write_bytes(b"output-normalizer")
            payload = write_training_config(
                cfg=cfg,
                args=args,
                model_version="v2",
                version_dir=version_dir,
                dataset_metadata={
                    "manifest": {"dataset_fingerprint": "sha256:dataset"},
                    "training_mode": "correction_full_retrain",
                    "parent_model_version": "v1",
                    "target_names": ["freq", "damping", "rho", "sin_theta", "cos_theta"],
                },
            )
            self.assertEqual(payload["training"]["learning_rate"], 0.0123)
            self.assertEqual(payload["training"]["batch_size"], 7)
            self.assertEqual(payload["normalizers"]["fit_on"], "training_split_only")
            self.assertTrue(payload["normalizers"]["input_sha256"].startswith("sha256:"))
            self.assertEqual(payload["selection"]["best_epoch"], 3)

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
            self.assertEqual(checkpoint["initialization_mode"], "random")
            self.assertFalse(checkpoint["parent_checkpoint_used_for_weights"])
            for key in ("peft_type", "beta_kd", "gamma_replay", "guard_ratio", "lora_rank"):
                self.assertNotIn(key, checkpoint)


if __name__ == "__main__":
    unittest.main()
