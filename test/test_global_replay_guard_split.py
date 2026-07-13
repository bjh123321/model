from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import torch

from surrogate_model.cli.train_incremental import (
    evaluate_representation_loss,
    iter_incremental_epoch_batches,
    resolve_history_dataset_path,
)
from surrogate_model.config import Config
from surrogate_model.data.build_replay_data import (
    build_replay_data,
    split_history_groups_for_replay_guard,
    stable_group_key,
)
from surrogate_model.models.blocks import (
    MFDNN,
    freeze_base_train_lora_only,
    load_base_weights_into_lora_model,
)


def _groups(excitation_count: int = 6, points_per_excitation: int = 8):
    groups = {}
    for excitation_id in range(excitation_count):
        for index in range(points_per_excitation):
            h = float(3 + (index % 4) * 2)
            m = float(0.7 + (index // 4) * 0.3 + (index % 2) * 0.05)
            key = (h, m, excitation_id)
            groups[key] = [
                f"{h}\t{m}\t{excitation_id}\t{channel}\n" for channel in range(3)
            ]
    return groups


def _write_history(root: Path, points_per_excitation: int = 8) -> Path:
    history = root / "history"
    history.mkdir()
    for excitation_id, prefix in enumerate(
        ("FXD_FDC", "FXD_DC", "PW_FDC", "PW_DC", "JFY_FDC", "JFY_DC")
    ):
        for index in range(points_per_excitation):
            h = float(3 + (index % 4) * 2)
            m = float(0.7 + (index // 4) * 0.3 + (index % 2) * 0.05)
            filename = f"SF_XGD_{prefix}_H{int(round(h * 100)):04d}_M{int(round(m * 100)):03d}.txt"
            rows = [f"{h}\t{m}\t{excitation_id}\t{channel}\n" for channel in range(3)]
            (history / filename).write_text("".join(rows), encoding="utf-8")
    return history


class GlobalReplayGuardSplitTests(unittest.TestCase):
    def test_replay_guard_are_groupwise_and_disjoint(self):
        groups = _groups()
        result = split_history_groups_for_replay_guard(groups, replay_fraction=0.75, seed=42)
        replay = set(result["replay_group_keys"])
        guard = set(result["guard_group_keys"])
        self.assertTrue(replay.isdisjoint(guard))
        self.assertEqual(replay | guard, set(groups))
        for key, rows in groups.items():
            expected_rows = result["replay_rows"] if key in replay else result["guard_rows"]
            encoded = stable_group_key(key)
            self.assertEqual(
                sum(stable_group_key((row[0], row[1], int(row[2]))) == encoded for row in expected_rows),
                len(rows),
            )

    def test_ratio_and_all_excitation_coverage(self):
        result = split_history_groups_for_replay_guard(_groups(), replay_fraction=0.75, seed=42)
        self.assertAlmostEqual(len(result["replay_group_keys"]) / 48.0, 0.75)
        self.assertEqual({key[2] for key in result["replay_group_keys"]}, set(range(6)))
        self.assertEqual({key[2] for key in result["guard_group_keys"]}, set(range(6)))

    def test_guard_covers_boundaries_when_quota_allows(self):
        groups = _groups(excitation_count=1, points_per_excitation=20)
        result = split_history_groups_for_replay_guard(groups, replay_fraction=0.75, seed=42)
        guard = result["guard_group_keys"]
        self.assertEqual(min(key[0] for key in guard), min(key[0] for key in groups))
        self.assertEqual(max(key[0] for key in guard), max(key[0] for key in groups))
        self.assertEqual(min(key[1] for key in guard), min(key[1] for key in groups))
        self.assertEqual(max(key[1] for key in guard), max(key[1] for key in groups))

    def test_split_is_deterministic(self):
        groups = _groups()
        first = split_history_groups_for_replay_guard(groups, replay_fraction=0.75, seed=17)
        second = split_history_groups_for_replay_guard(groups, replay_fraction=0.75, seed=17)
        self.assertEqual(first["replay_group_keys"], second["replay_group_keys"])
        self.assertEqual(first["guard_group_keys"], second["guard_group_keys"])

    def test_invalid_replay_fraction_is_rejected(self):
        for fraction in (0.0, 1.0, -0.1, 1.1):
            with self.subTest(fraction=fraction):
                with self.assertRaises(ValueError):
                    split_history_groups_for_replay_guard(
                        _groups(), replay_fraction=fraction, seed=42
                    )

    def test_small_strata_keep_both_sets_when_possible(self):
        groups = {
            (3.0, 0.8, 0): ["3 0.8 0 0\n"],
            (5.0, 1.0, 0): ["5 1.0 0 0\n"],
            (3.0, 0.8, 1): ["3 0.8 1 0\n"],
        }
        result = split_history_groups_for_replay_guard(groups)
        self.assertEqual(sum(key[2] == 0 for key in result["replay_group_keys"]), 1)
        self.assertEqual(sum(key[2] == 0 for key in result["guard_group_keys"]), 1)
        self.assertEqual(sum(key[2] == 1 for key in result["replay_group_keys"]), 1)
        self.assertEqual(sum(key[2] == 1 for key in result["guard_group_keys"]), 0)

    def test_new_group_is_excluded(self):
        groups = _groups()
        excluded = {(3.0, 0.7, 0)}
        result = split_history_groups_for_replay_guard(groups, excluded_group_keys=excluded)
        selected = set(result["replay_group_keys"]) | set(result["guard_group_keys"])
        self.assertTrue(excluded.isdisjoint(selected))

    def test_manifest_is_reused(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            history = _write_history(root)
            manifest = root / "split.json"
            kwargs = dict(
                base_dir=root,
                history_dataset_path=history,
                selected_names=["SF_XGD_FXD_DC_H0300_M070.txt"],
                structured=True,
                split_manifest_path=manifest,
            )
            first = build_replay_data(**kwargs)
            first_manifest = manifest.read_text(encoding="utf-8")
            second = build_replay_data(**kwargs)
            self.assertEqual(first["replay_group_keys"], second["replay_group_keys"])
            self.assertEqual(first_manifest, manifest.read_text(encoding="utf-8"))

    def test_reused_manifest_excludes_a_later_new_group_without_resplitting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            history = _write_history(root)
            manifest = root / "split.json"
            common = dict(
                base_dir=root,
                history_dataset_path=history,
                structured=True,
                split_manifest_path=manifest,
            )
            build_replay_data(
                **common, selected_names=["SF_XGD_FXD_DC_H0300_M070.txt"]
            )
            original_manifest = manifest.read_text(encoding="utf-8")
            later = build_replay_data(
                **common, selected_names=["SF_XGD_FXD_DC_H0500_M075.txt"]
            )
            selected = {
                stable_group_key(key)
                for key in later["replay_group_keys"] + later["guard_group_keys"]
            }
            self.assertNotIn((300, 70, 1), selected)
            self.assertNotIn((500, 75, 1), selected)
            self.assertEqual(original_manifest, manifest.read_text(encoding="utf-8"))

    def test_fingerprint_change_requires_rebuild(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            history = _write_history(root)
            manifest = root / "split.json"
            kwargs = dict(
                base_dir=root,
                history_dataset_path=history,
                selected_names=["SF_XGD_FXD_DC_H0300_M070.txt"],
                structured=True,
                split_manifest_path=manifest,
            )
            build_replay_data(**kwargs)
            target = next(history.glob("*.txt"))
            target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "fingerprint"):
                build_replay_data(**kwargs)
            rebuilt = build_replay_data(**kwargs, rebuild_split=True)
            self.assertTrue(rebuilt["replay_group_keys"])

    def test_manifest_contains_auditable_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            history = _write_history(root)
            manifest = root / "split.json"
            build_replay_data(
                root,
                history_dataset_path=history,
                selected_names=["SF_XGD_FXD_DC_H0300_M070.txt"],
                structured=True,
                split_manifest_path=manifest,
            )
            data = json.loads(manifest.read_text(encoding="utf-8"))
            for field in (
                "version", "seed", "replay_fraction", "history_dataset_path",
                "history_dataset_fingerprint", "excluded_group_keys",
                "replay_group_keys", "guard_group_keys", "created_at",
            ):
                self.assertIn(field, data)

    def test_epoch_steps_cover_longer_replay_loader(self):
        new_loader = [["new-0"], ["new-1"]]
        replay_loader = [[f"replay-{index}"] for index in range(5)]
        batches = list(iter_incremental_epoch_batches(new_loader, replay_loader))
        self.assertEqual(len(batches), 5)
        self.assertEqual([batch_r[0] for _, batch_r in batches], [f"replay-{i}" for i in range(5)])
        self.assertEqual([batch_n[0] for batch_n, _ in batches], ["new-0", "new-1", "new-0", "new-1", "new-0"])

    def test_direct_and_cg_modes_resolve_history_independently(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cfg = Config()
            cfg.incremental_history_dataset_dir = "configured_history"
            configured = resolve_history_dataset_path(root, None, cfg)
            override = resolve_history_dataset_path(root, str(root / "explicit_history"), cfg)
            self.assertEqual(configured, (root / "configured_history").resolve())
            self.assertEqual(override, (root / "explicit_history").resolve())
            correction_parent = root / "new_data_parent"
            self.assertNotEqual(configured, correction_parent.resolve())

    def test_plain_and_lora_checkpoints_remain_compatible(self):
        cfg = Config()
        cfg.low_hidden = [8, 8]
        cfg.dropout_prob = 0.0
        cfg.use_lora = False
        teacher = MFDNN(cfg)
        student_cfg = deepcopy(cfg)
        student_cfg.use_lora = True
        student = MFDNN(student_cfg)
        load_base_weights_into_lora_model(student, teacher.state_dict())
        h, m = torch.randn(2, 1), torch.randn(2, 1)
        excitation = torch.tensor([0, 1])
        channel = torch.tensor([0, 1])
        with torch.no_grad():
            teacher_output = teacher(h, m, excitation, channel)
            student_output = student(h, m, excitation, channel)
        for name in teacher_output:
            torch.testing.assert_close(teacher_output[name], student_output[name])
        freeze_base_train_lora_only(student)
        trainable = [name for name, parameter in student.named_parameters() if parameter.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(".A." in name or ".B." in name for name in trainable))
        continued_student = MFDNN(student_cfg)
        continued_student.load_state_dict(student.state_dict(), strict=True)

    def test_guard_evaluation_has_gradients_disabled(self):
        class RecordingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.grad_flags = []

            def forward(self, h, m, excitation, channel):
                self.grad_flags.append(torch.is_grad_enabled())
                values = torch.zeros((h.size(0), 1), dtype=h.dtype)
                return {
                    "alpha": values,
                    "omega": values,
                    "rho": values,
                    "sin_theta": values,
                    "cos_theta": torch.ones_like(values),
                }

        model = RecordingModel()
        batch = {
            "H": torch.zeros((2, 1)), "M": torch.zeros((2, 1)),
            "exc": torch.zeros(2, dtype=torch.long), "chn": torch.zeros(2, dtype=torch.long),
            "yN": {"alpha": torch.zeros((2, 1))}, "mask": torch.ones((2, 1)),
        }
        loss = evaluate_representation_loss(
            model, [batch], torch.device("cpu"), torch.zeros(1), torch.ones(1), ["alpha"]
        )
        self.assertEqual(loss, 0.0)
        self.assertEqual(model.grad_flags, [False])


if __name__ == "__main__":
    unittest.main()
