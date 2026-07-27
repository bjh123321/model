from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from surrogate_model.cli.train_incremental import _resolve_sim_dataset_path


class IncrementalSimDatasetTests(unittest.TestCase):
    def test_parent_snapshot_has_priority(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_version = root / "models" / "base_v1"
            parent_snapshot = base_version / "sim_dataset_snapshot"
            external = root / "program" / "sim_dataset"
            parent_snapshot.mkdir(parents=True)
            external.mkdir(parents=True)

            selected = _resolve_sim_dataset_path(
                base_version, root / "program"
            )

            self.assertEqual(selected, parent_snapshot)

    def test_external_dataset_is_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_version = root / "models" / "legacy_v1"
            external = root / "program" / "sim_dataset"
            external.mkdir(parents=True)

            selected = _resolve_sim_dataset_path(
                base_version, root / "program"
            )

            self.assertEqual(selected, external)

    def test_missing_datasets_report_both_locations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_version = root / "models" / "base_v1"
            program_dir = root / "program"

            with self.assertRaises(FileNotFoundError) as context:
                _resolve_sim_dataset_path(base_version, program_dir)

            message = str(context.exception)
            self.assertIn(str(base_version / "sim_dataset_snapshot"), message)
            self.assertIn(str(program_dir / "sim_dataset"), message)


if __name__ == "__main__":
    unittest.main()
