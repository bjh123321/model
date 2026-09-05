import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from surrogate_model.cli.train_incremental import _resolve_sim_dataset_path


class IncrementalSimDatasetTests(unittest.TestCase):
    def test_current_generation_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ini = root / "paths.ini"
            exe = root / "configured_exe"
            ini.write_text("[EXE_XGD]\npath_exe_xgd={}\n".format(exe), encoding="utf-8")
            selected = exe / "GENERATION" / "sim_dataset_20260905"
            with patch("surrogate_model.cli.train_incremental.prepare_sim_dataset", return_value=selected) as prepare:
                self.assertEqual(_resolve_sim_dataset_path(ini), selected)
                prepare.assert_called_once_with(exe / "GENERATION" / "user_input.dat")

    def test_no_legacy_fallback_without_generation_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            ini = Path(temp) / "paths.ini"
            ini.write_text("[CG]\npath_cg=CG\n", encoding="utf-8")
            with self.assertRaises(KeyError):
                _resolve_sim_dataset_path(ini)


if __name__ == "__main__":
    unittest.main()
