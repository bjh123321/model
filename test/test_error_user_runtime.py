from __future__ import annotations

import configparser
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from surrogate_model.test import ERROR_user


class ErrorUserRuntimeTests(unittest.TestCase):
    def test_main_accepts_positional_user_input_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prediction_root = root / "predictions"
            paths_ini = root / "paths.ini"
            parser = configparser.ConfigParser()
            parser["PR_XGD"] = {"path_PR_XGD": str(prediction_root)}
            with paths_ini.open("w", encoding="utf-8") as handle:
                parser.write(handle)

            a_file = root / "A.dat"
            a_file.write_text("test\n", encoding="utf-8")
            user_input = root / "job" / "user_input.dat"
            user_input.parent.mkdir()
            user_input.write_text(
                "{}\nXGD_model_v1\n{}\n".format(paths_ini, a_file),
                encoding="utf-8",
            )

            with mock.patch.object(
                ERROR_user, "batch_process_all_a_files", return_value=[]
            ) as process:
                exit_code = ERROR_user.main([str(user_input)])

            self.assertEqual(exit_code, 0)
            process.assert_called_once_with(
                a_files=[a_file],
                pred_root=prediction_root / "XGD_model_v1",
                out_dir=(
                    prediction_root
                    / "XGD_model_v1"
                    / "ERROR_PR_XGD"
                ),
            )


if __name__ == "__main__":
    unittest.main()
