import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from surrogate_model.cli import train
from surrogate_model.utils.channel_schema import read_dataset_list


class _DatasetPathsCaptured(RuntimeError):
    pass


class GenerationPathTests(unittest.TestCase):
    def test_ini_data_root_is_independent_of_script_task_list_and_cwd(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            generation = root / "fixed_data" / "exe" / "GENERATION"
            real_root = generation / "real_dataset"
            real_root.mkdir(parents=True)
            sim_root = generation / "sim_dataset"
            sim_root.mkdir()
            runtime = root / "scripts"
            runtime.mkdir()
            task_dir = root / "tasks"
            task_dir.mkdir()
            elsewhere = root / "elsewhere"
            elsewhere.mkdir()
            ini = root / "paths.ini"
            ini.write_text(
                "[EXE_XGD]\npath_exe_xgd=fixed_data/exe\n"
                "[MX_XGD]\npath_mx_xgd={}\n".format(root / "models"),
                encoding="utf-8",
            )
            names = ["SF_XGD_FXD_DC_H0300_M068.txt", "SF_XGD_PW_DC_H0300_M068.txt"]
            for name in names:
                (real_root / name).write_text("selected dataset\n", encoding="utf-8")
                (task_dir / name).write_text("wrong dataset\n", encoding="utf-8")
                (elsewhere / name).write_text("wrong dataset\n", encoding="utf-8")
            listing = task_dir / "selected.txt"
            listing.write_text(
                "# filename-only entries\n{} # comment\n{}\n".format(*names),
                encoding="utf-8",
            )
            task = task_dir / "user_input.dat"
            task.write_text(
                "../paths.ini\nmodel\nselected.txt\nSAMPLE_FOR_XGD\n",
                encoding="utf-8",
            )
            prepare = Mock(return_value=sim_root)
            read_list = Mock(wraps=read_dataset_list)
            load = Mock(side_effect=_DatasetPathsCaptured)
            mocks = {
                "get_exe_dir": Mock(return_value=runtime),
                "_ddp_setup": Mock(return_value=(False, 0, 1)),
                "_enable_fast_cuda": Mock(),
                "prepare_sim_dataset": prepare,
                "read_real_dataset": Mock(return_value=[]),
                "source_channel_names_for_dataset": Mock(return_value=[]),
                "channel_detail_path_from_ini": Mock(return_value=root / "channels.xlsx"),
                "channel_names_present": Mock(return_value=[]),
                "load_channel_schema": Mock(return_value=SimpleNamespace(channel_names=())),
                "configure_channel_names": Mock(),
                "remap_dataset_rows": Mock(return_value=([], {})),
                "print_remap_reports": Mock(),
                "read_dataset_list": read_list,
                "load_and_remap_datasets": load,
            }
            previous_cwd = Path.cwd()
            try:
                os.chdir(str(elsewhere))
                with patch.multiple(train, **mocks):
                    with self.assertRaises(_DatasetPathsCaptured):
                        train.main(["--user_input", str(task), "--val_split_strategy="])
            finally:
                os.chdir(str(previous_cwd))

            prepare.assert_called_once_with(task, generation)
            read_list.assert_called_once_with(listing.resolve(), dataset_root=real_root)
            self.assertEqual(load.call_args[0][0], [real_root / name for name in names])


if __name__ == "__main__":
    unittest.main()
