import tempfile
import unittest
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

from surrogate_model.utils.exe_runtime import (
    run_executable,
    split_user_input_arg,
)


class ExecutableRuntimeTests(unittest.TestCase):
    def test_user_input_argument_and_default_are_compatible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            program_dir = Path(temp_dir) / "program"
            selected = Path(temp_dir) / "jobs" / "run1.dat"
            default_path, default_args = split_user_input_arg(
                ["--epochs", "2"], program_dir
            )
            selected_path, selected_args = split_user_input_arg(
                ["--user_input", str(selected), "--epochs", "2"],
                program_dir,
            )

            self.assertEqual(default_path, program_dir / "user_input.dat")
            self.assertEqual(default_args, ["--epochs", "2"])
            self.assertEqual(selected_path, selected)
            self.assertEqual(selected_args, ["--epochs", "2"])

    def test_success_and_failure_write_valid_word_logs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            user_input = root / "job" / "user_input.dat"
            user_input.parent.mkdir(parents=True)
            user_input.write_text("paths.ini\nmodel_v1\n", encoding="utf-8")
            arguments = ["--user_input", str(user_input)]

            def succeed(_argv):
                print("progress\x1b[0m")
                return 0

            success = run_executable(
                succeed, "success_case", root, arguments
            )

            def fail(_argv):
                raise RuntimeError("expected failure")

            failure = run_executable(
                fail, "failure_case", root, arguments
            )

            self.assertEqual(success, 0)
            self.assertEqual(failure, -1)
            logs = sorted((user_input.parent / "logs").glob("*.docx"))
            self.assertEqual(len(logs), 2)
            for path in logs:
                with zipfile.ZipFile(str(path), "r") as archive:
                    self.assertIn("word/document.xml", archive.namelist())
                    document_xml = archive.read("word/document.xml")
                    ET.fromstring(document_xml)
                    self.assertIn(
                        "代理模型运行日志", document_xml.decode("utf-8")
                    )


if __name__ == "__main__":
    unittest.main()
