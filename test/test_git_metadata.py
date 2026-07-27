from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from surrogate_model.utils.model_artifacts import read_code_metadata


class GitMetadataTests(unittest.TestCase):
    def test_git_failure_is_silent_and_returns_unknown_values(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch(
            "surrogate_model.utils.model_artifacts.subprocess.check_output",
            side_effect=OSError(267, "invalid directory"),
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            metadata = read_code_metadata(Path(temp_dir))

        self.assertIsNone(metadata["git_branch"])
        self.assertIsNone(metadata["git_commit"])
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
