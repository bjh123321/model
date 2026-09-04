from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from openpyxl import Workbook

from surrogate_model.data.xlsx_utils import load_signal_type_rows


class SignalTypeWorkbookTests(unittest.TestCase):
    def test_new_headers_with_units_are_matched_by_keyword(self):
        with TemporaryDirectory() as temp_dir:
            workbook_path = Path(temp_dir) / "signal_type.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.append([
                "信号编号",
                "信号类型",
                "推力持续时间(ms)",
                "力矩(N)",
                "幅值（°）",
                "起始频率（Hz）",
                "截止频率（Hz）",
                "持续时间（s）",
            ])
            sheet.append(["E", "扫频", "/", 0.0, 1.0, 5.0, 60.0, 12.5])
            workbook.save(str(workbook_path))

            rows = load_signal_type_rows(workbook_path)

            self.assertEqual(rows, [("E", 1.0, 5.0, 60.0, 12.5)])


if __name__ == "__main__":
    unittest.main()
