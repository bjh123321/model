from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from openpyxl import Workbook

from surrogate_model.utils.channel_schema import (
    LEGACY_CHANNEL_NAMES,
    load_channel_schema,
    read_dataset_list,
    remap_dataset_rows,
)


class ChannelSchemaTests(unittest.TestCase):
    def _write_workbook(self, path: Path) -> None:
        workbook = Workbook()
        sheet = workbook.active
        sheet.append(["序号", "表头（机载）", "相关激励位置"])
        sheet.append([1, "TIME", "/"])
        sheet.append([2, "LW1", "FXD"])
        sheet.append([3, "LW4", "JFY"])
        sheet.append([4, "F1Z", ""])
        workbook.save(str(path))

    def test_excel_order_remap_and_sim_only_exception(self):
        with TemporaryDirectory() as temp_dir:
            workbook_path = Path(temp_dir) / "channels.xlsx"
            self._write_workbook(workbook_path)
            schema = load_channel_schema(workbook_path, ["F1Z"])

            self.assertEqual(schema.output_headers, ("TIME", "LW1", "LW4", "F1Z"))
            self.assertEqual(schema.channel_names, ("LW1", "LW4", "F1Z"))
            self.assertTrue(schema.channels[-1].sim_only_exception)

            source_rows = [
                [1.0, 0.5, 4.0, 0.0] + [0.0] * 12,
                [1.0, 0.5, 0.0, 28.0] + [0.0] * 12,
            ]
            sim_rows, _ = remap_dataset_rows(
                source_rows,
                schema,
                LEGACY_CHANNEL_NAMES,
                include_sim_only_exceptions=True,
                source_label="sim",
            )
            real_rows, report = remap_dataset_rows(
                source_rows,
                schema,
                LEGACY_CHANNEL_NAMES,
                include_sim_only_exceptions=False,
                source_label="real",
            )

            self.assertEqual(
                [(int(row[2]), int(row[3])) for row in sim_rows],
                [(0, 0), (0, 2)],
            )
            self.assertEqual(
                [(int(row[2]), int(row[3])) for row in real_rows],
                [(0, 0)],
            )
            self.assertEqual(report["skipped_sim_only_real"], 1)

    def test_dataset_list_supports_relative_paths_comments_and_empty_files(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "data.txt"
            dataset.write_text("1 2 3 4\n", encoding="utf-8")
            listing = root / "datasets.txt"
            listing.write_text(
                "# comment\n"
                "data.txt  # inline comment\n",
                encoding="utf-8",
            )
            self.assertEqual(read_dataset_list(listing), [dataset.resolve()])

            listing.write_text("\n# no datasets\n", encoding="utf-8")
            self.assertEqual(read_dataset_list(listing), [])


if __name__ == "__main__":
    unittest.main()
