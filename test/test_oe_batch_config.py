from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from surrogate_model.data.OE_BATCH_PROCESS import (
    PEAK_PARAMS,
    load_modal_band_config,
)


class OEBatchConfigTests(unittest.TestCase):
    def test_xlsx_uses_app_table_columns_and_keeps_prom_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "极点频段配置.xlsx"
            pandas_row = {
                "通道": "LVT1",
                "频段1": "5-15",
                "频段2": "20-30",
                "频段3": "40-50",
                "允许误差": "1-2-3",
                "OE阶数": "12-14",
            }
            pd.DataFrame([pandas_row]).to_excel(path, index=False)
            config = load_modal_band_config(path)

        self.assertEqual(
            config["LVT1"]["bands"],
            [(5.0, 15.0), (20.0, 30.0), (40.0, 50.0)],
        )
        self.assertEqual(config["LVT1"]["tolerances"], (1.0, 2.0, 3.0))
        self.assertEqual(config["LVT1"]["model_order"], (12, 14))
        self.assertEqual(PEAK_PARAMS["min_prominence_db"], 0.0)


if __name__ == "__main__":
    unittest.main()
