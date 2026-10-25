import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from openpyxl import Workbook
from scipy import signal

from surrogate_model.data import sim_dataset as sim
from surrogate_model.utils.channel_schema import load_channel_schema, remap_dataset_rows


class SimulationDatasetTests(unittest.TestCase):
    def setup_files(self, root, suffix="_20260905", count=3):
        source = root / "sensor" / ("SAMPLE_FOR_XGD" + suffix)
        source.mkdir(parents=True)
        generation = root / "exe" / "GENERATION"
        generation.mkdir(parents=True)
        workbook = root / "ChnTableHeader.xlsx"
        book = Workbook()
        book.active.append(["表头（机载）", "相关激励位置"])
        book.active.append(["TIME", ""])
        book.active.append(["B", "FXD类"])
        book.active.append(["A", "PW类"])
        book.save(str(workbook))
        ini = root / "global.ini"
        ini.write_text("[SensorMap]\npath_sensormap={}\n[EXE_XGD]\npath_exe_xgd={}\n[CurChannelDetailFile]\npath_curchanneldetailfile={}\n".format(source.parent, generation.parent, workbook), encoding="utf-8")
        task = generation / "user_input.dat"
        task.write_text("{}\nmodel\nreal_list.txt\n{}\n".format(ini, source.name), encoding="utf-8")
        raw = source / "FZ_FXD_DC_H0300_M029.dat"
        row = " ".join(["0"] + [str(i + 1) for i in range(count)]) + "\n"
        raw.write_text(row * sim.SAMPLES + "invalid ignored tail\n", encoding="utf-8")
        return task, raw, workbook, ini

    @staticmethod
    def fake_rows(path, values):
        return ["3 .29 1 {} -1 100 2 0 nan nan nan nan nan nan nan nan".format(i)
                for i in range(values.shape[1])]

    def test_sweep(self):
        sweep = sim.sweep_signal()
        self.assertEqual(len(sweep), 25600)
        self.assertEqual(sweep[0], 0)
        self.assertLessEqual(np.max(np.abs(sweep)), .3)
        t = np.arange(25600) / 512.
        np.testing.assert_allclose(sweep, .3 * np.sin(np.pi * t ** 2), atol=1e-12)

    def test_actual_oe_recovers_three_synthetic_modes(self):
        excitation = sim.sweep_signal()
        response = np.zeros_like(excitation)
        for frequency in (15., 30., 45.):
            pole = np.exp(complex(-1., 2 * np.pi * frequency) / sim.FS)
            response += signal.lfilter([.01], np.poly([pole, pole.conjugate()]).real, excitation)
        row = np.array([float(v) for v in sim.extract_sim_rows(
            Path("FZ_FXD_DC_H0300_M029.dat"), response[:, None])[0].split()])
        np.testing.assert_allclose(row[[5, 9, 13]] / (2 * np.pi), [15, 30, 45], atol=2.)

    def test_short_or_inconsistent_response_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "short.dat"
            path.write_text("time,A\n0,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "at least"):
                sim.read_sim_response(path)
            path.write_text("0 1 2\n1 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                sim.read_sim_response(path)

    def test_dynamic_columns_mapping_and_extra_survives(self):
        with tempfile.TemporaryDirectory() as temp:
            task, raw, workbook, ini = self.setup_files(Path(temp))
            self.assertEqual(sim.read_sim_response(raw).shape, (25600, 3))
            schema = sim.simulation_schema(workbook, 3)
            self.assertEqual(schema.channel_names, ("B", "A", "SIM_EXTRA_2"))
            target = load_channel_schema(workbook, schema.channel_names)
            rows, _ = remap_dataset_rows([[3, .29, 1, 2, -1, 100, 2, 0]], target,
                                         schema.channel_names, include_sim_only_exceptions=True,
                                         source_label="simulation")
            self.assertEqual(rows[0][3], 2)
            self.assertEqual(sim.simulation_schema(workbook, 1).channel_names, ("B",))
            self.assertEqual(sim.generation_user_input(ini), task)

    def test_existing_dataset_is_reused_without_source_or_manifest_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            task, raw, workbook, _ = self.setup_files(Path(temp))
            with patch.object(sim, "extract_sim_rows", side_effect=self.fake_rows) as extract:
                dataset = sim.prepare_sim_dataset(task)
                self.assertEqual(dataset.name, "sim_dataset_20260905")
                self.assertEqual(extract.call_count, 1)
            # Changes to the outputs, manifest, source list, or workbook must
            # never cause automatic re-extraction of an existing directory.
            output = dataset / sim.output_name(raw)
            output.write_text("", encoding="utf-8")
            (dataset / sim.MANIFEST).write_text("broken", encoding="utf-8")
            raw.unlink()
            (raw.parent / "another.dat").write_text("not raw samples", encoding="utf-8")
            workbook.unlink()
            with patch.object(sim, "read_sim_response", side_effect=AssertionError("read raw")), \
                    patch.object(sim, "extract_sim_rows", side_effect=AssertionError("run OE")), \
                    patch.object(sim, "read_real_dataset", side_effect=AssertionError("validate dataset")):
                self.assertEqual(sim.prepare_sim_dataset(task), dataset)
            self.assertEqual(output.read_text(encoding="utf-8"), "")
            self.assertEqual((dataset / sim.MANIFEST).read_text(encoding="utf-8"), "broken")

    def test_empty_and_legacy_directories_are_used_as_is(self):
        with tempfile.TemporaryDirectory() as temp:
            task, _, _, _ = self.setup_files(Path(temp), suffix="")
            dataset = task.parent / "sim_dataset"
            dataset.mkdir()
            with patch.object(sim, "extract_sim_rows", side_effect=AssertionError("run OE")):
                self.assertEqual(sim.prepare_sim_dataset(task), dataset)
                self.assertEqual(list(dataset.iterdir()), [])
                legacy = dataset / "FZ_XGD_PW_FDC_H1100_M130.txt"
                legacy.write_text("existing parameter data", encoding="utf-8")
                self.assertEqual(sim.prepare_sim_dataset(task), dataset)
                self.assertEqual(legacy.read_text(encoding="utf-8"), "existing parameter data")
                self.assertFalse((dataset / sim.MANIFEST).exists())

    def test_workbook_reorder_does_not_refresh_existing_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            task, _, workbook, _ = self.setup_files(Path(temp))
            with patch.object(sim, "extract_sim_rows", side_effect=self.fake_rows) as extract:
                dataset = sim.prepare_sim_dataset(task)
                saved_schema = (dataset / "channel_schema.json").read_bytes()
                book = Workbook()
                book.active.append(["表头（机载）", "相关激励位置"])
                book.active.append(["A", "JFY"])
                book.active.append(["B", "PW"])
                book.save(str(workbook))
                self.assertEqual(sim.prepare_sim_dataset(task), dataset)
                self.assertEqual(extract.call_count, 1)
                self.assertEqual((dataset / "channel_schema.json").read_bytes(), saved_schema)

    def test_folder_name_is_resolved_against_configured_sensormap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task, raw, workbook, ini = self.setup_files(root, suffix="_20261025")
            # Both the INI filename and SensorMap root may be relative.
            ini.write_text(
                "[SensorMap]\npath_sensormap=sensor\n"
                "[CurChannelDetailFile]\npath_curchanneldetailfile={}\n".format(workbook),
                encoding="utf-8")
            task.write_text("../../global.ini\nmodel\nreal_list.txt\nSAMPLE_FOR_XGD_20261025\n",
                            encoding="utf-8")
            with patch.object(sim, "extract_sim_rows", side_effect=self.fake_rows) as extract:
                dataset = sim.prepare_sim_dataset(task)
                self.assertEqual(dataset, task.parent / "sim_dataset_20261025")
                self.assertEqual(extract.call_args[0][0], raw.resolve())
                self.assertTrue((dataset / sim.output_name(raw)).is_file())

    def test_existing_dataset_uses_runtime_output_directory_without_raw_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task, raw, workbook, _ = self.setup_files(root, suffix="_20261025")
            raw.unlink()
            raw.parent.rmdir()
            workbook.unlink()
            runtime_dir = root / "runtime"
            dataset = runtime_dir / "sim_dataset_20261025"
            dataset.mkdir(parents=True)
            with patch.object(sim, "extract_sim_rows", side_effect=AssertionError("run OE")):
                self.assertEqual(sim.prepare_sim_dataset(task, runtime_dir), dataset)
            self.assertFalse((task.parent / "sim_dataset_20261025").exists())

    def test_fourth_line_rejects_paths_and_unexpected_folder_names(self):
        with tempfile.TemporaryDirectory() as temp:
            task, raw, _, ini = self.setup_files(Path(temp))
            for selection in (str(raw.parent), "../SAMPLE_FOR_XGD",
                              "nested/SAMPLE_FOR_XGD", "nested\\\\SAMPLE_FOR_XGD",
                              "SAMPLE_FOR_XGD_foo/bar", "other"):
                with self.subTest(selection=selection):
                    task.write_text("{}\nmodel\nreal_list.txt\n{}\n".format(ini, selection),
                                    encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "folder name"):
                        sim.prepare_sim_dataset(task)

    def test_missing_dataset_generation_failure_does_not_publish_partial_output(self):
        with tempfile.TemporaryDirectory() as temp:
            task, _, _, _ = self.setup_files(Path(temp))
            with patch.object(sim, "extract_sim_rows", side_effect=RuntimeError("OE failed")):
                with self.assertRaisesRegex(RuntimeError, "OE failed"):
                    sim.prepare_sim_dataset(task)
            self.assertFalse((task.parent / "sim_dataset_20260905").exists())
            self.assertFalse(list(task.parent.glob(".sim_build_*")))

    def test_band_tolerance_and_missing_slot(self):
        freq = np.array([15., 34., 45.])
        continuous = -np.ones(3) + 1j * np.sqrt((2 * np.pi * freq) ** 2 - 1)
        poles = np.exp(np.concatenate([continuous, continuous.conjugate()]) / sim.FS)
        with patch.object(sim, "marker_slots_from_bands", return_value=[15., 30., 45.]), patch.object(
            sim, "oe_extract_modes", return_value=(poles, np.ones(6, complex), np.array([]))) as oe:
            row = [float(v) for v in sim.extract_sim_rows(
                Path("FZ_FXD_DC_H0300_M029.dat"), np.ones((sim.SAMPLES, 1)))[0].split()]
        self.assertEqual(row[:4], [3, .29, 1, 0])
        self.assertTrue(np.isfinite(row[4:8]).all())
        self.assertTrue(np.isnan(row[8:12]).all())
        self.assertTrue(np.isfinite(row[12:16]).all())
        self.assertEqual(oe.call_args[1]["model_order"], (30, 8))
        np.testing.assert_array_equal(oe.call_args[0][0], sim.sweep_signal())


if __name__ == "__main__":
    unittest.main()
