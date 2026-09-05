"""Versioned simulation OE datasets selected by GENERATION/user_input.dat."""
from __future__ import annotations

import configparser
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

from surrogate_model.data.OE import amplitude_spectrum, marker_slots_from_bands, match_modes_to_marker_frequencies
from surrogate_model.data.OE_BATCH_PROCESS import PEAK_PARAMS, oe_extract_modes, parse_excitation_from_filename, format_row
from surrogate_model.data.io import SPLIT_DATASET_NAME_RE, read_real_dataset
from surrogate_model.utils.channel_schema import ChannelDefinition, ChannelSchema, channel_detail_path_from_ini, load_channel_schema, save_channel_schema

FS = 512
DURATION = 50
SAMPLES = FS * DURATION
BANDS = ((10.0, 20.0), (20.0, 40.0), (40.0, 50.0))
MANIFEST = "simulation_generation.json"


def ini_directory(ini_path: Path, section: str) -> Path:
    parser = configparser.ConfigParser()
    if not parser.read(str(ini_path), encoding="utf-8-sig"):
        raise FileNotFoundError(str(ini_path))
    value = parser[section]["path_" + section.lower()].strip()
    if not value:
        raise ValueError("Empty [{}] path in {}".format(section, ini_path))
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(ini_path).parent / path).resolve()


def generation_user_input(ini_path: Path) -> Path:
    return ini_directory(ini_path, "EXE_XGD") / "GENERATION" / "user_input.dat"


def sweep_signal() -> np.ndarray:
    time = np.arange(SAMPLES, dtype=float) / FS
    return 0.3 * np.sin(2.0 * np.pi * 0.5 * (50.0 / DURATION) * time ** 2)


def read_sim_response(path: Path) -> np.ndarray:
    """Read comma/tab/space numeric data, allowing one optional header row."""
    rows = []
    width = None
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        header_seen = False
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            cells = re.split(r"[,\s]+", raw.strip())
            try:
                row = [float(cell) for cell in cells]
            except ValueError:
                if not rows and not header_seen:
                    header_seen = True
                    continue
                raise ValueError("{}:{} nonnumeric simulation sample".format(path, line_number))
            if width is None:
                width = len(row)
            if len(row) != width or width < 2:
                raise ValueError("{}:{} inconsistent time/response columns".format(path, line_number))
            rows.append(row)
            if len(rows) == SAMPLES:
                break
    if len(rows) != SAMPLES:
        raise ValueError("{} needs at least {} samples (50 s at 512 Hz)".format(path, SAMPLES))
    values = np.asarray(rows, dtype=float)[:, 1:]
    if not np.isfinite(values).all():
        raise ValueError("{} contains nonfinite response samples".format(path))
    return values


def simulation_schema(workbook: Path, count: int) -> ChannelSchema:
    current = load_channel_schema(workbook)
    definitions = list(current.channels[:count])
    names = set(current.output_headers)
    for channel_id in range(len(definitions), count):
        name = "SIM_EXTRA_{}".format(channel_id)
        if name in names:
            raise ValueError("Reserved simulation channel name: {}".format(name))
        definitions.append(ChannelDefinition(channel_id, name, None, True))
    return ChannelSchema(str(workbook), current.output_headers, tuple(definitions))


def output_name(path: Path) -> str:
    meta = parse_excitation_from_filename(path)
    phase = "FDC" if meta.excite_id % 2 == 0 else "DC"
    return "FZ_XGD_{}_{}_H{:04d}_M{:03d}.txt".format(
        meta.excite_pos, phase, int(round(meta.H * 100)), int(round(meta.M * 100)))


def extract_sim_rows(path: Path, values: np.ndarray):
    meta = parse_excitation_from_filename(path)
    excitation = sweep_signal()
    rows = []
    has_modes = False
    for channel_id in range(values.shape[1]):
        response = values[:, channel_id]
        frequency, amplitude = amplitude_spectrum(response, FS)
        markers = marker_slots_from_bands(frequency, amplitude, list(BANDS), peak_params=PEAK_PARAMS)
        poles, residues, _ = oe_extract_modes(excitation, response, FS, model_order=(30, 8))
        fields = format_row(meta, channel_id, np.array([]), np.array([]), np.array([]))
        for marker, (low, high) in zip(markers, BANDS):
            matched = [] if marker is None else match_modes_to_marker_frequencies(
                np.array([marker]), poles, residues, FS, f_threshold=2.0,
                min_frequency=low, max_frequency=high)
            # Preserve all three modal slots; unmatched bands remain missing.
            if not matched:
                fields.extend(["nan"] * 4)
            else:
                has_modes = True
                mode = matched[0][1]
                fields.extend("{:.10g}".format(value) for value in
                              (mode.alpha, mode.omega, mode.residue_abs, mode.residue_phase))
        rows.append("\t".join(fields))
    if not has_modes:
        raise RuntimeError("No usable simulation modes extracted from {}".format(path))
    return rows


def _digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _complete(dataset: Path, expected, schema: ChannelSchema) -> bool:
    try:
        manifest = json.loads((dataset / MANIFEST).read_text(encoding="utf-8"))
        if manifest["expected"] != expected or manifest["channels"] != schema.to_dict():
            return False
        actual = {path.name for path in dataset.iterdir()
                  if path.is_file() and SPLIT_DATASET_NAME_RE.match(path.name)}
        if actual != set(expected):
            return False
        hashes = manifest["sha256"]
        if set(hashes) != set(expected) | {"channel_schema.json"}:
            return False
        if any(_digest(dataset / name) != digest for name, digest in hashes.items()):
            return False
        read_real_dataset(dataset)
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def prepare_sim_dataset(user_input_path: Path, output_dir: Path = None) -> Path:
    user_input_path = Path(user_input_path)
    lines = user_input_path.read_text(encoding="utf-8-sig").splitlines()
    if len(lines) < 4 or not lines[0].strip() or not lines[3].strip():
        raise ValueError("GENERATION/user_input.dat requires a simulation directory on line 4")
    ini_path = Path(lines[0].strip()).expanduser()
    if not ini_path.is_absolute():
        ini_path = user_input_path.parent / ini_path
    source = Path(lines[3].strip()).expanduser()
    if not source.is_absolute():
        raise ValueError("GENERATION/user_input.dat line 4 must be an absolute path")
    source = source.resolve()
    sensor_root = ini_directory(ini_path, "SensorMap")
    try:
        source.relative_to(sensor_root)
    except ValueError:
        raise ValueError("Simulation source must be inside configured SensorMap: {}".format(sensor_root))
    if not re.fullmatch(r"SAMPLE_FOR_XGD(?:_.+)?", source.name, flags=re.IGNORECASE):
        raise ValueError("Unexpected simulation source directory: {}".format(source))
    files = sorted(source.glob("*.dat"))
    if not files:
        raise FileNotFoundError("No simulation .dat files in {}".format(source))
    expected = [output_name(path) for path in files]
    if len(set(expected)) != len(expected):
        raise ValueError("Simulation files have duplicate excitation/H/M output names")
    workbook = channel_detail_path_from_ini(ini_path)
    # Reading the first file determines column identities without running OE.
    count = read_sim_response(files[0]).shape[1]
    schema = simulation_schema(workbook, count)
    destination = Path(output_dir or user_input_path.parent) / ("sim_dataset" + source.name[len("SAMPLE_FOR_XGD"):])
    if _complete(destination, expected, schema):
        print("[SIM-DATA] reuse={}".format(destination), flush=True)
        return destination
    # Validate legacy caches too; do not rerun complete pre-manifest datasets.
    if destination.is_dir() and not (destination / MANIFEST).exists():
        try:
            actual = sorted(p.name for p in destination.iterdir()
                            if p.is_file() and SPLIT_DATASET_NAME_RE.match(p.name))
            complete = actual == sorted(expected)
            for name in expected:
                file_rows = read_real_dataset(destination / name)
                complete = complete and sorted(int(row[3]) for row in file_rows) == list(range(count))
                complete = complete and all(len(row) >= 8 and (len(row) - 4) % 4 == 0
                                            for row in file_rows)
            if complete:
                save_channel_schema(schema, destination)
                _write_manifest(destination, expected, schema)
                return destination
        except (OSError, ValueError):
            pass
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".sim_build_", dir=str(destination.parent)))
    backup = None
    try:
        for path, name in zip(files, expected):
            values = read_sim_response(path)
            if values.shape[1] != count:
                raise ValueError("Simulation files must share one channel layout: {}".format(path))
            rows = extract_sim_rows(path, values)
            (staging / name).write_text("\n".join(rows) + "\n", encoding="utf-8")
            print("[SIM-OE] {} channels={} -> {}".format(path.name, count, name), flush=True)
        save_channel_schema(schema, staging)
        read_real_dataset(staging)
        _write_manifest(staging, expected, schema)
        if destination.exists():
            backup = Path(tempfile.mkdtemp(prefix=".sim_previous_", dir=str(destination.parent)))
            backup.rmdir()
            os.replace(str(destination), str(backup))
        try:
            os.replace(str(staging), str(destination))
        except Exception:
            if backup is not None:
                os.replace(str(backup), str(destination))
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(str(backup))
    finally:
        if staging.exists():
            shutil.rmtree(str(staging))
    return destination


def _write_manifest(directory: Path, expected, schema: ChannelSchema) -> None:
    payload = {"expected": expected, "channels": schema.to_dict(),
               "sha256": {name: _digest(directory / name) for name in expected + ["channel_schema.json"]}}
    (directory / MANIFEST).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
