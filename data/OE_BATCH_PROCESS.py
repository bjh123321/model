"""Batch OE modal extraction with the standalone app-equivalent kernel."""

from __future__ import annotations

import json
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from surrogate_model.data.OE import (
    DEFAULT_MODAL_BANDS,
    DEFAULT_MODAL_TOLERANCES,
    DEFAULT_PEAK_PARAMS,
    DEFAULT_POLE_FILTER_PARAMS,
    amplitude_spectrum,
    fit_oe_modal,
    marker_slots_from_bands,
    match_modes_to_marker_frequencies,
    robust_spectrum_peaks,
)


FS = 512
MAX_MODES = 6
INPUT_SIGNAL_NAME = "PM_O1"

OE_KWARGS = {
    "na": 30,
    "nb": 30,
    "nk": 0,
    "max_nfev": 50,
    "init_method": "arx",
    "rpm_khat": 75,
    "demean": False,
}

PEAK_PARAMS = dict(DEFAULT_PEAK_PARAMS)
PEAK_PARAMS["min_prominence_db"] = 0.0
POLE_FILTER_PARAMS = dict(DEFAULT_POLE_FILTER_PARAMS)
MODAL_BAND_CONFIG_PATH = Path.home() / ".analyze_signal_app" / "oe_batch_mode_bands.json"

EXCITATION_ID = {
    "FXD_FDC": 0,
    "FXD_DC": 1,
    "PW_FDC": 2,
    "PW_DC": 3,
    "JFY_FDC": 4,
    "JFY_DC": 5,
}

EXCITATION_CODE_MAP = {
    "FXD_FDC": "FXD_同",
    "FXD_DC": "FXD_反",
    "PW_FDC": "PW_同",
    "PW_DC": "PW_反",
    "JFY_FDC": "JFY_同",
    "JFY_DC": "JFY_反",
}

CHANNELS_BY_POS: Dict[str, List[str]] = {
    "FXD": ["LVT1", "LVT2", "LVT3", "LVT4", "LRY", "LRY1", "RVT1", "RVT2", "RVT3", "RVT4", "RRY", "RRY1"],
    "PW": ["LHT1", "LHT4", "LHT2", "LHT3", "RHT1", "RHT4", "RHT2", "RHT3"],
    "JFY": ["LW1", "LW2", "LW3", "LAI", "RW1", "RW2", "RW3", "RAI"],
}

CHANNEL_ID_BASE = {"JFY": 0, "FXD": 8, "PW": 20}


@dataclass
class ParsedMeta:
    H: float
    M: float
    excite_pos: str
    excite_phase: str
    excite_key: str
    excite_id: int


def _decode_hm_token(token: str) -> float:
    token = token.strip()
    if "d" in token:
        return float(token.replace("d", "."))
    if len(token) >= 2:
        return float(f"{token[:-2]}.{token[-2:]}")
    return float(token)


def parse_HM_from_filename(file_path: Path) -> Tuple[float, float]:
    name = Path(file_path).stem
    height_match = re.search(r"_H([0-9d]+)(?=_|$)", name)
    mach_match = re.search(r"_M([0-9d]+)(?=_|$)", name)
    if height_match is None or mach_match is None:
        raise ValueError(f"Cannot parse H/M from filename: {file_path}")
    return _decode_hm_token(height_match.group(1)), _decode_hm_token(mach_match.group(1))


def parse_excitation_from_filename(file_path: Path) -> ParsedMeta:
    match = re.search(r"_(FXD|PW|JFY)_(FDC|DC)(?=_|$)", Path(file_path).stem)
    if match is None:
        raise ValueError(f"Cannot parse excitation from filename: {file_path}")
    position = match.group(1)
    phase_code = match.group(2)
    code = f"{position}_{phase_code}"
    excitation_key = EXCITATION_CODE_MAP[code]
    height, mach = parse_HM_from_filename(Path(file_path))
    return ParsedMeta(
        H=height,
        M=mach,
        excite_pos=position,
        excite_phase=excitation_key.split("_", 1)[1],
        excite_key=excitation_key,
        excite_id=EXCITATION_ID[code],
    )


def read_table_dat(file_path: Path) -> Tuple[np.ndarray, List[str]]:
    dataframe = pd.read_csv(file_path, encoding="utf-8-sig")
    dataframe.columns = [str(column).strip() for column in dataframe.columns]
    keep_columns: List[str] = []
    keep_series = []
    for column in dataframe.columns:
        numeric = pd.to_numeric(dataframe[column], errors="coerce")
        if numeric.notna().any():
            keep_columns.append(column)
            keep_series.append(numeric.astype(np.float64))
    if not keep_columns:
        raise ValueError(f"No numeric columns found in: {file_path}")
    numeric_frame = pd.concat(keep_series, axis=1)
    numeric_frame.columns = keep_columns
    values = numeric_frame.to_numpy(dtype=np.float64)
    values = values - values.mean(axis=0, keepdims=True)
    return values, keep_columns


def get_input_signal(header: List[str], values: np.ndarray, input_name: str = INPUT_SIGNAL_NAME) -> np.ndarray:
    name_to_index = {name: index for index, name in enumerate(header)}
    if input_name not in name_to_index:
        raise ValueError(f"Input column not found: {input_name}")
    return np.asarray(values[:, name_to_index[input_name]], dtype=float).ravel()


def select_channel_indices(header: List[str], excite_pos: str) -> List[Tuple[int, str, int]]:
    if excite_pos not in CHANNELS_BY_POS or excite_pos not in CHANNEL_ID_BASE:
        raise ValueError(f"Unsupported excitation position: {excite_pos}")
    name_to_index = {name: index for index, name in enumerate(header)}
    base = CHANNEL_ID_BASE[excite_pos]
    selected = []
    for local_index, name in enumerate(CHANNELS_BY_POS[excite_pos]):
        if name in name_to_index:
            selected.append((base + local_index, name, name_to_index[name]))
    return selected


def _coerce_band(value: object) -> Optional[Tuple[float, float]]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        low, high = float(value[0]), float(value[1])
    else:
        text = str(value or "").strip()
        if not text:
            return None
        parts = re.split(r"\s*[-~:,，；;]\s*", text)
        if len(parts) != 2:
            return None
        low, high = float(parts[0]), float(parts[1])
    return (low, high) if np.isfinite(low) and np.isfinite(high) and high > low else None


def _coerce_tolerances(value: object) -> Tuple[float, float, float]:
    parts = list(value) if isinstance(value, (list, tuple)) else re.split(r"\s*[-~:,，；;]\s*", str(value or "").strip())
    if len(parts) != 3:
        return DEFAULT_MODAL_TOLERANCES
    values = tuple(float(item) for item in parts)
    if not all(np.isfinite(item) and item > 0 for item in values):
        return DEFAULT_MODAL_TOLERANCES
    return values


def load_modal_band_config() -> Dict[str, Dict[str, object]]:
    config = {
        channel: {"bands": list(DEFAULT_MODAL_BANDS), "tolerances": DEFAULT_MODAL_TOLERANCES}
        for channels in CHANNELS_BY_POS.values()
        for channel in channels
    }
    try:
        if not MODAL_BAND_CONFIG_PATH.exists():
            return config
        raw = json.loads(MODAL_BAND_CONFIG_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return config
        for channel, value in raw.items():
            if isinstance(value, dict):
                band_values = value.get("bands", [])
                tolerance_values = value.get("tolerances")
            else:
                band_values = value if isinstance(value, list) else []
                tolerance_values = None
            bands = [_coerce_band(item) for item in list(band_values)[:3]]
            while len(bands) < 3:
                bands.append(None)
            config[str(channel)] = {
                "bands": bands[:3],
                "tolerances": _coerce_tolerances(tolerance_values),
            }
    except Exception:
        return config
    return config


def _modal_config_for_channel(channel_name: Optional[str]) -> Dict[str, object]:
    config = load_modal_band_config()
    if channel_name is not None and str(channel_name) in config:
        return config[str(channel_name)]
    return {"bands": list(DEFAULT_MODAL_BANDS), "tolerances": DEFAULT_MODAL_TOLERANCES}


def find_spectrum_peaks(y: np.ndarray, fs: int, channel_name: Optional[str] = None) -> np.ndarray:
    frequency, amplitude = amplitude_spectrum(y, fs)
    robust_spectrum_peaks(frequency, amplitude, **PEAK_PARAMS)
    entry = _modal_config_for_channel(channel_name)
    slots = marker_slots_from_bands(
        frequency,
        amplitude,
        list(entry["bands"]),
        peak_params=PEAK_PARAMS,
    )
    return np.array([value for value in slots if value is not None], dtype=float)


def oe_extract_modes(u: np.ndarray, y: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = fit_oe_modal(
        u,
        y,
        fs,
        nb=int(OE_KWARGS["nb"]),
        nf=int(OE_KWARGS["na"]),
        nk=int(OE_KWARGS["nk"]),
        max_iter=int(OE_KWARGS["max_nfev"]),
    )
    poles = np.asarray(result["poles"], dtype=np.complex128)
    residues = np.asarray(result["residues"], dtype=np.complex128)
    with np.errstate(divide="ignore", invalid="ignore"):
        continuous_poles = np.log(poles) * float(fs)
    return poles, residues, continuous_poles


def match_peaks_to_oe(
    peak_freqs: np.ndarray,
    poles: np.ndarray,
    residues: np.ndarray,
    continuous_poles: np.ndarray,
    fs: int,
    tolerances: Optional[Tuple[float, float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    del continuous_poles
    markers = np.asarray(peak_freqs, dtype=float).ravel()
    thresholds = np.asarray(tolerances or DEFAULT_MODAL_TOLERANCES, dtype=float)[: markers.size]
    matched = match_modes_to_marker_frequencies(
        markers,
        np.asarray(poles, dtype=complex),
        np.asarray(residues, dtype=complex),
        fs,
        f_threshold=5.0,
        f_thresholds=thresholds,
        **POLE_FILTER_PARAMS,
    )
    if not matched:
        empty = np.array([], dtype=np.complex128)
        return empty, empty.copy(), empty.copy()
    selected_s = np.array([complex(mode.alpha, mode.omega) for _, mode in matched], dtype=np.complex128)
    selected_p = np.exp(selected_s / float(fs))
    selected_r = np.array(
        [mode.residue_abs * np.exp(1j * mode.residue_phase) for _, mode in matched],
        dtype=np.complex128,
    )
    return selected_p, selected_r, selected_s


def format_row(
    meta: ParsedMeta,
    channel_id: int,
    p_sel: np.ndarray,
    r_sel: np.ndarray,
    s_sel: np.ndarray,
) -> List[str]:
    del p_sel
    output = [f"{meta.H:.10g}", f"{meta.M:.10g}", str(meta.excite_id), str(channel_id)]
    for continuous_pole, residue in zip(s_sel, r_sel):
        output.extend([
            f"{float(np.real(continuous_pole)):.10g}",
            f"{float(np.imag(continuous_pole)):.10g}",
            f"{float(np.abs(residue)):.10g}",
            f"{float(np.angle(residue)):.10g}",
        ])
    return output


def read_selected_dat_files(root_dir: Path, selected_list_file: Path) -> List[Path]:
    root_dir = Path(root_dir)
    selected_list_file = Path(selected_list_file)
    if not selected_list_file.exists():
        raise FileNotFoundError(f"Selected-file list not found: {selected_list_file}")
    files: List[Path] = []
    seen = set()
    with selected_list_file.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            name = line.strip()
            if not name:
                continue
            path = root_dir / name
            if path in seen:
                continue
            if not path.exists():
                raise FileNotFoundError(f"Line {line_number} does not exist: {path}")
            if path.suffix.lower() != ".dat":
                raise ValueError(f"Line {line_number} is not a .dat file: {path}")
            files.append(path)
            seen.add(path)
    if not files:
        raise ValueError(f"Selected-file list is empty: {selected_list_file}")
    return files


def _extract_rows_from_dat(file_path: Path) -> List[str]:
    meta = parse_excitation_from_filename(file_path)
    values, header = read_table_dat(file_path)
    input_signal = get_input_signal(header, values, INPUT_SIGNAL_NAME)
    channels = select_channel_indices(header, meta.excite_pos)
    if not channels:
        print(f"[SKIP] no channel matched | {file_path}")
        return []
    rows: List[str] = []
    for channel_id, channel_name, column_index in channels:
        try:
            output_signal = np.asarray(values[:, column_index], dtype=float).ravel()
            valid = np.isfinite(input_signal) & np.isfinite(output_signal)
            paired_input = input_signal[valid]
            paired_output = output_signal[valid]
            markers = find_spectrum_peaks(paired_output, FS, channel_name)
            poles, residues, continuous_poles = oe_extract_modes(paired_input, paired_output, FS)
            entry = _modal_config_for_channel(channel_name)
            selected = match_peaks_to_oe(
                markers,
                poles,
                residues,
                continuous_poles,
                FS,
                tolerances=tuple(entry["tolerances"]),
            )
            if selected[0].size:
                rows.append("\t".join(format_row(meta, channel_id, *selected)))
        except Exception as exc:
            print(f"[FAIL-CH] {file_path} | channel_id={channel_id} col={column_index} name={channel_name} | {exc!r}")
            print(traceback.format_exc())
    return rows


def _split_dataset_stem(file_path: Path) -> str:
    meta = parse_excitation_from_filename(file_path)
    match = re.search(r"_(FXD|PW|JFY)_(FDC|DC)(?=_|$)", file_path.stem, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Cannot derive split-dataset excitation code from: {file_path.name}")
    excitation_code = f"{match.group(1).upper()}_{match.group(2).upper()}"
    return f"SF_XGD_{excitation_code}_H{int(round(meta.H * 100)):04d}_M{int(round(meta.M * 100)):03d}"


def _next_split_dataset_path(output_dir: Path, stem: str, reserved_dir: Optional[Path] = None) -> Path:
    reserved_dir = Path(reserved_dir) if reserved_dir is not None else output_dir

    def is_available(path: Path) -> bool:
        return not path.exists() and not (reserved_dir / path.name).exists()

    candidate = output_dir / f"{stem}.txt"
    if is_available(candidate):
        return candidate
    suffix = 2
    while True:
        candidate = output_dir / f"{stem}_{suffix}.txt"
        if is_available(candidate):
            return candidate
        suffix += 1


def batch_process_split(
    root_dir: Path,
    output_dir: Path,
    selected_list_file: Path,
    reserved_names_dir: Optional[Path] = None,
) -> List[Path]:
    root_dir, output_dir = Path(root_dir), Path(output_dir)
    files = read_selected_dat_files(root_dir, Path(selected_list_file))
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []
    for file_path in files:
        try:
            rows = _extract_rows_from_dat(file_path)
            if not rows:
                raise RuntimeError(f"No usable modal rows extracted from: {file_path}")
            output_path = _next_split_dataset_path(output_dir, _split_dataset_stem(file_path), reserved_names_dir)
            with output_path.open("x", encoding="utf-8") as stream:
                stream.write("\n".join(rows) + "\n")
            generated.append(output_path)
            print(f"[NEW-DATA] source={file_path.name} rows={len(rows)} -> {output_path.name}", flush=True)
        except Exception as exc:
            print(f"[FAIL-FILE] {file_path} | {exc!r}")
            print(traceback.format_exc())
    if not generated:
        raise RuntimeError("No split-dataset files were generated from selected_CG_path.txt")
    return generated


def batch_process(root_dir: Path, out_file: Path, selected_list_file: Path) -> None:
    root_dir, out_file = Path(root_dir), Path(out_file)
    files = read_selected_dat_files(root_dir, Path(selected_list_file))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as stream:
        for file_path in files:
            try:
                rows = _extract_rows_from_dat(file_path)
                if rows:
                    stream.write("\n".join(rows) + "\n")
            except Exception as exc:
                print(f"[FAIL-FILE] {file_path} | {exc!r}")
                print(traceback.format_exc())


if __name__ == "__main__":
    ROOT = Path(r"C:\Users\Administrator\Desktop\试飞数据(训练用 - 副本")
    OUT = Path(r"C:\Users\Administrator\Desktop") / "new_real.txt"
    SELECTED_LIST = Path(r"C:\Users\Administrator\Desktop\selected_CG_path.txt")
    batch_process(ROOT, OUT, SELECTED_LIST)
    print(f"Done. Output: {OUT}")
