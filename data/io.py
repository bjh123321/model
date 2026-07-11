from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import math
import re

import numpy as np


FIXED_K_MAX = 3
FS_HZ = 512.0
OMEGA_N = math.pi * FS_HZ


def _parse_float_cell(cell: str) -> float:
    cell = cell.strip()
    if cell == "":
        return float("nan")
    return float(cell)


SPLIT_DATASET_NAME_RE = re.compile(
    r"^SF_XGD_(FXD|PW|JFY)_(FDC|DC)_H(\d+)_M(\d+)(?:_(\d+))?\.txt$",
    re.IGNORECASE,
)
SPLIT_EXCITATION_IDS = {
    "FXD_FDC": 0,
    "FXD_DC": 1,
    "PW_FDC": 2,
    "PW_DC": 3,
    "JFY_FDC": 4,
    "JFY_DC": 5,
}


def read_variable_txt(path: Path, strict: bool = False) -> List[List[float]]:
    path = Path(path)
    rows: List[List[float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            ln = raw.rstrip("\r\n")
            if not ln.strip() or ln.lstrip().startswith("#"):
                continue
            try:
                if "\t" in ln:
                    row = [_parse_float_cell(x) for x in ln.split("\t")]
                else:
                    row = [float(x) for x in ln.replace(",", " ").split()]
            except ValueError as exc:
                if strict:
                    raise ValueError(f"{path}:{line_no} contains a non-numeric data row") from exc
                continue
            if len(row) >= 4:
                rows.append(row)
            elif strict:
                raise ValueError(f"{path}:{line_no} has {len(row)} columns; expected at least 4")
    return rows


def _validate_split_dataset_file(path: Path, rows: List[List[float]]) -> None:
    match = SPLIT_DATASET_NAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected split-dataset filename: {path.name}")
    if not rows:
        raise ValueError(f"Split-dataset file is empty: {path}")

    excitation_name = f"{match.group(1).upper()}_{match.group(2).upper()}"
    expected_exc_id = SPLIT_EXCITATION_IDS[excitation_name]
    expected_h_code = int(match.group(3))
    expected_m_code = int(match.group(4))
    first_h, first_m = float(rows[0][0]), float(rows[0][1])

    # Filenames store H*100 and M*100 as rounded integer codes. Keep the more
    # precise H/M values from each row as the actual model inputs.
    if abs(first_h * 100.0 - expected_h_code) > 0.500001:
        raise ValueError(f"{path.name}: filename H code does not match row height H={first_h:g}")
    if abs(first_m * 100.0 - expected_m_code) > 0.500001:
        raise ValueError(f"{path.name}: filename M code does not match row Mach M={first_m:g}")

    for line_no, row in enumerate(rows, start=1):
        if abs(float(row[0]) - first_h) > 1e-9 or abs(float(row[1]) - first_m) > 1e-9:
            raise ValueError(f"{path}:{line_no} has H/M inconsistent with the first row")
        exc_id = int(round(float(row[2])))
        if exc_id != expected_exc_id:
            raise ValueError(
                f"{path}:{line_no} has exc_id={exc_id}, but filename {excitation_name} expects {expected_exc_id}"
            )


def read_real_dataset(path: Path) -> List[List[float]]:
    """Read either one legacy txt file or a directory of SF_XGD split txt files."""
    path = Path(path)
    if path.is_file():
        rows = read_variable_txt(path, strict=True)
        if not rows:
            raise ValueError(f"Real dataset file contains no usable rows: {path}")
        if SPLIT_DATASET_NAME_RE.match(path.name):
            _validate_split_dataset_file(path, rows)
        print(f"[DATA] loaded real dataset file: files=1, rows={len(rows)}, path={path}", flush=True)
        return rows
    if not path.is_dir():
        raise FileNotFoundError(f"Real dataset path is neither a file nor a directory: {path}")

    files = sorted(
        (p for p in path.iterdir() if p.is_file() and SPLIT_DATASET_NAME_RE.match(p.name)),
        key=lambda p: p.name.lower(),
    )
    if not files:
        raise FileNotFoundError(f"No SF_XGD_*.txt split-dataset files found in: {path}")

    rows: List[List[float]] = []
    for txt_path in files:
        file_rows = read_variable_txt(txt_path, strict=True)
        _validate_split_dataset_file(txt_path, file_rows)
        file_seen_keys = set()
        for line_no, row in enumerate(file_rows, start=1):
            key = (float(row[0]), float(row[1]), int(round(float(row[2]))), int(round(float(row[3]))))
            if key in file_seen_keys:
                raise ValueError(
                    f"Duplicate (H, M, excitation, channel) sample {key} within {txt_path}:{line_no}"
                )
            file_seen_keys.add(key)
        rows.extend(file_rows)

    print(f"[DATA] loaded split real dataset: files={len(files)}, rows={len(rows)}, path={path}", flush=True)
    return rows


@dataclass
class ParsedRow:
    H: float
    M: float
    exc_id: int
    ch_id: int
    K_valid: int
    alpha: np.ndarray
    omega: np.ndarray
    rho: np.ndarray
    sin_theta: np.ndarray
    cos_theta: np.ndarray
    mask: np.ndarray
    freq: np.ndarray
    damping: np.ndarray
    damping_percent: np.ndarray
    residue_re: np.ndarray
    residue_im: np.ndarray


def _modal_tail(r: List[float]) -> List[float]:
    tail = r[4:]
    if len(tail) % 4 == 0:
        return tail
    if len(tail) >= 6 and (len(tail) - 2) % 4 == 0:
        return tail[2:]
    raise ValueError(
        f"Unexpected row width: total_cols={len(r)}, modal_tail_cols={len(tail)}. "
        "Expected 4 + 4*K or 4 + 2 + 4*K columns."
    )


def _safe_mode(Re: float, Im: float, mag: float, theta: float) -> Tuple[bool, tuple]:
    eps = 1e-12
    if not (np.isfinite(Re) and np.isfinite(Im)):
        return False, ()
    if abs(float(Re)) <= eps and abs(float(Im)) <= eps:
        return False, ()
    if abs(float(Im)) <= eps:
        return False, ()

    re_value = float(Re)
    im_value = float(Im)
    alpha = -abs(re_value)
    omega = abs(im_value) / OMEGA_N

    sigma = alpha
    omega_d = abs(im_value)
    denom = math.sqrt(sigma * sigma + omega_d * omega_d)
    if denom <= eps:
        return False, ()
    damping = -sigma / denom
    damping = float(np.clip(damping, 1e-6, 0.999999))
    freq = omega_d / (2.0 * math.pi)

    if np.isfinite(mag):
        mag_value = max(abs(float(mag)), eps)
    else:
        mag_value = eps
    rho = math.log(mag_value)

    if np.isfinite(theta):
        theta_value = float(theta)
        sin_theta = math.sin(theta_value)
        cos_theta = math.cos(theta_value)
        residue_re = mag_value * cos_theta
        residue_im = mag_value * sin_theta
    else:
        sin_theta = 0.0
        cos_theta = 1.0
        residue_re = mag_value
        residue_im = 0.0

    return True, (
        alpha,
        omega,
        rho,
        sin_theta,
        cos_theta,
        freq,
        damping,
        damping * 100.0,
        residue_re,
        residue_im,
    )


def parse_rows_to_struct(rows: List[List[float]]) -> Tuple[List[ParsedRow], dict]:
    ch_list, exc_list = [], []
    for r in rows:
        _modal_tail(r)
        exc_list.append(int(round(r[2])))
        ch_list.append(int(round(r[3])))

    channel_num = max(ch_list) + 1 if ch_list else 0
    excitation_num = max(exc_list) + 1 if exc_list else 0

    parsed: List[ParsedRow] = []
    dropped_bad = 0
    ignored_extra = 0
    missing_slots = 0

    for r in rows:
        H, M = float(r[0]), float(r[1])
        exc_id = int(round(r[2]))
        ch_id = int(round(r[3]))
        tail = _modal_tail(r)
        K_all = len(tail) // 4

        alpha = np.zeros((FIXED_K_MAX,), dtype=np.float32)
        omega = np.zeros((FIXED_K_MAX,), dtype=np.float32)
        rho = np.zeros((FIXED_K_MAX,), dtype=np.float32)
        sin_theta = np.zeros((FIXED_K_MAX,), dtype=np.float32)
        cos_theta = np.zeros((FIXED_K_MAX,), dtype=np.float32)
        freq = np.zeros((FIXED_K_MAX,), dtype=np.float32)
        damping = np.zeros((FIXED_K_MAX,), dtype=np.float32)
        damping_percent = np.zeros((FIXED_K_MAX,), dtype=np.float32)
        residue_re = np.zeros((FIXED_K_MAX,), dtype=np.float32)
        residue_im = np.zeros((FIXED_K_MAX,), dtype=np.float32)
        mask = np.zeros((FIXED_K_MAX,), dtype=np.float32)

        ignored_extra += max(0, K_all - FIXED_K_MAX)
        missing_slots += max(0, FIXED_K_MAX - K_all)

        for slot in range(FIXED_K_MAX):
            if slot >= K_all:
                continue
            base = 4 * slot
            ok, mode = _safe_mode(
                float(tail[base + 0]),
                float(tail[base + 1]),
                float(tail[base + 2]),
                float(tail[base + 3]),
            )
            if not ok:
                dropped_bad += 1
                continue

            (
                alpha[slot],
                omega[slot],
                rho[slot],
                sin_theta[slot],
                cos_theta[slot],
                freq[slot],
                damping[slot],
                damping_percent[slot],
                residue_re[slot],
                residue_im[slot],
            ) = mode
            mask[slot] = 1.0

        parsed.append(
            ParsedRow(
                H,
                M,
                exc_id,
                ch_id,
                int(mask.sum()),
                alpha,
                omega,
                rho,
                sin_theta,
                cos_theta,
                mask,
                freq,
                damping,
                damping_percent,
                residue_re,
                residue_im,
            )
        )

    if dropped_bad:
        print(f"[fixed-slots] skipped {dropped_bad} invalid pole tuples while keeping raw slot positions.")
    if ignored_extra:
        print(f"[fixed-slots] ignored {ignored_extra} modal tuples beyond the first {FIXED_K_MAX} raw slots.")
    if missing_slots:
        print(f"[fixed-slots] filled {missing_slots} missing raw slots with zeros and mask=0.")

    stats = {
        "K_max": FIXED_K_MAX,
        "channel_num": channel_num,
        "excitation_num": excitation_num,
    }
    return parsed, stats


if __name__ == "__main__":
    test_path = Path(__file__).parent / "real_dataset_oe.txt"
    test_rows = read_variable_txt(test_path)
    parsed, stats = parse_rows_to_struct(test_rows)
    print(stats)
    print(f"Total rows parsed: {len(parsed)}")
