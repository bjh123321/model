from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import math

import numpy as np


FIXED_K_MAX = 3
FS_HZ = 512.0
OMEGA_N = math.pi * FS_HZ


def _parse_float_cell(cell: str) -> float:
    cell = cell.strip()
    if cell == "":
        return float("nan")
    return float(cell)


def read_variable_txt(path: Path) -> List[List[float]]:
    rows: List[List[float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            ln = raw.rstrip("\r\n")
            if not ln.strip() or ln.lstrip().startswith("#"):
                continue
            try:
                if "\t" in ln:
                    row = [_parse_float_cell(x) for x in ln.split("\t")]
                else:
                    row = [float(x) for x in ln.replace(",", " ").split()]
            except ValueError:
                continue
            if len(row) >= 4:
                rows.append(row)
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
