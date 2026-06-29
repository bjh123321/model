from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import math

import numpy as np


FIXED_K_MAX = 3
FS_HZ = 512.0


def read_variable_txt(path: Path) -> List[List[float]]:
    rows: List[List[float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.replace(",", " ").split()
            try:
                row = [float(x) for x in parts]
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
    freq: np.ndarray
    damping: np.ndarray
    damping_percent: np.ndarray
    residue_re: np.ndarray
    residue_im: np.ndarray
    mask: np.ndarray
    # Compatibility fields for older visualization/postprocess helpers.
    alpha: np.ndarray
    omega: np.ndarray
    rho: np.ndarray
    sin_theta: np.ndarray
    cos_theta: np.ndarray


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
    vals = (Re, Im, mag, theta)
    if not all(np.isfinite(v) for v in vals):
        return False, ()
    if mag <= eps:
        return False, ()
    if abs(Im) <= eps:
        return False, ()

    sigma = -abs(float(Re))
    omega_d = abs(float(Im))
    denom = math.sqrt(sigma * sigma + omega_d * omega_d)
    if denom <= eps:
        return False, ()
    damping = -sigma / denom
    if not np.isfinite(damping):
        return False, ()
    damping = float(np.clip(damping, 1e-6, 0.999999))
    freq = omega_d / (2.0 * math.pi)
    residue_re = float(mag) * math.cos(float(theta))
    residue_im = float(mag) * math.sin(float(theta))
    rho_log = math.log(max(float(mag), eps))
    sin_theta = math.sin(float(theta))
    cos_theta = math.cos(float(theta))
    return True, (
        omega_d,
        freq,
        damping,
        damping * 100.0,
        residue_re,
        residue_im,
        sigma,
        omega_d,
        rho_log,
        sin_theta,
        cos_theta,
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
    dropped_extra = 0

    for r in rows:
        H, M = float(r[0]), float(r[1])
        exc_id = int(round(r[2]))
        ch_id = int(round(r[3]))
        tail = _modal_tail(r)
        K_all = len(tail) // 4

        modes = []
        for k in range(K_all):
            base = 4 * k
            ok, mode = _safe_mode(
                float(tail[base + 0]),
                float(tail[base + 1]),
                float(tail[base + 2]),
                float(tail[base + 3]),
            )
            if ok:
                modes.append(mode)
            else:
                dropped_bad += 1

        modes.sort(key=lambda item: item[0])
        if len(modes) > FIXED_K_MAX:
            dropped_extra += len(modes) - FIXED_K_MAX
        modes = modes[:FIXED_K_MAX]

        freq = np.zeros((FIXED_K_MAX,), dtype=np.float32)
        damping = np.zeros((FIXED_K_MAX,), dtype=np.float32)
        damping_percent = np.zeros((FIXED_K_MAX,), dtype=np.float32)
        residue_re = np.zeros((FIXED_K_MAX,), dtype=np.float32)
        residue_im = np.zeros((FIXED_K_MAX,), dtype=np.float32)
        alpha = np.zeros((FIXED_K_MAX,), dtype=np.float32)
        omega = np.zeros((FIXED_K_MAX,), dtype=np.float32)
        rho = np.zeros((FIXED_K_MAX,), dtype=np.float32)
        sin_theta = np.zeros((FIXED_K_MAX,), dtype=np.float32)
        cos_theta = np.zeros((FIXED_K_MAX,), dtype=np.float32)
        mask = np.zeros((FIXED_K_MAX,), dtype=np.float32)

        for slot, mode in enumerate(modes):
            (
                _sort_omega,
                freq[slot],
                damping[slot],
                damping_percent[slot],
                residue_re[slot],
                residue_im[slot],
                alpha[slot],
                omega[slot],
                rho[slot],
                sin_theta[slot],
                cos_theta[slot],
            ) = mode
            mask[slot] = 1.0

        parsed.append(
            ParsedRow(
                H,
                M,
                exc_id,
                ch_id,
                int(mask.sum()),
                freq,
                damping,
                damping_percent,
                residue_re,
                residue_im,
                mask,
                alpha,
                omega,
                rho,
                sin_theta,
                cos_theta,
            )
        )

    if dropped_bad:
        print(f"[fixed-slots] skipped {dropped_bad} invalid modal tuples while parsing.")
    if dropped_extra:
        print(f"[fixed-slots] truncated {dropped_extra} modal tuples beyond the first {FIXED_K_MAX} frequency slots.")

    stats = {
        "K_max": FIXED_K_MAX,
        "channel_num": channel_num,
        "excitation_num": excitation_num,
    }
    return parsed, stats


if __name__ == "__main__":
    test_path = Path(__file__).parent / "sim_dataset_new.txt"
    test_rows = read_variable_txt(test_path)
    parsed, stats = parse_rows_to_struct(test_rows)
    print(stats)
    print(f"Total rows parsed: {len(parsed)}")
