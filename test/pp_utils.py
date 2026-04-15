from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class ExcSegInfo:
    f0: float
    f1: float
    dur: float
    t0: float


def build_exc_seg_list(exc_segments) -> List[ExcSegInfo]:
    """
    根据激励表构造带起始时间偏移的激励段列表
    """
    seg_list: List[ExcSegInfo] = []
    t_off = 0.0
    for seg in exc_segments:
        seg_list.append(
            ExcSegInfo(
                f0=float(seg.f0),
                f1=float(seg.f1),
                dur=float(seg.dur),
                t0=float(t_off),
            )
        )
        t_off += float(seg.dur)
    return seg_list


def calc_pp_for_freq(
    y: np.ndarray,
    f_hz: float,
    Ts: float,
    seg_list: List[ExcSegInfo],
    peak_win_sec: float = 1,
):
    """
    返回:
        pp_str : 峰值字符串
        best_idx : 峰值所在的全局索引
    """
    if y is None:
        return "", None

    y = np.asarray(y, dtype=np.float64).ravel()
    N = y.size
    if N == 0 or len(seg_list) == 0:
        return "", None

    fs = 1.0 / Ts
    half = int(round(peak_win_sec * fs))
    if half < 1:
        half = 1

    best_pp = None
    best_idx = None

    for seg in seg_list:
        f0 = seg.f0
        f1 = seg.f1
        dur = seg.dur
        t0 = seg.t0

        f_lo = min(f0, f1)
        f_hi = max(f0, f1)

        if not (f_lo <= f_hz <= f_hi):
            continue

        if abs(f1 - f0) <= 1e-12:
            t_g = t0 + 0.5 * dur
        else:
            t_local = dur * (f_hz - f0) / (f1 - f0)
            t_g = t0 + t_local

        idx0 = int(round(t_g * fs))
        i0 = max(0, idx0 - half)
        i1 = min(N, idx0 + half + 1)

        if i1 <= i0:
            continue

        y_win = y[i0:i1]

        local_idx = np.argmax(np.abs(y_win))
        global_idx = i0 + local_idx
        pp = float(np.abs(y_win[local_idx]))

        if (best_pp is None) or (pp > best_pp):
            best_pp = pp
            best_idx = global_idx

    if best_pp is None:
        return ""

    return f"{best_pp:.6f}"

if __name__ == '__main__':
    from pathlib import Path
    import pandas as pd

    class ExcSegment:
        def __init__(self, amp_base, f0, f1, dur):
            self.amp_base = amp_base
            self.f0 = f0
            self.f1 = f1
            self.dur = dur

    data_file = Path(r"D:\Projects\flightdata\PR_PW_FDC_C_1d5_H0800_M120_XGD_mx_20260130_v1.txt")
    fs = 512
    Ts = 1/fs

    df = pd.read_table(data_file, sep=r"[,\t ]+", engine="python")
    Y = df.to_numpy(dtype=np.float64)
    Y = Y - Y.mean(axis=0, keepdims=True)
    y = Y[:, 19]

    segs = [
        ExcSegment(1.0, 0.1, 30.0, 30.0),   # 第一段扫频
        ExcSegment(0.0, 0.0, 0.0, 1.0),     # 中间停1秒
        ExcSegment(1.0, 25.0, 45.0, 20.0),  # 第二段扫频
    ]

    seg_list = build_exc_seg_list(segs)

    # 取 10Hz 对应位置附近的峰峰值
    pp_str, idx, t_g, t_local = calc_pp_for_freq(
        y=y,
        f_hz=36,
        Ts=Ts,
        seg_list=seg_list,
        peak_win_sec=1,
    )

    print("峰值 =", pp_str)
    print("峰值索引 =", idx)
    print("响应峰值 =", np.max(abs(y)))

    if idx is not None:
        print("峰值时间 =", idx * Ts)
        print("激励时间 =", t_g)