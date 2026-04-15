# OE_BATCH_PROCESS.py
from __future__ import annotations

import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .envelop import compute_spectrum, envelope_rms_tv
from .OE import OEModel


# =========================
# 固定配置（按你确认的默认）
# =========================
FS = 512
MAX_MODES = 6

# 峰检测（沿用你 OE_PROCESS.py 的默认风格）
F_MAX_HZ = 60.0
MATCH_HALF_WIDTH_HZ = 2.0          # 峰 ↔ OE 频率匹配窗 ±2 Hz
PROM_SCALE = 0.01                  # prom = 0.01*(P95-P5)
MIN_WIDTH_HZ = 1.0
MIN_DIST_HZ = 3.0

# OE参数
OE_KWARGS = dict(
    na=30, nb=30, nk=0,
    max_nfev=50,
    init_method="arx",
    rpm_khat=75,
    demean=False,
)

# 激励方式编码（按你给的映射）
EXCITATION_ID = {
    "FXD_同": 0,
    "FXD_反": 1,
    "PW_同": 2,
    "PW_反": 3,
    "JFY_同": 4,
    "JFY_反": 5,
}

# 文件名中的英文编码 -> 程序内部激励键
EXCITATION_CODE_MAP = {
    "FXD_FDC": "FXD_同",
    "FXD_DC": "FXD_反",
    "PW_FDC": "PW_同",
    "PW_DC": "PW_反",
    "JFY_FDC": "JFY_同",
    "JFY_DC": "JFY_反",
}

# 激励输入列名
INPUT_SIGNAL_NAME = "PM_O1"

# 你要给的通道名列表
# 注意：这里的名字要与 dat 第一行表头“完全相等”
CHANNELS_BY_POS: Dict[str, List[str]] = {
    "FXD": ['LVT1', 'LVT2', 'LVT3', 'LVT4', 'LRY', 'LRY1', 'RVT1', 'RVT2', 'RVT3', 'RVT4', 'RRY', 'RRY1'],
    "PW":  ['LHT1', 'LHT4', 'LHT2', 'LHT3', 'RHT1', 'RHT4', 'RHT2', 'RHT3'],
    "JFY": ['LW1', 'LW2', 'LW3', 'LAI', 'RW1', 'RW2', 'RW3', 'RAI'],
}

# 输出 txt 时的固定通道编号起点：
# JFY: 0-7, FXD: 8-19, PW: 20-27
CHANNEL_ID_BASE = {
    "JFY": 0,
    "FXD": 8,
    "PW": 20,
}


# =========================
# 工具函数
# =========================
@dataclass
class ParsedMeta:
    H: float
    M: float
    excite_pos: str   # FXD/PW/JFY
    excite_phase: str # 同/反
    excite_key: str   # e.g. "FXD_反"
    excite_id: int


def _decode_hm_token(token: str) -> float:
    """
    H/M 数值解码：
    - 含 d：直接把 d 当小数点，如 8d23 -> 8.23
    - 不含 d 且长度>=2：默认最后两位为小数，如 0823 -> 8.23, 140 -> 1.40
    - 其他情况：直接转 float
    """
    token = token.strip()
    if "d" in token:
        return float(token.replace("d", "."))
    if len(token) >= 2:
        return float(f"{token[:-2]}.{token[-2:]}")
    return float(token)


def parse_HM_from_filename(file_path: Path) -> Tuple[float, float]:
    """
    以文件名为准解析 H/M。
    支持：
    - H8d23 / M1d40
    - H0823 / M140
    """
    s = file_path.stem

    mH = re.search(r"_H([0-9d]+)(?=_|$)", s)
    mM = re.search(r"_M([0-9d]+)(?=_|$)", s)
    if not mH or not mM:
        raise ValueError(f"无法从文件名解析 H/M: {file_path}")

    H = _decode_hm_token(mH.group(1))
    M = _decode_hm_token(mM.group(1))
    return H, M


def parse_excitation_from_filename(file_path: Path) -> ParsedMeta:
    """
    从文件名提取激励位置与相位：
    例如：
    - FXD_DC  -> FXD_反
    - FXD_FDC -> FXD_同
    """
    s = file_path.stem

    m = re.search(r"_(FXD|PW|JFY)_(FDC|DC)(?=_|$)", s)
    if not m:
        raise ValueError(f"无法从文件名解析激励方式(FXD/PW/JFY + FDC/DC): {file_path}")

    code = f"{m.group(1)}_{m.group(2)}"
    if code not in EXCITATION_CODE_MAP:
        raise ValueError(f"激励方式未在映射表中: {code} ({file_path})")

    key = EXCITATION_CODE_MAP[code]
    pos, phase = key.split("_", 1)

    H, M = parse_HM_from_filename(file_path)
    return ParsedMeta(
        H=H, M=M,
        excite_pos=pos,
        excite_phase=phase,
        excite_key=key,
        excite_id=EXCITATION_ID[key],
    )


def read_table_dat(file_path: Path) -> Tuple[np.ndarray, List[str]]:
    """
    读取 dat：第一行为表头，逗号分隔。
    兼容首列/若干列为时间字符串等非数值内容的情况：
    - 逐列尝试转为数值
    - 全列都无法转为数值的列直接丢弃（如 Time=0:0:0:000）
    - 其余列保留，并转成 float64
    返回：Y (float64) 与保留下来的 header list
    """
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    df.columns = [str(c).strip() for c in df.columns]

    keep_cols = []
    keep_series = []
    for col in df.columns:
        s_num = pd.to_numeric(df[col], errors='coerce')
        # 像时间列 '0:0:0:000' 这种会整列都是 NaN，直接丢弃
        if s_num.notna().any():
            keep_cols.append(col)
            keep_series.append(s_num.astype(np.float64))

    if not keep_cols:
        raise ValueError(f"文件中未找到可用于计算的数值列: {file_path}")

    df_num = pd.concat(keep_series, axis=1)
    df_num.columns = keep_cols

    Y = df_num.to_numpy(dtype=np.float64)
    # 去均值（与你原脚本一致）
    Y = Y - Y.mean(axis=0, keepdims=True)
    return Y, keep_cols


def get_input_signal(header: List[str], Y: np.ndarray, input_name: str = INPUT_SIGNAL_NAME) -> np.ndarray:
    """
    通过表头精确匹配激励输入列，只取 PM_O1。
    """
    name_to_idx = {name: i for i, name in enumerate(header)}
    if input_name not in name_to_idx:
        raise ValueError(f"未找到激励输入列: {input_name}")
    return Y[:, name_to_idx[input_name]].astype(float).ravel()


def select_channel_indices(header: List[str], excite_pos: str) -> List[Tuple[int, str, int]]:
    """
    按激励位置选择要处理的通道名列表，并在 header 中做“完全相等”匹配。
    返回 [(channel_id, ch_name, col_idx_in_file), ...]

    channel_id 保持与旧 txt 一致：
    - JFY 按其通道字符串顺序映射为 0~7
    - FXD 按其通道字符串顺序映射为 8~19
    - PW  按其通道字符串顺序映射为 20~27
    """
    if excite_pos not in CHANNELS_BY_POS:
        raise ValueError(f"CHANNELS_BY_POS 未配置该激励位置: {excite_pos}")
    if excite_pos not in CHANNEL_ID_BASE:
        raise ValueError(f"CHANNEL_ID_BASE 未配置该激励位置: {excite_pos}")

    targets = CHANNELS_BY_POS[excite_pos]
    base_id = CHANNEL_ID_BASE[excite_pos]
    name_to_idx = {name: i for i, name in enumerate(header)}

    out: List[Tuple[int, str, int]] = []
    for local_idx, name in enumerate(targets):
        if name not in name_to_idx:
            continue
        channel_id = base_id + local_idx
        out.append((channel_id, name, name_to_idx[name]))
    return out


def find_spectrum_peaks(y: np.ndarray, fs: int) -> np.ndarray:
    """
    返回 0~F_MAX_HZ 范围内的谱包络峰频率数组 peak_freqs (Hz)。
    若失败，则返回空数组。
    """
    try:
        f, y_fft = compute_spectrum(y, fs, window=False, mode="amplitude")
        mask = f <= F_MAX_HZ
        f = f[mask]
        y_fft = y_fft[mask]

        # 频域包络（沿用你脚本）
        _, env = envelope_rms_tv(y_fft, rms_win=32, n_iter=50, lam=50, rho=2)

        # prom 自适应
        prom = PROM_SCALE * (np.percentile(env, 95) - np.percentile(env, 5))
        if not np.isfinite(prom) or prom <= 0:
            return np.array([], dtype=float)

        df = f[1] - f[0]
        min_width = max(1, int(MIN_WIDTH_HZ / df))
        min_dist = max(1, int(MIN_DIST_HZ / df))

        idx, _ = find_peaks(env, prominence=prom, width=min_width, distance=min_dist)
        return f[idx].astype(float)
    except Exception:
        return np.array([], dtype=float)


def oe_extract_modes(u: np.ndarray, y: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    运行 OE，返回初筛后的 (p, r, s)：
    - p: 离散极点(复数)
    - r: 留数(复数)
    - s: 连续域极点 log(p)/Ts (复数)
    初筛规则：正频支 + Re(s)<0 + 0<=zeta<=0.8 + 有限性
    """
    model = OEModel(**OE_KWARGS).fit(y, u)
    r, p, _k = model.residues()

    r = np.asarray(r, dtype=np.complex128)
    p = np.asarray(p, dtype=np.complex128)

    Ts = 1.0 / fs
    s = np.log(p + 1e-12) / Ts
    sigma = np.real(s)
    omega_d = np.imag(s)

    freq_hz = np.abs(omega_d) / (2 * np.pi)
    omega_n = np.sqrt(sigma**2 + omega_d**2)

    valid = np.isfinite(freq_hz) & np.isfinite(omega_n) & (omega_n > 0)
    zeta = np.full_like(freq_hz, np.nan, dtype=float)
    zeta[valid] = -sigma[valid] / omega_n[valid]

    mask_pos = omega_d > 0
    mask_stable = sigma < 0
    mask_zeta = np.isfinite(zeta) & (zeta <= 0.8) & (zeta >= 0.0)

    mask = valid & mask_pos & mask_stable & mask_zeta

    return p[mask], r[mask], s[mask]


def match_peaks_to_oe(peak_freqs: np.ndarray, p: np.ndarray, r: np.ndarray, s: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将谱峰与 OE 模态按频率匹配（±MATCH_HALF_WIDTH_HZ），冲突时选 |r| 最大。
    返回匹配后的 (p_sel, r_sel, s_sel)（数量<=谱峰数，也可能为0）。
    若 peak_freqs 为空，则 fallback：按 |r| 取前 MAX_MODES。
    """
    if p.size == 0:
        return np.array([], dtype=np.complex128), np.array([], dtype=np.complex128), np.array([], dtype=np.complex128)

    # OE 频率（来自连续域）
    omega_d = np.imag(s)
    oe_freqs = np.abs(omega_d) / (2 * np.pi)

    if peak_freqs.size == 0:
        # fallback：按 |r| 取前 MAX_MODES
        order = np.argsort(np.abs(r))[::-1]
        order = order[:MAX_MODES]
        return p[order], r[order], s[order]

    peak_order = np.arange(len(peak_freqs))

    used = np.zeros(len(oe_freqs), dtype=bool)
    chosen: List[int] = []

    for jj in peak_order:
        fp = float(peak_freqs[jj])
        cand = np.where((np.abs(oe_freqs - fp) <= MATCH_HALF_WIDTH_HZ) & (~used))[0]
        if cand.size == 0:
            continue
        best = cand[np.argmax(np.abs(r[cand]))]
        used[best] = True
        chosen.append(int(best))

        if len(chosen) >= MAX_MODES:
            break

    if len(chosen) == 0:
        return np.array([], dtype=np.complex128), np.array([], dtype=np.complex128), np.array([], dtype=np.complex128)

    chosen = np.array(chosen, dtype=int)
    return p[chosen], r[chosen], s[chosen]


def format_row(meta: ParsedMeta, channel_id: int, p_sel: np.ndarray, r_sel: np.ndarray, s_sel: np.ndarray) -> List[str]:
    """
    组织一行输出：
    H, M, excite_id, channel_id, (sigma1, omegad1, |R1|, angle1), ...
    - 极点用连续域 s_sel：sigma<0、omega_d>0
    - 留数幅值/相位用离散域 r_sel：abs/angle
    - 不足 MAX_MODES 的不补（你要求“没有就不写”）
    """
    out: List[str] = [
        f"{meta.H:.10g}",
        f"{meta.M:.10g}",
        str(meta.excite_id),
        str(channel_id),
    ]

    for si, ri in zip(s_sel, r_sel):
        sigma = float(np.real(si))
        omega_d = float(np.imag(si))
        Rabs = float(np.abs(ri))
        Rph = float(np.angle(ri))  # (-pi, pi]
        out += [f"{sigma:.10g}", f"{omega_d:.10g}", f"{Rabs:.10g}", f"{Rph:.10g}"]

    return out


def read_selected_dat_files(root_dir: Path, selected_list_file: Path) -> List[Path]:
    """
    从 selected_CG_path.txt 读取待处理文件名列表。
    文件中每行都是 root_dir 下的文件名，例如：
    CG_SC_FXD_DC_E_1d0_H0823_M140_20260306_JiZai.dat

    返回与 root_dir 拼接后的绝对路径列表。
    """
    root_dir = Path(root_dir)
    selected_list_file = Path(selected_list_file)

    if not selected_list_file.exists():
        raise FileNotFoundError(f"未找到选取文件列表: {selected_list_file}")

    dat_files: List[Path] = []
    seen = set()

    with selected_list_file.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            name = line.strip()
            if not name:
                continue

            fp = root_dir / name
            if fp in seen:
                continue
            if not fp.exists():
                raise FileNotFoundError(
                    f"selected_CG_path.txt 第 {line_no} 行指定的文件不存在: {fp}"
                )
            if fp.suffix.lower() != ".dat":
                raise ValueError(
                    f"selected_CG_path.txt 第 {line_no} 行不是 .dat 文件: {fp}"
                )

            dat_files.append(fp)
            seen.add(fp)

    if len(dat_files) == 0:
        raise ValueError(f"选取文件列表为空: {selected_list_file}")

    return dat_files


# =========================
# 主流程
# =========================
def batch_process(root_dir: Path, out_file: Path, selected_list_file: Path) -> None:
    """
    根据 selected_CG_path.txt 指定的文件名列表，
    从 root_dir 下选取对应 .dat 文件进行处理，并写入 out_file。
    不再遍历 root_dir 下全部 .dat，也不再生成 log 文件。
    """
    root_dir = Path(root_dir)
    out_file = Path(out_file)
    selected_list_file = Path(selected_list_file)

    dat_files = read_selected_dat_files(root_dir, selected_list_file)

    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("w", encoding="utf-8") as fout:
        for fp in dat_files:
            try:
                meta = parse_excitation_from_filename(fp)
                Y, header = read_table_dat(fp)

                # u 通过表头匹配，只用 PM_O1
                u = get_input_signal(header, Y, INPUT_SIGNAL_NAME)

                # 按激励位置选通道（精确匹配表头）
                ch_list = select_channel_indices(header, meta.excite_pos)
                if len(ch_list) == 0:
                    print(f"[SKIP] no channel matched | {fp}")
                    continue

                for channel_id, ch_name, col_idx in ch_list:
                    try:
                        y = Y[:, col_idx].astype(float).ravel()

                        # 谱峰
                        peak_freqs = find_spectrum_peaks(y, FS)

                        # OE + 初筛
                        p0, r0, s0 = oe_extract_modes(u, y, FS)

                        # 峰匹配（或 fallback）
                        p_sel, r_sel, s_sel = match_peaks_to_oe(peak_freqs, p0, r0, s0, FS)

                        # 没有模态：按你要求“什么都不写”
                        if p_sel.size == 0:
                            continue

                        row = format_row(meta, channel_id, p_sel, r_sel, s_sel)
                        fout.write("\t".join(row) + "\n")

                    except Exception as e_ch:
                        print(f"[FAIL-CH] {fp} | channel_id={channel_id} col={col_idx} name={ch_name} | {repr(e_ch)}")
                        print(traceback.format_exc())
                        continue

            except Exception as e_fp:
                print(f"[FAIL-FILE] {fp} | {repr(e_fp)}")
                print(traceback.format_exc())
                continue


if __name__ == "__main__":
    # ====== 你只需要改这里 ======
    ROOT = Path(r"C:\Users\Administrator\Desktop\试飞数据(训练用) - 副本")
    OUT = Path(r"C:\Users\Administrator\Desktop") / "new_real.txt"

    # 示例：
    # selected_CG_path.txt 中每行写一个 ROOT 文件夹下待处理的 dat 文件名，例如：
    # CG_SC_FXD_DC_E_1d0_H0823_M140_20260306_JiZai.dat
    SELECTED_LIST = Path(r"C:\Users\Administrator\Desktop\selected_CG_path.txt")

    batch_process(ROOT, OUT, SELECTED_LIST)
    print(f"完成. 输出: {OUT}")
