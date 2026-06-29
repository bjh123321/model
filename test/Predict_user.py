# %%
import argparse
import re
import sys
import configparser
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from scipy.signal import lfilter

from surrogate_model.config import Config
from surrogate_model.models.blocks import MFDNN
from surrogate_model.data.normalizer import Normalizer
from surrogate_model.data.xlsx_utils import load_signal_type_rows
from surrogate_model.test.residue_pred import build_BA_from_residues
# from Error import calc_relative_error
from surrogate_model.test.pp_utils import build_exc_seg_list, calc_pp_for_freq


def get_exe_dir():
    """
    返回 exe 所在目录：
    - 打包后（exe）：exe 文件所在目录
    - 未打包（python 运行）：当前脚本所在目录
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    else:
        return Path(__file__).resolve().parent



from functools import lru_cache

@lru_cache(maxsize=None)
def _load_channel_freq_ranges(csv_path: Path) -> Dict[str, List[Optional[Tuple[float, float]]]]:
    """读取通道频率范围表（csv）。每行格式：channel, r1, r2, r3
    其中 r? 形如 '0-20'，允许为空。返回 {channel: [ (fmin,fmax) or None ] * 3 }。
    """
    df = pd.read_csv(csv_path, header=None, dtype=str, encoding="utf-8-sig")
    mp: Dict[str, List[Optional[Tuple[float, float]]]] = {}
    for _, row in df.iterrows():
        ch = (row.iloc[0] or "").strip()
        if not ch:
            continue
        ranges: List[Optional[Tuple[float, float]]] = []
        for k in range(1, 4):
            val = ""
            if k < len(row):
                val = "" if row.iloc[k] is None else str(row.iloc[k]).strip()
            if not val or val.lower() == "nan":
                ranges.append(None)
                continue
            m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*[-~]\s*([0-9]+(?:\.[0-9]+)?)\s*$", val)
            if not m:
                ranges.append(None)
                continue
            fmin = float(m.group(1))
            fmax = float(m.group(2))
            if fmax < fmin:
                fmin, fmax = fmax, fmin
            ranges.append((fmin, fmax))
        while len(ranges) < 3:
            ranges.append(None)
        mp[ch] = ranges[:3]
    return mp


def _select_modes_by_freq_ranges(
    freqs_all: np.ndarray,
    alpha_all: np.ndarray,
    omega_all: np.ndarray,
    rho_all: np.ndarray,
    ranges: List[Optional[Tuple[float, float]]],
) -> Tuple[List[int], List[str], List[str]]:
    """按 ranges（最多3个频率段）筛选模态：
    - 每个频率段至多取 1 个模态（rho 最大）
    - 输出：所选模态在原数组中的索引列表（去重）、频率槽位字符串（3个）、阻尼槽位字符串（3个）
    """
    used = set()
    chosen_idx: List[int] = []
    freq_slots: List[str] = ["" for _ in range(3)]
    damp_slots: List[str] = ["" for _ in range(3)]

    for i in range(3):
        rg = ranges[i] if i < len(ranges) else None
        if rg is None:
            continue
        fmin, fmax = rg
        cand = (freqs_all >= fmin) & (freqs_all <= fmax)
        if not np.any(cand):
            continue
        cand_idx = np.where(cand)[0]
        cand_idx = np.array([ix for ix in cand_idx.tolist() if ix not in used], dtype=int)
        if cand_idx.size == 0:
            continue
        best_local = cand_idx[int(np.argmax(rho_all[cand_idx]))]
        used.add(int(best_local))
        chosen_idx.append(int(best_local))
        chosen_freq = float(freqs_all[best_local])
        a = float(alpha_all[best_local])
        w = float(omega_all[best_local])
        damping = (-a / np.sqrt(a * a + w * w)) * 200.0
        freq_slots[i] = f"{chosen_freq:.2f}"
        damp_slots[i] = f"{damping:.6f}"

    if chosen_idx:
        chosen_idx = sorted(chosen_idx, key=lambda ix: float(freqs_all[ix]))
    return chosen_idx, freq_slots, damp_slots


# ============================================================
# 1) 规则函数：通道筛选 / tag 格式化
# ============================================================
channels = ["LW1", "LW2", "LW3", "LAI", "RW1", "RW2", "RW3", "RAI",
            "LVT1", "LVT2", "LVT3", "LVT4", "LRY", "LRY1", "RVT1", "RVT2",
            "RVT3", "RVT4", "RRY", "RRY1", "LHT1", "LHT4", "LHT2", "LHT3",
            "RHT1", "RHT4", "RHT2", "RHT3", "F1Z", "F1Y", "ZWL1Y", "ZWL1Z",
            "YWL1Y", "YWL1Z"
            ]

def channels_for_exc_plot(exc_id: int) -> np.ndarray:
    """
    与训练数据的激励编号保持一致：
      0/1 -> FXD -> 8..19
      2/3 -> PW  -> 20..27
      4/5 -> JFY -> 0..7
    """
    if exc_id in (0, 1):
        return np.arange(8, 20, dtype=np.int64)
    if exc_id in (2, 3):
        return np.arange(20, 28, dtype=np.int64)
    if exc_id in (4, 5):
        return np.arange(0, 8, dtype=np.int64)
    return np.arange(0, 0, dtype=np.int64)


def format_H_tag(H_km: float) -> str:
    """
    高度 tag：两位整数 + 两位小数（保持 4 位），例如 8.23 -> H0823
    """
    s = f"{H_km:05.2f}"  # '08.23'
    return "H" + s.replace(".", "")


def format_M_tag(M: float) -> str:
    """
    马赫 tag：一位整数 + 两位小数（保持 3 位），例如 1.00 -> M100, 0.85 -> M085
    """
    s = f"{M:04.2f}"  # '1.00'
    return "M" + s.replace(".", "")


def parse_amp_tag(tag: str) -> float:
    """
    解析 0d5 / 1d0 / 1d5 -> 0.5 / 1.0 / 1.5
    """
    m = re.fullmatch(r"(\d+)d(\d+)", tag.strip())
    if not m:
        raise ValueError(f"无法解析幅值倍率: {tag}")
    a = int(m.group(1))
    b = int(m.group(2))
    # 约定 d 后面是一位小数（0..9）。如果你以后出现两位小数再改这里
    return a + b / 10.0


# ============================================================
# 2) ini 解析：把 DataX 解析成 entry
# ============================================================

@dataclass
class TableEntry:
    key: str          # Data1/Data2...
    raw: str          # 原始字符串 FXD_FDC_A_0d5_H1111_M100
    sortie_num: str   # '002'
    exc_name: str     # FXD_FDC / JFY_DC
    kind: str         # A / B / AB ...
    amp_tag: str      # 0d5 / 1d0 ...
    amp: float        # 0.5 / 1.0 ...
    H_tag: str        # H1111
    M_tag: str        # M100
    H_km: float       # 11.11
    M: float          # 1.00
    exc_id: int       # 0..5


def parse_data_string(data_str: str) -> Tuple[str, str, str, str, str]:
    """
    把 'FXD_FDC_A_0d5_H1111_M100' 拆成：
      exc_name='FXD_FDC'
      kind='A'
      amp_tag='0d5'
      H_tag='H1111'
      M_tag='M100'
    规则：从右往左拆，最后两段一定是 Hxxxx、Mxxx。
    """
    parts = data_str.strip().split("_")
    if len(parts) < 5:
        raise ValueError(f"Data 格式不合法: {data_str}")

    M_tag = parts[-1]
    H_tag = parts[-2]
    amp_tag = parts[-3]
    kind = parts[-4]
    exc_name = "_".join(parts[:-4])

    if not re.fullmatch(r"H\d{4}", H_tag):
        raise ValueError(f"H_tag 不合法: {H_tag} in {data_str}")
    if not re.fullmatch(r"M\d{3}", M_tag):
        raise ValueError(f"M_tag 不合法: {M_tag} in {data_str}")

    return exc_name, kind, amp_tag, H_tag, M_tag


def decode_HM(H_tag: str, M_tag: str) -> Tuple[float, float]:
    """
    H1111 -> 11.11 km
    M100 -> 1.00 Ma
    """
    H_num = int(H_tag[1:])  # 1111
    M_num = int(M_tag[1:])  # 100
    H_km = H_num / 100.0
    M = M_num / 100.0
    return H_km, M


def load_task_from_ini(task_ini_path: str, exc_name_to_id: Dict[str, int]) -> Tuple[str, List[TableEntry]]:
    """
    读取 ini：
      [Sortie] Num=002
      [TABLE] Data1=... Data2=...
    返回：
      sortie_num, entries(list)
    """
    cp = configparser.ConfigParser()
    cp.read(task_ini_path, encoding="utf-8")

    sortie_num = cp.get("Sortie", "Num", fallback="000").strip()

    if "TABLE" not in cp:
        raise ValueError("ini 中缺少 [TABLE] 段")

    entries: List[TableEntry] = []
    for k, v in cp["TABLE"].items():
        # configparser 会把 key 变小写，这里保留原 key 风格即可
        key = k  # 'data1'
        raw = v.strip()

        exc_name, kind, amp_tag, H_tag, M_tag = parse_data_string(raw)
        H_km, M = decode_HM(H_tag, M_tag)
        amp = parse_amp_tag(amp_tag)

        exc_id = int(exc_name_to_id[exc_name])

        entries.append(TableEntry(
            key=key,
            raw=raw,
            sortie_num=sortie_num,
            exc_name=exc_name,
            kind=kind,
            amp_tag=amp_tag,
            amp=amp,
            H_tag=H_tag,
            M_tag=M_tag,
            H_km=H_km,
            M=M,
            exc_id=exc_id,
        ))

    return sortie_num, entries


# ============================================================
# 3) 激励表：读取 + 生成 u(t)
# ============================================================

@dataclass
class ExcSegment:
    amp_base: float
    f0: float
    f1: float
    dur: float


def load_excitation_table(xlsx_path: str, sheet_name: str = "Sheet1") -> Dict[str, List[ExcSegment]]:
    table: Dict[str, List[ExcSegment]] = {}
    for kind, amp, f0, f1, dur in load_signal_type_rows(Path(xlsx_path)):
        seg = ExcSegment(
            amp_base=float(amp),
            f0=float(f0),
            f1=float(f1),
            dur=float(dur),
        )
        table.setdefault(kind, []).append(seg)
    return table



# ============================================================
# 3.5) 路径 ini：读取输出/配置目录
# ============================================================

def load_paths_ini(paths_ini: str) -> Dict[str, str]:
    """读取“路径 ini”（你给的那类 GeneralParameter/.../CONFIG/xxx.ini 之外的路径总表）。

    返回一个扁平 dict，例如：
      {
        "CONFIG": "D:/.../CONFIG",
        "PR_XGD": "D:/.../PR/XGD",
        "VG_CG": ".../VG_CG",
        "VF_CG": ".../VF_CG",
        "HG_CG": ".../HG_CG",
        "HF_CG": ".../HF_CG",
        ...
      }

    规则：
    - 读取每个 section 里第一个以 'path_' 开头的 key。
    - 返回的 key 为 section 名（去掉前后空格）。
    """
    cp = configparser.ConfigParser()
    cp.read(paths_ini, encoding="utf-8")

    out: Dict[str, str] = {}
    for sec in cp.sections():
        # 找形如 path_XXX 的键
        kv = [(k, v) for k, v in cp[sec].items() if k.lower().startswith("path_")]
        if not kv:
            continue
        # 取第一个 path_* 的值
        _, v = kv[0]
        out[sec.strip()] = str(v).strip()
    return out


def _chirp_linear(f0: float, f1: float, dur: float, fs: int) -> np.ndarray:
    N = int(round(dur * fs))
    if N <= 0 or dur <= 0:
        return np.zeros((0,), dtype=np.float64)
    t = np.arange(N, dtype=np.float64) / fs
    k = (f1 - f0) / dur
    phi = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)
    return np.sin(phi).astype(np.float64)


def build_u_from_kind(kind: str, amp_scale: float, fs: int, table: Dict[str, List[ExcSegment]]) -> np.ndarray:
    """
    kind: A/B/AB...
    amp_scale: 0.5/1.0/1.5...
    """
    if kind not in table:
        raise KeyError(f"激励种类 {kind} 不在激励表中")
    u_list = []
    for seg in table[kind]:
        N = int(round(seg.dur * fs))
        if N <= 0:
            continue

        if seg.amp_base == 0.0 or (seg.f0 == 0.0 and seg.f1 == 0.0):
            u_seg = np.zeros((N,), dtype=np.float64)
        else:
            u_seg = _chirp_linear(seg.f0, seg.f1, seg.dur, fs)
            u_seg *= (seg.amp_base * amp_scale)

        u_list.append(u_seg)

    return np.concatenate(u_list, axis=0) if u_list else np.zeros((0,), dtype=np.float64)


# ============================================================
# 4) 网络推理：batched 推理并返回 numpy（含 omega/rho 物理变换）
# ============================================================

@torch.no_grad()
def run_infer_batched(
    model,
    in_norm,
    omegaN: float,
    device: torch.device,
    H_all: np.ndarray,
    M_all: np.ndarray,
    exc_all: np.ndarray,
    ch_all: np.ndarray,
    batch_size: int = 20000,
    predict_output: str = "high",
):
    """
    输入：
      H_all/M_all: (N,) float32
      exc_all/ch_all: (N,) int64
    输出：
      dict: alpha/omega/rho/sin_theta/cos_theta/valid_mask 均为 numpy，形状 (N, S)
    注意：
      - omega 输出已乘 omegaN（rad/s）
      - rho 输出已 exp
    """
    N = len(H_all)
    outs_acc = {k: [] for k in ["alpha", "omega", "rho", "sin_theta", "cos_theta", "valid_mask"]}

    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)

        x = torch.tensor(np.stack([H_all[s:e], M_all[s:e]], axis=1), dtype=torch.float32)
        x_norm = in_norm.transform(x).to(device)
        Hn = x_norm[:, :1]
        Mn = x_norm[:, 1:]

        exc_t = torch.tensor(exc_all[s:e], dtype=torch.long, device=device)
        ch_t = torch.tensor(ch_all[s:e], dtype=torch.long, device=device)

        model_out = model(Hn, Mn, exc_t, ch_t)
        if isinstance(model_out, tuple):
            outL, outH = model_out
            out = outL if predict_output == "low" else outH
        else:
            if predict_output != "low":
                raise RuntimeError("Model returned a single output; use --predict_output low for this checkpoint.")
            out = model_out

        freq = out["freq"].detach().cpu().numpy()
        damping_percent = out["damping"].detach().cpu().numpy()
        residue_re = out["residue_re"].detach().cpu().numpy()
        residue_im = out["residue_im"].detach().cpu().numpy()

        freq = np.nan_to_num(freq, nan=0.0, posinf=0.0, neginf=0.0)
        zeta = np.clip(
            np.nan_to_num(damping_percent / 100.0, nan=0.0, posinf=0.999, neginf=0.0),
            1e-6,
            0.999,
        )
        omega = np.maximum(freq, 0.0) * (2.0 * np.pi)
        omega_n = omega / np.sqrt(np.maximum(1.0 - zeta * zeta, 1e-8))
        alpha = -zeta * omega_n
        rho = np.sqrt(np.maximum(residue_re * residue_re + residue_im * residue_im, 0.0))
        safe_rho = np.maximum(rho, 1e-12)
        sin_t = residue_im / safe_rho
        cos_t = residue_re / safe_rho
        valid_mask = (
            np.isfinite(alpha)
            & np.isfinite(omega)
            & np.isfinite(rho)
            & (freq > 0.0)
            & (zeta > 0.0)
            & (zeta < 1.0)
            & (rho > 1e-12)
        )

        outs_acc["alpha"].append(alpha)
        outs_acc["omega"].append(omega)
        outs_acc["rho"].append(rho)
        outs_acc["sin_theta"].append(sin_t)
        outs_acc["cos_theta"].append(cos_t)
        outs_acc["valid_mask"].append(valid_mask.astype(np.float32))

    outs = {k: np.concatenate(v, axis=0) for k, v in outs_acc.items()}
    return outs


def load_response_output_header(header_path: Path) -> List[str]:
    with open(header_path, "r", encoding="utf-8-sig") as f:
        line = f.readline().strip()
    header = [col.strip() for col in line.split(",") if col.strip()]
    if not header:
        raise ValueError(f"表头文件为空: {header_path}")
    return header


def prediction_file_name(prefix: str, entry: TableEntry, model_version: str) -> str:
    return (
        f"{prefix}_PR_{entry.exc_name}_{entry.kind}_{entry.amp_tag}_"
        f"{entry.H_tag}_{entry.M_tag}_{model_version}.dat"
    )


def _path_from_config(cp: configparser.ConfigParser, paths_ini_path: Path, section: str) -> Path:
    key = f"path_{section}".lower()
    try:
        return Path(cp[section][key])
    except KeyError:
        raise KeyError(f"配置文件 {paths_ini_path} 中缺少 [{section}] 部分或 {key} 键") from None


# ============================================================
# 5) 文件写入：RESP / HFHG / VFVG
# ============================================================

def save_response_dat(
    entry: TableEntry,
    Ts: float,
    u: np.ndarray,
    outs_resp: Dict[str, np.ndarray],
    csv_path: Path,
    out_root: Path, 
    model_version: str,
    exc_table: Dict[str, List[ExcSegment]],
    response_output_header: List[str],
    exist_thresh: float = 0.5,
    peak_win_sec: float = 1,
):
    """
    保存响应文件：一个 entry 一个文件

    输出列：
      HH, MM, SS, MS, <34个通道预测>
    其中通道表头使用全局 channels 列表中的 34 个字符串；分隔符为逗号。
    """
    resp_dir = out_root / "PR_XGD"
    resp_dir.mkdir(parents=True, exist_ok=True)
    resp_result_dir = out_root / "RESULT_PR_XGD"
    resp_result_dir.mkdir(parents=True, exist_ok=True)
    # resp_error_dir = out_root / "ERROR_PR_XGD"
    # result_cg_dir = out_root.parents[2] / "RESULT_CG"

    C_full = 34
    if len(channels) != C_full:
        raise ValueError(f"channels 长度应为 {C_full}，但当前为 {len(channels)}")

    N = u.shape[0]

    # 生成从 0 开始的时分秒毫秒（基于采样间隔 Ts）
    t_ms = np.round(np.arange(N, dtype=np.float64) * Ts * 1000.0).astype(np.int64)
    HH = (t_ms // 3600000).astype(np.int64)
    MM = ((t_ms % 3600000) // 60000).astype(np.int64)
    SS = ((t_ms % 60000) // 1000).astype(np.int64)
    MS = (t_ms % 1000).astype(np.int64)

    theta = np.arctan2(outs_resp["sin_theta"], outs_resp["cos_theta"])  # (34,S)

    # 频率范围表（csv）：用于按通道频段筛选模态
    fr_map = _load_channel_freq_ranges(csv_path)

    Y_hat = np.zeros((N, C_full), dtype=np.float64)
    for j in range(C_full):
        # 先按存在性筛选
        mask_ex = outs_resp["valid_mask"][j] > 0.5
        if not np.any(mask_ex):
            continue

        a_all = outs_resp["alpha"][j][mask_ex]
        w_all = outs_resp["omega"][j][mask_ex]
        r_all = outs_resp["rho"][j][mask_ex]
        th_all = theta[j][mask_ex]

        freqs_all = w_all / (2 * np.pi)

        # 按通道频率范围(csv)筛选：每个频段最多取 1 个（rho 最大）
        ch_name = channels[j]
        ranges = fr_map.get(ch_name, [None, None, None])

        chosen_idx, _, _ = _select_modes_by_freq_ranges(
            freqs_all=freqs_all,
            alpha_all=a_all,
            omega_all=w_all,
            rho_all=r_all,
            ranges=ranges,
        )
        if not chosen_idx:
            continue

        a_sel = a_all[chosen_idx]
        w_sel = w_all[chosen_idx]
        r_sel = r_all[chosen_idx]
        th_sel = th_all[chosen_idx]

        B, A = build_BA_from_residues(
            a_sel,
            w_sel,
            r_sel,
            th_sel,
            Ts,
        )
        Y_hat[:, j] = lfilter(B, A, u)

    fname = f"PR_{entry.exc_name}_{entry.kind}_{entry.amp_tag}_{entry.H_tag}_{entry.M_tag}_{model_version}.dat"
    fpath = resp_dir / fname

    header_cols = response_output_header
    header = ",".join(header_cols)

    data = np.empty((N, len(header_cols)), dtype=object)
    data[:, :] = 0.0

    col_index = {name: idx for idx, name in enumerate(header_cols)}
    time_idx = col_index.get("TIME")
    if time_idx is not None:
        data[:, time_idx] = [
            f"{hh:02d}:{mm:02d}:{ss:02d}:{ms:03d}"
            for hh, mm, ss, ms in zip(HH, MM, SS, MS)
        ]
    else:
        for name, values in (("HH", HH), ("MM", MM), ("SS", SS), ("MS", MS)):
            idx = col_index.get(name)
            if idx is not None:
                data[:, idx] = values

    for ch_idx, ch_name in enumerate(channels):
        target_idx = col_index.get(ch_name)
        if target_idx is not None:
            data[:, target_idx] = Y_hat[:, ch_idx]

    fmt = ["%s" if col == "TIME" else "%.10e" for col in header_cols]
    np.savetxt(fpath, data, fmt=fmt, delimiter=",", header=header, comments="")

    # 同步输出：用于生成该响应的频率/阻尼/峰峰值参数（按激励方式选择通道）
    # 每行：ch,freq1,damp1,pp1,freq2,damp2,pp2...
    chs = channels_for_exc_plot(entry.exc_id)
    if chs.size == 0:
        chs = np.arange(C_full, dtype=int)

    fd_fname = f"RESULT_PR_{entry.exc_name}_{entry.kind}_{entry.amp_tag}_{entry.H_tag}_{entry.M_tag}_{model_version}.dat"
    fd_path = resp_result_dir / fd_fname
    fd_path.parent.mkdir(parents=True, exist_ok=True)

    fd_lines = ["ch,freq1,damp1,pp1,freq2,damp2,pp2,freq3,damp3,pp3\n"]
    for ch in chs:
        ch_idx = int(ch)
        ch_name = channels[ch_idx]

        mask = outs_resp["valid_mask"][ch_idx] > 0.5
        if not np.any(mask):
            fd_lines.append(f"{ch_name}," + ",".join([""] * 9) + "\n")
            continue

        w_all = outs_resp["omega"][ch_idx, mask]
        a_all = outs_resp["alpha"][ch_idx, mask]
        r_all = outs_resp["rho"][ch_idx, mask]

        freqs_all = w_all / (2 * np.pi)

        # 按通道频率范围(csv)筛选：每个频段最多取 1 个（rho 最大），并按 csv 列索引占位输出
        ranges = fr_map.get(ch_name, [None, None, None])

        _, freq_slots, damp_slots = _select_modes_by_freq_ranges(
            freqs_all=freqs_all,
            alpha_all=a_all,
            omega_all=w_all,
            rho_all=r_all,
            ranges=ranges,
        )

        # 交替写入 freq,damp,pp（3个槽位 -> 9列；空槽位用逗号占位）
        segs = exc_table.get(entry.kind, [])
        seg_list = build_exc_seg_list(segs)

        tail = []
        for i in range(3):
            f_str = freq_slots[i] if i < len(freq_slots) else ""
            d_str = damp_slots[i] if i < len(damp_slots) else ""

            pp_str = ""
            if f_str != "":
                try:
                    f_hz = float(f_str)
                    pp_str = calc_pp_for_freq(
                        y=Y_hat[:, ch_idx],
                        f_hz=f_hz,
                        Ts=Ts,
                        seg_list=seg_list,
                        peak_win_sec=peak_win_sec,
                    )
                except Exception:
                    pp_str = ""

            tail.append(f_str)
            tail.append(d_str)
            tail.append(pp_str)

        fd_lines.append(f"{ch_name}," + ",".join(tail) + "\n")

    with open(fd_path, "w", encoding="utf-8") as f:
        f.writelines(fd_lines)

    # error_fname = f"ERROR_PR_{entry.exc_name}_{entry.kind}_{entry.amp_tag}_{entry.H_tag}_{entry.M_tag}_{model_version}.dat"
    # error_path = resp_error_dir / error_fname
    # error_path.parent.mkdir(parents=True, exist_ok=True)
    # calc_relative_error(fd_path, result_cg_dir, error_path)




def _build_yhat_and_slot_strings(
    alpha_row: np.ndarray,
    omega_row: np.ndarray,
    rho_row: np.ndarray,
    sin_row: np.ndarray,
    cos_row: np.ndarray,
    valid_mask_row: np.ndarray,
    ch_name: str,
    csv_path: Path,
    u: np.ndarray,
    Ts: float,
    exist_thresh: float = 0.5,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """对单个扫描点/单个通道：
    1) 按存在性与频段范围筛选槽位；
    2) 重构该通道时域响应 y_hat；
    3) 返回 y_hat、freq_slots、damp_slots。
    """
    y_hat = np.zeros((u.shape[0],), dtype=np.float64)
    freq_slots = ["", "", ""]
    damp_slots = ["", "", ""]

    mask = valid_mask_row > 0.5
    if not np.any(mask):
        return y_hat, freq_slots, damp_slots

    a_all = alpha_row[mask]
    w_all = omega_row[mask]
    r_all = rho_row[mask]
    th_all = np.arctan2(sin_row[mask], cos_row[mask])
    freqs_all = w_all / (2 * np.pi)

    fr_map = _load_channel_freq_ranges(csv_path)
    ranges = fr_map.get(ch_name, [None, None, None])

    chosen_idx, freq_slots, damp_slots = _select_modes_by_freq_ranges(
        freqs_all=freqs_all,
        alpha_all=a_all,
        omega_all=w_all,
        rho_all=r_all,
        ranges=ranges,
    )
    if not chosen_idx:
        return y_hat, freq_slots, damp_slots

    a_sel = a_all[chosen_idx]
    w_sel = w_all[chosen_idx]
    r_sel = r_all[chosen_idx]
    th_sel = th_all[chosen_idx]

    B, A = build_BA_from_residues(
        a_sel,
        w_sel,
        r_sel,
        th_sel,
        Ts,
    )
    y_hat = lfilter(B, A, u)
    return y_hat, freq_slots, damp_slots

def save_HF_HG_for_entry(
    entry: TableEntry,
    H_axis: np.ndarray,
    outs_hscan: Dict[str, np.ndarray],
    csv_path: Path,
    hf_dir: Path,
    hg_dir: Path,
    ha_dir: Path,
    model_version: str,
    Ts: float,
    exc_table: Dict[str, List[ExcSegment]],
    u: np.ndarray,
    exist_thresh: float = 0.5,
    peak_win_sec: float = 1,
):
    """
    文件每行：H_km, M, ch, freqs... 或 damping...
    """
    hf_dir.mkdir(parents=True, exist_ok=True)
    hg_dir.mkdir(parents=True, exist_ok=True)
    ha_dir.mkdir(parents=True, exist_ok=True)

    chs = channels_for_exc_plot(entry.exc_id)
    if chs.size == 0:
        return

    alpha = outs_hscan["alpha"]   # (n_case,S)
    omega = outs_hscan["omega"]
    rho = outs_hscan["rho"]
    valid_mask = outs_hscan["valid_mask"]

    hf_path = hf_dir / prediction_file_name("HF", entry, model_version)
    hg_path = hg_dir / prediction_file_name("HG", entry, model_version)
    ha_path = ha_dir / prediction_file_name("HA", entry, model_version)

    hf_lines = ["H,M,响应,wn_1,wn_2,wn_3\n"]
    hg_lines = ["H,M,响应,zeta_1,zeta_2,zeta_3\n"]
    ha_lines = ["H,M,响应,t_val_1,t_val_2,t_val_3\n"]

    nC = len(chs)

    segs = exc_table.get(entry.kind, [])
    seg_list = build_exc_seg_list(segs)

    for iH, H in enumerate(H_axis):
        for iC, ch in enumerate(chs):
            row = iH * nC + iC
            ch_idx = int(ch)
            ch_name = channels[ch_idx] if 0 <= ch_idx < len(channels) else f"C{ch_idx:02d}"

            y_hat, freq_slots, damp_slots = _build_yhat_and_slot_strings(
                alpha_row=alpha[row],
                omega_row=omega[row],
                rho_row=rho[row],
                sin_row=outs_hscan["sin_theta"][row],
                cos_row=outs_hscan["cos_theta"][row],
                valid_mask_row=valid_mask[row],
                ch_name=ch_name,
                csv_path=csv_path,
                u=u,
                Ts=Ts,
                exist_thresh=exist_thresh,
            )

            pp_slots = ["", "", ""]
            for i in range(3):
                f_str = freq_slots[i]
                if f_str == "":
                    continue
                try:
                    f_hz = float(f_str)
                    pp_slots[i] = calc_pp_for_freq(
                        y=y_hat,
                        f_hz=f_hz,
                        Ts=Ts,
                        seg_list=seg_list,
                        peak_win_sec=peak_win_sec,
                    )
                except Exception:
                    pp_slots[i] = ""

            hf_lines.append(f"{H:.2f},{entry.M:.2f},{ch_name}," + ",".join(freq_slots) + "\n")
            hg_lines.append(f"{H:.2f},{entry.M:.2f},{ch_name}," + ",".join(damp_slots) + "\n")
            ha_lines.append(f"{H:.2f},{entry.M:.2f},{ch_name}," + ",".join(pp_slots) + "\n")


    with open(hf_path, "w", encoding="utf-8") as f:
        f.writelines(hf_lines)
    with open(hg_path, "w", encoding="utf-8") as f:
        f.writelines(hg_lines)
    with open(ha_path, "w", encoding="utf-8") as f:
        f.writelines(ha_lines)


def save_VF_VG_for_entry(
    entry: TableEntry,
    M_axis: np.ndarray,
    outs_mscan: Dict[str, np.ndarray],
    csv_path: Path,
    vf_dir: Path,
    vg_dir: Path,
    va_dir: Path,
    model_version: str,
    Ts: float,
    exc_table: Dict[str, List[ExcSegment]],
    u: np.ndarray,
    exist_thresh: float = 0.5,
    peak_win_sec: float = 1,
):
    """
    固定 H=entry.H_km 扫 M_axis；通道 = channels_for_exc_plot(entry.exc_id)
    文件每行：H_km, M, ch, freqs... 或 damping...
    其中 ch 使用 channels 列表中对应下标的字符串；分隔符为逗号；表头不带 '#'
    """
    vf_dir.mkdir(parents=True, exist_ok=True)
    vg_dir.mkdir(parents=True, exist_ok=True)
    va_dir.mkdir(parents=True, exist_ok=True)

    chs = channels_for_exc_plot(entry.exc_id)
    if chs.size == 0:
        return

    alpha = outs_mscan["alpha"]
    omega = outs_mscan["omega"]
    rho = outs_mscan["rho"]
    valid_mask = outs_mscan["valid_mask"]

    vf_path = vf_dir / prediction_file_name("VF", entry, model_version)
    vg_path = vg_dir / prediction_file_name("VG", entry, model_version)
    va_path = va_dir / prediction_file_name("VA", entry, model_version)

    vf_lines = ["H,M,响应,wn_1,wn_2,wn_3\n"]
    vg_lines = ["H,M,响应,zeta_1,zeta_2,zeta_3\n"]
    va_lines = ["H,M,响应,t_val_1,t_val_2,t_val_3\n"]

    nC = len(chs)

    segs = exc_table.get(entry.kind, [])
    seg_list = build_exc_seg_list(segs)

    for iM, M in enumerate(M_axis):
        for iC, ch in enumerate(chs):
            row = iM * nC + iC
            ch_idx = int(ch)
            ch_name = channels[ch_idx] if 0 <= ch_idx < len(channels) else f"C{ch_idx:02d}"

            y_hat, freq_slots, damp_slots = _build_yhat_and_slot_strings(
                alpha_row=alpha[row],
                omega_row=omega[row],
                rho_row=rho[row],
                sin_row=outs_mscan["sin_theta"][row],
                cos_row=outs_mscan["cos_theta"][row],
                valid_mask_row=valid_mask[row],
                ch_name=ch_name,
                csv_path=csv_path,
                u=u,
                Ts=Ts,
                exist_thresh=exist_thresh,
            )

            pp_slots = ["", "", ""]
            for i in range(3):
                f_str = freq_slots[i]
                if f_str == "":
                    continue
                try:
                    f_hz = float(f_str)
                    pp_slots[i] = calc_pp_for_freq(
                        y=y_hat,
                        f_hz=f_hz,
                        Ts=Ts,
                        seg_list=seg_list,
                        peak_win_sec=peak_win_sec,
                    )
                except Exception:
                    pp_slots[i] = ""

            vf_lines.append(f"{entry.H_km:.2f},{M:.2f},{ch_name}," + ",".join(freq_slots) + "\n")
            vg_lines.append(f"{entry.H_km:.2f},{M:.2f},{ch_name}," + ",".join(damp_slots) + "\n")
            va_lines.append(f"{entry.H_km:.2f},{M:.2f},{ch_name}," + ",".join(pp_slots) + "\n")


    with open(vf_path, "w", encoding="utf-8") as f:
        f.writelines(vf_lines)
    with open(vg_path, "w", encoding="utf-8") as f:
        f.writelines(vg_lines)
    with open(va_path, "w", encoding="utf-8") as f:
        f.writelines(va_lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        "--ckpt_path",
        dest="model_path",
        type=str,
        default=None,
        help="Optional checkpoint path. Defaults to MX_XGD/model_version/model_version.pth from user_input.dat.",
    )
    parser.add_argument(
        "--predict_output",
        choices=["high", "low"],
        default="high",
        help="Use high-fidelity or low-fidelity model output for prediction. Default: high.",
    )
    args, _unknown = parser.parse_known_args()

    fs = 512
    Ts = 1.0 / fs
    omegaN = np.pi * fs
    exist_thresh = 0.2

    base_dir = get_exe_dir()
    user_input_path = base_dir / "user_input.dat"
    csv_path = base_dir / "模态频率范围.csv"
    with open(user_input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) < 4:
        raise ValueError("user_input.dat 至少需要 4 行：路径配置ini、模型版本、任务单文件名、输出文件夹名")

    # 拿到路径ini的路径
    paths_ini_path = Path(lines[0].strip()).expanduser()
    cp = configparser.ConfigParser()
    cp.read(paths_ini_path, encoding="utf-8")
    # 拿到模型版本
    model_version = lines[1].strip()
    # 拿到任务单名字
    task_ini = lines[2].strip()
    output_folder = lines[3].strip()

    # ===== 模型文件路径 =====
    model_dir = _path_from_config(cp, paths_ini_path, "MX_XGD")

    ckpt_path = Path(args.model_path).expanduser() if args.model_path else model_dir / model_version / (model_version + ".pth")
    in_norm_path =  model_dir / "Normalizer_Input.pth"
    out_norm_path = model_dir / "Normalizer_Output.pth"
    data_header_dir = _path_from_config(cp, paths_ini_path, "DataHeader")
    response_output_header = load_response_output_header(data_header_dir / "sc.dat")

    signaltype_dir = _path_from_config(cp, paths_ini_path, "SignalType")
    exc_table_xlsx = signaltype_dir / "signal_type.xlsx"

    # ==== 任务单路径 =====
    task_dir = _path_from_config(cp, paths_ini_path, "Task")
    task_ini_path = task_dir / task_ini

    pr_dir = _path_from_config(cp, paths_ini_path, "PR_XGD")

    # 输出：预测响应
    out_root = pr_dir / output_folder

    # 输出：VF/VG/HF/HG 分别走路径 ini 里对应目录（若缺失则回退到 out_root/hfg 或 out_root/vfg）
    vf_dir = out_root / "VF_PR_XGD"
    vg_dir = out_root / "VG_PR_XGD"
    hf_dir = out_root / "HF_PR_XGD"
    hg_dir = out_root / "HG_PR_XGD"
    ha_dir = out_root / "HA_PR_XGD"
    va_dir = out_root / "VA_PR_XGD"

    # ===== mesh 轴范围（用于 H-scan / M-scan）=====
    H_min, H_max, dH = 1.00, 12.00, 0.1
    M_min, M_max, dM = 0.30, 2.00, 0.01
    H_axis = np.arange(H_min, H_max + 1e-12, dH, dtype=np.float32)
    M_axis = np.arange(M_min, M_max + 1e-12, dM, dtype=np.float32)

    # ===== 你必须补全这个映射：exc_name -> exc_id(0..5) =====
    # 例子里你说：FXD_FDC 对应激励方式1（这里我先写成 0，你自己改成正确的）
    EXC_NAME_TO_ID = {
        "FXD_FDC": 0,
        "FXD_DC":  1,
        "PW_FDC":  2,
        "PW_DC":   3,
        "JFY_FDC": 4,
        "JFY_DC":  5,
    }

    # ===== 加载模型 =====
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_norm = Normalizer.load(out_norm_path)
    cfg.K_max = 3
    cfg.residue_target = "fixed"

    print(f"==> Loading ckpt: {ckpt_path}")
    print(f"==> Prediction output: {args.predict_output}")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)

    model = MFDNN(cfg).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    model.eval()

    in_norm = Normalizer.load(in_norm_path)
    if len(in_norm.mean) != 2 or len(in_norm.std) != 2:
        raise RuntimeError(f"Normalizer_Input.pth 应为二维 [H, M]，当前 mean={in_norm.mean}, std={in_norm.std}")

    # ===== 读取 ini → entries =====
    sortie_num, entries = load_task_from_ini(task_ini_path, EXC_NAME_TO_ID)
    print(f"Sortie Num = {sortie_num}, entries = {len(entries)}")

    # ===== 激励表 =====
    exc_table = load_excitation_table(exc_table_xlsx)
    exc_kinds = sorted(exc_table.keys())
    print(f"Signal type file = {exc_table_xlsx}")
    print(f"Signal type kinds = {exc_kinds}")
    missing_kinds = sorted({entry.kind for entry in entries if entry.kind not in exc_table})
    if missing_kinds:
        raise KeyError(
            f"任务单中的激励种类不在激励表中: {missing_kinds}; "
            f"激励表文件: {exc_table_xlsx}; 已读取种类: {exc_kinds}"
        )

    date = datetime.now().strftime("%Y%m%d")

    # ========================================================
    # 核心循环：每条 Data 做三类输出
    # ========================================================
    for entry in entries:
        print(f"[{entry.key}] {entry.raw}  -> exc_id={entry.exc_id}, kind={entry.kind}, amp={entry.amp}")

        # ---------- A) RESP：固定 (H,M,exc) 预测 34 通道响应 ----------
        # 1) 网络推理（34通道）
        C_full = 34
        H_all = np.full((C_full,), entry.H_km, dtype=np.float32)
        M_all = np.full((C_full,), entry.M, dtype=np.float32)
        exc_all = np.full((C_full,), entry.exc_id, dtype=np.int64)
        ch_all = np.arange(C_full, dtype=np.int64)

        outs_resp = run_infer_batched(
            model=model, in_norm=in_norm, omegaN=omegaN, device=device,
            H_all=H_all, M_all=M_all, exc_all=exc_all, ch_all=ch_all,
            batch_size=4096,
            predict_output=args.predict_output,
        )

        # 2) 生成激励 u(t)
        u = build_u_from_kind(entry.kind, entry.amp, fs, exc_table)
        if u.size == 0:
            print(f"  [WARN] u(t) 长度为 0，跳过 RESP 保存：{entry.raw}")
        else:
            save_response_dat(
                entry, Ts, u, outs_resp, csv_path, out_root, model_version,
                exc_table, response_output_header, exist_thresh
            )

        # ---------- B) HF/HG：固定 M 扫 H_axis（裁剪通道） ----------
        chs_plot = channels_for_exc_plot(entry.exc_id)
        if chs_plot.size > 0:
            nH = len(H_axis)
            nC = len(chs_plot)
            Hs = np.repeat(H_axis, nC).astype(np.float32)
            Ms = np.full((nH * nC,), entry.M, dtype=np.float32)
            excs = np.full((nH * nC,), entry.exc_id, dtype=np.int64)
            chs = np.tile(chs_plot, nH).astype(np.int64)

            outs_hscan = run_infer_batched(
                model=model, in_norm=in_norm, omegaN=omegaN, device=device,
                H_all=Hs, M_all=Ms, exc_all=excs, ch_all=chs,
                batch_size=20000,
                predict_output=args.predict_output,
            )
            save_HF_HG_for_entry(
                entry, H_axis, outs_hscan, csv_path,
                hf_dir, hg_dir, ha_dir, model_version,
                Ts, exc_table, u, exist_thresh
            )
        else:
            print(f"  [WARN] exc_id={entry.exc_id} 无 plot 通道，跳过 HF/HG")

        # ---------- C) VF/VG：固定 H 扫 M_axis（裁剪通道） ----------
        if chs_plot.size > 0:
            nM = len(M_axis)
            nC = len(chs_plot)
            Ms = np.repeat(M_axis, nC).astype(np.float32)
            Hs = np.full((nM * nC,), entry.H_km, dtype=np.float32)
            excs = np.full((nM * nC,), entry.exc_id, dtype=np.int64)
            chs = np.tile(chs_plot, nM).astype(np.int64)

            outs_mscan = run_infer_batched(
                model=model, in_norm=in_norm, omegaN=omegaN, device=device,
                H_all=Hs, M_all=Ms, exc_all=excs, ch_all=chs,
                batch_size=20000,
                predict_output=args.predict_output,
            )
            save_VF_VG_for_entry(
                entry, M_axis, outs_mscan, csv_path,
                vf_dir, vg_dir, va_dir, model_version,
                Ts, exc_table, u, exist_thresh
            )
        else:
            print(f"  [WARN] exc_id={entry.exc_id} 无 plot 通道，跳过 VF/VG")

    print("Done.")


    import socket
    def udp_sender():
        # 配置参数
        HOST = 'localhost'
        PORT = 12345
        MESSAGE = "XGD_PREDICTION_EXE_FINISHED"

        # 创建UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        try:
            print("UDP发送端已启动")
            print(f"目标地址: {HOST}:{PORT}")
            print(f"发送内容: {MESSAGE}")
            print("\n正在发送...")

            # 发送消息
            sock.sendto(MESSAGE.encode('utf-8'), (HOST, PORT))

            print("消息发送成功！")

        except Exception as e:
            print(f"发送失败: {e}")
        finally:
            sock.close()
            print("发送端已关闭")

    udp_sender()
if __name__ == "__main__":
    main()
