# %%
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
from residue_pred import build_BA_from_residues


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
    按你给的规则：
      exc 0/1 -> 前 8 个通道: 0..7
      exc 2/3 -> 通道 8..19 (12个)
      exc 4/5 -> 通道 20..27 (8个)
    """
    if exc_id in (0, 1):
        return np.arange(8, 20, dtype=np.int64)
    if exc_id in (2, 3):
        return np.arange(0, 8, dtype=np.int64)
    if exc_id in (4, 5):
        return np.arange(20, 28, dtype=np.int64)
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
    """
    Excel 列名要求：
      第1列：激励种类（A/B/AB...）——列名可以是 Unnamed: 0（默认）
      幅值 / 开始频率 / 结束频率 / 时长
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    kind_col = df.columns[0]

    amp_col = "幅值"
    f0_col = "开始频率"
    f1_col = "结束频率"
    dur_col = "时长"

    table: Dict[str, List[ExcSegment]] = {}
    for _, row in df.iterrows():
        kind = str(row[kind_col]).strip()
        seg = ExcSegment(
            amp_base=float(row[amp_col]),
            f0=float(row[f0_col]),
            f1=float(row[f1_col]),
            dur=float(row[dur_col]),
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
):
    """
    输入：
      H_all/M_all: (N,) float32
      exc_all/ch_all: (N,) int64
    输出：
      dict: alpha/omega/rho/sin_theta/cos_theta/p_exist 均为 numpy，形状 (N, S)
    注意：
      - omega 输出已乘 omegaN（rad/s）
      - rho 输出已 exp
    """
    N = len(H_all)
    outs_acc = {k: [] for k in ["alpha", "omega", "rho", "sin_theta", "cos_theta", "p_exist"]}

    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)

        x = torch.tensor(np.stack([H_all[s:e], M_all[s:e]], axis=1), dtype=torch.float32)
        x_norm = in_norm.transform(x).to(device)
        Hn = x_norm[:, :1]
        Mn = x_norm[:, 1:]

        exc_t = torch.tensor(exc_all[s:e], dtype=torch.long, device=device)
        ch_t = torch.tensor(ch_all[s:e], dtype=torch.long, device=device)

        outL, outH = model(Hn, Mn, exc_t, ch_t)
        out = outH  # 你一直用 outH

        alpha = out["alpha"].detach().cpu().numpy()
        omega = out["omega"].detach().cpu().numpy() * omegaN
        rho = np.exp(out["rho"].detach().cpu().numpy())
        sin_t = out["sin_theta"].detach().cpu().numpy()
        cos_t = out["cos_theta"].detach().cpu().numpy()
        p_exist = out["p_exist"].detach().cpu().numpy()

        outs_acc["alpha"].append(alpha)
        outs_acc["omega"].append(omega)
        outs_acc["rho"].append(rho)
        outs_acc["sin_theta"].append(sin_t)
        outs_acc["cos_theta"].append(cos_t)
        outs_acc["p_exist"].append(p_exist)

    outs = {k: np.concatenate(v, axis=0) for k, v in outs_acc.items()}
    return outs


# ============================================================
# 5) 文件写入：RESP / HFHG / VFVG
# ============================================================

def save_response_dat(
    entry: TableEntry,
    Ts: float,
    u: np.ndarray,
    outs_resp: Dict[str, np.ndarray],
    csv_path: Path,
    resp_dir: Path,
    resp_result_dir: Path,
    model_version: str,
    exc_table: Dict[str, List[ExcSegment]],
    exist_thresh: float = 0.5,
    peak_win_sec: float = 0.5,
):
    """
    保存响应文件：一个 entry 一个文件

    输出列：
      HH, MM, SS, MS, <34个通道预测>
    其中通道表头使用全局 channels 列表中的 34 个字符串；分隔符为逗号。
    """
    resp_dir.mkdir(parents=True, exist_ok=True)

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
        mask_ex = outs_resp["p_exist"][j] > exist_thresh
        if not np.any(mask_ex):
            continue

        a_all = outs_resp["alpha"][j][mask_ex]
        w_all = outs_resp["omega"][j][mask_ex]
        r_all = outs_resp["rho"][j][mask_ex]
        th_all = theta[j][mask_ex]
        p_all = outs_resp["p_exist"][j][mask_ex]

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
        p_sel = p_all[chosen_idx]

        B, A = build_BA_from_residues(
            a_sel,
            w_sel,
            r_sel,
            th_sel,
            Ts,
            p_exist=p_sel,
            exist_thresh=exist_thresh,
        )
        Y_hat[:, j] = lfilter(B, A, u)

    fname = f"PR_{entry.exc_name}_{entry.kind}_{entry.amp_tag}_{entry.H_tag}_{entry.M_tag}_{model_version}.dat"
    fpath = resp_dir / fname

    header_cols = ["HH", "MM", "SS", "MS"] + list(channels)
    header = ",".join(header_cols)

    data = np.column_stack([HH, MM, SS, MS, Y_hat])

    fmt = ["%d", "%d", "%d", "%d"] + ["%.10e"] * C_full
    np.savetxt(fpath, data, fmt=fmt, delimiter=",", header=header, comments="")

    # 同步输出：用于生成该响应的频率/阻尼参数（按激励方式选择通道）
    # 每行：ch,freq1,damp1,freq2,damp2,...
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

        mask = outs_resp["p_exist"][ch_idx] > exist_thresh
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
        fs = 1.0 / Ts
        segs = exc_table.get(entry.kind, [])
        seg_list = []
        t_off = 0.0
        for seg in segs:
            seg_list.append((seg.f0, seg.f1, seg.dur, t_off))
            t_off += seg.dur

        tail = []
        for i in range(3):
            f_str = freq_slots[i] if i < len(freq_slots) else ""
            d_str = damp_slots[i] if i < len(damp_slots) else ""

            # 默认空 pp
            pp_str = ""

            if f_str != "":
                try:
                    f_hz = float(f_str)
                except Exception:
                    f_hz = None

                if f_hz is not None and len(seg_list) > 0:
                    idx0 = None
                    for (f0, f1, dur, t0) in seg_list:
                        f_lo = min(f0, f1)
                        f_hi = max(f0, f1)
                        if f_lo <= f_hz <= f_hi and abs(f1 - f0) > 1e-12:
                            t_local = dur * (f_hz - f0) / (f1 - f0)
                            t_g = t0 + t_local
                            idx0 = int(round(t_g * fs))
                            break

                    if idx0 is not None:
                        half = int(round(peak_win_sec * fs))
                        i0 = max(0, idx0 - half)
                        i1 = min(N, idx0 + half + 1)
                        if i1 > i0:
                            pp = float(np.max(np.abs(Y_hat[i0:i1, ch_idx])))
                            pp_str = f"{pp:.6f}"

            tail.append(f_str)
            tail.append(d_str)
            tail.append(pp_str)

        fd_lines.append(f"{ch_name}," + ",".join(tail) + "\n")

    with open(fd_path, "w", encoding="utf-8") as f:
        f.writelines(fd_lines)


def save_HF_HG_for_entry(
    entry: TableEntry,
    H_axis: np.ndarray,
    outs_hscan: Dict[str, np.ndarray],
    csv_path: Path,
    hf_dir: Path,
    hg_dir: Path,
    model_version: str,
):
    """
    文件每行：H_km, M, ch, freqs... 或 damping...
    """
    hf_dir.mkdir(parents=True, exist_ok=True)
    hg_dir.mkdir(parents=True, exist_ok=True)

    chs = channels_for_exc_plot(entry.exc_id)
    if chs.size == 0:
        return

    alpha = outs_hscan["alpha"]   # (n_case,S)
    omega = outs_hscan["omega"]
    rho = outs_hscan["rho"]
    p_ex = outs_hscan["p_exist"]

    M_tag = format_M_tag(entry.M)
    exc_name = entry.exc_name

    hf_path = hf_dir / f"HF_PR_{M_tag}_{exc_name}_{model_version}.dat"
    hg_path = hg_dir / f"HG_PR_{M_tag}_{exc_name}_{model_version}.dat"

    hf_lines = ["H_km,M,ch,freqs...\n"]
    hg_lines = ["H_km,M,ch,damping...\n"]

    nC = len(chs)

    for iH, H in enumerate(H_axis):
        for iC, ch in enumerate(chs):
            row = iH * nC + iC
            ch_idx = int(ch)
            ch_name = channels[ch_idx] if 0 <= ch_idx < len(channels) else f"C{ch_idx:02d}"

            mask = p_ex[row] > 0.5

            if not np.any(mask):
                hf_lines.append(f"{H:.2f},{entry.M:.2f},{ch_name},,,\n")
                hg_lines.append(f"{H:.2f},{entry.M:.2f},{ch_name},,,\n")
                continue

            w_all = omega[row, mask]
            a_all = alpha[row, mask]
            r_all = rho[row, mask]

            freqs_all = w_all / (2 * np.pi)

            # 按通道频率范围(csv)筛选：每个频段最多取 1 个（rho 最大）
            fr_map = _load_channel_freq_ranges(csv_path)
            ranges = fr_map.get(ch_name, [None, None, None])

            _, freq_slots, damp_slots = _select_modes_by_freq_ranges(
                freqs_all=freqs_all,
                alpha_all=a_all,
                omega_all=w_all,
                rho_all=r_all,
                ranges=ranges,
            )

            # 固定输出 3 列（按 csv 列索引占位）
            hf_lines.append(f"{H:.2f},{entry.M:.2f},{ch_name}," + ",".join(freq_slots) + "\n")
            hg_lines.append(f"{H:.2f},{entry.M:.2f},{ch_name}," + ",".join(damp_slots) + "\n")


    with open(hf_path, "w", encoding="utf-8") as f:
        f.writelines(hf_lines)
    with open(hg_path, "w", encoding="utf-8") as f:
        f.writelines(hg_lines)


def save_VF_VG_for_entry(
    entry: TableEntry,
    M_axis: np.ndarray,
    outs_mscan: Dict[str, np.ndarray],
    csv_path: Path,
    vf_dir: Path,
    vg_dir: Path,
    model_version: str,
):
    """
    固定 H=entry.H_km 扫 M_axis；通道 = channels_for_exc_plot(entry.exc_id)
    文件每行：H_km, M, ch, freqs... 或 damping...
    其中 ch 使用 channels 列表中对应下标的字符串；分隔符为逗号；表头不带 '#'
    """
    vf_dir.mkdir(parents=True, exist_ok=True)
    vg_dir.mkdir(parents=True, exist_ok=True)

    chs = channels_for_exc_plot(entry.exc_id)
    if chs.size == 0:
        return

    alpha = outs_mscan["alpha"]
    omega = outs_mscan["omega"]
    rho = outs_mscan["rho"]
    p_ex = outs_mscan["p_exist"]

    H_tag = format_H_tag(entry.H_km)
    exc_name = entry.exc_name

    vf_path = vf_dir / f"VF_PR_{H_tag}_{exc_name}_{model_version}.dat"
    vg_path = vg_dir / f"VG_PR_{H_tag}_{exc_name}_{model_version}.dat"

    vf_lines = ["H_km,M,ch,freqs...\n"]
    vg_lines = ["H_km,M,ch,damping...\n"]

    nC = len(chs)

    for iM, M in enumerate(M_axis):
        for iC, ch in enumerate(chs):
            row = iM * nC + iC
            ch_idx = int(ch)
            ch_name = channels[ch_idx] if 0 <= ch_idx < len(channels) else f"C{ch_idx:02d}"

            mask = p_ex[row] > 0.5
            if not np.any(mask):
                vf_lines.append(f"{entry.H_km:.2f},{M:.2f},{ch_name},,,\n")
                vg_lines.append(f"{entry.H_km:.2f},{M:.2f},{ch_name},,,\n")
                continue

            w_all = omega[row, mask]
            a_all = alpha[row, mask]
            r_all = rho[row, mask]

            freqs_all = w_all / (2 * np.pi)

            # 按通道频率范围(csv)筛选：每个频段最多取 1 个（rho 最大）
            fr_map = _load_channel_freq_ranges(csv_path)
            ranges = fr_map.get(ch_name, [None, None, None])

            _, freq_slots, damp_slots = _select_modes_by_freq_ranges(
                freqs_all=freqs_all,
                alpha_all=a_all,
                omega_all=w_all,
                rho_all=r_all,
                ranges=ranges,
            )

            # 固定输出 3 列（按 csv 列索引占位）
            vf_lines.append(f"{entry.H_km:.2f},{M:.2f},{ch_name}," + ",".join(freq_slots) + "\n")
            vg_lines.append(f"{entry.H_km:.2f},{M:.2f},{ch_name}," + ",".join(damp_slots) + "\n")


    with open(vf_path, "w", encoding="utf-8") as f:
        f.writelines(vf_lines)
    with open(vg_path, "w", encoding="utf-8") as f:
        f.writelines(vg_lines)

def main():
    fs = 512
    Ts = 1.0 / fs
    omegaN = np.pi * fs
    exist_thresh = 0.5

    base_dir = get_exe_dir()
    user_input_path = base_dir / "user_input.dat"
    csv_path = base_dir / "模态频率范围.csv"
    with open(user_input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 拿到路径ini的路径
    paths_ini_path = Path(lines[0].strip()).expanduser()
    # 拿到模型版本
    model_version = lines[1].strip()
    # 拿到任务单名字
    task_ini = lines[2].strip()

    # 拿到路径ini
    paths_map = load_paths_ini(paths_ini_path)

    # ===== 模型文件路径 =====
    model_dir = Path(paths_map.get("MX_XGD", "")).expanduser()

    ckpt_path = model_dir / model_version / (model_version + ".pth")
    in_norm_path =  model_dir / "Normalizer_Input.pth"
    exc_table_xlsx = model_dir / "激励类型.xlsx"

    # ==== 任务单路径 =====
    config_dir = Path(paths_map.get("CONFIG", "")).expanduser()
    task_ini_path = config_dir / task_ini

    # 激励类型表：默认放在 PR 目录下（PR/XGD 的上一级），文件名固定“激励类型.xlsx”
    pr_dir = Path(paths_map.get("PR_XGD", "")).expanduser()

    # 输出：预测响应
    out_root = pr_dir / model_version
    resp_dir = out_root / "PR_XGD"
    resp_result_dir = out_root / "RESULT_PR_XGD"

    # 输出：VF/VG/HF/HG 分别走路径 ini 里对应目录（若缺失则回退到 out_root/hfg 或 out_root/vfg）
    vf_dir = out_root / "VF_PR_XGD"
    vg_dir = out_root / "VG_PR_XGD"
    hf_dir = out_root / "HF_PR_XGD"
    hg_dir = out_root / "HG_PR_XGD"

    # ===== mesh 轴范围（用于 H-scan / M-scan）=====
    H_min, H_max, dH = 1.00, 12.00, 0.1
    M_min, M_max, dM = 0.30, 2.00, 0.01
    H_axis = np.arange(H_min, H_max + 1e-12, dH, dtype=np.float32)
    M_axis = np.arange(M_min, M_max + 1e-12, dM, dtype=np.float32)

    # ===== 你必须补全这个映射：exc_name -> exc_id(0..5) =====
    # 例子里你说：FXD_FDC 对应激励方式1（这里我先写成 0，你自己改成正确的）
    EXC_NAME_TO_ID = {
        "FXD_DC": 0,
        "FXD_FDC": 1,
        "JFY_DC":  2,
        "JFY_FDC": 3,
        "PW_DC":  4,
        "PW_FDC": 5,
    }

    # ===== 加载模型 =====
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("==> Loading ckpt...")
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

    # ===== 读取 ini → entries =====
    sortie_num, entries = load_task_from_ini(task_ini_path, EXC_NAME_TO_ID)
    print(f"Sortie Num = {sortie_num}, entries = {len(entries)}")

    # ===== 激励表 =====
    exc_table = load_excitation_table(exc_table_xlsx)

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
            batch_size=4096
        )

        # 2) 生成激励 u(t)
        u = build_u_from_kind(entry.kind, entry.amp, fs, exc_table)
        if u.size == 0:
            print(f"  [WARN] u(t) 长度为 0，跳过 RESP 保存：{entry.raw}")
        else:
            save_response_dat(entry, Ts, u, outs_resp, csv_path, resp_dir, resp_result_dir, model_version, exc_table, exist_thresh)

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
                batch_size=20000
            )
            save_HF_HG_for_entry(entry, H_axis, outs_hscan, csv_path, hf_dir, hg_dir, model_version)
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
                batch_size=20000
            )
            save_VF_VG_for_entry(entry, M_axis, outs_mscan, csv_path, vf_dir, vg_dir, model_version)
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

# %%
