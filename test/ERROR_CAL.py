# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import configparser
import numpy as np
import pandas as pd


def get_exe_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


def load_paths_ini(paths_ini: str) -> Dict[str, str]:
    cp = configparser.ConfigParser()
    cp.read(paths_ini, encoding="utf-8")
    out: Dict[str, str] = {}
    for sec in cp.sections():
        kv = [(k, v) for k, v in cp[sec].items() if k.lower().startswith("path_")]
        if not kv:
            continue
        _, v = kv[0]
        out[sec.strip()] = str(v).strip()
    return out


base_dir = get_exe_dir()
user_input_path = base_dir / "user_input.dat"
with open(user_input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

paths_ini_path = Path(lines[0].strip()).expanduser()
model_name = lines[1].strip()
data = [line.strip() for line in lines[2:] if line.strip()]
paths_map = load_paths_ini(paths_ini_path)

model_prefix = model_name.split("_", 1)[0].upper()
if model_prefix == "NH":
    pred_root_dir = Path(paths_map.get("PR_NH")) / model_name
    pred_out_dir = pred_root_dir / "ERROR_PR_NH"
elif model_prefix == "BH":
    pred_root_dir = Path(paths_map.get("PR_BH")) / model_name
    pred_out_dir = pred_root_dir / "ERROR_PR_BH"
elif model_prefix == "XGD":
    pred_root_dir = Path(paths_map.get("PR_XGD")) / model_name
    pred_out_dir = pred_root_dir / "ERROR_PR_XGD"
else:
    raise ValueError(f"无法根据模型名判断路径类型：{model_name}")

VERBOSE = True
SCAN_TYPES = ("HF", "HG", "HA", "VF", "VG", "VA")
H_SCAN_TYPES = ("HF", "HG", "HA")
V_SCAN_TYPES = ("VF", "VG", "VA")


def log(msg: str) -> None:
    if VERBOSE:
        print(msg)


def read_dat_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=",",
        engine="python",
        skiprows=1,
        header=None,
        names=["H", "M", "ch", "val1", "val2", "val3"],
        skipinitialspace=True,
        encoding="utf-8-sig",
    )
    df = df.dropna(how="all")
    return df

def clean_channel(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def to_float(x) -> float:
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def rel_err(pred_val: float, ref_val: float) -> float:
    pred_val = to_float(pred_val)
    ref_val = to_float(ref_val)
    if np.isnan(pred_val) or np.isnan(ref_val) or ref_val == 0:
        return np.nan
    return abs(pred_val - ref_val) / abs(ref_val) * 100.0


def h_token_to_value(h_token: str) -> float:
    m = re.fullmatch(r"H(\d+)", h_token)
    if not m:
        raise ValueError(f"非法高度 token: {h_token}")
    return int(m.group(1)) / 100.0


def m_token_to_value(m_token: str) -> float:
    m = re.fullmatch(r"M(\d+)", m_token)
    if not m:
        raise ValueError(f"非法马赫 token: {m_token}")
    return int(m.group(1)) / 100.0


def nearest_value(values: np.ndarray, target: float) -> float:
    values = np.asarray(values, dtype=float)
    idx = int(np.argmin(np.abs(values - target)))
    return float(values[idx])


def parse_a_filename(path: Path) -> Dict[str, object]:
    """
    例如：
    HF_CG_SC_M139_FXD_FDC_20260313_JiZai.dat
    VF_CG_SC_H1100_FXD_FDC_20260313_JiZai.dat
    """
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 8:
        raise ValueError(f"A 文件名格式异常：{path.name}")

    scan_type = parts[0]
    if scan_type not in SCAN_TYPES:
        raise ValueError(f"A 文件 scan_type 非法：{path.name}")

    fixed_token = parts[3]
    excitation = f"{parts[4]}_{parts[5]}"
    date_idx = next((i for i, p in enumerate(parts) if re.fullmatch(r"\d{8}", p)), None)
    if date_idx is None:
        raise ValueError(f"A 文件名中未找到日期：{path.name}")

    info: Dict[str, object] = {
        "stem": stem,
        "name": path.name,
        "scan_type": scan_type,
        "excitation": excitation,
        "fixed_token": fixed_token,
        "date": parts[date_idx],
    }

    if scan_type in H_SCAN_TYPES:
        if not re.fullmatch(r"M\d+", fixed_token):
            raise ValueError(f"{path.name} 应包含固定马赫 Mxxx")
        info["fixed_axis"] = "M"
        info["fixed_val"] = m_token_to_value(fixed_token)
        info["scan_axis"] = "H"
        info["M_token"] = fixed_token
        info["M_val"] = info["fixed_val"]
        info["H_token"] = None
        info["H_val"] = None
    else:
        if not re.fullmatch(r"H\d+", fixed_token):
            raise ValueError(f"{path.name} 应包含固定高度 Hxxxx")
        info["fixed_axis"] = "H"
        info["fixed_val"] = h_token_to_value(fixed_token)
        info["scan_axis"] = "M"
        info["H_token"] = fixed_token
        info["H_val"] = info["fixed_val"]
        info["M_token"] = None
        info["M_val"] = None

    return info


def parse_pred_filename(path: Path) -> Dict[str, object]:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 7:
        raise ValueError(f"预测文件名格式异常：{path.name}")

    scan_type = parts[0]
    if scan_type not in SCAN_TYPES or parts[1] != "PR":
        raise ValueError(f"非法预测文件名：{path.name}")

    fixed_token = parts[2]
    excitation = f"{parts[3]}_{parts[4]}"
    date_idx = next((i for i, p in enumerate(parts) if re.fullmatch(r"\d{8}", p)), None)
    if date_idx is None:
        raise ValueError(f"预测文件名中未找到日期：{path.name}")

    info: Dict[str, object] = {
        "stem": stem,
        "name": path.name,
        "scan_type": scan_type,
        "excitation": excitation,
        "fixed_token": fixed_token,
        "date": parts[date_idx],
        "tail": "_".join(parts[5:]),
        "path": path,
    }
    if scan_type in H_SCAN_TYPES:
        if not re.fullmatch(r"M\d+", fixed_token):
            raise ValueError(f"{path.name} 应包含 Mxxx 固定字段")
        info["fixed_axis"] = "M"
        info["fixed_val"] = m_token_to_value(fixed_token)
        info["scan_axis"] = "H"
    else:
        if not re.fullmatch(r"H\d+", fixed_token):
            raise ValueError(f"{path.name} 应包含 Hxxxx 固定字段")
        info["fixed_axis"] = "H"
        info["fixed_val"] = h_token_to_value(fixed_token)
        info["scan_axis"] = "M"
    return info


def pick_first_existing(df: pd.DataFrame, candidates: List[str], file_path: Path) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"{file_path.name} 缺少列，候选列名为：{candidates}，实际列名为：{list(df.columns)}")


def get_fixed_columns(df: pd.DataFrame, path: Path) -> Dict[str, str]:
    if df.shape[1] < 6:
        raise ValueError(f"{path.name} 列数不足，至少需要 6 列，实际只有 {df.shape[1]} 列")
    cols = list(df.columns)
    return {
        "H": cols[0],
        "M": cols[1],
        "ch": cols[2],
        "val1": cols[3],
        "val2": cols[4],
        "val3": cols[5],
    }


def collect_pred_candidates(folder: Path, a_info: Dict[str, object], scan_type: str) -> List[Dict[str, object]]:
    out = []
    for fp in folder.rglob("*.dat"):
        try:
            info = parse_pred_filename(fp)
        except Exception:
            continue
        if info["scan_type"] != scan_type:
            continue
        if info["excitation"] != a_info["excitation"]:
            continue
        out.append(info)
    return out


def choose_nearest_pred_file(folder: Path, a_info: Dict[str, object], scan_type: str) -> Optional[Dict[str, object]]:
    cands = collect_pred_candidates(folder, a_info, scan_type)
    if not cands:
        return None
    target_fixed_val = float(a_info["fixed_val"])
    return min(cands, key=lambda x: abs(float(x["fixed_val"]) - target_fixed_val))


def build_output_filename(scan_type: str, out_dir: Path) -> Path:
    return out_dir / f"ERROR_{scan_type}.dat"

def calc_error_for_one_pair(a_file: Path, pred_info: Dict[str, object], out_dir: Path) -> Optional[Path]:
    df_a = read_dat_table(a_file)
    df_pred = read_dat_table(pred_info["path"])

    a_info = parse_a_filename(a_file)
    a_cols = {"H":"H", "M":"M", "ch":"ch", "val1":"val1", "val2":"val2", "val3":"val3"}
    pred_cols = {"H":"H", "M":"M", "ch":"ch", "val1":"val1", "val2":"val2", "val3":"val3"}

    df_a = df_a.copy()
    df_pred = df_pred.copy()
    df_a[a_cols["ch"]] = df_a[a_cols["ch"]].map(clean_channel)
    df_pred[pred_cols["ch"]] = df_pred[pred_cols["ch"]].map(clean_channel)
    df_a[a_cols["H"]] = pd.to_numeric(df_a[a_cols["H"]], errors="coerce")
    df_a[a_cols["M"]] = pd.to_numeric(df_a[a_cols["M"]], errors="coerce")
    for col in ["H", "M", "val1", "val2", "val3"]:
        df_pred[col] = pd.to_numeric(df_pred[col], errors="coerce")
        df_a[col] = pd.to_numeric(df_a[col], errors="coerce")

    scan_axis = str(a_info["scan_axis"])
    a_scan_col = a_cols[scan_axis]
    p_scan_col = pred_cols[scan_axis]
    fixed_axis = str(a_info["fixed_axis"])
    a_fixed_col = a_cols[fixed_axis]
    p_fixed_col = pred_cols[fixed_axis]
    fixed_val = float(a_info["fixed_val"])

    # 仅保留预测文件中固定轴最近的子集；正常情况下整文件都相同，这里做稳健处理
    pred_fixed_vals = df_pred[p_fixed_col].dropna().to_numpy(dtype=float)
    if pred_fixed_vals.size == 0:
        raise ValueError(f"{pred_info['name']} 中固定轴列 {p_fixed_col} 没有有效数值")
    pred_fixed_nearest = nearest_value(pred_fixed_vals, fixed_val)
    df_pred_fixed = df_pred[np.isclose(df_pred[p_fixed_col].astype(float), pred_fixed_nearest, atol=1e-12, rtol=0.0)].copy()

    # 仅保留 A 文件中固定轴最近的子集；应对文件内存在轻微浮动
    a_fixed_vals = df_a[a_fixed_col].dropna().to_numpy(dtype=float)
    if a_fixed_vals.size == 0:
        raise ValueError(f"{a_file.name} 中固定轴列 {a_fixed_col} 没有有效数值")
    a_fixed_nearest = nearest_value(a_fixed_vals, fixed_val)
    df_a_fixed = df_a[np.isclose(df_a[a_fixed_col].astype(float), a_fixed_nearest, atol=1e-12, rtol=0.0)].copy()

    pred_scan_vals = df_pred_fixed[p_scan_col].dropna().to_numpy(dtype=float)
    if pred_scan_vals.size == 0:
        raise ValueError(f"{pred_info['name']} 中扫描轴列 {p_scan_col} 没有有效数值")

    result_rows = []
    unmatched = []
    unique_targets = sorted(df_a_fixed[a_scan_col].dropna().astype(float).unique().tolist())

    for target_scan in unique_targets:
        matched_scan = nearest_value(pred_scan_vals, target_scan)
        df_a_grp = df_a_fixed[np.isclose(df_a_fixed[a_scan_col].astype(float), target_scan, atol=1e-12, rtol=0.0)].copy()
        df_p_grp = df_pred_fixed[np.isclose(df_pred_fixed[p_scan_col].astype(float), matched_scan, atol=1e-12, rtol=0.0)].copy()
        df_p_grp = df_p_grp.set_index(pred_cols["ch"], drop=False)

        for _, a_row in df_a_grp.iterrows():
            ch = clean_channel(a_row[a_cols["ch"]])
            if ch not in df_p_grp.index:
                unmatched.append((target_scan, ch))
                result_rows.append([
                    to_float(a_row[a_cols["H"]]),
                    to_float(a_row[a_cols["M"]]),
                    ch,
                    np.nan,
                    np.nan,
                    np.nan,
                ])
                continue

            p_row = df_p_grp.loc[ch]
            if isinstance(p_row, pd.DataFrame):
                p_row = p_row.iloc[0]

            errs = []
            for a_col, p_col in zip([a_cols["val1"], a_cols["val2"], a_cols["val3"]], [pred_cols["val1"], pred_cols["val2"], pred_cols["val3"]]):
                errs.append(rel_err(p_row[p_col], a_row[a_col]))

            result_rows.append([
                to_float(a_row[a_cols["H"]]),
                to_float(a_row[a_cols["M"]]),
                ch,
                errs[0],
                errs[1],
                errs[2],
            ])

    df_out = pd.DataFrame(
        result_rows,
        columns=["H", "M", "ch", "err1", "err2", "err3"],
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = build_output_filename(pred_info["scan_type"], out_dir)
    df_out.to_csv(out_path, index=False)

    if unmatched:
        log(f"[WARN] {a_file.name} vs {pred_info['name']} 未匹配到的通道数：{len(unmatched)}")
    log(f"[OK] 已保存：{out_path}")
    return out_path


def process_one_a_file(a_file: Path, pred_root: Path, out_dir: Path) -> List[Path]:
    a_info = parse_a_filename(a_file)
    log(f"\n========== 处理 A 文件：{a_file.name} ==========")
    log(
        f"提取信息：scan_type={a_info['scan_type']}, excitation={a_info['excitation']}, "
        f"fixed_axis={a_info['fixed_axis']}, fixed_val={float(a_info['fixed_val']):.2f}, date={a_info['date']}"
    )

    outputs: List[Path] = []
    pred_info = choose_nearest_pred_file(pred_root, a_info, str(a_info["scan_type"]))
    if pred_info is None:
        log(f"[SKIP] 未找到 {a_info['scan_type']} 对应文件")
        return outputs

    log(
        f"[MATCH] {a_info['scan_type']} 目标固定值={float(a_info['fixed_val']):.2f}, "
        f"命中文件={pred_info['name']} (固定{pred_info['fixed_axis']}={float(pred_info['fixed_val']):.2f})"
    )
    out_path = calc_error_for_one_pair(a_file=a_file, pred_info=pred_info, out_dir=out_dir)
    if out_path is not None:
        outputs.append(out_path)
    return outputs


def batch_process_all_a_files(a_files: List[Path], pred_root: Path, out_dir: Path) -> List[Path]:
    if not a_files:
        raise FileNotFoundError("user_input.dat 第 3 行起未提供任何 A 文件路径")

    all_outputs: List[Path] = []
    for a_file in a_files:
        try:
            a_path = Path(a_file)
            if not a_path.exists():
                raise FileNotFoundError(f"A 文件不存在：{a_path}")
            outs = process_one_a_file(a_path, pred_root=pred_root, out_dir=out_dir)
            all_outputs.extend(outs)
        except Exception as e:
            print(f"[ERROR] 处理失败：{a_file}\n{e}")
    print(f"\n全部完成，共生成 {len(all_outputs)} 个误差文件。")
    return all_outputs


if __name__ == "__main__":
    batch_process_all_a_files(
        a_files=[Path(x) for x in data],
        pred_root=pred_root_dir,
        out_dir=pred_out_dir,
    )
