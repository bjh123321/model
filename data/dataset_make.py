#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从“多文件夹/多文件”的结构构建数据集（单通道一行、变长模态串），并将：
- 激励方式用固定数字（0~5）
- 通道用你手动定义的列表映射为数字（0 ~ len(CHANNEL_LIST)-1）

输出格式：
H, M, excitation_id, channel_id, K1, Re1, Im1, Z1, K2, Re2, Im2, Z2, ...
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Tuple

# =====================【可编辑区域】=====================

# 激励方式固定映射（按你给的顺序）
EXCITATION_LIST = [
    "方向舵对称",
    "方向舵反对称",
    "襟副翼对称",
    "襟副翼反对称",
    "平尾对称",
    "平尾反对称",
]
EXC_TO_ID = {name: i for i, name in enumerate(EXCITATION_LIST)}

CHANNEL_LIST = [
    "LW1", "LW2", "LW3", "LAI", "RW1", "RW2", "RW3", "RAI", "LVT1", "LVT2",
    "LVT3", "LVT4", "LRY", "LRY1", "RVT1", "RVT2", "RVT3", "RVT4", "RRY", "RRY1",
    "LHT1", "LHT4", "LHT2", "LHT3", "RHT1", "RHT4", "RHT2", "RHT3", "F1Z", "F1Y",
    "ZWL1Y", "ZWL1Z", "YWL1Y", "YWL1Z"
]
CH_TO_ID = {name: i for i, name in enumerate(CHANNEL_LIST)}

# 其他设置
USE_ABS_IMAG = True       # True -> Im 取绝对值，上半平面
SEP_OUT = "\t"            # 输出分隔符
ENCODING = "utf-8"
FLOAT_FMT = "{:.6g}"      # 浮点数格式

# ======================================================

def parse_folder_name(folder: str) -> Tuple[str, str]:
    """解析子文件夹名: "<exc1>_<channel>_<exc2>" -> (excitation_text, channel_name)"""
    parts = folder.split("_")
    if len(parts) < 3:
        raise ValueError(f"子文件夹名不符合约定: {folder}")
    exc1, ch, exc2 = parts[0], parts[1], parts[2]
    return exc1 + exc2, ch

def parse_file_mode_index(filename: str) -> int:
    """从文件名开头解析纯数字模态索引，例如 '1方向舵_F1Y_对称.txt' -> 1"""
    m = re.match(r"^(\d+)", filename)
    if not m:
        raise ValueError(f"文件名不含前导数字（模态索引）: {filename}")
    return int(m.group(1))

def read_matrix_whitespace(path: Path) -> List[List[float]]:
    """读取以空白分隔的 txt 文件为二维矩阵"""
    rows: List[List[float]] = []
    with path.open("r", encoding=ENCODING) as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            rows.append([float(x) for x in parts])
    if not rows:
        raise ValueError(f"文件为空: {path}")
    ncols = len(rows[0])
    for i, r in enumerate(rows):
        if len(r) != ncols:
            raise ValueError(f"行列数不一致: 第1行={ncols}, 第{i+1}行={len(r)}")
    return rows

def is_placeholder_sample(H: float, M: float) -> bool:
    return abs(H) < 1e-12 and abs(M) < 1e-12

def f6(x: float) -> str:
    return FLOAT_FMT.format(x)

def int_or_f6(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else FLOAT_FMT.format(x)

def build_dataset_from_root(root_dir: str, output_path: str) -> None:
    """主函数"""
    root = Path(root_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"root_dir 不存在或不是目录: {root}")

    groups: Dict[Tuple[float, float, int, int], List[dict]] = {}
    n_files = n_cols_total = n_cols_kept = 0

    for folder in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
        try:
            exc_text, ch_name = parse_folder_name(folder.name)
        except Exception as e:
            print(f"[跳过子文件夹] {folder.name}: {e}")
            continue

        if exc_text not in EXC_TO_ID:
            print(f"[跳过子文件夹] {folder.name}: 未在激励方式表中 -> {exc_text}")
            continue
        if ch_name not in CH_TO_ID:
            print(f"[跳过子文件夹] {folder.name}: 未在通道表中 -> {ch_name}")
            continue

        exc_id = EXC_TO_ID[exc_text]
        ch_id = CH_TO_ID[ch_name]

        for txt in sorted(folder.glob("*.txt"), key=lambda p: p.name):
            n_files += 1
            try:
                mode_idx = parse_file_mode_index(txt.name)
                rows = read_matrix_whitespace(txt)
            except Exception as e:
                print(f"[跳过文件] {txt.name}: {e}")
                continue

            if len(rows) < 11:
                print(f"[跳过文件] {txt.name}: 行数不足 11（实际 {len(rows)}）")
                continue

            H_row, M_row = rows[0], rows[1]
            K_row, freq_row = rows[2], rows[3]
            alpha_row, omega_row, zero_row = rows[5], rows[7], rows[9]
            ncols = len(H_row)
            n_cols_total += ncols

            for j in range(ncols):
                H, M = H_row[j], M_row[j]
                if is_placeholder_sample(H, M):
                    continue
                K, fr, Re, Im, Z = K_row[j], freq_row[j], alpha_row[j], omega_row[j], zero_row[j]
                if USE_ABS_IMAG:
                    Im = abs(Im)
                key = (H, M, exc_id, ch_id)
                groups.setdefault(key, []).append({
                    "mode_idx": mode_idx,
                    "freq": fr,
                    "K": K,
                    "Re": Re,
                    "Im": Im,
                    "Z": Z,
                })
                n_cols_kept += 1

    # 写出结果
    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding=ENCODING) as f:
        for key, modes in groups.items():
            H, M, exc_id, ch_id = key
            modes_sorted = sorted(modes, key=lambda d: (d["freq"], d["mode_idx"]))
            header = [f6(H), f6(M), str(exc_id), str(ch_id)]
            payload = []
            for m in modes_sorted:
                payload.extend([
                    int_or_f6(m["K"]),
                    f6(m["Re"]),
                    f6(m["Im"]),
                    f6(m["Z"]),
                ])
            f.write(SEP_OUT.join(header + payload) + "\n")

    print(f"[完成] 文件数={n_files}, 样本列(原始)={n_cols_total}, 样本列(有效)={n_cols_kept}")
    print(f"[输出数据] {outp}")

if __name__ == "__main__":
    root_dir   = r"C:\Users\Leo\Desktop\MFDNN_datamake\试飞参数35A（训练用）_sorted"
    output_txt = r"C:\Users\Leo\Desktop\MFDNN_datamake\dataset_merged.txt"
    build_dataset_from_root(root_dir, output_txt)
