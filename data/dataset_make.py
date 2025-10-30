#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将“单模态逐行”的原始试飞数据，合并为“单通道一行、变长模态串”的格式。

输入列定义（每行=一个模态）：
1:H, 2:M, 3:excitation, 4:Channel, 5:mode_num, 6:K, 7:freq, 8:zeta,
9:alpha(极点实部), 10:alpha_conj, 11:omega(极点虚部), 12:omega_conj,
13:zero(零点), 14:占位0

输出（每行=一个通道）：
H, M, excitation, Channel, K1, Re1, Im1, Z1, K2, Re2, Im2, Z2, ...

要点：
- 同一 (H,M,exc,Ch) 内按 freq 升序；
- “前五列完全相同”的重复行只保留一条；
- Im 默认取 abs(omega)，统一到上半平面（可配置）；
- 过滤 H=M=exc=ch=0 的占位样本。
"""

from collections import defaultdict, OrderedDict
from typing import List, Tuple, Optional, Iterable
import math

# 1-based 列号（便于与你沟通一致）
COL_H, COL_M, COL_EXC, COL_CH = 1, 2, 3, 4
COL_MODE_NUM, COL_K, COL_FREQ, COL_ZETA = 5, 6, 7, 8
COL_RE, COL_RE_CONJ, COL_IM, COL_IM_CONJ, COL_ZERO, COL_PAD0 = 9, 10, 11, 12, 13, 14

def _split_line(line: str, sep: Optional[str]) -> List[str]:
    line = line.strip()
    if not line:
        return []
    return line.split() if sep is None else line.split(sep)

def _to_float_safe(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None

def _key4_numeric(cols: List[str]) -> Optional[Tuple[float,float,int,int]]:
    if len(cols) < 4:
        return None
    H = _to_float_safe(cols[COL_H-1])
    M = _to_float_safe(cols[COL_M-1])
    exc = _to_float_safe(cols[COL_EXC-1])
    ch  = _to_float_safe(cols[COL_CH-1])
    if H is None or M is None or exc is None or ch is None:
        return None
    return (H, M, int(round(exc)), int(round(ch)))

def _key5_text(cols: List[str]) -> Tuple[str,...]:
    # 用于“前五列完全相同”的去重（保留首条）
    return tuple(cols[i-1] for i in (COL_H, COL_M, COL_EXC, COL_CH, COL_MODE_NUM))

def merge_file(
    input_path: str,
    output_path: str,
    *,
    sep_in: Optional[str] = None,  # 默认 None：任意空白分隔（推荐）
    sep_out: str = "\t",
    use_abs_imag: bool = True,     # True -> Im 取上半平面
) -> None:

    # 分组存放：key4 -> 列数组列表
    groups: "OrderedDict[Tuple[float,float,int,int], List[List[str]]]" = OrderedDict()
    seen5: "OrderedDict[Tuple[float,float,int,int], set]" = OrderedDict()

    # 读取与分组
    with open(input_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            cols = _split_line(line, sep_in)
            if not cols:
                continue
            if len(cols) < COL_PAD0:
                raise ValueError(f"第 {ln} 行列数不足 {COL_PAD0}：{cols}")

            k4 = _key4_numeric(cols)
            if k4 is None:
                # 非法行，跳过
                continue

            # 过滤占位行 (0,0,0,0)
            if all(abs(v) < 1e-12 for v in k4):
                continue

            if k4 not in groups:
                groups[k4] = []
                seen5[k4] = set()

            k5 = _key5_text(cols)
            if k5 in seen5[k4]:
                # 前五列完全相同 -> 视为重复，跳过
                continue
            seen5[k4].add(k5)
            groups[k4].append(cols)

    # 写出（每组一行）
    with open(output_path, "w", encoding="utf-8") as out:
        for k4, rows in groups.items():
            # 组内按 freq 升序（更稳）；若 freq 缺失则退化用 mode_num
            def _freq_or_mode_num(r: List[str]) -> Tuple[float, float]:
                f  = _to_float_safe(r[COL_FREQ-1])
                mn = _to_float_safe(r[COL_MODE_NUM-1])
                return (f if f is not None else float('inf'),
                        mn if mn is not None else float('inf'))
            rows_sorted = sorted(rows, key=_freq_or_mode_num)

            # 组装输出头（H M exc ch）
            H, M, exc, ch = k4
            header = [f"{H:.6g}", f"{M:.6g}", str(int(exc)), str(int(ch))]

            # 逐模态抽取 K, Re, Im, Zero
            payload: List[str] = []
            for r in rows_sorted:
                K = _to_float_safe(r[COL_K-1])
                Re = _to_float_safe(r[COL_RE-1])
                Im = _to_float_safe(r[COL_IM-1])
                Z  = _to_float_safe(r[COL_ZERO-1])

                # 健壮性：若某一项解析失败，跳过该模态
                if K is None or Re is None or Im is None or Z is None:
                    continue

                if use_abs_imag:
                    Im = abs(Im)  # 上半平面

                # 保持紧凑表示：K/通道索引等若为整数按 int 输出
                K_str  = str(int(K)) if float(K).is_integer() else f"{K:.6g}"
                Re_str = f"{Re:.6g}"
                Im_str = f"{Im:.6g}"
                Z_str  = f"{Z:.6g}"

                payload.extend([K_str, Re_str, Im_str, Z_str])

            # 写出：若该通道无有效模态（payload 空），也写头（后续可据此统计/过滤）
            out.write(sep_out.join(header + payload) + "\n")


if __name__ == "__main__":
    # 示例：按任意空白分隔读取，制表符写出
    # Windows 下记得用原始字符串 r"..." 或双反斜杠
    in_path  = r"C:\Users\Leo\Desktop\MFDNN_datamake\sim_dataset.txt"
    out_path = r"C:\Users\Leo\Desktop\MFDNN_datamake\sim_dataset_new.txt"
    merge_file(in_path, out_path, sep_in=None, sep_out="\t", use_abs_imag=True)
