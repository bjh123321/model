from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List
import numpy as np

def read_variable_txt(path: Path) -> List[List[float]]:
    rows: List[List[float]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            parts = ln.replace(',', ' ').split()
            try:
                row = [float(x) for x in parts]
            except ValueError:
                continue
            if len(row) < 4:
                continue
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
    rho:  np.ndarray
    sin_theta:  np.ndarray
    cos_theta: np.ndarray
    mask:  np.ndarray

def parse_rows_to_struct(rows: List[List[float]]) -> List[ParsedRow]:
    import math
    fs = 512
    omegaN = math.pi * fs
    K_list, ch_list, exc_list = [], [], []
    for r in rows:
        tail = r[4:]
        K_list.append(len(tail) // 4)
        exc_list.append(int(round(r[2])))
        ch_list.append(int(round(r[3])))
        
    K_max = max(K_list)
    channel_num = max(ch_list) + 1
    excitation_num = max(exc_list) + 1
    eps = 1e-12

    parsed: List[ParsedRow] = []
    for r in rows:
        H, M = float(r[0]), float(r[1])
        exc_id = int(round(r[2]))
        ch_id = int(round(r[3]))
        tail = r[4:]
        K_all = len(tail) // 4
        
        valid_idx = []
        for k in range(K_all):
            Rek   = float(tail[4*k + 0])
            Imk  = float(tail[4*k + 1])
            rk  = float(tail[4*k + 2])
            thk   = float(tail[4*k + 3])
            # 判定该模态是否为“有效（非补位）”
            if (abs(Rek) > eps) or (abs(Imk) > eps) or (abs(rk) > eps) or (abs(thk) > eps):
                valid_idx.append(k)
        K = len(valid_idx)

        alpha = np.zeros((K_max,), dtype=np.float32)
        omega = np.zeros((K_max,), dtype=np.float32)
        rho  = np.zeros((K_max,), dtype=np.float32)
        sin_theta = np.zeros((K_max,), dtype=np.float32)
        cos_theta = np.zeros((K_max,), dtype=np.float32)
        mask  = np.zeros((K_max,), dtype=np.float32)

        for k in range(K):
            src = valid_idx[k]
            base = 4 * src
            Re  = float(tail[base + 0])
            Im  = float(tail[base + 1])
            rh  = float(tail[base + 2])
            th  = float(tail[base + 3])

            alpha[k]  = -abs(Re)
            omega[k] = abs(Im)/omegaN
            rho[k] = np.log(rh)
            sin_theta[k] = np.sin(th)
            cos_theta[k] = np.cos(th)
            mask[k]  = 1.0

        parsed.append(ParsedRow(H, M, exc_id, ch_id, K, alpha, omega, rho, sin_theta, cos_theta, mask))
    stats = {
        "K_max": K_max,
        "channel_num": channel_num,
        "excitation_num": excitation_num,
    }
    return parsed, stats

if __name__ == '__main__':
    # 测试代码
    test_path = Path(__file__).parent / 'sim_dataset_new.txt'
    test_rows = read_variable_txt(test_path)
    parsed, stats = parse_rows_to_struct(test_rows)
    print(f'Total rows parsed: {parsed[1000]}')
    # for p in parsed:
        # print(p)