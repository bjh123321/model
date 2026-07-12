from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Set, Any
import math
import re

from surrogate_model.data.io import read_real_dataset

# replay 选取参数（按 LoRA 微调的局部 replay 思路给的默认值）
PRIMARY_H_THR = 1.5   # km
PRIMARY_M_THR = 0.10
FALLBACK_H_THR = 2.5  # km
FALLBACK_M_THR = 0.15
MAX_GROUPS_PER_SELECTED = 6   # 每个新样本最多取多少个 replay 状态点组
MIN_GROUPS_PER_SELECTED = 3   # 若主窗口内太少，则启用 fallback

# 文件名英文编码 -> 程序内部激励键
EXCITATION_CODE_MAP: Dict[str, str] = {
    "FXD_FDC": "FXD_同",
    "FXD_DC": "FXD_反",
    "PW_FDC": "PW_同",
    "PW_DC": "PW_反",
    "JFY_FDC": "JFY_同",
    "JFY_DC": "JFY_反",
}

EXCITATION_ID: Dict[str, int] = {
    "FXD_同": 0,
    "FXD_反": 1,
    "PW_同": 2,
    "PW_反": 3,
    "JFY_同": 4,
    "JFY_反": 5,
}


def _decode_hm_token(token: str) -> float:
    """
    H/M 数值解码：
    - 含 d：把 d 当小数点，如 8d23 -> 8.23
    - 不含 d 且长度>=2：默认最后两位为小数，如 0823 -> 8.23, 140 -> 1.40
    - 其他情况：直接转 float
    """
    token = token.strip()
    if "d" in token:
        return float(token.replace("d", "."))
    if len(token) >= 2:
        return float(f"{token[:-2]}.{token[-2:]}")
    return float(token)


def parse_selected_filename(name: str) -> Tuple[float, float, str, int]:
    """
    从选中的 dat 文件名解析：H, M, excite_key, excite_id
    示例：CG_SC_FXD_DC_E_1d0_H0823_M140_20260306_JiZai.dat
    """
    base = Path(name).name.strip()

    # 优先匹配长编码，避免 FXD_DC 被拆坏
    excite_code = None
    for code in sorted(EXCITATION_CODE_MAP.keys(), key=len, reverse=True):
        if code in base:
            excite_code = code
            break
    if excite_code is None:
        raise ValueError(f"文件名无法解析激励方式: {base}")

    excite_key = EXCITATION_CODE_MAP[excite_code]
    excite_id = EXCITATION_ID[excite_key]

    mh = re.search(r"_H([0-9d]+)", base, flags=re.IGNORECASE)
    mm = re.search(r"_M([0-9d]+)", base, flags=re.IGNORECASE)
    if mh is None or mm is None:
        raise ValueError(f"文件名无法解析 H/M: {base}")

    H = _decode_hm_token(mh.group(1))
    M = _decode_hm_token(mm.group(1))
    return H, M, excite_key, excite_id


def load_selected_targets(selected_list_file: Path) -> List[Tuple[str, float, float, str, int]]:
    """读取 selected_CG_path.txt，返回 [(filename, H, M, excite_key, excite_id), ...]"""
    lines = selected_list_file.read_text(encoding="utf-8").splitlines()
    out: List[Tuple[str, float, float, str, int]] = []
    seen: Set[str] = set()
    for raw in lines:
        name = raw.strip()
        if not name or name.startswith("#"):
            continue
        if name in seen:
            continue
        H, M, excite_key, excite_id = parse_selected_filename(name)
        out.append((name, H, M, excite_key, excite_id))
        seen.add(name)
    return out


def load_real_dataset_groups(real_dataset_file: Path) -> Dict[Tuple[float, float, int], List[str]]:
    """
    读取 real_dataset.txt，并按 (H, M, excite_id) 分组，保留原始行字符串。
    每组内通常包含该状态点下该激励的全部通道行。
    """
    groups: Dict[Tuple[float, float, int], List[str]] = {}
    with open(real_dataset_file, "r", encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 4:
                print(f"[WARN-LINE] 第 {ln} 行列数不足，已跳过")
                continue
            try:
                H = float(parts[0])
                M = float(parts[1])
                excite_id = int(float(parts[2]))
            except Exception as e:
                print(f"[WARN-LINE] 第 {ln} 行前4列解析失败，已跳过 | {e!r}")
                continue
            key = (H, M, excite_id)
            groups.setdefault(key, []).append(line)
    return groups


def load_replay_dataset_groups(dataset_path: Path) -> Dict[Tuple[float, float, int], List[str]]:
    """Load replay groups from either a legacy txt file or a split-dataset directory."""
    groups: Dict[Tuple[float, float, int], List[str]] = {}
    for row in read_real_dataset(Path(dataset_path)):
        H = float(row[0])
        M = float(row[1])
        excite_id = int(round(float(row[2])))
        line = "\t".join(f"{float(value):.17g}" for value in row) + "\n"
        groups.setdefault((H, M, excite_id), []).append(line)
    return groups


def normalized_distance(h: float, m: float, h0: float, m0: float,
                        h_scale: float = PRIMARY_H_THR,
                        m_scale: float = PRIMARY_M_THR) -> float:
    dh = (h - h0) / h_scale
    dm = (m - m0) / m_scale
    return math.sqrt(dh * dh + dm * dm)


def _same_filename_operating_point(h: float, m: float, h0: float, m0: float) -> bool:
    """Compare H/M by the rounded Hxxxx/Mxxx filename codes used by split datasets."""
    return int(round(h * 100.0)) == int(round(h0 * 100.0)) and int(round(m * 100.0)) == int(round(m0 * 100.0))


def pick_replay_groups_for_target(
    target_H: float,
    target_M: float,
    target_excite_id: int,
    all_group_keys: List[Tuple[float, float, int]],
    primary_h_thr: float = PRIMARY_H_THR,
    primary_m_thr: float = PRIMARY_M_THR,
    fallback_h_thr: float = FALLBACK_H_THR,
    fallback_m_thr: float = FALLBACK_M_THR,
    max_groups: int = MAX_GROUPS_PER_SELECTED,
    min_groups: int = MIN_GROUPS_PER_SELECTED,
) -> List[Tuple[float, float, int]]:
    """
    为单个新样本挑选 replay 状态点组。

    规则：
    1) 激励方式必须相同
    2) 优先取主窗口 |ΔH|<=1.5, |ΔM|<=0.10
    3) 若主窗口内组数不足 MIN_GROUPS_PER_SELECTED，则放宽到 fallback 窗口
    4) 按归一化二维距离排序，最多取 MAX_GROUPS_PER_SELECTED 组
    """
    same_excite = [k for k in all_group_keys if k[2] == target_excite_id]
    if not same_excite:
        return []

    primary = []
    for H, M, excite_id in same_excite:
        if abs(H - target_H) <= primary_h_thr and abs(M - target_M) <= primary_m_thr:
            primary.append((H, M, excite_id))

    use_pool = primary
    if len(primary) < min_groups:
        fallback = []
        for H, M, excite_id in same_excite:
            if abs(H - target_H) <= fallback_h_thr and abs(M - target_M) <= fallback_m_thr:
                fallback.append((H, M, excite_id))
        use_pool = fallback if fallback else same_excite

    use_pool = sorted(
        use_pool,
        key=lambda k: normalized_distance(k[0], k[1], target_H, target_M, primary_h_thr, primary_m_thr)
    )

    # 若 real_dataset 中恰好包含与新样本同一 H/M/excite_id 的历史行，排除掉“完全同点”可避免重复强化
    filtered = [
        k for k in use_pool
        if not (_same_filename_operating_point(k[0], k[1], target_H, target_M) and k[2] == target_excite_id)
    ]
    if len(filtered) >= 1:
        use_pool = filtered

    return use_pool[:max_groups]


def select_global_guard_groups(
    all_group_keys: Iterable[Tuple[float, float, int]],
    targets: Iterable[Tuple[str, float, float, str, int]],
    max_groups: int = 48,
    min_groups_per_excitation: int = 2,
    seed: int = 42,
) -> List[Tuple[float, float, int]]:
    """Deterministically retain excitation strata, boundaries, near and far states."""
    import random

    target_points = [(t[1], t[2], t[4]) for t in targets]
    keys = sorted(set(all_group_keys))
    keys = [k for k in keys if not any(k[2] == e and _same_filename_operating_point(k[0], k[1], h, m) for h, m, e in target_points)]
    if not keys:
        # Small-data fallback: never return an empty guard solely due to exclusion.
        keys = sorted(set(all_group_keys))
    if not keys:
        return []
    rng = random.Random(int(seed))
    by_exc: Dict[int, List[Tuple[float, float, int]]] = {}
    for key in keys:
        by_exc.setdefault(key[2], []).append(key)
    required_coverage = sum(min(len(group), max(1, int(min_groups_per_excitation))) for group in by_exc.values())
    limit = max(1, min(max(int(max_groups), required_coverage), len(keys)))

    chosen: List[Tuple[float, float, int]] = []
    def add(key):
        if key not in chosen and len(chosen) < limit:
            chosen.append(key)

    # Boundaries per excitation provide H/M range and cross-excitation coverage.
    for exc in sorted(by_exc):
        group = by_exc[exc]
        boundary = [min(group, key=lambda k: (k[0], k[1])), max(group, key=lambda k: (k[0], k[1])),
                    min(group, key=lambda k: (k[1], k[0])), max(group, key=lambda k: (k[1], k[0]))]
        for key in boundary:
            add(key)
        remaining = [k for k in group if k not in chosen]
        rng.shuffle(remaining)
        for key in remaining[:max(0, int(min_groups_per_excitation) - sum(k[2] == exc for k in chosen))]:
            add(key)

    # Explicitly mix states nearest to and farthest from the new points.
    if target_points:
        distance = lambda k: min(normalized_distance(k[0], k[1], h, m) for h, m, _ in target_points)
        for key in sorted(keys, key=lambda k: (distance(k), k)):
            add(key)
        for key in sorted(keys, key=lambda k: (-distance(k), k)):
            add(key)
    remaining = [k for k in keys if k not in chosen]
    rng.shuffle(remaining)
    for key in remaining:
        add(key)
    return chosen


def _targets_from_names(names: Iterable[str]) -> List[Tuple[str, float, float, str, int]]:
    targets: List[Tuple[str, float, float, str, int]] = []
    seen: Set[str] = set()
    for raw_name in names:
        name = Path(raw_name).name.strip()
        if not name or name in seen:
            continue
        H, M, excite_key, excite_id = parse_selected_filename(name)
        targets.append((name, H, M, excite_key, excite_id))
        seen.add(name)
    return targets


def build_replay_data(
    base_dir: Path,
    model_dir: Path | None = None,
    replay_dataset_path: Path | None = None,
    selected_names: Iterable[str] | None = None,
    *,
    structured: bool = False,
    guard_max_groups: int = 48,
    guard_min_groups_per_excitation: int = 2,
    selection_seed: int = 42,
    primary_h_thr: float = PRIMARY_H_THR,
    primary_m_thr: float = PRIMARY_M_THR,
    fallback_h_thr: float = FALLBACK_H_THR,
    fallback_m_thr: float = FALLBACK_M_THR,
    max_groups_per_selected: int = MAX_GROUPS_PER_SELECTED,
    min_groups_per_selected: int = MIN_GROUPS_PER_SELECTED,
) -> List[List[float]] | Dict[str, Any]:
    base_dir = Path(base_dir)
    selected_list_file = base_dir / "selected_CG_path.txt"
    if replay_dataset_path is None:
        replay_dataset_path = Path(model_dir) / "real_dataset.txt" if model_dir is not None else base_dir.parent / "GENERATION" / "real_dataset.txt"
    replay_dataset_path = Path(replay_dataset_path)
    targets = load_selected_targets(selected_list_file) if selected_names is None else _targets_from_names(selected_names)
    if not targets:
        raise RuntimeError("selected_CG_path.txt 中没有可用文件名")

    groups = load_replay_dataset_groups(replay_dataset_path)
    if not groups:
        raise RuntimeError("real_dataset.txt 中没有可用数据组")

    all_group_keys = list(groups.keys())
    chosen_group_keys: List[Tuple[float, float, int]] = []
    chosen_group_key_set: Set[Tuple[float, float, int]] = set()

    print(f"[INFO] 选中新的 dat 文件数: {len(targets)}")
    print(f"[INFO] real_dataset 中可用状态点组数: {len(all_group_keys)}")

    for filename, H, M, excite_key, excite_id in targets:
        selected_keys = pick_replay_groups_for_target(
            H, M, excite_id, all_group_keys, primary_h_thr, primary_m_thr,
            fallback_h_thr, fallback_m_thr, max_groups_per_selected, min_groups_per_selected,
        )
        if not selected_keys:
            print(f"[WARN] 未找到 replay 组 | {filename} | H={H:.3f}, M={M:.3f}, excite={excite_key}")
            continue

        print(f"[TARGET] {filename} | H={H:.3f}, M={M:.3f}, excite={excite_key} -> replay组数={len(selected_keys)}")
        for key in selected_keys:
            if key not in chosen_group_key_set:
                chosen_group_key_set.add(key)
                chosen_group_keys.append(key)
                d = normalized_distance(key[0], key[1], H, M)
                print(f"    [REPLAY] H={key[0]:.3f}, M={key[1]:.3f}, excite_id={key[2]}, d={d:.3f}, 行数={len(groups[key])}")

    if not chosen_group_keys:
        raise RuntimeError("最终没有选中任何 replay 数据组")

    replay_rows: List[List[float]] = []
    print(f"[REPLAY-USED] groups={len(chosen_group_keys)}")
    for index, key in enumerate(chosen_group_keys, start=1):
        group_lines = groups[key]
        print(
            f"    [{index}] H={key[0]:.6g}, M={key[1]:.6g}, "
            f"excite_id={key[2]}, rows={len(group_lines)}"
        )
        for line in group_lines:
            replay_rows.append([float(value) for value in line.replace(",", " ").split()])

    print(
        f"[DONE] replay kept in memory: groups={len(chosen_group_keys)}, "
        f"rows={len(replay_rows)}; no replay_data.txt written"
    )
    if not structured:
        return replay_rows
    guard_keys = select_global_guard_groups(
        all_group_keys, targets, guard_max_groups, guard_min_groups_per_excitation, selection_seed
    )
    guard_rows = [
        [float(value) for value in line.replace(",", " ").split()]
        for key in guard_keys for line in groups[key]
    ]
    return {
        "local_replay_rows": replay_rows,
        "global_guard_rows": guard_rows,
        "local_group_keys": chosen_group_keys,
        "guard_group_keys": guard_keys,
    }


def main():
    base_dir = Path.cwd()
    build_replay_data(base_dir)


if __name__ == "__main__":
    main()
