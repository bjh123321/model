from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

from surrogate_model.data.io import SPLIT_DATASET_NAME_RE, read_real_dataset


GroupKey = Tuple[float, float, int]
StableGroupKey = Tuple[int, int, int]

# Deprecated local-replay parameters. Kept for callers that still import the
# diagnostic helper; they no longer control the production replay set.
PRIMARY_H_THR = 1.5
PRIMARY_M_THR = 0.10
FALLBACK_H_THR = 2.5
FALLBACK_M_THR = 0.15
MAX_GROUPS_PER_SELECTED = 6
MIN_GROUPS_PER_SELECTED = 3

EXCITATION_CODE_TO_ID: Dict[str, int] = {
    "FXD_FDC": 0,
    "FXD_DC": 1,
    "PW_FDC": 2,
    "PW_DC": 3,
    "JFY_FDC": 4,
    "JFY_DC": 5,
}


def _decode_hm_token(token: str) -> float:
    token = token.strip()
    if "d" in token.lower():
        return float(token.lower().replace("d", "."))
    if len(token) >= 2:
        return float(f"{token[:-2]}.{token[-2:]}")
    return float(token)


def parse_selected_filename(name: str) -> Tuple[float, float, str, int]:
    base = Path(name).name.strip()
    excitation_code = next(
        (code for code in sorted(EXCITATION_CODE_TO_ID, key=len, reverse=True) if code in base.upper()),
        None,
    )
    if excitation_code is None:
        raise ValueError(f"Cannot parse excitation type from selected filename: {base}")
    mh = re.search(r"_H([0-9d]+)", base, flags=re.IGNORECASE)
    mm = re.search(r"_M([0-9d]+)", base, flags=re.IGNORECASE)
    if mh is None or mm is None:
        raise ValueError(f"Cannot parse H/M from selected filename: {base}")
    return (
        _decode_hm_token(mh.group(1)),
        _decode_hm_token(mm.group(1)),
        excitation_code,
        EXCITATION_CODE_TO_ID[excitation_code],
    )


def load_selected_targets(selected_list_file: Path) -> List[Tuple[str, float, float, str, int]]:
    targets: List[Tuple[str, float, float, str, int]] = []
    seen: Set[str] = set()
    for raw in Path(selected_list_file).read_text(encoding="utf-8").splitlines():
        name = raw.strip()
        if not name or name.startswith("#") or name in seen:
            continue
        h, m, excitation_code, excitation_id = parse_selected_filename(name)
        targets.append((name, h, m, excitation_code, excitation_id))
        seen.add(name)
    return targets


def _targets_from_names(names: Iterable[str]) -> List[Tuple[str, float, float, str, int]]:
    targets: List[Tuple[str, float, float, str, int]] = []
    seen: Set[str] = set()
    for raw_name in names:
        name = Path(raw_name).name.strip()
        if not name or name in seen:
            continue
        h, m, excitation_code, excitation_id = parse_selected_filename(name)
        targets.append((name, h, m, excitation_code, excitation_id))
        seen.add(name)
    return targets


def load_replay_dataset_groups(dataset_path: Path) -> Dict[GroupKey, List[str]]:
    """Load a legacy dataset file or a directory and preserve complete H/M/excitation groups."""
    groups: Dict[GroupKey, List[str]] = {}
    for row in read_real_dataset(Path(dataset_path)):
        h = float(row[0])
        m = float(row[1])
        excitation_id = int(round(float(row[2])))
        line = "\t".join(f"{float(value):.17g}" for value in row) + "\n"
        groups.setdefault((h, m, excitation_id), []).append(line)
    return groups


def normalized_distance(
    h: float,
    m: float,
    h0: float,
    m0: float,
    h_scale: float = PRIMARY_H_THR,
    m_scale: float = PRIMARY_M_THR,
) -> float:
    return math.hypot((h - h0) / h_scale, (m - m0) / m_scale)


def _same_filename_operating_point(h: float, m: float, h0: float, m0: float) -> bool:
    return int(round(h * 100.0)) == int(round(h0 * 100.0)) and int(round(m * 100.0)) == int(round(m0 * 100.0))


def stable_group_key(key: GroupKey) -> StableGroupKey:
    """Encode an operating point using the precision of split-dataset filenames."""
    return int(round(float(key[0]) * 100.0)), int(round(float(key[1]) * 100.0)), int(key[2])


def pick_replay_groups_for_target(
    target_H: float,
    target_M: float,
    target_excite_id: int,
    all_group_keys: List[GroupKey],
    primary_h_thr: float = PRIMARY_H_THR,
    primary_m_thr: float = PRIMARY_M_THR,
    fallback_h_thr: float = FALLBACK_H_THR,
    fallback_m_thr: float = FALLBACK_M_THR,
    max_groups: int = MAX_GROUPS_PER_SELECTED,
    min_groups: int = MIN_GROUPS_PER_SELECTED,
) -> List[GroupKey]:
    """Deprecated diagnostic helper for inspecting nearby operating points."""
    same_excitation = [key for key in all_group_keys if key[2] == target_excite_id]
    primary = [
        key for key in same_excitation
        if abs(key[0] - target_H) <= primary_h_thr and abs(key[1] - target_M) <= primary_m_thr
    ]
    pool = primary
    if len(primary) < min_groups:
        fallback = [
            key for key in same_excitation
            if abs(key[0] - target_H) <= fallback_h_thr and abs(key[1] - target_M) <= fallback_m_thr
        ]
        pool = fallback if fallback else same_excitation
    pool = [
        key for key in pool
        if not (
            key[2] == target_excite_id
            and _same_filename_operating_point(key[0], key[1], target_H, target_M)
        )
    ]
    return sorted(
        pool,
        key=lambda key: normalized_distance(
            key[0], key[1], target_H, target_M, primary_h_thr, primary_m_thr
        ),
    )[:max_groups]


def _canonical_group_map(groups: Dict[GroupKey, List[str]]) -> Dict[StableGroupKey, GroupKey]:
    canonical: Dict[StableGroupKey, GroupKey] = {}
    for key in groups:
        encoded = stable_group_key(key)
        previous = canonical.get(encoded)
        if previous is not None and previous != key:
            raise RuntimeError(
                f"Multiple history groups collapse to filename-precision key {encoded}: "
                f"{previous} and {key}"
            )
        canonical[encoded] = key
    return canonical


def _guard_target_count(group_count: int, replay_fraction: float) -> int:
    if group_count <= 1:
        return 0
    # Use conventional half-up rounding; Python's bankers rounding would bias
    # repeated x.5 excitation strata away from the requested global fraction.
    replay_count = int(math.floor(group_count * replay_fraction + 0.5))
    replay_count = min(max(replay_count, 1), group_count - 1)
    return group_count - replay_count


def _select_guard_for_excitation(keys: List[GroupKey], count: int, seed: int) -> List[GroupKey]:
    if count <= 0:
        return []
    keys = sorted(keys)
    h_min, h_max = min(k[0] for k in keys), max(k[0] for k in keys)
    m_min, m_max = min(k[1] for k in keys), max(k[1] for k in keys)

    boundary = [
        min(keys, key=lambda k: (k[0], k[1])),
        max(keys, key=lambda k: (k[0], k[1])),
        min(keys, key=lambda k: (k[1], k[0])),
        max(keys, key=lambda k: (k[1], k[0])),
    ]
    unique_boundary: List[GroupKey] = []
    for key in boundary:
        if key not in unique_boundary:
            unique_boundary.append(key)

    selected = unique_boundary[:count]
    rng = random.Random(int(seed))
    tie_order = list(keys)
    rng.shuffle(tie_order)
    tie_rank = {key: index for index, key in enumerate(tie_order)}

    def point(key: GroupKey) -> Tuple[float, float]:
        hn = 0.0 if h_max == h_min else (key[0] - h_min) / (h_max - h_min)
        mn = 0.0 if m_max == m_min else (key[1] - m_min) / (m_max - m_min)
        return hn, mn

    while len(selected) < count:
        candidates = [key for key in keys if key not in selected]
        if not selected:
            selected.append(min(candidates, key=lambda key: tie_rank[key]))
            continue

        def min_distance(key: GroupKey) -> float:
            kh, km = point(key)
            return min(math.hypot(kh - sh, km - sm) for sh, sm in map(point, selected))

        selected.append(max(candidates, key=lambda key: (min_distance(key), -tie_rank[key])))
    return selected


def _rows_for_keys(groups: Dict[GroupKey, List[str]], keys: Iterable[GroupKey]) -> List[List[float]]:
    return [
        [float(value) for value in line.replace(",", " ").split()]
        for key in keys
        for line in groups[key]
    ]


def _range(values: List[float]) -> Optional[List[float]]:
    return [min(values), max(values)] if values else None


def _build_coverage(
    groups: Dict[GroupKey, List[str]],
    replay_keys: List[GroupKey],
    guard_keys: List[GroupKey],
    excluded_keys: Set[StableGroupKey],
) -> Dict[str, Any]:
    replay_set, guard_set = set(replay_keys), set(guard_keys)
    excitation_ids = sorted({key[2] for key in groups})
    per_excitation: Dict[str, Dict[str, Any]] = {}
    for excitation_id in excitation_ids:
        all_keys = [key for key in groups if key[2] == excitation_id and stable_group_key(key) not in excluded_keys]
        replay_exc = [key for key in replay_keys if key[2] == excitation_id]
        guard_exc = [key for key in guard_keys if key[2] == excitation_id]
        per_excitation[str(excitation_id)] = {
            "group_count": len(all_keys),
            "replay_count": len(replay_exc),
            "guard_count": len(guard_exc),
            "guard_coverage_unavailable": len(all_keys) == 1,
            "H_range": _range([key[0] for key in all_keys]),
            "M_range": _range([key[1] for key in all_keys]),
            "replay_H_range": _range([key[0] for key in replay_exc]),
            "replay_M_range": _range([key[1] for key in replay_exc]),
            "guard_H_range": _range([key[0] for key in guard_exc]),
            "guard_M_range": _range([key[1] for key in guard_exc]),
        }
    replay_rows = sum(len(groups[key]) for key in replay_keys)
    guard_rows = sum(len(groups[key]) for key in guard_keys)
    total_selected = len(replay_keys) + len(guard_keys)
    return {
        "total_history_groups": len(groups),
        "excluded_new_groups": len(excluded_keys),
        "replay_groups": len(replay_keys),
        "guard_groups": len(guard_keys),
        "replay_rows": replay_rows,
        "guard_rows": guard_rows,
        "actual_replay_fraction": len(replay_keys) / max(total_selected, 1),
        "overlap_groups": len(replay_set & guard_set),
        "replay_unique_heights": len({key[0] for key in replay_keys}),
        "guard_unique_heights": len({key[0] for key in guard_keys}),
        "replay_unique_mach": len({key[1] for key in replay_keys}),
        "guard_unique_mach": len({key[1] for key in guard_keys}),
        "per_excitation": per_excitation,
    }


def split_history_groups_for_replay_guard(
    groups: Dict[GroupKey, List[str]],
    *,
    replay_fraction: float = 0.75,
    seed: int = 42,
    excluded_group_keys: Optional[Set[GroupKey]] = None,
) -> Dict[str, Any]:
    """Split complete operating-point groups into mutually exclusive global replay/guard sets."""
    if not 0.0 < float(replay_fraction) < 1.0:
        raise ValueError(f"replay_fraction must satisfy 0 < value < 1, got {replay_fraction}")
    if not groups:
        raise ValueError("History dataset contains no operating-point groups")

    canonical = _canonical_group_map(groups)
    excluded_stable = {
        stable_group_key(key)
        for key in (excluded_group_keys or set())
        if stable_group_key(key) in canonical
    }
    candidate_keys = [key for encoded, key in sorted(canonical.items()) if encoded not in excluded_stable]
    if len(candidate_keys) < 2:
        raise RuntimeError(
            "At least two historical operating-point groups must remain after excluding new data"
        )

    by_excitation: Dict[int, List[GroupKey]] = {}
    for key in candidate_keys:
        by_excitation.setdefault(key[2], []).append(key)

    guard_keys: List[GroupKey] = []
    for excitation_id in sorted(by_excitation):
        stratum = sorted(by_excitation[excitation_id])
        guard_count = _guard_target_count(len(stratum), float(replay_fraction))
        guard_keys.extend(
            _select_guard_for_excitation(stratum, guard_count, int(seed) + excitation_id * 1009)
        )

    guard_set = set(guard_keys)
    replay_keys = [key for key in candidate_keys if key not in guard_set]
    guard_keys = sorted(guard_keys)
    replay_keys = sorted(replay_keys)
    overlap = set(replay_keys) & set(guard_keys)
    if overlap:
        raise RuntimeError(f"Replay/guard group leakage detected: {sorted(overlap)}")
    if not replay_keys or not guard_keys:
        raise RuntimeError("History split must produce non-empty replay and guard sets")

    replay_row_keys = {stable_group_key((row[0], row[1], int(round(row[2])))) for row in _rows_for_keys(groups, replay_keys)}
    guard_row_keys = {stable_group_key((row[0], row[1], int(round(row[2])))) for row in _rows_for_keys(groups, guard_keys)}
    row_overlap = replay_row_keys & guard_row_keys
    if row_overlap:
        raise RuntimeError(f"Replay/guard row-source leakage detected: {sorted(row_overlap)}")

    return {
        "replay_group_keys": replay_keys,
        "guard_group_keys": guard_keys,
        "replay_rows": _rows_for_keys(groups, replay_keys),
        "guard_rows": _rows_for_keys(groups, guard_keys),
        "coverage": _build_coverage(groups, replay_keys, guard_keys, excluded_stable),
    }


def history_dataset_fingerprint(dataset_path: Path, groups: Dict[GroupKey, List[str]]) -> str:
    dataset_path = Path(dataset_path).resolve()
    if dataset_path.is_dir():
        files = sorted(
            (path for path in dataset_path.iterdir() if path.is_file() and SPLIT_DATASET_NAME_RE.match(path.name)),
            key=lambda path: path.name.lower(),
        )
        file_records = [
            {"relative_path": path.relative_to(dataset_path).as_posix(), "size": path.stat().st_size}
            for path in files
        ]
    else:
        file_records = [{"relative_path": dataset_path.name, "size": dataset_path.stat().st_size}]
    group_records = [
        {"key": list(stable_group_key(key)), "rows": len(groups[key])}
        for key in sorted(groups, key=stable_group_key)
    ]
    payload = json.dumps(
        {"files": file_records, "groups": group_records},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _group_hash(keys: Iterable[GroupKey]) -> str:
    payload = json.dumps(
        [list(stable_group_key(key)) for key in sorted(keys, key=stable_group_key)],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_coverage_reports(coverage: Dict[str, Any], manifest_path: Path) -> Tuple[Path, Path]:
    json_path = manifest_path.parent / "replay_guard_coverage.json"
    csv_path = manifest_path.parent / "replay_guard_coverage.csv"
    json_path.write_text(json.dumps(coverage, indent=2, ensure_ascii=False), encoding="utf-8")
    columns = [
        "excitation_id", "group_count", "replay_count", "guard_count",
        "H_range", "M_range", "replay_H_range", "replay_M_range",
        "guard_H_range", "guard_M_range", "guard_coverage_unavailable",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for excitation_id, values in sorted(coverage["per_excitation"].items(), key=lambda item: int(item[0])):
            row = {"excitation_id": excitation_id}
            row.update({name: values.get(name) for name in columns if name != "excitation_id"})
            writer.writerow(row)
    return json_path, csv_path


def _manifest_default_path(history_dataset_path: Path) -> Path:
    path = Path(history_dataset_path)
    return (path if path.is_dir() else path.parent) / ".incremental_replay_guard_split.json"


def _keys_from_manifest(
    encoded_keys: Iterable[Iterable[int]],
    canonical: Dict[StableGroupKey, GroupKey],
    label: str,
) -> List[GroupKey]:
    result: List[GroupKey] = []
    for raw in encoded_keys:
        encoded = tuple(int(value) for value in raw)
        if len(encoded) != 3 or encoded not in canonical:
            raise RuntimeError(f"Split manifest contains unknown {label} group key: {list(raw)}")
        result.append(canonical[encoded])
    return result


def build_replay_data(
    base_dir: Path,
    model_dir: Optional[Path] = None,
    replay_dataset_path: Optional[Path] = None,
    selected_names: Optional[Iterable[str]] = None,
    *,
    structured: bool = False,
    history_dataset_path: Optional[Path] = None,
    replay_fraction: float = 0.75,
    split_seed: int = 42,
    split_manifest_path: Optional[Path] = None,
    rebuild_split: bool = False,
    reuse_split: bool = True,
    **_deprecated_local_selection_options: Any,
) -> Union[List[List[float]], Dict[str, Any]]:
    """Build global replay/guard from one stable, group-wise history split."""
    base_dir = Path(base_dir)
    history_path = Path(history_dataset_path or replay_dataset_path or (
        Path(model_dir) / "real_dataset.txt" if model_dir is not None
        else base_dir.parent / "GENERATION" / "real_dataset.txt"
    )).expanduser().resolve()
    if not history_path.exists():
        raise FileNotFoundError(f"History dataset does not exist: {history_path}")

    targets = (
        load_selected_targets(base_dir / "selected_CG_path.txt")
        if selected_names is None
        else _targets_from_names(selected_names)
    )
    if not targets:
        raise RuntimeError("No usable new-data operating points were provided")
    groups = load_replay_dataset_groups(history_path)
    if not groups:
        raise RuntimeError(f"History dataset contains no usable groups: {history_path}")

    excluded_raw: Set[GroupKey] = {(target[1], target[2], target[4]) for target in targets}
    canonical = _canonical_group_map(groups)
    requested_excluded = {
        stable_group_key(key) for key in excluded_raw if stable_group_key(key) in canonical
    }
    fingerprint = history_dataset_fingerprint(history_path, groups)
    manifest_path = Path(split_manifest_path).expanduser().resolve() if split_manifest_path else _manifest_default_path(history_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any]
    if manifest_path.exists() and not rebuild_split:
        if not reuse_split:
            raise RuntimeError(
                f"Split manifest already exists: {manifest_path}. Use --rebuild_split to replace it."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("version") != 1:
            raise RuntimeError(f"Unsupported split manifest version in {manifest_path}")
        if manifest.get("history_dataset_fingerprint") != fingerprint:
            raise RuntimeError(
                "History dataset fingerprint differs from the saved replay/guard split. "
                "Use --rebuild_split only after confirming the history-data change."
            )
        if not math.isclose(float(manifest.get("replay_fraction")), float(replay_fraction), abs_tol=1e-12):
            raise RuntimeError("Requested replay_fraction differs from the manifest; use --rebuild_split")
        if int(manifest.get("seed")) != int(split_seed):
            raise RuntimeError("Requested split_seed differs from the manifest; use --rebuild_split")
        replay_keys = _keys_from_manifest(manifest.get("replay_group_keys", []), canonical, "replay")
        guard_keys = _keys_from_manifest(manifest.get("guard_group_keys", []), canonical, "guard")
        manifest_excluded = {tuple(int(value) for value in key) for key in manifest.get("excluded_group_keys", [])}
        effective_excluded = manifest_excluded | requested_excluded
        replay_keys = [key for key in replay_keys if stable_group_key(key) not in effective_excluded]
        guard_keys = [key for key in guard_keys if stable_group_key(key) not in effective_excluded]
        print(f"[SPLIT] reused existing manifest: {manifest_path}", flush=True)
    else:
        split = split_history_groups_for_replay_guard(
            groups,
            replay_fraction=float(replay_fraction),
            seed=int(split_seed),
            excluded_group_keys=excluded_raw,
        )
        replay_keys = split["replay_group_keys"]
        guard_keys = split["guard_group_keys"]
        effective_excluded = requested_excluded
        manifest = {
            "version": 1,
            "seed": int(split_seed),
            "replay_fraction": float(replay_fraction),
            "history_dataset_path": str(history_path),
            "history_dataset_fingerprint": fingerprint,
            "excluded_group_keys": [list(key) for key in sorted(effective_excluded)],
            "replay_group_keys": [list(stable_group_key(key)) for key in replay_keys],
            "guard_group_keys": [list(stable_group_key(key)) for key in guard_keys],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[SPLIT] created manifest: {manifest_path}", flush=True)

    overlap = set(replay_keys) & set(guard_keys)
    if overlap:
        raise RuntimeError(f"Replay/guard group leakage detected: {sorted(overlap)}")
    for encoded in effective_excluded:
        if any(stable_group_key(key) == encoded for key in replay_keys + guard_keys):
            raise RuntimeError(f"New-data group leaked into history split: {encoded}")
    if not replay_keys or not guard_keys:
        raise RuntimeError("Effective replay or guard set is empty after excluding new-data groups")

    replay_rows = _rows_for_keys(groups, replay_keys)
    guard_rows = _rows_for_keys(groups, guard_keys)
    replay_row_keys = {stable_group_key((row[0], row[1], int(round(row[2])))) for row in replay_rows}
    guard_row_keys = {stable_group_key((row[0], row[1], int(round(row[2])))) for row in guard_rows}
    row_overlap = replay_row_keys & guard_row_keys
    if row_overlap:
        raise RuntimeError(f"Replay/guard row-source leakage detected: {sorted(row_overlap)}")

    coverage = _build_coverage(groups, replay_keys, guard_keys, effective_excluded)
    coverage["history_dataset_path"] = str(history_path)
    coverage["history_dataset_fingerprint"] = fingerprint
    coverage["replay_fraction_requested"] = float(replay_fraction)
    coverage["split_seed"] = int(split_seed)
    coverage["replay_group_hash"] = _group_hash(replay_keys)
    coverage["guard_group_hash"] = _group_hash(guard_keys)
    coverage_json, coverage_csv = _write_coverage_reports(coverage, manifest_path)

    print(f"[HISTORY] dataset={history_path}", flush=True)
    print(f"[SPLIT] manifest={manifest_path}", flush=True)
    print(f"[SPLIT] replay_fraction={replay_fraction}", flush=True)
    print(f"[SPLIT] replay rows/groups={len(replay_rows)}/{len(replay_keys)}", flush=True)
    print(f"[SPLIT] guard rows/groups={len(guard_rows)}/{len(guard_keys)}", flush=True)
    print(f"[SPLIT] overlap={len(overlap)}", flush=True)
    print(f"[SPLIT] excluded new groups={len(effective_excluded)}", flush=True)
    historical_excitations = {key[2] for key in groups}
    for label, keys in (("replay", replay_keys), ("guard", guard_keys)):
        missing = historical_excitations - {key[2] for key in keys}
        if missing:
            print(f"[WARN] global {label} is missing historical excitation IDs: {sorted(missing)}", flush=True)

    result = {
        "global_replay_rows": replay_rows,
        "global_guard_rows": guard_rows,
        "replay_group_keys": replay_keys,
        "guard_group_keys": guard_keys,
        "split_manifest_path": str(manifest_path),
        "coverage_json_path": str(coverage_json),
        "coverage_csv_path": str(coverage_csv),
        "history_dataset_path": str(history_path),
        "history_dataset_fingerprint": fingerprint,
        "replay_group_hash": coverage["replay_group_hash"],
        "guard_group_hash": coverage["guard_group_hash"],
        "excluded_group_keys": sorted(effective_excluded),
        "coverage": coverage,
        # Temporary compatibility aliases for external callers.
        "local_replay_rows": replay_rows,
        "local_group_keys": replay_keys,
    }
    return result if structured else replay_rows


def main() -> None:
    build_replay_data(Path.cwd())


if __name__ == "__main__":
    main()
