from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import torch

from surrogate_model.data.io import SPLIT_DATASET_NAME_RE, SPLIT_EXCITATION_IDS, read_real_dataset


SCHEMA_VERSION = 1
EXCITATION_NAMES = {value: key for key, value in SPLIT_EXCITATION_IDS.items()}


def group_key_from_row(row: Sequence[float]) -> Tuple[int, int, int]:
    return int(round(float(row[0]) * 100)), int(round(float(row[1]) * 100)), int(round(float(row[2])))


def group_key_from_parsed(row) -> Tuple[int, int, int]:
    return int(round(float(row.H) * 100)), int(round(float(row.M) * 100)), int(row.exc_id)


def group_rows(
    rows: Iterable[Sequence[float]], *, source: str, allow_duplicate_keys: bool = False
) -> Dict[Tuple[int, int, int], List[List[float]]]:
    grouped: Dict[Tuple[int, int, int], List[List[float]]] = {}
    seen: Set[Tuple[int, int, int, int]] = set()
    for row in rows:
        if len(row) < 4:
            raise ValueError(f"{source}: data row has fewer than four columns")
        key = group_key_from_row(row)
        channel = int(round(float(row[3])))
        sample_key = (*key, channel)
        if sample_key in seen and not allow_duplicate_keys:
            raise ValueError(f"{source}: duplicate group/channel row {sample_key}")
        seen.add(sample_key)
        grouped.setdefault(key, []).append([float(value) for value in row])
    if not grouped:
        raise ValueError(f"{source}: dataset contains no usable rows")
    for values in grouped.values():
        values.sort(key=lambda row: int(round(row[3])))
    return grouped


def load_dataset_rows(path: Path) -> List[List[float]]:
    return [list(row) for row in read_real_dataset(Path(path))]


def merge_dataset_rows(
    parent_rows: Iterable[Sequence[float]], correction_rows: Iterable[Sequence[float]]
) -> Tuple[List[List[float]], dict]:
    parent = group_rows(parent_rows, source="parent dataset")
    correction = group_rows(correction_rows, source="correction dataset")
    replaced: List[Tuple[int, int, int]] = []
    added: List[Tuple[int, int, int]] = []
    changes: List[dict] = []
    merged = {key: [list(row) for row in rows] for key, rows in parent.items()}
    expected_channels_by_excitation: Dict[int, Set[int]] = {}
    for excitation_id in {key[2] for key in parent}:
        channel_sets = [
            frozenset(int(round(row[3])) for row in values)
            for key, values in parent.items()
            if key[2] == excitation_id
        ]
        if channel_sets:
            expected_channels_by_excitation[excitation_id] = set(Counter(channel_sets).most_common(1)[0][0])
    for key, new_rows in correction.items():
        new_channels = {int(round(row[3])) for row in new_rows}
        if key in parent:
            old_channels = {int(round(row[3])) for row in parent[key]}
            if new_channels != old_channels:
                raise ValueError(
                    f"Correction group {list(key)} is incomplete: channels={sorted(new_channels)}, "
                    f"expected complete parent channels={sorted(old_channels)}"
                )
            replaced.append(key)
            changes.append(
                {
                    "group_key": list(key),
                    "old_sha256": group_content_sha256(parent[key]),
                    "new_sha256": group_content_sha256(new_rows),
                    "reason": "selected operating-point correction",
                }
            )
        else:
            expected_channels = expected_channels_by_excitation.get(key[2])
            if expected_channels is not None and new_channels != expected_channels:
                raise ValueError(
                    f"Added correction group {list(key)} is incomplete: channels={sorted(new_channels)}, "
                    f"expected excitation channel set={sorted(expected_channels)}"
                )
            added.append(key)
            changes.append(
                {
                    "group_key": list(key),
                    "new_sha256": group_content_sha256(new_rows),
                    "reason": "selected operating-point addition",
                }
            )
        merged[key] = [list(row) for row in new_rows]
    flattened = [row for key in sorted(merged) for row in merged[key]]
    group_rows(flattened, source="merged dataset")
    return flattened, {
        "correction_group_keys": sorted(correction),
        "replaced_group_keys": sorted(replaced),
        "added_group_keys": sorted(added),
        "change_records": changes,
    }


def _stable_number(value: float) -> str:
    if value != value:
        return "nan"
    return format(float(value), ".17g")


def _stable_rows_bytes(rows: Iterable[Sequence[float]]) -> bytes:
    ordered = sorted(rows, key=lambda row: int(round(float(row[3]))))
    return ("\n".join("\t".join(_stable_number(value) for value in row) for row in ordered) + "\n").encode("utf-8")


def group_content_sha256(rows: Iterable[Sequence[float]]) -> str:
    return "sha256:" + hashlib.sha256(_stable_rows_bytes(rows)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def load_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _snapshot_filename(key: Tuple[int, int, int]) -> str:
    h_code, m_code, excitation_id = key
    name = EXCITATION_NAMES.get(excitation_id, f"EXC_{excitation_id}")
    return f"SF_XGD_{name}_H{h_code:04d}_M{m_code:03d}.txt"


def build_dataset_snapshot(
    rows: Iterable[Sequence[float]], snapshot_dir: Path
) -> Dict[Tuple[int, int, int], Path]:
    snapshot_dir = Path(snapshot_dir)
    grouped = group_rows(rows, source="snapshot source")
    snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{snapshot_dir.name}.", dir=str(snapshot_dir.parent)))
    try:
        result: Dict[Tuple[int, int, int], Path] = {}
        for key, values in sorted(grouped.items()):
            destination = temp_dir / _snapshot_filename(key)
            destination.write_bytes(_stable_rows_bytes(values))
            result[key] = destination
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
        os.replace(temp_dir, snapshot_dir)
        return {key: snapshot_dir / path.name for key, path in result.items()}
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def build_input_dataset_snapshot(
    input_path: Path, snapshot_dir: Path, *, allow_duplicate_keys: bool = False
) -> Dict[Tuple[int, int, int], List[str]]:
    """Copy the exact files used by base training and index groups to relative snapshot files."""
    input_path = Path(input_path)
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{snapshot_dir.name}.", dir=str(snapshot_dir.parent)))
    relative_files: Dict[Tuple[int, int, int], List[str]] = {}
    try:
        if input_path.is_file():
            sources = [(input_path, "real_dataset.txt")]
        else:
            sources = [
                (path, path.name)
                for path in sorted(input_path.iterdir(), key=lambda item: item.name.lower())
                if path.is_file() and SPLIT_DATASET_NAME_RE.match(path.name)
            ]
        if not sources:
            raise FileNotFoundError(f"No fixed-SF dataset files found in: {input_path}")
        for source, destination_name in sources:
            shutil.copy2(source, temp_dir / destination_name)
            file_groups = group_rows(
                read_real_dataset(source, allow_duplicate_keys=allow_duplicate_keys),
                source=str(source),
                allow_duplicate_keys=allow_duplicate_keys,
            )
            for key in file_groups:
                relative_files.setdefault(key, []).append(
                    f"{snapshot_dir.name}/{destination_name}"
                )
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
        os.replace(temp_dir, snapshot_dir)
        return relative_files
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def dataset_fingerprint(groups: Sequence[Mapping], splits: Mapping) -> str:
    stable = {
        "groups": [
            {
                "group_key": group["group_key"],
                "sha256": group["sha256"],
                "rows": group["rows"],
            }
            for group in sorted(groups, key=lambda item: item["group_key"])
        ],
        "train_group_keys": sorted(splits.get("train_group_keys", [])),
        "validation_group_keys": sorted(splits.get("validation_group_keys", [])),
    }
    encoded = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def build_dataset_manifest(
    *,
    rows: Iterable[Sequence[float]],
    model_version: str,
    snapshot_dir: Path,
    train_group_keys: Iterable[Sequence[int]],
    validation_group_keys: Iterable[Sequence[int]],
    correction_group_keys: Iterable[Sequence[int]] = (),
    parent_manifest: Optional[Mapping] = None,
    parent_model_version: Optional[str] = None,
    source_types: Optional[Mapping[Tuple[int, int, int], str]] = None,
    changes: Optional[Mapping] = None,
    split_strategy: str = "",
    relative_files_by_group: Optional[
        Mapping[Tuple[int, int, int], Sequence[str]]
    ] = None,
    allow_duplicate_keys: bool = False,
) -> dict:
    grouped = group_rows(
        rows,
        source="manifest dataset",
        allow_duplicate_keys=allow_duplicate_keys,
    )
    source_types = source_types or {}
    groups = []
    for key, values in sorted(grouped.items()):
        file_path = Path(snapshot_dir) / _snapshot_filename(key)
        groups.append(
            {
                "group_key": list(key),
                "height_km": float(values[0][0]),
                "mach": float(values[0][1]),
                "excitation_id": int(key[2]),
                "excitation_name": EXCITATION_NAMES.get(key[2], f"EXC_{key[2]}"),
                "relative_files": list(
                    (relative_files_by_group or {}).get(key, [f"dataset_snapshot/{file_path.name}"])
                ),
                "rows": len(values),
                "channels": len({int(round(row[3])) for row in values}),
                "sha256": group_content_sha256(values),
                "source_type": source_types.get(key, "original"),
            }
        )
    correction = sorted([list(map(int, key)) for key in correction_group_keys])
    train = sorted([list(map(int, key)) for key in train_group_keys])
    validation = sorted([list(map(int, key)) for key in validation_group_keys])
    removed_from_validation = sorted(key for key in correction if key in validation)
    split_changes = {
        "correction_groups_forced_to_train": correction,
        "removed_from_validation": removed_from_validation,
    }
    split_payload = {
        "split_strategy": split_strategy,
        "train_group_keys": train,
        "validation_group_keys": validation,
        "correction_group_keys": correction,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": f"{model_version}_dataset",
        "parent_dataset_version": None if parent_manifest is None else parent_manifest.get("dataset_version"),
        "created_from_model": parent_model_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_root": "dataset_snapshot",
        "group_key_definition": ["height_code", "mach_code", "excitation_id"],
        "groups": groups,
        "splits": split_payload,
        "split_changes": split_changes,
        "changes": changes or {"replaced_groups": [], "added_groups": [], "removed_groups": []},
    }
    manifest["dataset_fingerprint"] = dataset_fingerprint(groups, split_payload)
    return manifest


def read_code_metadata(repo_dir: Path) -> dict:
    def git_value(*args: str):
        try:
            return subprocess.check_output(
                ["git", *args], cwd=str(repo_dir), stderr=subprocess.DEVNULL, text=True, timeout=5
            ).strip() or None
        except Exception as exc:
            print(f"[WARN] unable to read Git metadata: {exc}", flush=True)
            return None

    return {
        "git_branch": git_value("branch", "--show-current"),
        "git_commit": git_value("rev-parse", "HEAD"),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }


def resolve_parent_dataset(
    base_version_dir: Path, history_dataset: Optional[Path]
) -> Tuple[Path, Optional[dict]]:
    snapshot = Path(base_version_dir) / "dataset_snapshot"
    manifest_path = Path(base_version_dir) / "dataset_manifest.json"
    if snapshot.is_dir() and manifest_path.is_file():
        return snapshot, load_json(manifest_path)
    if history_dataset is None:
        raise FileNotFoundError(
            "Parent model has no dataset snapshot. Supply --history_dataset or "
            "Config.incremental_history_dataset_dir for one legacy migration."
        )
    history_dataset = Path(history_dataset)
    if not history_dataset.exists():
        raise FileNotFoundError(f"Legacy history dataset does not exist: {history_dataset}")
    print(
        "[LEGACY-DATA] Parent model has no dataset snapshot.\n"
        f"Using external history dataset: {history_dataset}",
        flush=True,
    )
    return history_dataset, None


def validate_new_version_directory(base_version_dir: Path, new_version_dir: Path) -> None:
    """Refuse correction targets that could overwrite a parent or an existing version."""
    base_version_dir = Path(base_version_dir).resolve()
    new_version_dir = Path(new_version_dir).resolve()
    if new_version_dir == base_version_dir:
        raise ValueError("new_model_version must differ from base_model_version")
    if new_version_dir.exists():
        raise FileExistsError(
            f"New model version already exists; refusing to overwrite version artifacts: {new_version_dir}"
        )
