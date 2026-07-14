from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from surrogate_model.utils.model_artifacts import atomic_write_json, load_json


SCHEMA_VERSION = 1
MODEL_INVENTORY_NAME = "models_manifest.json"
TRAINING_FAILURES_NAME = "training_failures.json"
EXCITATION_NAMES = {
    0: "FXD_FDC",
    1: "FXD_DC",
    2: "PW_FDC",
    3: "PW_DC",
    4: "JFY_FDC",
    5: "JFY_DC",
}

CHANNEL_NAMES = (
    "LW1", "LW2", "LW3", "LAI", "RW1", "RW2", "RW3", "RAI",
    "LVT1", "LVT2", "LVT3", "LVT4", "LRY", "LRY1", "RVT1", "RVT2",
    "RVT3", "RVT4", "RRY", "RRY1", "LHT1", "LHT4", "LHT2", "LHT3",
    "RHT1", "RHT4", "RHT2", "RHT3", "F1Z", "F1Y", "ZWL1Y", "ZWL1Z",
    "YWL1Y", "YWL1Z",
)


def excitation_name(excitation_id: int) -> str:
    excitation_id = int(excitation_id)
    return EXCITATION_NAMES.get(excitation_id, f"EXC_{excitation_id}")


def channel_name(channel_id: int) -> str:
    channel_id = int(channel_id)
    if 0 <= channel_id < len(CHANNEL_NAMES):
        return CHANNEL_NAMES[channel_id]
    return f"channel_{channel_id:02d}"


def combination_from_row(row) -> tuple[int, int]:
    return int(row.exc_id), int(row.ch_id)


def combinations_in_rows(rows: Iterable) -> list[tuple[int, int]]:
    return sorted({combination_from_row(row) for row in rows})


def combination_dir(version_dir: Path, excitation_id: int, channel_id: int) -> Path:
    return Path(version_dir) / excitation_name(excitation_id) / channel_name(channel_id)


def combination_relative_dir(excitation_id: int, channel_id: int) -> str:
    return f"{excitation_name(excitation_id)}/{channel_name(channel_id)}"


def model_entry(
    *,
    excitation_id: int,
    channel_id: int,
    row_count: int,
    status: str,
    source_model_version: str | None = None,
) -> dict:
    relative_dir = combination_relative_dir(excitation_id, channel_id)
    entry = {
        "excitation_id": int(excitation_id),
        "excitation_name": excitation_name(excitation_id),
        "channel_id": int(channel_id),
        "channel_name": channel_name(channel_id),
        "relative_dir": relative_dir,
        "checkpoint": f"{relative_dir}/model.pth",
        "normalizer_input": f"{relative_dir}/Normalizer_Input.pth",
        "normalizer_output": f"{relative_dir}/Normalizer_Output.pth",
        "training_config": f"{relative_dir}/training_config.json",
        "train_log": f"{relative_dir}/train_log.csv",
        "row_count": int(row_count),
        "status": str(status),
    }
    if source_model_version:
        entry["source_model_version"] = str(source_model_version)
    return entry


def write_model_inventory(
    version_dir: Path,
    *,
    model_version: str,
    models: Iterable[Mapping],
    failures: Iterable[Mapping],
    parent_model_version: str | None = None,
) -> dict:
    models = sorted(
        (dict(item) for item in models),
        key=lambda item: (int(item["excitation_id"]), int(item["channel_id"])),
    )
    failures = [dict(item) for item in failures]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "layout": "one_model_per_excitation_channel",
        "model_version": str(model_version),
        "parent_model_version": parent_model_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_count": len(models),
        "failure_count": len(failures),
        "models": models,
    }
    version_dir = Path(version_dir)
    atomic_write_json(version_dir / MODEL_INVENTORY_NAME, payload)
    atomic_write_json(
        version_dir / TRAINING_FAILURES_NAME,
        {"model_version": str(model_version), "failures": failures},
    )
    return payload


def load_model_inventory(version_dir: Path) -> dict:
    path = Path(version_dir) / MODEL_INVENTORY_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Excitation/channel model inventory is missing: {path}. "
            "Create a base model version with the excitation-channel training branch first."
        )
    payload = load_json(path)
    if payload.get("layout") != "one_model_per_excitation_channel":
        raise ValueError(f"Unsupported model inventory layout in {path}: {payload.get('layout')!r}")
    return payload


def inventory_index(inventory: Mapping) -> dict[tuple[int, int], dict]:
    return {
        (int(item["excitation_id"]), int(item["channel_id"])): dict(item)
        for item in inventory.get("models", [])
    }


def copy_parent_model_directories(
    parent_version_dir: Path,
    new_version_dir: Path,
    parent_inventory: Mapping,
) -> list[dict]:
    copied = []
    for item in parent_inventory.get("models", []):
        source = Path(parent_version_dir) / str(item["relative_dir"])
        destination = Path(new_version_dir) / str(item["relative_dir"])
        if not source.is_dir():
            raise FileNotFoundError(f"Parent channel-model directory is missing: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
        copied_item = dict(item)
        copied_item["status"] = "inherited"
        copied_item["source_model_version"] = str(parent_inventory.get("model_version", ""))
        copied.append(copied_item)
    return copied
