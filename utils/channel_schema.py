from __future__ import annotations

import configparser
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from surrogate_model.data.io import SPLIT_EXCITATION_IDS, read_real_dataset


SUPPORTED_POSITIONS = ("JFY", "FXD", "PW")
EXCITATION_NAMES = {value: key for key, value in SPLIT_EXCITATION_IDS.items()}
LEGACY_CHANNEL_NAMES = (
    "LW1", "LW2", "LW3", "LAI", "RW1", "RW2", "RW3", "RAI",
    "LVT1", "LVT2", "LVT3", "LVT4", "LRY", "LRY1", "RVT1", "RVT2",
    "RVT3", "RVT4", "RRY", "RRY1", "LHT1", "LHT4", "LHT2", "LHT3",
    "RHT1", "RHT4", "RHT2", "RHT3", "F1Z", "F1Y", "ZWL1Y", "ZWL1Z",
    "YWL1Y", "YWL1Z",
)
CHANNEL_SCHEMA_FILENAME = "channel_schema.json"


@dataclass(frozen=True)
class ChannelDefinition:
    channel_id: int
    name: str
    excitation_position: Optional[str]
    sim_only_exception: bool


@dataclass(frozen=True)
class ChannelSchema:
    source_excel: str
    output_headers: Tuple[str, ...]
    channels: Tuple[ChannelDefinition, ...]

    @property
    def channel_names(self) -> Tuple[str, ...]:
        return tuple(item.name for item in self.channels)

    @property
    def id_by_name(self) -> Dict[str, int]:
        return {item.name: item.channel_id for item in self.channels}

    @property
    def definition_by_name(self) -> Dict[str, ChannelDefinition]:
        return {item.name: item for item in self.channels}

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "source_excel": self.source_excel,
            "output_headers": list(self.output_headers),
            "channels": [
                {
                    "channel_id": item.channel_id,
                    "name": item.name,
                    "excitation_position": item.excitation_position,
                    "sim_only_exception": item.sim_only_exception,
                }
                for item in self.channels
            ],
        }


def channel_detail_path_from_ini(paths_ini_path: Path) -> Path:
    paths_ini_path = Path(paths_ini_path).expanduser()
    parser = configparser.ConfigParser()
    loaded = parser.read(str(paths_ini_path), encoding="utf-8")
    if not loaded:
        raise FileNotFoundError("Path configuration INI does not exist: {}".format(paths_ini_path))
    section = "CurChannelDetailFile"
    key = "path_{}".format(section).lower()
    try:
        raw_path = parser[section][key].strip()
    except KeyError:
        raise KeyError(
            "{} is missing [{}] / {}".format(paths_ini_path, section, key)
        ) from None
    if not raw_path:
        raise ValueError("{} [{}] / {} is empty".format(paths_ini_path, section, key))
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (paths_ini_path.parent / path).resolve()
    if not path.is_file():
        raise FileNotFoundError("Channel detail workbook does not exist: {}".format(path))
    return path


def _cell_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _supported_position(value: object) -> Optional[str]:
    text = _cell_text(value).upper()
    matched = [position for position in SUPPORTED_POSITIONS if position in text]
    if len(matched) > 1:
        raise ValueError(
            "A channel cannot contain multiple supported excitation positions: {!r}".format(
                value
            )
        )
    return matched[0] if matched else None


def channel_names_present(
    rows: Iterable[Sequence[float]], source_channel_names: Sequence[str]
) -> List[str]:
    present = set()
    for row in rows:
        if len(row) < 4:
            continue
        channel_id = int(round(float(row[3])))
        if 0 <= channel_id < len(source_channel_names):
            present.add(str(source_channel_names[channel_id]))
    return [name for name in source_channel_names if name in present]


def load_channel_schema(
    workbook_path: Path, sim_channel_names: Iterable[str] = ()
) -> ChannelSchema:
    from openpyxl import load_workbook

    workbook_path = Path(workbook_path)
    workbook = load_workbook(str(workbook_path), read_only=True, data_only=True)
    try:
        sheet = workbook.worksheets[0]
        header_cells = next(sheet.iter_rows(min_row=1, max_row=1, values_only=True))
        headers = [_cell_text(value) for value in header_cells]
        try:
            name_column = headers.index("表头（机载）") + 1
        except ValueError:
            name_column = 2
        try:
            position_column = headers.index("相关激励位置") + 1
        except ValueError:
            raise KeyError(
                "{} first sheet is missing column '相关激励位置'".format(workbook_path)
            ) from None

        excel_rows: List[Tuple[str, Optional[str]]] = []
        output_headers: List[str] = []
        seen_names = set()
        max_column = max(name_column, position_column)
        for values in sheet.iter_rows(
            min_row=2, max_col=max_column, values_only=True
        ):
            name = _cell_text(values[name_column - 1])
            if not name:
                continue
            output_headers.append(name)
            if name in seen_names:
                raise ValueError(
                    "{} contains duplicate channel/header name {!r}".format(
                        workbook_path, name
                    )
                )
            seen_names.add(name)
            excel_rows.append((name, _supported_position(values[position_column - 1])))
    finally:
        workbook.close()

    sim_names = {str(name) for name in sim_channel_names}
    normal = [(name, position) for name, position in excel_rows if position is not None]
    exceptions = [
        (name, None)
        for name, position in excel_rows
        if position is None and name in sim_names
    ]
    # Extra simulation columns have no workbook row but must survive remapping.
    exceptions.extend(
        (name, None)
        for name in sorted(sim_names, key=lambda value: int(value.rsplit("_", 1)[-1])
                           if re.fullmatch(r"SIM_EXTRA_\d+", value) else -1)
        if name not in seen_names and re.fullmatch(r"SIM_EXTRA_\d+", name)
    )
    definitions = []
    for channel_id, (name, position) in enumerate(normal + exceptions):
        definitions.append(
            ChannelDefinition(
                channel_id=channel_id,
                name=name,
                excitation_position=position,
                sim_only_exception=position is None,
            )
        )
    return ChannelSchema(
        source_excel=str(workbook_path),
        output_headers=tuple(output_headers),
        channels=tuple(definitions),
    )


def save_channel_schema(schema: ChannelSchema, version_dir: Path) -> Path:
    path = Path(version_dir) / CHANNEL_SCHEMA_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(schema.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_saved_channel_schema(path: Path) -> ChannelSchema:
    path = Path(path)
    if path.is_dir():
        path = path / CHANNEL_SCHEMA_FILENAME
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    channels = tuple(
        ChannelDefinition(
            channel_id=int(item["channel_id"]),
            name=str(item["name"]),
            excitation_position=(
                None
                if item.get("excitation_position") in (None, "")
                else str(item["excitation_position"])
            ),
            sim_only_exception=bool(item.get("sim_only_exception", False)),
        )
        for item in payload.get("channels", [])
    )
    return ChannelSchema(
        source_excel=str(payload.get("source_excel", "")),
        output_headers=tuple(str(value) for value in payload.get("output_headers", [])),
        channels=channels,
    )


def source_channel_names_for_dataset(
    dataset_path: Path, default: Sequence[str] = LEGACY_CHANNEL_NAMES
) -> Tuple[str, ...]:
    dataset_path = Path(dataset_path)
    candidates = []
    if dataset_path.is_dir():
        candidates.extend(
            [
                dataset_path / CHANNEL_SCHEMA_FILENAME,
                dataset_path.parent / CHANNEL_SCHEMA_FILENAME,
            ]
        )
    else:
        candidates.extend(
            [
                dataset_path.parent / CHANNEL_SCHEMA_FILENAME,
                dataset_path.parent.parent / CHANNEL_SCHEMA_FILENAME,
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return load_saved_channel_schema(candidate).channel_names
    return tuple(str(name) for name in default)


def _target_excitation_id(
    source_excitation_id: int, target_position: Optional[str]
) -> Optional[int]:
    source_name = EXCITATION_NAMES.get(int(source_excitation_id))
    if source_name is None:
        return None
    if target_position is None:
        return int(source_excitation_id)
    phase = source_name.rsplit("_", 1)[-1]
    return SPLIT_EXCITATION_IDS.get("{}_{}".format(target_position, phase))


def remap_dataset_rows(
    rows: Iterable[Sequence[float]],
    schema: ChannelSchema,
    source_channel_names: Sequence[str],
    *,
    include_sim_only_exceptions: bool,
    source_label: str,
) -> Tuple[List[List[float]], dict]:
    target_by_name = schema.definition_by_name
    remapped: List[List[float]] = []
    skipped_unknown_channel = 0
    skipped_not_selected = 0
    skipped_sim_only_real = 0
    skipped_unknown_excitation = 0
    for source_row in rows:
        row = [float(value) for value in source_row]
        source_channel_id = int(round(row[3]))
        if not (0 <= source_channel_id < len(source_channel_names)):
            skipped_unknown_channel += 1
            continue
        name = str(source_channel_names[source_channel_id])
        definition = target_by_name.get(name)
        if definition is None:
            skipped_not_selected += 1
            continue
        if definition.sim_only_exception and not include_sim_only_exceptions:
            skipped_sim_only_real += 1
            continue
        target_excitation_id = _target_excitation_id(
            int(round(row[2])), definition.excitation_position
        )
        if target_excitation_id is None:
            skipped_unknown_excitation += 1
            continue
        row[2] = float(target_excitation_id)
        row[3] = float(definition.channel_id)
        remapped.append(row)
    stats = {
        "source": str(source_label),
        "input_rows": len(remapped)
        + skipped_unknown_channel
        + skipped_not_selected
        + skipped_sim_only_real
        + skipped_unknown_excitation,
        "output_rows": len(remapped),
        "skipped_unknown_channel": skipped_unknown_channel,
        "skipped_not_selected": skipped_not_selected,
        "skipped_sim_only_real": skipped_sim_only_real,
        "skipped_unknown_excitation": skipped_unknown_excitation,
    }
    return remapped, stats


def read_dataset_list(list_path: Path) -> List[Path]:
    list_path = Path(list_path).expanduser()
    if not list_path.is_file():
        raise FileNotFoundError("Training dataset list does not exist: {}".format(list_path))
    paths = []
    for line_number, raw_line in enumerate(
        list_path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        value = raw_line.strip()
        if not value or value.startswith("#"):
            continue
        value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        if not value:
            continue
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = (list_path.parent / candidate).resolve()
        if not candidate.exists():
            raise FileNotFoundError(
                "{}:{} dataset path does not exist: {}".format(
                    list_path, line_number, candidate
                )
            )
        if not candidate.is_file() and not candidate.is_dir():
            raise ValueError(
                "{}:{} dataset path is not a file or directory: {}".format(
                    list_path, line_number, candidate
                )
            )
        paths.append(candidate)
    return paths


def load_and_remap_datasets(
    dataset_paths: Iterable[Path],
    schema: ChannelSchema,
    *,
    include_sim_only_exceptions: bool,
    allow_duplicate_keys: bool,
) -> Tuple[List[List[float]], List[dict]]:
    combined: List[List[float]] = []
    reports = []
    for dataset_path in dataset_paths:
        dataset_path = Path(dataset_path)
        raw_rows = read_real_dataset(
            dataset_path, allow_duplicate_keys=allow_duplicate_keys
        )
        source_names = source_channel_names_for_dataset(dataset_path)
        rows, report = remap_dataset_rows(
            raw_rows,
            schema,
            source_names,
            include_sim_only_exceptions=include_sim_only_exceptions,
            source_label=str(dataset_path),
        )
        combined.extend(rows)
        reports.append(report)
    return combined, reports


def print_remap_reports(reports: Iterable[Mapping[str, object]]) -> None:
    for report in reports:
        print(
            "[CHANNEL-MAP] source={source} input={input_rows} output={output_rows} "
            "skip_unknown={skipped_unknown_channel} "
            "skip_not_selected={skipped_not_selected} "
            "skip_sim_only_real={skipped_sim_only_real} "
            "skip_excitation={skipped_unknown_excitation}".format(**report),
            flush=True,
        )
