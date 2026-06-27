from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
from xml.etree import ElementTree as ET
import zipfile


_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _col_to_index(ref: str) -> int:
    idx = 0
    for ch in ref:
        if not ch.isalpha():
            break
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return max(idx - 1, 0)


def _load_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except KeyError:
        return []

    values: List[str] = []
    for si in root.findall(".//{%s}si" % _MAIN_NS):
        texts = [node.text or "" for node in si.findall(".//{%s}t" % _MAIN_NS)]
        values.append("".join(texts))
    return values


def _get_first_sheet_path(zf: zipfile.ZipFile) -> str:
    wb_root = ET.fromstring(zf.read("xl/workbook.xml"))
    rel_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))

    first_sheet = wb_root.find(".//{%s}sheets/{%s}sheet" % (_MAIN_NS, _MAIN_NS))
    if first_sheet is None:
        raise RuntimeError("xlsx workbook does not contain any sheet")

    rel_id = first_sheet.attrib.get("{%s}id" % _REL_NS)
    if not rel_id:
        raise RuntimeError("xlsx first sheet is missing relationship id")

    for rel in rel_root.findall("{%s}Relationship" % _PKG_REL_NS):
        if rel.attrib.get("Id") != rel_id:
            continue
        target = rel.attrib.get("Target", "")
        if target.startswith("/"):
            return target.lstrip("/")
        if target.startswith("xl/"):
            return target
        return "xl/" + target

    raise RuntimeError("xlsx first sheet relationship target not found")


def _cell_value(cell: ET.Element, shared_strings: List[str]) -> str:
    cell_type = cell.attrib.get("t")

    if cell_type == "inlineStr":
        texts = [node.text or "" for node in cell.findall(".//{%s}t" % _MAIN_NS)]
        return "".join(texts)

    value_node = cell.find("{%s}v" % _MAIN_NS)
    if value_node is None:
        return ""

    raw = value_node.text or ""
    if cell_type == "s":
        idx = int(raw)
        return shared_strings[idx] if 0 <= idx < len(shared_strings) else ""
    return raw


def _parse_float(value: object) -> float:
    text = str(value).strip().replace("\u3000", "").replace("\xa0", "").replace("，", ".")
    if text == "" or text.lower() == "nan":
        raise ValueError("empty numeric value")
    return float(text)


def _required_signal_type_headers() -> Dict[str, str]:
    return {
        "kind": "信号编号",
        "amp": "幅值",
        "f0": "开始频率",
        "f1": "结束频率",
        "dur": "激励时间",
    }


def _load_signal_type_rows_with_pandas(xlsx_path: Path) -> List[Tuple[str, float, float, float, float]]:
    try:
        import pandas as pd
    except Exception:
        return []

    required_headers = _required_signal_type_headers()
    try:
        df = pd.read_excel(xlsx_path, sheet_name=0, dtype=object)
    except Exception:
        return []

    df.columns = [str(col).strip() for col in df.columns]
    if any(name not in df.columns for name in required_headers.values()):
        return []

    rows: List[Tuple[str, float, float, float, float]] = []
    for _, row in df.iterrows():
        kind = str(row[required_headers["kind"]]).strip()
        if not kind or kind.lower() == "nan" or kind == "Unnamed: 0":
            continue
        try:
            amp = _parse_float(row[required_headers["amp"]])
            f0 = _parse_float(row[required_headers["f0"]])
            f1 = _parse_float(row[required_headers["f1"]])
            dur = _parse_float(row[required_headers["dur"]])
        except ValueError:
            continue
        rows.append((kind, amp, f0, f1, dur))
    return rows


def load_signal_type_rows(xlsx_path: Path) -> List[Tuple[str, float, float, float, float]]:
    xlsx_path = Path(xlsx_path)
    rows = _load_signal_type_rows_with_pandas(xlsx_path)
    if rows:
        return rows

    rows: List[Tuple[str, float, float, float, float]] = []
    required_headers = _required_signal_type_headers()

    with zipfile.ZipFile(str(xlsx_path), "r") as zf:
        shared_strings = _load_shared_strings(zf)
        sheet_path = _get_first_sheet_path(zf)
        root = ET.fromstring(zf.read(sheet_path))

    row_nodes = root.findall(".//{%s}sheetData/{%s}row" % (_MAIN_NS, _MAIN_NS))
    if not row_nodes:
        return rows

    header_values = {}
    for cell in row_nodes[0].findall("{%s}c" % _MAIN_NS):
        header_values[_col_to_index(cell.attrib.get("r", ""))] = _cell_value(cell, shared_strings).strip()
    header_index = {name: idx for idx, name in header_values.items() if name}

    missing = [name for name in required_headers.values() if name not in header_index]
    if missing:
        raise KeyError(f"{xlsx_path} missing signal type headers: {', '.join(missing)}")

    kind_col = header_index[required_headers["kind"]]
    amp_col = header_index[required_headers["amp"]]
    f0_col = header_index[required_headers["f0"]]
    f1_col = header_index[required_headers["f1"]]
    dur_col = header_index[required_headers["dur"]]

    skipped_examples = []
    for row in row_nodes[1:]:
        values = {}
        for cell in row.findall("{%s}c" % _MAIN_NS):
            values[_col_to_index(cell.attrib.get("r", ""))] = _cell_value(cell, shared_strings).strip()

        kind = values.get(kind_col, "")
        if not kind or kind.lower() == "nan" or kind == "Unnamed: 0":
            continue

        try:
            amp = _parse_float(values.get(amp_col, ""))
            f0 = _parse_float(values.get(f0_col, ""))
            f1 = _parse_float(values.get(f1_col, ""))
            dur = _parse_float(values.get(dur_col, ""))
        except ValueError:
            if len(skipped_examples) < 5:
                skipped_examples.append(
                    {
                        "row": row.attrib.get("r", ""),
                        "kind": kind,
                        "amp": values.get(amp_col, ""),
                        "f0": values.get(f0_col, ""),
                        "f1": values.get(f1_col, ""),
                        "dur": values.get(dur_col, ""),
                    }
                )
            continue

        rows.append((kind, amp, f0, f1, dur))

    if not rows:
        raise ValueError(
            f"{xlsx_path} contains no valid signal type rows; "
            f"sheet_rows={len(row_nodes)}, headers={header_values}, skipped={skipped_examples}"
        )

    return rows
