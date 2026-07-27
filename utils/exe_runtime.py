"""Shared executable runtime behavior for Win7/Python 3.8 entrypoints."""

from __future__ import annotations

import io
import os
import sys
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Sequence, TextIO, Tuple
from xml.sax.saxutils import escape


USER_INPUT_FLAGS = ("--user_input", "--user-input")


def _xml_safe_text(text: str) -> str:
    return "".join(
        character
        for character in text
        if character in "\t\n\r"
        or 0x20 <= ord(character) <= 0xD7FF
        or 0xE000 <= ord(character) <= 0xFFFD
        or 0x10000 <= ord(character) <= 0x10FFFF
    )


def split_user_input_arg(
    argv: Optional[Sequence[str]], program_dir: Path
) -> Tuple[Path, List[str]]:
    """Resolve and remove the optional user-input argument."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    remaining: List[str] = []
    selected: Optional[str] = None
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in USER_INPUT_FLAGS:
            if index + 1 >= len(arguments):
                raise ValueError("{} requires a file path".format(argument))
            selected = arguments[index + 1]
            index += 2
            continue
        matched = False
        for flag in USER_INPUT_FLAGS:
            prefix = flag + "="
            if argument.startswith(prefix):
                selected = argument[len(prefix):]
                matched = True
                break
        if not matched:
            remaining.append(argument)
        index += 1

    if selected is None:
        return Path(program_dir) / "user_input.dat", remaining
    if not selected.strip():
        raise ValueError("--user_input path must not be empty")
    path = Path(selected).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path, remaining


class _TeeStream:
    def __init__(self, original: TextIO, capture: io.StringIO):
        self.original = original
        self.capture = capture
        self.encoding = getattr(original, "encoding", "utf-8")
        self.errors = getattr(original, "errors", "replace")

    def write(self, text: str) -> int:
        self.original.write(text)
        self.capture.write(text)
        return len(text)

    def flush(self) -> None:
        self.original.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.original, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self.original.fileno()


def _paragraph_xml(text: str, *, bold: bool = False, monospace: bool = False) -> str:
    properties = ""
    if bold or monospace:
        run_properties = []
        if bold:
            run_properties.append("<w:b/>")
        if monospace:
            run_properties.append(
                '<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" '
                'w:eastAsia="SimSun"/>'
            )
        properties = "<w:rPr>{}</w:rPr>".format("".join(run_properties))
    return (
        '<w:p><w:r>{}<w:t xml:space="preserve">{}</w:t>'
        "</w:r></w:p>"
    ).format(properties, escape(_xml_safe_text(text)))


def _write_docx_log(
    path: Path,
    program_name: str,
    started_at: datetime,
    finished_at: datetime,
    exit_code: int,
    user_input_path: Path,
    command_args: Sequence[str],
    captured_text: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        _paragraph_xml("代理模型运行日志", bold=True),
        _paragraph_xml("程序：{}".format(program_name)),
        _paragraph_xml("开始时间：{}".format(started_at.strftime("%Y-%m-%d %H:%M:%S"))),
        _paragraph_xml("结束时间：{}".format(finished_at.strftime("%Y-%m-%d %H:%M:%S"))),
        _paragraph_xml("退出码：{}".format(exit_code)),
        _paragraph_xml("user_input：{}".format(user_input_path)),
        _paragraph_xml("参数：{}".format(" ".join(command_args))),
        _paragraph_xml("控制台输出", bold=True),
    ]
    lines = captured_text.splitlines()
    body.extend(_paragraph_xml(line, monospace=True) for line in (lines or [""]))
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/'
        'wordprocessingml/2006/main"><w:body>{}'
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" '
        'w:left="1134" w:header="708" w:footer="708" w:gutter="0"/>'
        "</w:sectPr></w:body></w:document>"
    ).format("".join(body))
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    relationships = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/></Relationships>'
    )
    with zipfile.ZipFile(str(path), "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types.encode("utf-8"))
        archive.writestr("_rels/.rels", relationships.encode("utf-8"))
        archive.writestr("word/document.xml", document_xml.encode("utf-8"))


def _safe_model_tag(user_input_path: Path) -> str:
    try:
        lines = user_input_path.read_text(encoding="utf-8").splitlines()
        raw = lines[1].strip() if len(lines) >= 2 else ""
    except Exception:
        raw = ""
    cleaned = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in raw
    ).strip("._")
    return cleaned[:60] or "unknown_model"


def run_executable(
    main_function: Callable[[List[str]], Optional[int]],
    program_name: str,
    program_dir: Path,
    argv: Optional[Sequence[str]] = None,
) -> int:
    """Run an EXE main function with Word logging and stable exit codes."""

    command_args = list(sys.argv[1:] if argv is None else argv)
    started_at = datetime.now()
    capture = io.StringIO()
    original_stdout, original_stderr = sys.stdout, sys.stderr
    exit_code = 0
    try:
        user_input_path, _remaining = split_user_input_arg(command_args, program_dir)
    except Exception:
        user_input_path = Path(program_dir) / "user_input.dat"

    sys.stdout = _TeeStream(original_stdout, capture)
    sys.stderr = _TeeStream(original_stderr, capture)
    try:
        print("[RUNTIME] user_input={}".format(user_input_path), flush=True)
        result = main_function(command_args)
        exit_code = int(result or 0)
    except SystemExit as exc:
        code = 0 if exc.code is None else int(exc.code)
        exit_code = 0 if code == 0 else -1
    except KeyboardInterrupt:
        print("[RUNTIME-ERROR] interrupted by user", file=sys.stderr, flush=True)
        exit_code = -1
    except Exception:
        traceback.print_exc()
        exit_code = -1
    finally:
        finished_at = datetime.now()
        print("[RUNTIME] exit_code={}".format(exit_code), flush=True)
        sys.stdout, sys.stderr = original_stdout, original_stderr

    tag = _safe_model_tag(user_input_path)
    timestamp = started_at.strftime("%Y%m%d_%H%M%S_%f")
    log_path = user_input_path.parent / "logs" / (
        "{}_{}_{}_{}.docx".format(
            program_name, tag, timestamp, os.getpid()
        )
    )
    try:
        _write_docx_log(
            log_path,
            program_name,
            started_at,
            finished_at,
            exit_code,
            user_input_path,
            command_args,
            capture.getvalue(),
        )
        print("[RUNTIME] Word log={}".format(log_path), flush=True)
    except Exception:
        traceback.print_exc(file=original_stderr)
        if exit_code == 0:
            exit_code = -1
    return exit_code
