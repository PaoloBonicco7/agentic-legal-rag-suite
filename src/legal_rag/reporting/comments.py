"""Durable interpretation notes stored independently of generated reports."""

import fcntl
import json
import os
from pathlib import Path

from pydantic import Field

from legal_rag.artifacts import read_jsonl
from legal_rag.contracts import Record


class ReportComment(Record):
    comment_id: str = Field(min_length=1)
    author: str = Field(min_length=1)
    text: str = Field(min_length=1)
    qid: str | None = None
    run_hash: str | None = None
    document_id: str | None = None


def _unique(rows):
    if len({row.comment_id for row in rows}) != len(rows):
        raise ValueError("Report comment IDs must be unique")
    return rows


def load_comments(path: Path | str) -> list[ReportComment]:
    return _unique(read_jsonl(Path(path), ReportComment))


def write_comments(path: Path | str, comments, *, append: bool = False) -> Path:
    rows = _unique([ReportComment.model_validate(row) for row in comments])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+" if append else "x", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if append:
            handle.seek(0)
            existing = [ReportComment.model_validate_json(line) for line in handle if line.strip()]
            _unique([*existing, *rows])
        handle.seek(0, os.SEEK_END)
        for row in rows:
            handle.write(json.dumps(row.to_json_record(), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path
