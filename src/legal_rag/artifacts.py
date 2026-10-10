"""Canonical artifact I/O and provenance."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, TypeVar

from .contracts import Record

T = TypeVar("T", bound=Record)


def digest(value: bytes | str) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def file_hash(path: Path) -> str:
    return digest(path.read_bytes())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict | Record) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.to_json_record() if isinstance(value, Record) else value
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Record | dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = row.to_json_record() if isinstance(row, Record) else row
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path, model: type[T]) -> list[T]:
    with path.open(encoding="utf-8") as handle:
        return [model.model_validate_json(line) for line in handle if line.strip()]


def code_identity() -> dict:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, True
    files = sorted((root / "src/legal_rag").rglob("*.py"))
    content = "\n".join(f"{p.relative_to(root)}:{file_hash(p)}" for p in files)
    return {"commit": commit, "dirty": dirty, "source_sha256": digest(content)}


def new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Artifact directory already exists: {path}")
    path.mkdir(parents=True)
