"""File IO helpers for oracle-context evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_utc() -> str:
    """Return a compact UTC timestamp for manifests."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    """Compute a SHA-256 digest for UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Compute a SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file as a list of objects."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            data = json.loads(text)
            if not isinstance(data, dict):
                raise TypeError(f"Expected JSON object at {path}:{line_no}")
            records.append(data)
    return records


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write stable JSON for manifest-like artifacts."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write JSONL records with stable key ordering."""
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_tmp_output_dir(output_dir: Path) -> Path:
    """Create a fresh sibling temporary output directory."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir.parent / f".{output_dir.name}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    return tmp_dir


def replace_output_dir(tmp_dir: Path, output_dir: Path) -> None:
    """Atomically replace the final output directory after successful writes."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    tmp_dir.replace(output_dir)
