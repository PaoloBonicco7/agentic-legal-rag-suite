"""Small IO helpers for indexing artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def now_utc() -> str:
    """Return an ISO UTC timestamp for manifest-like artifacts."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    """Compute a SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, data: Any) -> None:
    """Write stable, sorted JSON."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write JSONL records in the provided order."""
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def prepare_run_dir(root: Path, run_id: str) -> Path:
    """Create a fresh run directory and reject accidental run-id reuse."""
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Indexing run directory already exists: {run_dir}")
    tmp_dir = root / f".{run_id}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    return tmp_dir


def finalize_run_dir(tmp_dir: Path, run_dir: Path) -> None:
    """Move a completed temporary run directory into place."""
    tmp_dir.replace(run_dir)
