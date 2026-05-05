"""Validation and loading helpers for the clean legal dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import iter_jsonl, read_json
from .models import LIST_PAYLOAD_FIELDS, SOURCE_CHUNK_REQUIRED_FIELDS

REQUIRED_DATASET_FILES = ("manifest.json", "chunks.jsonl", "laws.jsonl", "articles.jsonl", "edges.jsonl")


@dataclass(frozen=True)
class DatasetValidationResult:
    """Outcome of validating the clean dataset before indexing."""

    dataset_dir: Path
    required_files: dict[str, bool]
    counts: dict[str, int]
    missing_chunk_fields: dict[str, int]
    duplicate_chunk_ids: tuple[str, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Return whether the dataset satisfies the indexing input contract."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize validation details for artifacts."""
        return {
            "dataset_dir": str(self.dataset_dir),
            "required_files": self.required_files,
            "counts": self.counts,
            "missing_chunk_fields": self.missing_chunk_fields,
            "duplicate_chunk_ids": list(self.duplicate_chunk_ids),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "ok": self.ok,
        }


def read_manifest(dataset_dir: str | Path) -> dict[str, Any]:
    """Read the clean dataset manifest."""
    return read_json(Path(dataset_dir) / "manifest.json")


def load_chunks(dataset_dir: str | Path) -> list[dict[str, Any]]:
    """Load clean chunk records in file order."""
    return list(iter_jsonl(Path(dataset_dir) / "chunks.jsonl"))


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def validate_clean_dataset(dataset_dir: str | Path, *, strict: bool = True) -> DatasetValidationResult:
    """Validate the source dataset contract required by indexing."""
    root = Path(dataset_dir).resolve()
    required_files = {name: (root / name).exists() for name in REQUIRED_DATASET_FILES}
    errors: list[str] = []
    warnings: list[str] = []

    for name, exists in required_files.items():
        if not exists:
            errors.append(f"Missing required dataset file: {name}")

    manifest: dict[str, Any] = {}
    if required_files["manifest.json"]:
        manifest = read_manifest(root)
        if manifest.get("ready_for_indexing") is not True:
            message = "manifest.json does not expose ready_for_indexing=true"
            if strict:
                errors.append(message)
            else:
                warnings.append(message)

    counts = {
        "laws": _count_jsonl(root / "laws.jsonl"),
        "articles": _count_jsonl(root / "articles.jsonl"),
        "edges": _count_jsonl(root / "edges.jsonl"),
        "chunks": _count_jsonl(root / "chunks.jsonl"),
    }
    manifest_counts = manifest.get("counts") if isinstance(manifest, dict) else {}
    if isinstance(manifest_counts, dict):
        for name in ("laws", "articles", "edges", "chunks"):
            expected = manifest_counts.get(name)
            if isinstance(expected, int) and expected != counts[name]:
                errors.append(f"{name}.jsonl count mismatch: manifest={expected}, actual={counts[name]}")

    missing_chunk_fields = {field: 0 for field in sorted(SOURCE_CHUNK_REQUIRED_FIELDS)}
    duplicate_chunk_ids: list[str] = []
    seen_chunk_ids: set[str] = set()
    chunks_path = root / "chunks.jsonl"
    if chunks_path.exists():
        for record in iter_jsonl(chunks_path):
            chunk_id = str(record.get("chunk_id") or "")
            if not chunk_id:
                missing_chunk_fields["chunk_id"] += 1
            elif chunk_id in seen_chunk_ids:
                duplicate_chunk_ids.append(chunk_id)
            else:
                seen_chunk_ids.add(chunk_id)

            for field in SOURCE_CHUNK_REQUIRED_FIELDS:
                value = record.get(field)
                if field in LIST_PAYLOAD_FIELDS:
                    if not isinstance(value, list):
                        missing_chunk_fields[field] += 1
                    continue
                if value is None:
                    missing_chunk_fields[field] += 1
                elif isinstance(value, str) and field != "structure_path" and not value.strip():
                    missing_chunk_fields[field] += 1

    for field, missing in missing_chunk_fields.items():
        if missing:
            errors.append(f"chunks.jsonl has {missing} records with missing/invalid {field!r}")
    if duplicate_chunk_ids:
        errors.append(f"Duplicate chunk_id values found: {len(set(duplicate_chunk_ids))}")
    if counts["chunks"] == 0:
        errors.append("chunks.jsonl is empty")

    return DatasetValidationResult(
        dataset_dir=root,
        required_files=required_files,
        counts=counts,
        missing_chunk_fields=missing_chunk_fields,
        duplicate_chunk_ids=tuple(sorted(set(duplicate_chunk_ids))),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
