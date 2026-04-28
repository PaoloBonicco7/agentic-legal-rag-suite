from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import pandas as pd


class HandoffValidationResult(TypedDict):
    contract_df: pd.DataFrame
    ready_to_embedding: bool
    checked_chunk_rows: int
    missing_chunk_fields: list[str]
    handoff_ok: bool


def validate_handoff_for_notebook_04(dataset_dir: Path, *, chunk_sample_size: int = 200) -> HandoffValidationResult:
    required_files = [
        "manifest.json",
        "laws.jsonl",
        "articles.jsonl",
        "notes.jsonl",
        "edges.jsonl",
        "events.jsonl",
        "chunks.jsonl",
    ]

    rows = []
    for name in required_files:
        path = dataset_dir / name
        rows.append({"file": name, "exists": path.exists(), "path": str(path)})
    contract_df = pd.DataFrame(rows)

    manifest = {}
    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    ready_flag = bool(manifest.get("ready_to_embedding"))

    required_chunk_fields = [
        "chunk_id",
        "passage_id",
        "article_id",
        "law_id",
        "chunk_seq",
        "text",
        "text_for_embedding",
        "law_status",
        "article_is_abrogated",
        "passage_label",
        "related_law_ids",
        "relation_types",
        "inbound_law_ids",
        "outbound_law_ids",
        "index_views",
    ]

    checked_rows = 0
    missing_chunk_fields: set[str] = set()
    chunks_path = dataset_dir / "chunks.jsonl"
    if chunks_path.exists():
        with chunks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                checked_rows += 1
                for field in required_chunk_fields:
                    if field not in record:
                        missing_chunk_fields.add(field)
                if checked_rows >= int(chunk_sample_size):
                    break

    missing_sorted = sorted(missing_chunk_fields)
    all_files_ok = bool(contract_df["exists"].all()) if not contract_df.empty else False
    handoff_ok = bool(all_files_ok and ready_flag and not missing_sorted)

    return {
        "contract_df": contract_df,
        "ready_to_embedding": ready_flag,
        "checked_chunk_rows": checked_rows,
        "missing_chunk_fields": missing_sorted,
        "handoff_ok": handoff_ok,
    }
