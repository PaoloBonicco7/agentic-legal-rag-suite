from __future__ import annotations

import json
from pathlib import Path


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def qa_artifacts(*, out_dir: Path, suspicious_note_chars: int = 5000) -> dict:
    """
    Minimal QA checks for ingestion outputs.
    """
    paths = {
        "laws": out_dir / "laws.jsonl",
        "articles": out_dir / "articles.jsonl",
        "passages": out_dir / "passages.jsonl",
        "notes": out_dir / "notes.jsonl",
        "edges": out_dir / "edges.jsonl",
        "chunks": out_dir / "chunks.jsonl",
        "manifest": out_dir / "manifest.json",
    }

    counts = {k: 0 for k in ("laws", "articles", "passages", "notes", "edges", "chunks")}
    dupes = {k: 0 for k in ("law_id", "article_id", "passage_id", "note_id", "edge_id", "chunk_id")}
    empty = {"article_text": 0, "passage_text": 0, "note_text": 0, "chunk_text": 0}
    suspicious_notes: list[dict] = []

    def count_dupes(field: str, ids: set[str], value: str) -> None:
        if not value:
            return
        if value in ids:
            dupes[field] += 1
        else:
            ids.add(value)

    law_ids: set[str] = set()
    article_ids: set[str] = set()
    passage_ids: set[str] = set()
    note_ids: set[str] = set()
    edge_ids: set[str] = set()
    chunk_ids: set[str] = set()

    for r in _iter_jsonl(paths["laws"]):
        counts["laws"] += 1
        count_dupes("law_id", law_ids, r.get("law_id") or "")

    for r in _iter_jsonl(paths["articles"]):
        counts["articles"] += 1
        count_dupes("article_id", article_ids, r.get("article_id") or "")
        if not (r.get("article_text") or "").strip():
            empty["article_text"] += 1

    for r in _iter_jsonl(paths["passages"]):
        counts["passages"] += 1
        count_dupes("passage_id", passage_ids, r.get("passage_id") or "")
        if not (r.get("passage_text") or "").strip():
            empty["passage_text"] += 1

    for r in _iter_jsonl(paths["notes"]):
        counts["notes"] += 1
        nid = r.get("note_id") or ""
        count_dupes("note_id", note_ids, nid)
        txt = (r.get("note_text") or "").strip()
        if not txt:
            empty["note_text"] += 1
        if len(txt) >= int(suspicious_note_chars):
            suspicious_notes.append(
                {
                    "note_id": nid,
                    "law_id": r.get("law_id"),
                    "note_anchor_name": r.get("note_anchor_name"),
                    "chars": len(txt),
                }
            )

    for r in _iter_jsonl(paths["edges"]):
        counts["edges"] += 1
        count_dupes("edge_id", edge_ids, r.get("edge_id") or "")

    for r in _iter_jsonl(paths["chunks"]):
        counts["chunks"] += 1
        count_dupes("chunk_id", chunk_ids, r.get("chunk_id") or "")
        if not (r.get("text") or "").strip():
            empty["chunk_text"] += 1

    warnings: list[str] = []
    if sum(dupes.values()) > 0:
        warnings.append("Duplicate IDs detected (see dupes breakdown).")
    if empty["chunk_text"] > 0:
        warnings.append("Some chunks have empty text.")
    if suspicious_notes:
        warnings.append("Some notes are unusually long (possible boundary issue).")

    return {
        "out_dir": str(out_dir),
        "paths": {k: str(v) for k, v in paths.items()},
        "counts": counts,
        "dupes": dupes,
        "empty": empty,
        "suspicious_notes": suspicious_notes[:50],
        "warnings": warnings,
    }

