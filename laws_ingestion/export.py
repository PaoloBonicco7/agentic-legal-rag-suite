from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from . import __version__
from .ingest import ingest_law
from .registry import build_corpus_registry
from .utils import compute_dataset_id


def ingest_and_write(
    *,
    html_dir: Path,
    out_dir: Path,
    backend: str = "auto",
    strict: bool = False,
    max_words: int = 600,
    overlap_words: int = 80,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = build_corpus_registry(html_dir)
    dataset_id = compute_dataset_id(html_dir.glob("*.html"))

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
    unresolved_refs_total = 0
    warnings: list[str] = []
    errors: list[str] = []
    parser_backend_used: str | None = None

    with (
        paths["laws"].open("w", encoding="utf-8") as laws_f,
        paths["articles"].open("w", encoding="utf-8") as articles_f,
        paths["passages"].open("w", encoding="utf-8") as passages_f,
        paths["notes"].open("w", encoding="utf-8") as notes_f,
        paths["edges"].open("w", encoding="utf-8") as edges_f,
        paths["chunks"].open("w", encoding="utf-8") as chunks_f,
    ):
        for law_id, law_file in sorted(registry.by_law_id.items(), key=lambda kv: kv[0]):
            try:
                ing = ingest_law(
                    law_file,
                    registry,
                    backend=backend,
                    strict=strict,
                    max_words=max_words,
                    overlap_words=overlap_words,
                )
            except Exception as e:
                msg = f"{law_file.source_file}: {type(e).__name__}: {e}"
                if strict:
                    raise
                errors.append(msg)
                continue

            parser_backend_used = parser_backend_used or ing.parser_backend
            unresolved_refs_total += int(ing.unresolved_refs or 0)
            warnings.extend(list(ing.warnings or []))

            laws_f.write(json.dumps(ing.law, ensure_ascii=False) + "\n")
            counts["laws"] += 1

            for a in ing.articles:
                articles_f.write(json.dumps(a, ensure_ascii=False) + "\n")
                counts["articles"] += 1

            for p in ing.passages:
                passages_f.write(json.dumps(p, ensure_ascii=False) + "\n")
                counts["passages"] += 1

            for n in ing.notes:
                notes_f.write(json.dumps(n, ensure_ascii=False) + "\n")
                counts["notes"] += 1

            for e in ing.edges:
                edges_f.write(json.dumps(e, ensure_ascii=False) + "\n")
                counts["edges"] += 1

            for c in ing.chunks:
                chunks_f.write(json.dumps(c, ensure_ascii=False) + "\n")
                counts["chunks"] += 1

    manifest = {
        "dataset_id": dataset_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "parser_version": __version__,
        "parser_backend": parser_backend_used or "unknown",
        "source_dir": str(html_dir),
        "out_dir": str(out_dir),
        "config": {
            "backend": backend,
            "strict": bool(strict),
            "max_words": int(max_words),
            "overlap_words": int(overlap_words),
        },
        "counts": counts,
        "unresolved_refs": unresolved_refs_total,
        "warnings_sample": warnings[:50],
        "errors_sample": errors[:50],
        "output_files": {k: str(v) for k, v in paths.items() if k != "manifest"},
    }

    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest

