from __future__ import annotations

from pathlib import Path

from .ingest import ingest_law
from .registry import build_corpus_registry


def debug_law(
    *,
    html_dir: Path,
    law_id: str = "",
    source_file: str = "",
    backend: str = "auto",
    strict: bool = False,
    preview_articles: int = 10,
    preview_chars: int = 500,
) -> dict:
    registry = build_corpus_registry(html_dir)

    law_file = None
    if law_id:
        law_file = registry.by_law_id.get(law_id)
        if not law_file:
            raise ValueError(f"Unknown law_id: {law_id}")
    elif source_file:
        for lf in registry.by_law_id.values():
            if lf.source_file == source_file:
                law_file = lf
                break
        if not law_file:
            raise ValueError(f"Unknown source_file: {source_file}")
    else:
        raise ValueError("Provide either law_id or source_file")

    ing = ingest_law(law_file, registry, backend=backend, strict=strict)

    previews = []
    for a in ing.articles[: max(0, int(preview_articles))]:
        previews.append(
            {
                "article_id": a.get("article_id"),
                "article_label_norm": a.get("article_label_norm"),
                "article_heading": a.get("article_heading"),
                "anchor_name": a.get("anchor_name"),
                "structure_path": a.get("structure_path"),
                "note_anchor_names": a.get("note_anchor_names"),
                "is_abrogated": a.get("is_abrogated"),
                "text_preview": (a.get("article_text") or "").strip()[: max(0, int(preview_chars))],
            }
        )

    return {
        "law": ing.law,
        "counts": {
            "articles": len(ing.articles),
            "passages": len(ing.passages),
            "notes": len(ing.notes),
            "edges": len(ing.edges),
            "chunks": len(ing.chunks),
            "unresolved_refs": ing.unresolved_refs,
        },
        "articles_preview": previews,
    }

