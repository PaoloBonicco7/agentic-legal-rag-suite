from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def article_key_from_ids(*, law_id: str, article_id: str | None) -> tuple[str, str] | None:
    if not law_id or not article_id:
        return None
    marker = "#art:"
    if marker not in article_id:
        return None
    article_label_norm = article_id.split(marker, 1)[1].strip()
    if not article_label_norm:
        return None
    return law_id, article_label_norm


@dataclass(frozen=True)
class ArtifactDoc:
    doc_id: str
    text: str
    meta: dict


def load_docs_from_out_dir(
    out_dir: Path,
    *,
    unit: str,
    text_field: str,
) -> list[ArtifactDoc]:
    if unit == "chunk":
        path = out_dir / "chunks.jsonl"
        docs: list[ArtifactDoc] = []
        for r in iter_jsonl(path):
            doc_id = r.get("chunk_id") or ""
            text = r.get(text_field) or r.get("text_for_embedding") or r.get("text") or ""
            law_id = r.get("law_id") or ""
            article_id = r.get("article_id") or ""
            key = article_key_from_ids(law_id=law_id, article_id=article_id)
            meta = {
                "unit": "chunk",
                "chunk_id": doc_id,
                "law_id": law_id,
                "article_id": article_id,
                "article_key": key,
                "passage_id": r.get("passage_id"),
            }
            docs.append(ArtifactDoc(doc_id=doc_id, text=text, meta=meta))
        return docs

    if unit == "passage":
        path = out_dir / "passages.jsonl"
        docs = []
        for r in iter_jsonl(path):
            doc_id = r.get("passage_id") or ""
            text = r.get(text_field) or r.get("passage_text") or ""
            law_id = r.get("law_id") or ""
            article_id = r.get("article_id") or ""
            key = article_key_from_ids(law_id=law_id, article_id=article_id)
            meta = {
                "unit": "passage",
                "law_id": law_id,
                "article_id": article_id,
                "article_key": key,
                "passage_id": doc_id,
                "passage_label": r.get("passage_label"),
            }
            docs.append(ArtifactDoc(doc_id=doc_id, text=text, meta=meta))
        return docs

    if unit == "article":
        path = out_dir / "articles.jsonl"
        docs = []
        for r in iter_jsonl(path):
            doc_id = r.get("article_id") or ""
            text = r.get(text_field) or r.get("article_text") or ""
            law_id = r.get("law_id") or ""
            article_id = r.get("article_id") or ""
            key = article_key_from_ids(law_id=law_id, article_id=article_id)
            meta = {"unit": "article", "law_id": law_id, "article_id": article_id, "article_key": key}
            docs.append(ArtifactDoc(doc_id=doc_id, text=text, meta=meta))
        return docs

    raise ValueError(f"Unknown unit: {unit!r}")

