from __future__ import annotations

from pathlib import Path

from .artifacts import load_docs_from_out_dir
from .bm25 import BM25Index


def build_bm25_index_from_out_dir(
    out_dir: Path,
    *,
    unit: str = "chunk",
    text_field: str = "text_for_embedding",
    k1: float = 1.2,
    b: float = 0.75,
) -> BM25Index:
    docs = load_docs_from_out_dir(out_dir, unit=unit, text_field=text_field)
    idx = BM25Index(k1=k1, b=b)
    idx.build(((d.doc_id, d.text, d.meta) for d in docs))
    return idx

