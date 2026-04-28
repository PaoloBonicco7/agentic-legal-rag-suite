from __future__ import annotations

import hashlib
import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from legal_indexing.pipeline import run_indexing_pipeline
from legal_indexing.settings import IndexingConfig, make_chunking_profile


class FakeEmbedder:
    def __init__(self, size: int = 8) -> None:
        self._size = size

    @property
    def model_name(self) -> str:
        return "fake-test-embedder"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vec = [((digest[i % len(digest)] / 255.0) * 2.0) - 1.0 for i in range(self._size)]
            vectors.append(vec)
        return vectors


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _dataset_with_current_and_historical(root: Path) -> Path:
    ds = root / "dataset"
    ds.mkdir(parents=True, exist_ok=True)

    laws = [
        {
            "law_id": "law:current",
            "law_date": "2025-01-01",
            "law_number": 1,
            "law_title": "Legge corrente",
            "status": "current",
        },
        {
            "law_id": "law:past",
            "law_date": "2001-01-01",
            "law_number": 2,
            "law_title": "Legge storica",
            "status": "past",
        },
    ]
    articles = [
        {
            "article_id": "law:current#art:1",
            "law_id": "law:current",
            "article_label_norm": "1",
            "is_abrogated": False,
        },
        {
            "article_id": "law:past#art:1",
            "law_id": "law:past",
            "article_label_norm": "1",
            "is_abrogated": False,
        },
    ]
    chunks = [
        {
            "chunk_id": "law:current#art:1#p:c1#chunk:0",
            "passage_id": "law:current#art:1#p:c1",
            "article_id": "law:current#art:1",
            "law_id": "law:current",
            "chunk_seq": 0,
            "text": "Norma corrente",
            "text_for_embedding": "Norma corrente",
            "law_date": "2025-01-01",
            "law_number": 1,
            "law_title": "Legge corrente",
            "law_status": "current",
            "article_label_norm": "1",
            "article_is_abrogated": False,
            "passage_label": "c1",
            "related_law_ids": [],
            "relation_types": [],
            "inbound_law_ids": [],
            "outbound_law_ids": [],
            "status_confidence": 0.9,
            "status_evidence": [],
            "index_views": ["historical", "current"],
        },
        {
            "chunk_id": "law:past#art:1#p:c1#chunk:0",
            "passage_id": "law:past#art:1#p:c1",
            "article_id": "law:past#art:1",
            "law_id": "law:past",
            "chunk_seq": 0,
            "text": "Norma storica",
            "text_for_embedding": "Norma storica",
            "law_date": "2001-01-01",
            "law_number": 2,
            "law_title": "Legge storica",
            "law_status": "past",
            "article_label_norm": "1",
            "article_is_abrogated": False,
            "passage_label": "c1",
            "related_law_ids": [],
            "relation_types": [],
            "inbound_law_ids": [],
            "outbound_law_ids": [],
            "status_confidence": 0.9,
            "status_evidence": [],
            "index_views": ["historical"],
        },
    ]

    manifest = {
        "schema_version": "laws-graph-pipeline-v1",
        "run_id": "filters_test",
        "ready_to_embedding": True,
        "counts": {
            "laws": len(laws),
            "articles": len(articles),
            "notes": 0,
            "edges": 0,
            "events": 0,
            "chunks": len(chunks),
        },
        "hashes": {
            "chunks": "hash_filters_chunks",
            "laws": "h1",
            "articles": "h2",
            "notes": "h3",
            "edges": "h4",
            "events": "h5",
        },
    }

    _write_jsonl(ds / "laws.jsonl", laws)
    _write_jsonl(ds / "articles.jsonl", articles)
    _write_jsonl(ds / "notes.jsonl", [])
    _write_jsonl(ds / "edges.jsonl", [])
    _write_jsonl(ds / "events.jsonl", [])
    _write_jsonl(ds / "chunks.jsonl", chunks)
    (ds / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return ds


def test_qdrant_metadata_filters(tmp_path: Path) -> None:
    dataset_dir = _dataset_with_current_and_historical(tmp_path)
    qdrant_path = tmp_path / "qdrant"

    config = IndexingConfig(
        dataset_dir=dataset_dir,
        qdrant_path=qdrant_path,
        artifacts_root=tmp_path / "artifacts",
        embedding_provider="utopia",
        embedding_model="test-model",
        embedding_api_key="unused-in-test",
        chunking_profile=make_chunking_profile("balanced", min_words_merge=2, max_words_split=40, overlap_words_split=5),
        run_id="filters_run",
    )

    summary = run_indexing_pipeline(config, embedder=FakeEmbedder(size=8))

    assert summary.filter_validation_ok
    assert summary.duplicate_chunk_ids_ok

    client = QdrantClient(path=str(qdrant_path))
    try:
        current_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="index_views", match=qmodels.MatchValue(value="current")
                ),
                qmodels.FieldCondition(
                    key="law_status", match=qmodels.MatchValue(value="current")
                ),
            ]
        )
        current_count = client.count(
            collection_name=summary.collection_name,
            count_filter=current_filter,
            exact=True,
        )
        assert current_count.count >= 1

        past_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="law_status", match=qmodels.MatchValue(value="past")
                )
            ]
        )
        past_count = client.count(
            collection_name=summary.collection_name,
            count_filter=past_filter,
            exact=True,
        )
        assert past_count.count >= 1
    finally:
        client.close()
