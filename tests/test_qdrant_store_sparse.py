from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from legal_indexing.qdrant_store import (
    PreparedPoint,
    ensure_collection,
    sync_points_incremental,
)
from legal_indexing.settings import IndexingConfig, make_chunking_profile
from legal_indexing.sparse import SparseVectorData


class FakeEmbedder:
    @property
    def model_name(self) -> str:
        return "fake"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i, _ in enumerate(texts, start=1):
            out.append([1.0 / i, 0.5, 0.25])
        return out


def _cfg(tmp_path: Path) -> IndexingConfig:
    return IndexingConfig(
        dataset_dir=tmp_path,
        qdrant_path=tmp_path / "qdrant",
        artifacts_root=tmp_path / "artifacts",
        embedding_provider="utopia",
        embedding_model="fake",
        embedding_api_key="x",
        chunking_profile=make_chunking_profile(
            "balanced",
            min_words_merge=2,
            max_words_split=40,
            overlap_words_split=5,
        ),
        sparse_enabled=True,
        sparse_vector_name="bm25",
        sparse_min_token_len=2,
        sparse_stopwords_lang="it",
    )


def test_qdrant_collection_supports_dense_and_sparse_vectors(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    client = QdrantClient(path=str(cfg.resolved_qdrant_path))
    try:
        ensure_collection(
            client,
            cfg,
            collection_name="test_sparse",
            vector_size=3,
        )

        points = [
                PreparedPoint(
                    chunk_id="c1",
                    point_id="00000000-0000-0000-0000-000000000001",
                embedding_text="testo uno",
                payload={
                    "chunk_id": "c1",
                    "law_id": "law:1",
                    "article_id": "law:1#art:1",
                    "text": "testo uno",
                    "source_chunk_ids": ["c1"],
                    "source_passage_ids": ["p:1"],
                    "index_views": ["current"],
                    "law_status": "current",
                    "content_hash": "h1",
                },
                content_hash="h1",
            ),
                PreparedPoint(
                    chunk_id="c2",
                    point_id="00000000-0000-0000-0000-000000000002",
                embedding_text="testo due",
                payload={
                    "chunk_id": "c2",
                    "law_id": "law:2",
                    "article_id": "law:2#art:1",
                    "text": "testo due",
                    "source_chunk_ids": ["c2"],
                    "source_passage_ids": ["p:2"],
                    "index_views": ["current"],
                    "law_status": "current",
                    "content_hash": "h2",
                },
                content_hash="h2",
            ),
        ]

        sparse_vectors = {
            "c1": SparseVectorData(indices=(1, 2), values=(0.8, 0.2)),
            "c2": SparseVectorData(indices=(3,), values=(1.0,)),
        }

        stats = sync_points_incremental(
            client,
            collection_name="test_sparse",
            points=points,
            embedder=FakeEmbedder(),
            force_reembed=False,
            embed_batch_size=2,
            dense_vector_name=None,
            sparse_vector_name="bm25",
            sparse_vectors_by_chunk=sparse_vectors,
        )
        assert stats.upserted == 2

        sparse_res = client.query_points(
            collection_name="test_sparse",
            query=qmodels.SparseVector(indices=[1], values=[1.0]),
            using="bm25",
            limit=2,
            with_payload=True,
            with_vectors=False,
        )
        assert len(sparse_res.points) >= 1
        assert (sparse_res.points[0].payload or {}).get("chunk_id") == "c1"
    finally:
        client.close()
