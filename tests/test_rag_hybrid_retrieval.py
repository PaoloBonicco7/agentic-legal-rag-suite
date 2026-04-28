from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from legal_indexing.rag_runtime.config import AdvancedHybridConfig, QdrantPayloadFieldMap
from legal_indexing.rag_runtime.qdrant_retrieval import QdrantRetriever
from legal_indexing.sparse import SparseEncoder


class KeywordEmbedder:
    @property
    def model_name(self) -> str:
        return "keyword"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            low = text.lower()
            if "sanzione" in low:
                out.append([1.0, 0.0, 0.0])
            elif "procedura" in low:
                out.append([0.0, 1.0, 0.0])
            else:
                out.append([0.0, 0.0, 1.0])
        return out


def _setup(path: Path) -> tuple[QdrantClient, SparseEncoder]:
    client = QdrantClient(path=str(path))
    if client.collection_exists("hybrid"):
        client.delete_collection("hybrid")
    client.create_collection(
        collection_name="hybrid",
        vectors_config=qmodels.VectorParams(size=3, distance=qmodels.Distance.COSINE),
        sparse_vectors_config={"bm25": qmodels.SparseVectorParams()},
    )

    docs = [
        ("c1", "Disciplina della sanzione amministrativa"),
        ("c2", "Procedura di presentazione della domanda"),
        ("c3", "Disposizioni finali"),
    ]
    sparse = SparseEncoder(min_token_len=2, stopwords_lang="it")
    sparse.fit([t for _, t in docs])

    dense_vectors = KeywordEmbedder().embed_texts([t for _, t in docs])
    points: list[qmodels.PointStruct] = []
    for idx, (chunk_id, text) in enumerate(docs, start=1):
        sv = sparse.transform(text)
        payload = {
            "chunk_id": chunk_id,
            "law_id": f"law:{idx}",
            "article_id": f"law:{idx}#art:1",
            "text": text,
            "source_chunk_ids": [chunk_id],
            "source_passage_ids": [f"p:{idx}"],
            "index_views": ["current"],
            "law_status": "current",
        }
        points.append(
            qmodels.PointStruct(
                id=idx,
                vector={
                    "": dense_vectors[idx - 1],
                    "bm25": qmodels.SparseVector(
                        indices=list(sv.indices),
                        values=list(sv.values),
                    ),
                },
                payload=payload,
            )
        )
    client.upsert(collection_name="hybrid", points=points, wait=True)
    return client, sparse


def test_hybrid_retrieval_fuses_dense_and_sparse(tmp_path: Path) -> None:
    client, sparse = _setup(tmp_path / "qdrant")
    try:
        retriever = QdrantRetriever(
            client=client,
            collection_name="hybrid",
            embedder=KeywordEmbedder(),
            field_map=QdrantPayloadFieldMap(),
            dense_vector_name=None,
            sparse_vector_name="bm25",
            sparse_encoder=sparse,
        )

        res = retriever.query_hybrid(
            "Qual e la sanzione prevista",
            top_k=3,
            query_filter=None,
            hybrid_config=AdvancedHybridConfig(enabled=True, dense_top_k=3, sparse_top_k=3),
        )

        assert res.retrieval_mode == "hybrid"
        assert len(res.retrieved) >= 1
        assert len(res.dense_retrieved) >= 1
        assert len(res.sparse_retrieved) >= 1
    finally:
        client.close()


def test_hybrid_retrieval_fallbacks_to_dense_when_sparse_unavailable(tmp_path: Path) -> None:
    client, sparse = _setup(tmp_path / "qdrant")
    _ = sparse
    try:
        retriever = QdrantRetriever(
            client=client,
            collection_name="hybrid",
            embedder=KeywordEmbedder(),
            field_map=QdrantPayloadFieldMap(),
            dense_vector_name=None,
            sparse_vector_name=None,
            sparse_encoder=None,
        )

        res = retriever.query_hybrid(
            "Qual e la procedura",
            top_k=2,
            query_filter=None,
            hybrid_config=AdvancedHybridConfig(enabled=True, dense_top_k=2, sparse_top_k=2),
        )

        assert res.retrieval_mode in {"dense_only", "fallback_dense"}
        assert len(res.retrieved) >= 1
        assert len(res.sparse_retrieved) == 0
    finally:
        client.close()
