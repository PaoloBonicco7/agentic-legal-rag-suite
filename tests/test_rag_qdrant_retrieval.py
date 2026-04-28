from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from legal_indexing.rag_runtime.config import QdrantPayloadFieldMap
from legal_indexing.rag_runtime.qdrant_retrieval import (
    QdrantRetriever,
    build_article_filter,
    build_law_date_filter,
    build_law_status_filter,
    build_relation_type_filter,
    retrieve_multi_queries,
)


class KeywordEmbedder:
    @property
    def model_name(self) -> str:
        return "keyword-test-embedder"

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


def _setup_collection(path: Path, name: str) -> QdrantClient:
    client = QdrantClient(path=str(path))
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(size=3, distance=qmodels.Distance.COSINE),
    )
    payloads = [
        {
            "chunk_id": "law:test#art:1#rc:0",
            "law_id": "law:test",
            "article_id": "law:test#art:1",
            "text": "Disciplina della sanzione amministrativa.",
            "source_chunk_ids": ["law:test#art:1#p:c1#chunk:0"],
            "source_passage_ids": ["law:test#art:1#p:c1"],
            "index_views": ["historical", "current"],
            "law_status": "current",
            "law_date": "2020-01-01",
            "relation_types": ["AMENDS"],
        },
        {
            "chunk_id": "law:test#art:2#rc:0",
            "law_id": "law:test",
            "article_id": "law:test#art:2",
            "text": "Procedura di presentazione della domanda.",
            "source_chunk_ids": ["law:test#art:2#p:c1#chunk:0"],
            "source_passage_ids": ["law:test#art:2#p:c1"],
            "index_views": ["historical", "current"],
            "law_status": "current",
            "law_date": "2021-01-01",
            "relation_types": ["REFERENCES"],
        },
        {
            "chunk_id": "law:test#art:3#rc:0",
            "law_id": "law:test",
            "article_id": "law:test#art:3",
            "text": "Disposizioni finali.",
            "source_chunk_ids": ["law:test#art:3#p:c1#chunk:0"],
            "source_passage_ids": ["law:test#art:3#p:c1"],
            "index_views": ["historical", "current"],
            "law_status": "current",
            "law_date": "2022-01-01",
            "relation_types": ["ABROGATED_BY"],
        },
    ]
    points = [
        qmodels.PointStruct(id=i + 1, vector=vec, payload=payload)
        for i, (vec, payload) in enumerate(
            zip(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                payloads,
                strict=True,
            )
        )
    ]
    client.upsert(collection_name=name, points=points, wait=True)
    return client


def test_qdrant_retrieval_returns_topk_and_provenance(tmp_path: Path) -> None:
    collection_name = "rag_retrieval"
    client = _setup_collection(tmp_path / "qdrant", collection_name)
    try:
        retriever = QdrantRetriever(
            client=client,
            collection_name=collection_name,
            embedder=KeywordEmbedder(),
            field_map=QdrantPayloadFieldMap(),
        )
        docs = retriever.query("Qual e la sanzione prevista?", top_k=2)
        assert len(docs) == 2
        assert docs[0].chunk_id == "law:test#art:1#rc:0"
        assert docs[0].law_id == "law:test"
        assert docs[0].article_id == "law:test#art:1"
        assert docs[0].source_chunk_ids == ("law:test#art:1#p:c1#chunk:0",)
        assert docs[0].source_passage_ids == ("law:test#art:1#p:c1",)

        by_id = retriever.retrieve_by_chunk_ids(
            ["law:test#art:2#rc:0", "law:test#art:3#rc:0"],
            limit=2,
        )
        got_ids = {d.chunk_id for d in by_id}
        assert got_ids == {"law:test#art:2#rc:0", "law:test#art:3#rc:0"}
    finally:
        client.close()


def test_qdrant_retrieval_supports_score_threshold_direction(tmp_path: Path) -> None:
    collection_name = "rag_retrieval_threshold"
    client = _setup_collection(tmp_path / "qdrant", collection_name)
    try:
        retriever = QdrantRetriever(
            client=client,
            collection_name=collection_name,
            embedder=KeywordEmbedder(),
            field_map=QdrantPayloadFieldMap(),
        )
        docs = retriever.query(
            "Qual e la sanzione prevista?",
            top_k=3,
            score_threshold=0.99,
            threshold_direction="gte",
        )
        assert len(docs) == 1
        assert docs[0].chunk_id == "law:test#art:1#rc:0"
    finally:
        client.close()


def test_qdrant_advanced_filters_and_multi_retrieval(tmp_path: Path) -> None:
    collection_name = "rag_retrieval_advanced"
    client = _setup_collection(tmp_path / "qdrant", collection_name)
    try:
        retriever = QdrantRetriever(
            client=client,
            collection_name=collection_name,
            embedder=KeywordEmbedder(),
            field_map=QdrantPayloadFieldMap(),
        )

        field_map = QdrantPayloadFieldMap()
        assert build_law_status_filter(field_map, "current") is not None
        assert build_relation_type_filter(field_map, ["AMENDS"]) is not None
        assert build_article_filter(field_map, ["law:test#art:2"]) is not None
        assert (
            build_law_date_filter(
                field_map,
                year_from=2020,
                year_to=2021,
            )
            is not None
        )

        merged, batches = retrieve_multi_queries(
            retriever,
            queries=[
                "Qual e la sanzione prevista?",
                "Qual e la procedura prevista?",
            ],
            top_k_primary=2,
            top_k_secondary=2,
            dedupe_by_chunk_id=True,
        )
        assert len(batches) == 2
        assert batches[0].name == "primary"
        assert len(merged) >= 2
    finally:
        client.close()
