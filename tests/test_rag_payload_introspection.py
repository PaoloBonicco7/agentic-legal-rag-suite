from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from legal_indexing.rag_runtime.config import QdrantPayloadFieldMap
from legal_indexing.rag_runtime.qdrant_retrieval import (
    assert_required_payload_fields,
    introspect_payload_schema,
)


def _make_collection_with_points(
    qdrant_path: Path,
    collection_name: str,
    payloads: list[dict],
) -> QdrantClient:
    client = QdrantClient(path=str(qdrant_path))
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(size=3, distance=qmodels.Distance.COSINE),
    )
    points = [
        qmodels.PointStruct(
            id=i,
            vector=[1.0, 0.0, 0.0] if i % 2 == 0 else [0.0, 1.0, 0.0],
            payload=payload,
        )
        for i, payload in enumerate(payloads, start=1)
    ]
    client.upsert(collection_name=collection_name, points=points, wait=True)
    return client


def test_payload_introspection_detects_required_fields(tmp_path: Path) -> None:
    payloads = [
        {
            "chunk_id": "law:test#art:1#rc:0",
            "law_id": "law:test",
            "article_id": "law:test#art:1",
            "text": "Norma testuale",
            "source_chunk_ids": ["law:test#art:1#p:c1#chunk:0"],
            "source_passage_ids": ["law:test#art:1#p:c1"],
            "index_views": ["historical", "current"],
            "law_status": "current",
        }
    ]
    client = _make_collection_with_points(tmp_path / "qdrant", "rag_payload_ok", payloads)
    try:
        inspection = introspect_payload_schema(
            client,
            collection_name="rag_payload_ok",
            field_map=QdrantPayloadFieldMap(),
            sample_size=8,
        )
        assert inspection.inspected_points == 1
        assert inspection.missing_required_fields == tuple()
        assert_required_payload_fields(inspection)
    finally:
        client.close()


def test_payload_introspection_fails_when_required_fields_missing(tmp_path: Path) -> None:
    payloads = [
        {
            "chunk_id": "law:test#art:1#rc:0",
            "law_id": "law:test",
            "article_id": "law:test#art:1",
            "text": "Norma senza source_passage_ids",
            "source_chunk_ids": ["law:test#art:1#p:c1#chunk:0"],
            "index_views": ["current"],
            "law_status": "current",
        }
    ]
    client = _make_collection_with_points(tmp_path / "qdrant", "rag_payload_missing", payloads)
    try:
        inspection = introspect_payload_schema(
            client,
            collection_name="rag_payload_missing",
            field_map=QdrantPayloadFieldMap(),
            sample_size=8,
        )
        assert "source_passage_ids" in inspection.missing_required_fields
        with pytest.raises(RuntimeError, match="missing required fields"):
            assert_required_payload_fields(inspection)
    finally:
        client.close()
