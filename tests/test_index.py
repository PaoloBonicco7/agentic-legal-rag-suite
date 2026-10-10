import json
from uuid import NAMESPACE_URL, uuid5

import pytest
from qdrant_client import models

from legal_rag.artifacts import file_hash, write_json, write_jsonl
from legal_rag.contracts import Chunk, EngineConfig
from legal_rag.indexing import build_index, migrate_vector_cache, open_index
from legal_rag.retrieval import Retriever


class Embedder:
    identity = {"model": "fixture", "dimension": 2, "idf": False}

    def __init__(self):
        self.calls = 0

    def embed_hybrid(self, texts, *, query=False):
        self.calls += 1
        return [[1., 0.] for _ in texts], [models.SparseVector(indices=[1], values=[1.]) for _ in texts]


def chunk(identifier, status="no_cessation_evidence"):
    return Chunk(chunk_id=identifier, document_id="law", snapshot_id="snapshot", title="Law", text="Source legal text",
                 embedding_text="Law Article 1 Source legal text", unit_ids=["law#art:1"], spans=[], source_url="https://example.org/law", status=status)


def corpus(tmp_path, chunks):
    path = tmp_path / "corpus"
    path.mkdir()
    write_jsonl(path / "chunks.jsonl", chunks)
    write_json(path / "manifest.json", {"snapshot_id": "snapshot", "ready_for_indexing": True, "files": {"chunks.jsonl": file_hash(path / "chunks.jsonl")}})
    return path


def test_immutable_index_native_rrf_and_policy_filters(tmp_path):
    source = corpus(tmp_path, [chunk("live"), chunk("unknown", "unknown"), chunk("ceased", "ceased_confirmed")])
    target = tmp_path / "index"
    embedder = Embedder()
    progress = []
    manifest = build_index(source, target, embedder, batch_size=2, progress=progress.append)
    assert manifest["ready_for_retrieval"]
    assert progress[0]["processed"] == 0 and not progress[0]["ready_for_retrieval"]
    assert progress[-1]["processed"] == 3 and progress[-1]["ready_for_retrieval"]
    client = open_index(target)
    try:
        retriever = Retriever(client, embedder, manifest=manifest)
        trace = []
        for mode in ("dense", "sparse", "hybrid"):
            hits = retriever.search("question", mode=mode, trace=trace)
            assert {c.chunk.chunk_id for c in hits} == {"live", "unknown"}
        assert len(retriever.search("question", mode="hybrid", config=EngineConfig(exclude_ceased=False))) == 3
        assert set(trace[-1]["branches"]) == {"dense", "sparse"}
    finally:
        client.close()
    with pytest.raises(FileExistsError):
        build_index(source, target, embedder)


def test_verified_vector_cache_reuses_only_identical_inputs(tmp_path):
    source = corpus(tmp_path, [chunk("one")])
    embedder = Embedder()
    build_index(source, tmp_path / "first", embedder)
    calls = embedder.calls
    rebuilt = build_index(source, tmp_path / "second", embedder)
    assert rebuilt["cached_vectors"] == 1
    assert embedder.calls == calls
    embedder.identity = {"model": "different", "dimension": 2, "idf": False}
    build_index(source, tmp_path / "third", embedder)
    assert embedder.calls == calls + 1


def test_upload_and_embedding_batches_are_recorded_separately(tmp_path):
    source = corpus(tmp_path, [chunk("one"), chunk("two")])
    embedder = Embedder()
    embedder.batch_size = 8
    manifest = build_index(source, tmp_path / "index", embedder, batch_size=64)
    assert manifest["batch_size"] == 64
    assert manifest["embedding_batch_size"] == 8
    assert "batch_size" not in manifest["embedding"]


def test_input_hash_mismatch_prevents_embedding(tmp_path):
    source = corpus(tmp_path, [chunk("one")])
    with (source / "chunks.jsonl").open("a") as handle:
        handle.write("\n")
    embedder = Embedder()
    with pytest.raises(ValueError, match="hash mismatch"):
        build_index(source, tmp_path / "index", embedder)
    assert embedder.calls == 0


def test_failed_collection_validation_releases_local_storage(tmp_path, monkeypatch):
    source = corpus(tmp_path, [chunk("one")])
    target = tmp_path / "index"
    build_index(source, target, Embedder())
    class BrokenClient:
        closed = False

        def get_collection(self, name):
            raise ValueError("Collection is missing")

        def close(self):
            self.closed = True

    client = BrokenClient()
    monkeypatch.setattr("legal_rag.indexing.index.QdrantClient", lambda **kwargs: client)
    with pytest.raises(ValueError, match="Collection is missing"):
        open_index(target)
    assert client.closed


def test_readiness_requires_exact_persisted_payloads(tmp_path, monkeypatch):
    from qdrant_client import QdrantClient

    source = corpus(tmp_path, [chunk("one")])
    original = QdrantClient.upsert

    def corrupted_upsert(client, collection_name, **kwargs):
        result = original(client, collection_name, **kwargs)
        client.set_payload(collection_name, payload={"title": "Corrupted title"}, points=[point.id for point in kwargs["points"]])
        return result

    monkeypatch.setattr(QdrantClient, "upsert", corrupted_upsert)
    target = tmp_path / "index"
    with pytest.raises(ValueError, match="exact frozen source chunk"):
        build_index(source, target, Embedder())
    assert not json.loads((target / "manifest.json").read_text())["ready_for_retrieval"]


def test_dense_dimensions_and_nonfinite_vectors_are_rejected(tmp_path):
    source = corpus(tmp_path, [chunk("one")])
    embedder = Embedder()
    embedder.embed_hybrid = lambda texts, query=False: ([[float("nan"), 1]], [models.SparseVector(indices=[1], values=[1])])
    with pytest.raises(ValueError, match="finite"):
        build_index(source, tmp_path / "index", embedder)
    manifest = json.loads((tmp_path / "index/manifest.json").read_text())
    assert not manifest["ready_for_retrieval"]


@pytest.mark.parametrize("mutation", ["nonfinite", "dimension", "duplicate_sparse_indices"])
def test_corrupted_cached_vectors_are_rejected_before_reuse(tmp_path, mutation):
    source = corpus(tmp_path, [chunk("one")])
    embedder = Embedder()
    build_index(source, tmp_path / "first", embedder)
    path = next((tmp_path / "vector_cache").rglob("*.json"))
    cached = json.loads(path.read_text())
    if mutation == "nonfinite":
        cached["dense"][0] = float("nan")
    elif mutation == "dimension":
        cached["dense"].append(1)
    else:
        cached["sparse"] = {"indices": [1, 1], "values": [1, 2]}
    write_json(path, cached)
    calls = embedder.calls
    with pytest.raises(ValueError):
        build_index(source, tmp_path / "second", embedder)
    assert embedder.calls == calls
    assert not json.loads((tmp_path / "second/manifest.json").read_text())["ready_for_retrieval"]


@pytest.mark.parametrize("payload", [{"embedding_identity_sha256": "different"}, {"snapshot_id": "other-snapshot"}])
def test_qdrant_payload_identity_is_checked_on_each_retrieval(tmp_path, payload):
    source = corpus(tmp_path, [chunk("one")])
    embedder = Embedder()
    target = tmp_path / "index"
    manifest = build_index(source, target, embedder)
    client = open_index(target)
    try:
        client.set_payload("chunks", payload=payload, points=[str(uuid5(NAMESPACE_URL, "one"))])
        with pytest.raises(ValueError):
            Retriever(client, embedder, manifest=manifest).search("question", mode="hybrid")
    finally:
        client.close()


def test_empty_sparse_representation_keeps_dense_hybrid_and_sparse_empty(tmp_path):
    source = corpus(tmp_path, [chunk("one")])
    embedder = Embedder()
    embedder.embed_hybrid = lambda texts, query=False: ([[1, 0] for _ in texts], [models.SparseVector(indices=[], values=[]) for _ in texts])
    target = tmp_path / "index"
    manifest = build_index(source, target, embedder)
    client = open_index(target)
    try:
        retriever = Retriever(client, embedder, manifest=manifest)
        assert retriever.search("question", mode="sparse") == []
        assert len(retriever.search("question", mode="hybrid")) == 1
    finally:
        client.close()


def test_cache_checksums_detect_finite_corruption(tmp_path):
    source = corpus(tmp_path, [chunk("one")])
    embedder = Embedder()
    build_index(source, tmp_path / "first", embedder)
    path = next((tmp_path / "vector_cache").rglob("*.json"))
    cached = json.loads(path.read_text())
    cached["dense"] = [0., 1.]
    write_json(path, cached)
    with pytest.raises(ValueError, match="checksum"):
        build_index(source, tmp_path / "second", embedder)


def test_legacy_cache_migration_verifies_against_exact_indexed_source(tmp_path):
    source = corpus(tmp_path, [chunk("one")])
    embedder = Embedder()
    target = tmp_path / "first"
    build_index(source, target, embedder)
    path = next((tmp_path / "vector_cache").rglob("*.json"))
    cached = json.loads(path.read_text())
    write_json(path, {"dense": cached["dense"], "sparse": cached["sparse"]})
    from legal_rag.indexing.cache import load_cache
    with pytest.raises(ValueError, match="Legacy cache"):
        load_cache(path, cached["embedding_identity_sha256"], "Law Article 1 Source legal text")
    report = migrate_vector_cache(target, tmp_path / "vector_cache")
    assert report.migrated_records == report.verified_records == 1
    assert json.loads(path.read_text())["schema_version"] == "vector-cache-v2"
    calls = embedder.calls
    rebuilt = build_index(source, tmp_path / "verified", embedder)
    assert rebuilt["cached_vectors"] == 1 and embedder.calls == calls


def test_unverified_legacy_cache_is_reembedded_by_default(tmp_path):
    source = corpus(tmp_path, [chunk("one")])
    embedder = Embedder()
    build_index(source, tmp_path / "first", embedder)
    path = next((tmp_path / "vector_cache").rglob("*.json"))
    record = json.loads(path.read_text())
    write_json(path, {"dense": record["dense"], "sparse": record["sparse"]})
    calls = embedder.calls
    rebuilt = build_index(source, tmp_path / "second", embedder)
    assert rebuilt["cached_vectors"] == 0 and embedder.calls == calls + 1


def test_cache_migration_rejects_a_different_finite_vector(tmp_path):
    source = corpus(tmp_path, [chunk("one")])
    target = tmp_path / "index"
    build_index(source, target, Embedder())
    path = next((tmp_path / "vector_cache").rglob("*.json"))
    cached = json.loads(path.read_text())
    write_json(path, {"dense": [0., 1.], "sparse": cached["sparse"]})
    with pytest.raises(ValueError, match="projection"):
        migrate_vector_cache(target, tmp_path / "vector_cache")
    assert "schema_version" not in json.loads(path.read_text())


def test_cache_migration_normalizes_both_persisted_and_cached_cosine_vectors(tmp_path):
    source = corpus(tmp_path, [chunk("one")])
    target = tmp_path / "index"
    embedder = Embedder()
    embedder.embed_hybrid = lambda texts, query=False: ([[3., 4.] for _ in texts], [models.SparseVector(indices=[1], values=[1.]) for _ in texts])
    build_index(source, target, embedder)
    report = migrate_vector_cache(target, tmp_path / "vector_cache")
    assert report.verified_records == 1
    assert report.normalization == "qdrant-local-cosine-both-vectors-float32"


def test_incomplete_cache_migration_only_upgrades_persisted_points(tmp_path):
    one, two = chunk("one"), chunk("two").model_copy(update={"embedding_text": "Different source text"})
    source = corpus(tmp_path, [one, two])
    target = tmp_path / "index"
    build_index(source, target, Embedder())
    paths = list((tmp_path / "vector_cache").rglob("*.json"))
    for path in paths:
        cached = json.loads(path.read_text())
        write_json(path, {"dense": cached["dense"], "sparse": cached["sparse"]})
    client = open_index(target)
    client.delete("chunks", points_selector=models.PointIdsList(points=[str(uuid5(NAMESPACE_URL, "two"))]))
    client.close()
    manifest = json.loads((target / "manifest.json").read_text())
    manifest.update(ready_for_retrieval=False, dimension=None)
    write_json(target / "manifest.json", manifest)
    with pytest.raises(ValueError, match="incomplete"):
        migrate_vector_cache(target, tmp_path / "vector_cache")
    report = migrate_vector_cache(target, tmp_path / "vector_cache", allow_incomplete=True)
    assert report.migrated_records == 1 and not report.source_index_complete
    assert sum("schema_version" in json.loads(path.read_text()) for path in paths) == 1


def test_open_index_validates_full_payload_against_frozen_source(tmp_path):
    source = corpus(tmp_path, [chunk("one")])
    target = tmp_path / "index"
    build_index(source, target, Embedder())
    client = open_index(target)
    client.set_payload("chunks", payload={"text": "Finite but fabricated source text"}, points=[str(uuid5(NAMESPACE_URL, "one"))])
    client.close()
    with pytest.raises(ValueError, match="payload differs"):
        open_index(target)


def test_separate_sparse_retrieval_does_not_call_dense_provider(tmp_path):
    class Dense:
        identity = {"provider": "fixture", "model": "dense", "dimension": 2}

        def embed_documents(self, texts):
            return [[1., 0.] for _ in texts]

        def embed_queries(self, texts):
            pytest.fail("A sparse-only profile must not contact its dense provider")

    class Lexical:
        identity = {"model": "bm25-fixture", "idf": True}

        def embed_documents(self, texts):
            return [models.SparseVector(indices=[1], values=[1.]) for _ in texts]

        embed_queries = embed_documents

    source = corpus(tmp_path, [chunk("one")])
    target = tmp_path / "index"
    dense, lexical = Dense(), Lexical()
    manifest = build_index(source, target, dense, sparse_embedder=lexical)
    client = open_index(target)
    try:
        retriever = Retriever(client, dense, sparse_embedder=lexical, manifest=manifest)
        assert retriever.search("question", mode="sparse")[0].chunk.chunk_id == "one"
    finally:
        client.close()
