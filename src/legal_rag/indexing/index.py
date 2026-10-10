"""Immutable persistent Qdrant indexes."""

import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from legal_rag.artifacts import code_identity, digest, file_hash, new_output, read_json, read_jsonl, utc_now, write_json
from legal_rag.contracts import Chunk, Record
from legal_rag.providers.embeddings import encode_vectors, validate_dense, validate_sparse
from .cache import CACHE_SCHEMA_VERSION, cache_record, load_cache

COLLECTION = "chunks"
INDEX_SCHEMA_VERSION = "index-v1"


class IndexManifest(Record):
    schema_version: str = INDEX_SCHEMA_VERSION
    created_at: str
    snapshot_id: str
    corpus_dir: str
    corpus_manifest_sha256: str
    chunks_sha256: str
    files: dict[str, str]
    embedding: dict
    sparse_embedding: dict | None
    embedding_identity_sha256: str
    collection: str = COLLECTION
    storage: str = "qdrant-local"
    count: int
    ready_for_retrieval: bool = False
    code: dict
    dimension: int | None = None
    cached_vectors: int = 0
    cache_schema_version: str | None = None
    unverified_legacy_cache_vectors: int = 0
    batch_size: int | None = None
    embedding_batch_size: int | None = None


def build_index(corpus_dir, index_dir, embedder, *, sparse_embedder=None, batch_size=2, cache_dir=None, progress=None, allow_legacy_cache=False):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    source, target = Path(corpus_dir), Path(index_dir)
    corpus_manifest = read_json(source / "manifest.json")
    if not corpus_manifest.get("ready_for_indexing"):
        raise ValueError("Corpus is not ready for indexing")
    expected = corpus_manifest.get("files", {})
    if "chunks.jsonl" not in expected:
        raise ValueError("Corpus manifest must hash chunks.jsonl")
    for name, sha in expected.items():
        if file_hash(source / name) != sha:
            raise ValueError(f"Corpus artifact hash mismatch: {name}")
    chunks = read_jsonl(source / "chunks.jsonl", Chunk)
    if not chunks or len({c.chunk_id for c in chunks}) != len(chunks) or any(not c.text.strip() or not c.embedding_text.strip() for c in chunks):
        raise ValueError("Index needs nonempty, unique source chunks")
    if any(c.snapshot_id != corpus_manifest["snapshot_id"] for c in chunks):
        raise ValueError("Chunk snapshot identity mismatch")
    if sparse_embedder is not None and hasattr(sparse_embedder, "fit"):
        sparse_embedder.fit([c.embedding_text for c in chunks])
    dense_identity = embedder.identity
    sparse_identity = sparse_embedder.identity if sparse_embedder is not None else (dense_identity if hasattr(embedder, "embed_hybrid") else None)
    identity_hash = digest(json.dumps({"dense": dense_identity, "sparse": sparse_identity}, sort_keys=True))
    cache = Path(cache_dir) if cache_dir is not None else target.parent / "vector_cache"
    new_output(target)
    client = None
    manifest = {"schema_version": INDEX_SCHEMA_VERSION, "created_at": utc_now(), "snapshot_id": corpus_manifest["snapshot_id"],
                "corpus_dir": str(source.resolve()), "corpus_manifest_sha256": file_hash(source / "manifest.json"),
                "chunks_sha256": file_hash(source / "chunks.jsonl"), "files": expected, "embedding": dense_identity,
                "sparse_embedding": sparse_identity, "embedding_identity_sha256": identity_hash, "collection": COLLECTION,
                "storage": "qdrant-local", "count": len(chunks), "ready_for_retrieval": False, "code": code_identity(),
                "cache_schema_version": CACHE_SCHEMA_VERSION, "batch_size": batch_size,
                "embedding_batch_size": getattr(embedder, "batch_size", None)}
    write_json(target / "manifest.json", IndexManifest.model_validate(manifest))
    try:
        client = QdrantClient(path=str(target / "storage"))
        dimension = None
        cached_count = 0
        legacy_count = 0
        if progress:
            progress({"processed": 0, "total": len(chunks), "cached_vectors": 0, "batches": 0, "dimension": None, "ready_for_retrieval": False})
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset:offset + batch_size]
            rows = [None] * len(batch)
            missing = []
            keys = []
            for i, chunk in enumerate(batch):
                key = digest(identity_hash + chunk.embedding_text)
                keys.append(key)
                path = cache / identity_hash / (key + ".json")
                if path.exists():
                    if read_json(path).get("schema_version") is None and not allow_legacy_cache:
                        missing.append(i)
                    else:
                        record, legacy = load_cache(path, identity_hash, chunk.embedding_text, allow_legacy=allow_legacy_cache)
                        rows[i] = record.to_json_record()
                        legacy_count += int(legacy)
                        cached_count += 1
                else:
                    missing.append(i)
            if missing:
                dense, sparse = encode_vectors(embedder, [batch[i].embedding_text for i in missing], sparse_embedder=sparse_embedder)
                for j, i in enumerate(missing):
                    record = cache_record(identity_hash, batch[i].embedding_text, dense[j], sparse[j] if sparse is not None else None)
                    rows[i] = record.to_json_record()
                    write_json(cache / identity_hash / (keys[i] + ".json"), record)
            dense = validate_dense([row["dense"] for row in rows], len(batch))
            sparse = validate_sparse([models.SparseVector.model_validate(row["sparse"]) for row in rows], len(batch)) if sparse_identity else None
            if dimension is None:
                dimension = len(dense[0])
                declared = dense_identity.get("dimension")
                if declared is not None and dimension != declared:
                    raise ValueError("Embedding dimension disagrees with model identity")
                client.create_collection(COLLECTION, vectors_config={"dense": models.VectorParams(size=dimension, distance=models.Distance.COSINE)},
                    sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF if sparse_identity.get("idf") else None)} if sparse_identity else None)
            if any(len(row) != dimension for row in dense):
                raise ValueError("Embedding dimension changed between batches")
            points = []
            for i, chunk in enumerate(batch):
                vectors = {"dense": dense[i]}
                if sparse is not None:
                    vectors["sparse"] = sparse[i]
                points.append(models.PointStruct(id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)), vector=vectors,
                                                payload={**chunk.to_json_record(), "embedding_identity_sha256": identity_hash}))
            client.upsert(COLLECTION, points=points, wait=True)
            batch_number = offset // batch_size + 1
            if progress and (offset + len(batch)) // 100 != offset // 100:
                progress({"processed": offset + len(batch), "total": len(chunks), "cached_vectors": cached_count,
                          "batches": batch_number, "dimension": dimension, "ready_for_retrieval": False})
        if client.count(COLLECTION, exact=True).count != len(chunks):
            raise ValueError("Qdrant count mismatch")
        verify_index_payloads(client, manifest)
        manifest.update(dimension=dimension, cached_vectors=cached_count, unverified_legacy_cache_vectors=legacy_count, ready_for_retrieval=True)
        write_json(target / "manifest.json", IndexManifest.model_validate(manifest))
        if progress:
            progress({"processed": len(chunks), "total": len(chunks), "cached_vectors": cached_count,
                      "batches": (len(chunks) + batch_size - 1) // batch_size, "dimension": dimension, "ready_for_retrieval": True})
        return manifest
    finally:
        if client is not None:
            client.close()


def verify_index_payloads(client, manifest, *, corpus_dir=None, allow_partial=False):
    source = Path(corpus_dir or manifest["corpus_dir"])
    if file_hash(source / "manifest.json") != manifest["corpus_manifest_sha256"] or file_hash(source / "chunks.jsonl") != manifest["chunks_sha256"]:
        raise ValueError("Index source corpus hashes differ from the frozen manifest")
    chunks = read_jsonl(source / "chunks.jsonl", Chunk)
    source_records = {chunk.chunk_id: chunk for chunk in chunks}
    if len(source_records) != len(chunks) or len(chunks) != manifest["count"]:
        raise ValueError("Index source IDs are duplicated or differ from the manifest count")
    seen, offset = set(), None
    while True:
        points, offset = client.scroll(manifest["collection"], limit=256, offset=offset, with_payload=True, with_vectors=False)
        for point in points:
            chunk_id = point.payload.get("chunk_id")
            chunk = source_records.get(chunk_id)
            if chunk is None or chunk_id in seen or str(point.id) != str(uuid5(NAMESPACE_URL, chunk_id)):
                raise ValueError("Index point IDs differ from the frozen corpus")
            expected = {**chunk.to_json_record(), "embedding_identity_sha256": manifest["embedding_identity_sha256"]}
            if point.payload != expected:
                raise ValueError("Index payload differs from the exact frozen source chunk")
            seen.add(chunk_id)
        if offset is None:
            break
    if not allow_partial and seen != set(source_records):
        raise ValueError("Index does not contain exactly the frozen corpus IDs")


def open_index(index_dir, *, corpus_dir=None, verify_payloads=True, allow_incomplete=False):
    root = Path(index_dir)
    manifest = IndexManifest.model_validate(read_json(root / "manifest.json")).to_json_record()
    if not manifest.get("ready_for_retrieval") and not allow_incomplete:
        raise ValueError("Index is incomplete")
    client = QdrantClient(path=str(root / "storage"))
    try:
        identity_hash = digest(json.dumps({"dense": manifest["embedding"], "sparse": manifest["sparse_embedding"]}, sort_keys=True))
        configuration = client.get_collection(manifest["collection"]).config.params
        vectors = configuration.vectors
        expected_dimension = manifest["dimension"] or (manifest["embedding"].get("dimension") if allow_incomplete else None)
        dimensions_match = isinstance(vectors, dict) and "dense" in vectors and (vectors["dense"].size == expected_dimension if expected_dimension else allow_incomplete)
        sparse_matches = ("sparse" in (configuration.sparse_vectors or {})) == (manifest["sparse_embedding"] is not None)
        if identity_hash != manifest["embedding_identity_sha256"] or not dimensions_match or not sparse_matches:
            raise ValueError("Index vector configuration or embedding identity differs from its manifest")
        count = client.count(manifest["collection"], exact=True).count
        if count > manifest["count"] or (not allow_incomplete and count != manifest["count"]):
            raise ValueError("Index count differs from its manifest")
        if verify_payloads:
            verify_index_payloads(client, manifest, corpus_dir=corpus_dir, allow_partial=allow_incomplete)
    except Exception:
        client.close()
        raise
    return client
