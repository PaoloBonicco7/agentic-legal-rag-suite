"""Checksummed vector caches and verified migration against a frozen index."""

import json
from pathlib import Path
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client import models

from legal_rag.artifacts import digest, file_hash, read_json, utc_now, write_json
from legal_rag.contracts import Chunk, Record
from legal_rag.providers.embeddings import validate_dense, validate_sparse

CACHE_SCHEMA_VERSION = "vector-cache-v2"


class VectorCacheRecord(Record):
    schema_version: Literal["vector-cache-v2"] = CACHE_SCHEMA_VERSION
    embedding_identity_sha256: str
    text_sha256: str
    vector_sha256: str
    dense: list[float]
    sparse: models.SparseVector | None


class CacheVerification(Record):
    schema_version: str = "cache-verification-v1"
    created_at: str
    index_manifest_sha256: str
    embedding_identity_sha256: str
    chunks_sha256: str
    normalization: str = "qdrant-local-cosine-both-vectors-float32"
    dense_absolute_tolerance: float = 1e-6
    dense_relative_tolerance: float = 0
    source_index_complete: bool
    verified_records: int
    migrated_records: int
    record_hashes: dict[str, str]


def vector_digest(dense, sparse) -> str:
    data = {"dense": dense, "sparse": sparse.model_dump(mode="json") if sparse is not None else None}
    return digest(json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False))


def cache_record(identity: str, text: str, dense, sparse) -> VectorCacheRecord:
    dense = validate_dense([dense], 1)[0]
    if sparse is not None:
        sparse = validate_sparse([models.SparseVector.model_validate(sparse)], 1)[0]
    return VectorCacheRecord(embedding_identity_sha256=identity, text_sha256=digest(text),
                             vector_sha256=vector_digest(dense, sparse), dense=dense, sparse=sparse)


def load_cache(path: Path, identity: str, text: str, *, allow_legacy=False):
    raw = read_json(path)
    if raw.get("schema_version") != CACHE_SCHEMA_VERSION:
        if not allow_legacy or set(raw) != {"dense", "sparse"}:
            raise ValueError("Legacy cache requires verified migration or explicit allow_legacy_cache")
        record = cache_record(identity, text, raw["dense"], raw["sparse"])
        return record, True
    record = VectorCacheRecord.model_validate(raw)
    verified = cache_record(identity, text, record.dense, record.sparse)
    if record != verified:
        raise ValueError("Vector cache checksum, text or embedding identity mismatch")
    return record, False


def migrate_vector_cache(index_dir, cache_dir, *, allow_incomplete=False):
    from .index import open_index

    root, cache = Path(index_dir), Path(cache_dir)
    manifest = read_json(root / "manifest.json")
    identity = manifest["embedding_identity_sha256"]
    client = open_index(root, allow_incomplete=allow_incomplete)
    verified, migrated, offset = {}, {}, None
    try:
        while True:
            points, offset = client.scroll(manifest["collection"], limit=256, offset=offset, with_payload=True, with_vectors=True)
            for point in points:
                chunk = Chunk.model_validate({key: value for key, value in point.payload.items() if key in Chunk.model_fields})
                if str(point.id) != str(uuid5(NAMESPACE_URL, chunk.chunk_id)) or point.payload.get("embedding_identity_sha256") != identity:
                    raise ValueError("Cache migration requires matching source IDs and embedding identity")
                key = digest(identity + chunk.embedding_text)
                path = cache / identity / (key + ".json")
                if not path.exists():
                    raise ValueError("A cache record required by the index is missing")
                record, legacy = load_cache(path, identity, chunk.embedding_text, allow_legacy=True)
                normalized = np.asarray(record.dense, dtype=np.float32)
                norm = np.linalg.norm(normalized)
                if norm > 0:
                    normalized = normalized / norm
                stored = np.asarray(point.vector["dense"], dtype=np.float32)
                stored_norm = np.linalg.norm(stored)
                if stored_norm > 0:
                    stored = stored / stored_norm
                if not np.allclose(normalized, stored, atol=1e-6, rtol=0):
                    raise ValueError("Cached dense vector differs from the Qdrant float32 cosine projection (absolute tolerance 1e-6)")
                stored_sparse = models.SparseVector.model_validate(point.vector["sparse"]) if "sparse" in point.vector else None
                if record.sparse != stored_sparse:
                    raise ValueError("Cached sparse vector differs from the indexed representation")
                verified[key] = path
                if legacy:
                    migrated[key] = record
            if offset is None:
                break
        for key, record in migrated.items():
            write_json(verified[key], record)
        report = CacheVerification(created_at=utc_now(), index_manifest_sha256=file_hash(root / "manifest.json"),
            embedding_identity_sha256=identity, chunks_sha256=manifest["chunks_sha256"], verified_records=len(verified), migrated_records=len(migrated),
            source_index_complete=bool(manifest["ready_for_retrieval"]),
            record_hashes={path.name: file_hash(path) for path in verified.values()})
        write_json(cache / identity / ("verification-" + report.index_manifest_sha256[:16] + ".json"), report)
        return report
    finally:
        client.close()
