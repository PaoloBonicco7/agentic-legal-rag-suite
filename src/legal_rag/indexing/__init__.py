"""Qdrant indexing contract implementation for the clean legal dataset."""

from __future__ import annotations

from .dataset import DatasetValidationResult, load_chunks, read_manifest, validate_clean_dataset
from .embeddings import (
    LocalEmbeddingBackend,
    SupportsEmbedding,
    SupportsSparseEmbedding,
    UtopiaOllamaEmbedder,
    build_embedder,
    debug_utopia_embedding_connection,
    discover_utopia_models,
    supports_sparse_embedding,
)
from .hashing import content_hash_for_text, point_id_from_chunk_id
from .models import (
    FILTERABLE_FIELDS,
    INDEXING_SCHEMA_VERSION,
    REQUIRED_PAYLOAD_FIELDS,
    IndexingConfig,
)
from .pipeline import run_indexing_pipeline
from .retrieval import RetrievedChunk, build_qdrant_filter, search_index

__all__ = [
    "DatasetValidationResult",
    "FILTERABLE_FIELDS",
    "INDEXING_SCHEMA_VERSION",
    "IndexingConfig",
    "LocalEmbeddingBackend",
    "REQUIRED_PAYLOAD_FIELDS",
    "RetrievedChunk",
    "SupportsEmbedding",
    "SupportsSparseEmbedding",
    "UtopiaOllamaEmbedder",
    "build_embedder",
    "build_qdrant_filter",
    "content_hash_for_text",
    "debug_utopia_embedding_connection",
    "discover_utopia_models",
    "load_chunks",
    "point_id_from_chunk_id",
    "read_manifest",
    "run_indexing_pipeline",
    "search_index",
    "supports_sparse_embedding",
    "validate_clean_dataset",
]
