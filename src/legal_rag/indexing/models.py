"""Typed configuration and constants for Qdrant indexing."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

INDEXING_SCHEMA_VERSION = "indexing-contract-v1"
LOCAL_DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
UTOPIA_DEFAULT_EMBEDDING_MODEL = "SLURM.nomic-embed-text:latest"

REQUIRED_PAYLOAD_FIELDS = {
    "chunk_id",
    "passage_id",
    "article_id",
    "law_id",
    "text",
    "law_date",
    "law_number",
    "law_title",
    "law_status",
    "article_status",
    "article_label_norm",
    "passage_label",
    "structure_path",
    "source_file",
    "index_views",
    "related_law_ids",
    "inbound_law_ids",
    "outbound_law_ids",
    "relation_types",
    "content_hash",
}

SOURCE_CHUNK_REQUIRED_FIELDS = REQUIRED_PAYLOAD_FIELDS - {"content_hash"} | {"text_for_embedding"}
LIST_PAYLOAD_FIELDS = {
    "index_views",
    "related_law_ids",
    "inbound_law_ids",
    "outbound_law_ids",
    "relation_types",
}
FILTERABLE_FIELDS = (
    "chunk_id",
    "law_id",
    "law_status",
    "index_views",
    "article_id",
    "article_status",
    "relation_types",
    "law_date",
    "law_number",
)

EmbeddingBackend = Literal["local", "utopia"]
ChunkSelectionMode = Literal["full", "sample"]
UtopiaEmbedMode = Literal["ollama"]


def _now_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


class IndexingConfig(BaseModel):
    """Runtime configuration for the clean dataset to Qdrant indexing step."""

    model_config = ConfigDict(extra="forbid")

    clean_dataset_dir: str = "data/laws_dataset_clean"
    index_dir: str = "data/indexes/qdrant"
    runs_dir: str = "data/indexing_runs"
    collection_name: str = "legal_chunks"
    qdrant_url: str | None = None
    qdrant_api_key: str = ""
    force_rebuild: bool = False
    chunk_selection_mode: ChunkSelectionMode = "full"
    sample_size: int | None = Field(default=None, gt=0)
    strict: bool = True
    run_id: str | None = None
    env_file: str | None = ".env"

    embedding_backend: EmbeddingBackend = "local"
    embedding_model: str = LOCAL_DEFAULT_EMBEDDING_MODEL
    embedding_dim: int | None = Field(default=None, gt=0)
    hybrid_enabled: bool = True
    utopia_base_url: str = "https://utopia.hpc4ai.unito.it/api"
    utopia_embed_url: str = "https://utopia.hpc4ai.unito.it/ollama/api/embeddings"
    utopia_embed_api_mode: UtopiaEmbedMode = "ollama"
    embedding_api_key: str = ""
    batch_size: int = Field(default=64, gt=0)
    embedding_timeout_seconds: float = Field(default=60.0, gt=0)

    upload_batch_size: int = Field(default=64, gt=0)
    upload_max_retries: int = Field(default=3, ge=1)
    qdrant_distance: Literal["cosine", "dot", "euclid", "manhattan"] = "cosine"
    qdrant_on_disk_payload: bool = True
    qdrant_hnsw_m: int = Field(default=16, ge=0)
    qdrant_hnsw_ef_construct: int = Field(default=100, gt=0)
    diagnostic_queries: list[str] = Field(
        default_factory=lambda: [
            "contributi regionali",
            "formazione professionale",
            "tutela dell'ambiente",
        ]
    )

    @field_validator("collection_name", "embedding_model")
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("value must be non-empty")
        return text

    @field_validator("embedding_backend", "chunk_selection_mode", mode="before")
    @classmethod
    def _normalize_literal(cls, value: str) -> str:
        return str(value or "").strip().lower()

    @field_validator("utopia_embed_api_mode", mode="before")
    @classmethod
    def _normalize_embed_mode(cls, value: str) -> str:
        return str(value or "ollama").strip().lower()

    @field_validator("hybrid_enabled")
    @classmethod
    def _hybrid_requires_local_backend_by_default(cls, value: bool) -> bool:
        return bool(value)

    @property
    def resolved_dataset_dir(self) -> Path:
        return Path(self.clean_dataset_dir).resolve()

    @property
    def resolved_index_dir(self) -> Path:
        return Path(self.index_dir).resolve()

    @property
    def resolved_artifacts_root(self) -> Path:
        return Path(self.runs_dir).resolve()

    @property
    def effective_run_id(self) -> str:
        return self.run_id or _now_run_id()

    @property
    def selection_limit(self) -> int | None:
        return self.sample_size if self.chunk_selection_mode == "sample" else None

    @property
    def resolved_embedding_model(self) -> str:
        if self.embedding_backend == "utopia" and self.embedding_model == LOCAL_DEFAULT_EMBEDDING_MODEL:
            return os.getenv("UTOPIA_EMBED_MODEL", UTOPIA_DEFAULT_EMBEDDING_MODEL)
        return self.embedding_model

    @property
    def resolved_utopia_base_url(self) -> str:
        return os.getenv("UTOPIA_BASE_URL", self.utopia_base_url)

    @property
    def resolved_utopia_embed_url(self) -> str:
        return os.getenv("UTOPIA_EMBED_URL", self.utopia_embed_url)

    def public_dict(self) -> dict[str, object]:
        """Return a manifest-safe config dictionary without secrets."""
        data = self.model_dump(mode="json")
        data["resolved_embedding_model"] = self.resolved_embedding_model
        data["resolved_utopia_base_url"] = self.resolved_utopia_base_url
        data["resolved_utopia_embed_url"] = self.resolved_utopia_embed_url
        data["embedding_api_key_set"] = bool(self.embedding_api_key)
        data["qdrant_api_key_set"] = bool(self.qdrant_api_key)
        data.pop("embedding_api_key", None)
        data.pop("qdrant_api_key", None)
        return data
