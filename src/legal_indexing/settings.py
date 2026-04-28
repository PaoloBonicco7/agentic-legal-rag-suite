from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import os
from pathlib import Path
import re
from typing import Any


_QDRANT_DISTANCE = "cosine"


def _slugify(value: str, *, max_len: int = 48) -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]", "_", (value or "").strip())
    text = re.sub(r"_+", "_", text).strip("_").lower()
    if not text:
        return "default"
    return text[:max_len]


@dataclass(frozen=True)
class ChunkingProfile:
    profile_id: str
    min_words_merge: int
    max_words_split: int
    overlap_words_split: int

    def validate(self) -> None:
        if self.min_words_merge <= 0:
            raise ValueError("ChunkingProfile.min_words_merge must be > 0")
        if self.max_words_split <= self.min_words_merge:
            raise ValueError(
                "ChunkingProfile.max_words_split must be > min_words_merge"
            )
        if self.overlap_words_split < 0:
            raise ValueError("ChunkingProfile.overlap_words_split must be >= 0")
        if self.overlap_words_split >= self.max_words_split:
            raise ValueError(
                "ChunkingProfile.overlap_words_split must be < max_words_split"
            )


_CHUNKING_PRESETS: dict[str, ChunkingProfile] = {
    "balanced": ChunkingProfile(
        profile_id="balanced", min_words_merge=20, max_words_split=220, overlap_words_split=40
    ),
    "conservative": ChunkingProfile(
        profile_id="conservative", min_words_merge=12, max_words_split=240, overlap_words_split=40
    ),
    "aggressive": ChunkingProfile(
        profile_id="aggressive", min_words_merge=30, max_words_split=180, overlap_words_split=50
    ),
}


@dataclass(frozen=True)
class IndexingConfig:
    dataset_dir: Path = Path("data/laws_dataset_clean")
    qdrant_path: Path = Path("data/indexes/qdrant")
    artifacts_root: Path = Path("data/qdrant_indexing")

    collection_name: str | None = None
    collection_prefix: str = "laws_clean"

    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "utopia")
    utopia_base_url: str = os.getenv("UTOPIA_BASE_URL", "https://utopia.hpc4ai.unito.it/api")
    utopia_embed_api_mode: str = os.getenv("UTOPIA_EMBED_API_MODE", "auto")
    utopia_embed_url: str = os.getenv(
        "UTOPIA_EMBED_URL", "https://utopia.hpc4ai.unito.it/ollama/api/embed"
    )
    embedding_model: str = os.getenv("UTOPIA_EMBED_MODEL", "SLURM.nomic-embed-text:latest")
    embedding_api_key: str = os.getenv("UTOPIA_API_KEY", "")
    embedding_batch_size: int = 32
    embedding_timeout_seconds: float = 60.0

    chunking_profile: ChunkingProfile = field(default_factory=lambda: make_chunking_profile("balanced"))

    force_reembed: bool = False
    subset_limit: int | None = None
    strict_validation: bool = True

    qdrant_distance: str = _QDRANT_DISTANCE
    qdrant_on_disk_payload: bool = True
    qdrant_hnsw_m: int = 16
    qdrant_hnsw_ef_construct: int = 100

    sparse_enabled: bool = True
    sparse_vector_name: str = "bm25"
    sparse_min_token_len: int = 2
    sparse_stopwords_lang: str = "it"
    sparse_store_artifacts: bool = True
    sparse_analyzer: str = "it_default"

    index_contract_min_eval_coverage: float = 0.95
    index_contract_enforce_eval_coverage: bool = False
    eval_questions_csv: Path = Path("data/evaluation/questions.csv")
    eval_questions_no_hint_csv: Path = Path("data/evaluation/questions_no_hint.csv")

    run_id: str | None = None

    def with_overrides(self, **overrides: Any) -> "IndexingConfig":
        data = asdict(self)
        data.update(overrides)
        if isinstance(data.get("chunking_profile"), dict):
            cp = data["chunking_profile"]
            data["chunking_profile"] = ChunkingProfile(
                profile_id=str(cp["profile_id"]),
                min_words_merge=int(cp["min_words_merge"]),
                max_words_split=int(cp["max_words_split"]),
                overlap_words_split=int(cp["overlap_words_split"]),
            )
        for key in (
            "dataset_dir",
            "qdrant_path",
            "artifacts_root",
            "eval_questions_csv",
            "eval_questions_no_hint_csv",
        ):
            if key in data and not isinstance(data[key], Path):
                data[key] = Path(data[key])
        cfg = IndexingConfig(**data)
        cfg.chunking_profile.validate()
        if cfg.embedding_batch_size <= 0:
            raise ValueError("IndexingConfig.embedding_batch_size must be > 0")
        if cfg.embedding_timeout_seconds <= 0:
            raise ValueError("IndexingConfig.embedding_timeout_seconds must be > 0")
        if cfg.sparse_min_token_len <= 0:
            raise ValueError("IndexingConfig.sparse_min_token_len must be > 0")
        if not str(cfg.sparse_vector_name or "").strip():
            raise ValueError("IndexingConfig.sparse_vector_name cannot be empty")
        if str(cfg.sparse_stopwords_lang or "").strip().lower() not in {"it", "none"}:
            raise ValueError("IndexingConfig.sparse_stopwords_lang must be one of: it, none")
        if str(cfg.sparse_analyzer or "").strip().lower() not in {"it_default", "it_legal"}:
            raise ValueError("IndexingConfig.sparse_analyzer must be one of: it_default, it_legal")
        if not (0.0 <= float(cfg.index_contract_min_eval_coverage) <= 1.0):
            raise ValueError("IndexingConfig.index_contract_min_eval_coverage must be in [0, 1]")
        return cfg

    def resolve_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return (Path.cwd() / path).resolve()

    @property
    def resolved_dataset_dir(self) -> Path:
        return self.resolve_path(self.dataset_dir)

    @property
    def resolved_qdrant_path(self) -> Path:
        return self.resolve_path(self.qdrant_path)

    @property
    def resolved_artifacts_root(self) -> Path:
        return self.resolve_path(self.artifacts_root)

    @property
    def resolved_eval_questions_csv(self) -> Path:
        return self.resolve_path(self.eval_questions_csv)

    @property
    def resolved_eval_questions_no_hint_csv(self) -> Path:
        return self.resolve_path(self.eval_questions_no_hint_csv)

    @property
    def effective_run_id(self) -> str:
        if self.run_id:
            return str(self.run_id)
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    def to_dict(self) -> dict[str, Any]:
        cp = self.chunking_profile
        return {
            "dataset_dir": str(self.dataset_dir),
            "qdrant_path": str(self.qdrant_path),
            "artifacts_root": str(self.artifacts_root),
            "collection_name": self.collection_name,
            "collection_prefix": self.collection_prefix,
            "embedding_provider": self.embedding_provider,
            "utopia_base_url": self.utopia_base_url,
            "utopia_embed_api_mode": self.utopia_embed_api_mode,
            "utopia_embed_url": self.utopia_embed_url,
            "embedding_model": self.embedding_model,
            "embedding_api_key_set": bool(self.embedding_api_key),
            "embedding_batch_size": self.embedding_batch_size,
            "embedding_timeout_seconds": self.embedding_timeout_seconds,
            "chunking_profile": {
                "profile_id": cp.profile_id,
                "min_words_merge": cp.min_words_merge,
                "max_words_split": cp.max_words_split,
                "overlap_words_split": cp.overlap_words_split,
            },
            "force_reembed": self.force_reembed,
            "subset_limit": self.subset_limit,
            "strict_validation": self.strict_validation,
            "qdrant_distance": self.qdrant_distance,
            "qdrant_on_disk_payload": self.qdrant_on_disk_payload,
            "qdrant_hnsw_m": self.qdrant_hnsw_m,
            "qdrant_hnsw_ef_construct": self.qdrant_hnsw_ef_construct,
            "sparse_enabled": self.sparse_enabled,
            "sparse_vector_name": self.sparse_vector_name,
            "sparse_min_token_len": self.sparse_min_token_len,
            "sparse_stopwords_lang": self.sparse_stopwords_lang,
            "sparse_store_artifacts": self.sparse_store_artifacts,
            "sparse_analyzer": self.sparse_analyzer,
            "index_contract_min_eval_coverage": self.index_contract_min_eval_coverage,
            "index_contract_enforce_eval_coverage": self.index_contract_enforce_eval_coverage,
            "eval_questions_csv": str(self.eval_questions_csv),
            "eval_questions_no_hint_csv": str(self.eval_questions_no_hint_csv),
            "run_id": self.run_id,
        }


def make_chunking_profile(
    profile: str,
    *,
    min_words_merge: int | None = None,
    max_words_split: int | None = None,
    overlap_words_split: int | None = None,
    profile_id: str | None = None,
) -> ChunkingProfile:
    base = _CHUNKING_PRESETS.get(profile.lower())
    if base is None:
        available = ", ".join(sorted(_CHUNKING_PRESETS))
        raise ValueError(f"Unknown chunking profile {profile!r}. Available: {available}")
    out = ChunkingProfile(
        profile_id=profile_id or base.profile_id,
        min_words_merge=min_words_merge if min_words_merge is not None else base.min_words_merge,
        max_words_split=max_words_split if max_words_split is not None else base.max_words_split,
        overlap_words_split=(
            overlap_words_split if overlap_words_split is not None else base.overlap_words_split
        ),
    )
    out.validate()
    return out


def safe_collection_component(value: str, *, max_len: int = 48) -> str:
    return _slugify(value, max_len=max_len)


__all__ = [
    "ChunkingProfile",
    "IndexingConfig",
    "make_chunking_profile",
    "safe_collection_component",
]
