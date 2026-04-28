from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
from typing import Any, Literal

from legal_indexing.settings import IndexingConfig, make_chunking_profile


ThresholdDirection = Literal["gte", "lte"]
ViewFilter = Literal["none", "current", "historical"]
PipelineMode = Literal["naive", "advanced"]
MetadataFilterMode = Literal["off", "explicit_only", "hybrid"]
RerankTieBreaker = Literal["chunk_id", "retrieval_score"]
HybridFusionMethod = Literal["rrf", "weighted_sum"]
HybridQueryAnalyzer = Literal["it_default", "it_legal"]


@dataclass(frozen=True)
class QdrantPayloadFieldMap:
    chunk_id: str = "chunk_id"
    law_id: str = "law_id"
    article_id: str = "article_id"
    text: str = "text"
    text_for_embedding: str = "text_for_embedding"
    source_chunk_ids: str = "source_chunk_ids"
    source_passage_ids: str = "source_passage_ids"
    source_passage_labels: str = "source_passage_labels"
    index_views: str = "index_views"
    law_status: str = "law_status"
    relation_types: str = "relation_types"
    related_law_ids: str = "related_law_ids"
    law_date: str = "law_date"
    law_year: str = "law_year"
    prev_chunk_id: str = "prev_chunk_id"
    next_chunk_id: str = "next_chunk_id"
    article_chunk_order: str = "article_chunk_order"

    def required_fields(self) -> tuple[str, ...]:
        return (
            self.chunk_id,
            self.law_id,
            self.article_id,
            self.text,
            self.source_chunk_ids,
            self.source_passage_ids,
            self.index_views,
            self.law_status,
        )


@dataclass(frozen=True)
class AdvancedRewriteConfig:
    enabled: bool = False
    use_llm: bool = True
    max_rewrites: int = 2
    max_subqueries: int = 3
    llm_timeout_seconds: float = 30.0
    fallback_to_original: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AdvancedRewriteConfig":
        return cls(
            enabled=bool(payload.get("enabled", False)),
            use_llm=bool(payload.get("use_llm", True)),
            max_rewrites=int(payload.get("max_rewrites", 2)),
            max_subqueries=int(payload.get("max_subqueries", 3)),
            llm_timeout_seconds=float(payload.get("llm_timeout_seconds", 30.0)),
            fallback_to_original=bool(payload.get("fallback_to_original", True)),
        )

    def validate(self) -> None:
        if self.max_rewrites < 0:
            raise ValueError("advanced.rewrite.max_rewrites must be >= 0")
        if self.max_subqueries <= 0:
            raise ValueError("advanced.rewrite.max_subqueries must be > 0")
        if self.llm_timeout_seconds <= 0:
            raise ValueError("advanced.rewrite.llm_timeout_seconds must be > 0")


@dataclass(frozen=True)
class AdvancedMetadataFilteringConfig:
    mode: MetadataFilterMode = "hybrid"
    enable_heuristics: bool = True
    explicit_view: ViewFilter | None = None
    explicit_law_status: str | None = None
    explicit_law_ids: tuple[str, ...] = ()
    explicit_relation_types: tuple[str, ...] = ()
    explicit_article_ids: tuple[str, ...] = ()
    explicit_year_from: int | None = None
    explicit_year_to: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AdvancedMetadataFilteringConfig":
        return cls(
            mode=str(payload.get("mode", "hybrid")).strip().lower(),  # type: ignore[arg-type]
            enable_heuristics=bool(payload.get("enable_heuristics", True)),
            explicit_view=payload.get("explicit_view"),
            explicit_law_status=(
                str(payload.get("explicit_law_status")).strip()
                if payload.get("explicit_law_status") is not None
                else None
            ),
            explicit_law_ids=tuple(
                [str(x).strip() for x in (payload.get("explicit_law_ids") or []) if str(x).strip()]
            ),
            explicit_relation_types=tuple(
                [str(x).strip() for x in (payload.get("explicit_relation_types") or []) if str(x).strip()]
            ),
            explicit_article_ids=tuple(
                [str(x).strip() for x in (payload.get("explicit_article_ids") or []) if str(x).strip()]
            ),
            explicit_year_from=(
                int(payload["explicit_year_from"])
                if payload.get("explicit_year_from") is not None
                else None
            ),
            explicit_year_to=(
                int(payload["explicit_year_to"])
                if payload.get("explicit_year_to") is not None
                else None
            ),
        )

    def validate(self) -> None:
        if self.mode not in {"off", "explicit_only", "hybrid"}:
            raise ValueError(
                "advanced.metadata_filtering.mode must be one of: off, explicit_only, hybrid"
            )
        if self.explicit_view is not None and self.explicit_view not in {"none", "current", "historical"}:
            raise ValueError(
                "advanced.metadata_filtering.explicit_view must be one of: "
                "none, current, historical"
            )
        if self.explicit_year_from is not None and self.explicit_year_from < 1800:
            raise ValueError("advanced.metadata_filtering.explicit_year_from must be >= 1800")
        if self.explicit_year_to is not None and self.explicit_year_to < 1800:
            raise ValueError("advanced.metadata_filtering.explicit_year_to must be >= 1800")
        if (
            self.explicit_year_from is not None
            and self.explicit_year_to is not None
            and self.explicit_year_from > self.explicit_year_to
        ):
            raise ValueError(
                "advanced.metadata_filtering.explicit_year_from cannot be > explicit_year_to"
            )


@dataclass(frozen=True)
class AdvancedMultiRetrievalConfig:
    top_k_primary: int = 8
    top_k_secondary: int = 4
    dedupe_by_chunk_id: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AdvancedMultiRetrievalConfig":
        return cls(
            top_k_primary=int(payload.get("top_k_primary", 8)),
            top_k_secondary=int(payload.get("top_k_secondary", 4)),
            dedupe_by_chunk_id=bool(payload.get("dedupe_by_chunk_id", True)),
        )

    def validate(self) -> None:
        if self.top_k_primary <= 0:
            raise ValueError("advanced.multi_retrieval.top_k_primary must be > 0")
        if self.top_k_secondary <= 0:
            raise ValueError("advanced.multi_retrieval.top_k_secondary must be > 0")


@dataclass(frozen=True)
class AdvancedGraphExpansionConfig:
    enabled: bool = True
    max_related_laws: int = 8
    graph_retrieval_top_k: int = 6
    include_related_articles: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AdvancedGraphExpansionConfig":
        return cls(
            enabled=bool(payload.get("enabled", True)),
            max_related_laws=int(payload.get("max_related_laws", 8)),
            graph_retrieval_top_k=int(payload.get("graph_retrieval_top_k", 6)),
            include_related_articles=bool(payload.get("include_related_articles", True)),
        )

    def validate(self) -> None:
        if self.max_related_laws <= 0:
            raise ValueError("advanced.graph_expansion.max_related_laws must be > 0")
        if self.graph_retrieval_top_k <= 0:
            raise ValueError("advanced.graph_expansion.graph_retrieval_top_k must be > 0")


@dataclass(frozen=True)
class AdvancedRerankConfig:
    enabled: bool = True
    weight_retrieval_score: float = 1.0
    weight_graph_bonus: float = 0.25
    weight_metadata_bonus: float = 0.2
    weight_lexical_overlap: float = 0.15
    weight_sparse_score: float = 0.2
    tie_breaker: RerankTieBreaker = "chunk_id"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AdvancedRerankConfig":
        return cls(
            enabled=bool(payload.get("enabled", True)),
            weight_retrieval_score=float(payload.get("weight_retrieval_score", 1.0)),
            weight_graph_bonus=float(payload.get("weight_graph_bonus", 0.25)),
            weight_metadata_bonus=float(payload.get("weight_metadata_bonus", 0.2)),
            weight_lexical_overlap=float(payload.get("weight_lexical_overlap", 0.15)),
            weight_sparse_score=float(payload.get("weight_sparse_score", 0.2)),
            tie_breaker=str(payload.get("tie_breaker", "chunk_id")).strip().lower(),  # type: ignore[arg-type]
        )

    def validate(self) -> None:
        if self.tie_breaker not in {"chunk_id", "retrieval_score"}:
            raise ValueError("advanced.rerank.tie_breaker must be chunk_id or retrieval_score")
        for key, value in {
            "weight_retrieval_score": self.weight_retrieval_score,
            "weight_graph_bonus": self.weight_graph_bonus,
            "weight_metadata_bonus": self.weight_metadata_bonus,
            "weight_lexical_overlap": self.weight_lexical_overlap,
            "weight_sparse_score": self.weight_sparse_score,
        }.items():
            if value < 0:
                raise ValueError(f"advanced.rerank.{key} must be >= 0")


@dataclass(frozen=True)
class AdvancedAnswerGuardConfig:
    retry_on_empty_answer: bool = True
    max_empty_retries: int = 1
    fallback_message_it: str = (
        "Non dispongo di elementi sufficienti nel contesto recuperato per una risposta "
        "affidabile; servono più riferimenti normativi specifici."
    )
    fallback_message_en: str = (
        "I do not have enough reliable evidence in the retrieved context to provide a "
        "trustworthy answer; more specific legal references are needed."
    )
    mark_empty_as_pipeline_error: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AdvancedAnswerGuardConfig":
        return cls(
            retry_on_empty_answer=bool(payload.get("retry_on_empty_answer", True)),
            max_empty_retries=int(payload.get("max_empty_retries", 1)),
            fallback_message_it=str(
                payload.get("fallback_message_it")
                or (
                    "Non dispongo di elementi sufficienti nel contesto recuperato per una risposta "
                    "affidabile; servono più riferimenti normativi specifici."
                )
            ).strip(),
            fallback_message_en=str(
                payload.get("fallback_message_en")
                or (
                    "I do not have enough reliable evidence in the retrieved context to provide a "
                    "trustworthy answer; more specific legal references are needed."
                )
            ).strip(),
            mark_empty_as_pipeline_error=bool(payload.get("mark_empty_as_pipeline_error", True)),
        )

    def validate(self) -> None:
        if self.max_empty_retries < 0:
            raise ValueError("advanced.answer_guard.max_empty_retries must be >= 0")
        if not self.fallback_message_it.strip():
            raise ValueError("advanced.answer_guard.fallback_message_it cannot be empty")
        if not self.fallback_message_en.strip():
            raise ValueError("advanced.answer_guard.fallback_message_en cannot be empty")


@dataclass(frozen=True)
class AdvancedHybridConfig:
    enabled: bool = True
    dense_top_k: int = 12
    sparse_top_k: int = 20
    fusion_method: HybridFusionMethod = "rrf"
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    min_sparse_score: float | None = None
    fallback_to_dense_only: bool = True
    query_analyzer: HybridQueryAnalyzer = "it_default"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AdvancedHybridConfig":
        return cls(
            enabled=bool(payload.get("enabled", True)),
            dense_top_k=int(payload.get("dense_top_k", 12)),
            sparse_top_k=int(payload.get("sparse_top_k", 20)),
            fusion_method=str(payload.get("fusion_method", "rrf")).strip().lower(),  # type: ignore[arg-type]
            rrf_k=int(payload.get("rrf_k", 60)),
            dense_weight=float(payload.get("dense_weight", 1.0)),
            sparse_weight=float(payload.get("sparse_weight", 1.0)),
            min_sparse_score=(
                float(payload["min_sparse_score"])
                if payload.get("min_sparse_score") is not None
                else None
            ),
            fallback_to_dense_only=bool(payload.get("fallback_to_dense_only", True)),
            query_analyzer=str(payload.get("query_analyzer", "it_default")).strip().lower(),  # type: ignore[arg-type]
        )

    def validate(self) -> None:
        if self.dense_top_k < 0:
            raise ValueError("advanced.hybrid.dense_top_k must be >= 0")
        if self.sparse_top_k < 0:
            raise ValueError("advanced.hybrid.sparse_top_k must be >= 0")
        if self.dense_top_k <= 0 and self.sparse_top_k <= 0:
            raise ValueError("advanced.hybrid requires at least one active channel")
        if self.fusion_method not in {"rrf", "weighted_sum"}:
            raise ValueError("advanced.hybrid.fusion_method must be rrf or weighted_sum")
        if self.rrf_k <= 0:
            raise ValueError("advanced.hybrid.rrf_k must be > 0")
        if self.dense_weight < 0:
            raise ValueError("advanced.hybrid.dense_weight must be >= 0")
        if self.sparse_weight < 0:
            raise ValueError("advanced.hybrid.sparse_weight must be >= 0")
        if self.query_analyzer not in {"it_default", "it_legal"}:
            raise ValueError("advanced.hybrid.query_analyzer must be it_default or it_legal")


@dataclass(frozen=True)
class AdvancedRagConfig:
    hybrid: AdvancedHybridConfig = field(default_factory=AdvancedHybridConfig)
    rewrite: AdvancedRewriteConfig = field(default_factory=AdvancedRewriteConfig)
    metadata_filtering: AdvancedMetadataFilteringConfig = field(
        default_factory=AdvancedMetadataFilteringConfig
    )
    multi_retrieval: AdvancedMultiRetrievalConfig = field(
        default_factory=AdvancedMultiRetrievalConfig
    )
    graph_expansion: AdvancedGraphExpansionConfig = field(
        default_factory=AdvancedGraphExpansionConfig
    )
    rerank: AdvancedRerankConfig = field(default_factory=AdvancedRerankConfig)
    answer_guard: AdvancedAnswerGuardConfig = field(default_factory=AdvancedAnswerGuardConfig)
    max_candidates: int = 48

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AdvancedRagConfig":
        return cls(
            hybrid=AdvancedHybridConfig.from_dict(payload.get("hybrid") or {}),
            rewrite=AdvancedRewriteConfig.from_dict(payload.get("rewrite") or {}),
            metadata_filtering=AdvancedMetadataFilteringConfig.from_dict(
                payload.get("metadata_filtering") or {}
            ),
            multi_retrieval=AdvancedMultiRetrievalConfig.from_dict(
                payload.get("multi_retrieval") or {}
            ),
            graph_expansion=AdvancedGraphExpansionConfig.from_dict(
                payload.get("graph_expansion") or {}
            ),
            rerank=AdvancedRerankConfig.from_dict(payload.get("rerank") or {}),
            answer_guard=AdvancedAnswerGuardConfig.from_dict(payload.get("answer_guard") or {}),
            max_candidates=int(payload.get("max_candidates", 48)),
        )

    def validate(self) -> None:
        self.hybrid.validate()
        self.rewrite.validate()
        self.metadata_filtering.validate()
        self.multi_retrieval.validate()
        self.graph_expansion.validate()
        self.rerank.validate()
        self.answer_guard.validate()
        if self.max_candidates <= 0:
            raise ValueError("advanced.max_candidates must be > 0")


@dataclass(frozen=True)
class RagRuntimeConfig:
    dataset_dir: Path = Path("data/laws_dataset_clean")
    qdrant_path: Path = Path("data/indexes/qdrant")
    indexing_artifacts_root: Path = Path("data/qdrant_indexing")
    indexing_run_id: str | None = None
    collection_name: str | None = None
    index_contract_min_eval_coverage: float = 0.95
    enforce_index_contract_coverage: bool = False

    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "utopia")
    utopia_base_url: str = os.getenv("UTOPIA_BASE_URL", "https://utopia.hpc4ai.unito.it/api")
    utopia_embed_api_mode: str = os.getenv("UTOPIA_EMBED_API_MODE", "auto")
    utopia_embed_url: str = os.getenv(
        "UTOPIA_EMBED_URL",
        "https://utopia.hpc4ai.unito.it/ollama/api/embed",
    )
    embedding_model: str = os.getenv(
        "UTOPIA_EMBED_MODEL",
        "SLURM.nomic-embed-text:latest",
    )
    embedding_api_key: str = os.getenv("UTOPIA_API_KEY", "")
    embedding_batch_size: int = 32
    embedding_timeout_seconds: float = 60.0

    llm_provider: str = os.getenv("RAG_LLM_PROVIDER", "utopia")
    llm_base_url: str = os.getenv("UTOPIA_BASE_URL", "https://utopia.hpc4ai.unito.it/api")
    llm_model: str = os.getenv("UTOPIA_CHAT_MODEL", "SLURM.gpt-oss:120b")
    llm_api_key: str = os.getenv("UTOPIA_API_KEY", "")
    llm_temperature: float = 0.0

    pipeline_mode: PipelineMode = "naive"

    top_k: int = 8
    view_filter: ViewFilter = "none"
    retrieval_score_threshold: float | None = None
    score_threshold_direction: ThresholdDirection | None = None
    query_language: str = "it"

    max_context_chunks: int = 12
    max_context_chars: int = 12_000
    per_chunk_max_chars: int = 1_200

    payload_introspection_sample_size: int = 128
    payload_fields: QdrantPayloadFieldMap = field(default_factory=QdrantPayloadFieldMap)
    advanced: AdvancedRagConfig = field(default_factory=AdvancedRagConfig)

    def with_overrides(self, **overrides: Any) -> "RagRuntimeConfig":
        data = asdict(self)
        data.update(overrides)
        if isinstance(data.get("payload_fields"), dict):
            data["payload_fields"] = QdrantPayloadFieldMap(**data["payload_fields"])
        if isinstance(data.get("advanced"), dict):
            data["advanced"] = AdvancedRagConfig.from_dict(data["advanced"])
        for key in ("dataset_dir", "qdrant_path", "indexing_artifacts_root"):
            if key in data and not isinstance(data[key], Path):
                data[key] = Path(data[key])
        cfg = RagRuntimeConfig(**data)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if self.max_context_chunks <= 0:
            raise ValueError("max_context_chunks must be > 0")
        if self.max_context_chars <= 0:
            raise ValueError("max_context_chars must be > 0")
        if self.per_chunk_max_chars <= 0:
            raise ValueError("per_chunk_max_chars must be > 0")
        if self.payload_introspection_sample_size <= 0:
            raise ValueError("payload_introspection_sample_size must be > 0")
        if self.pipeline_mode not in {"naive", "advanced"}:
            raise ValueError("pipeline_mode must be one of: naive, advanced")
        if not (0.0 <= float(self.index_contract_min_eval_coverage) <= 1.0):
            raise ValueError("index_contract_min_eval_coverage must be in [0, 1]")

        if self.score_threshold_direction is not None and self.retrieval_score_threshold is None:
            raise ValueError(
                "score_threshold_direction is set but retrieval_score_threshold is None"
            )
        if (
            self.retrieval_score_threshold is not None
            and self.score_threshold_direction is None
        ):
            raise ValueError(
                "retrieval_score_threshold is set but score_threshold_direction is None. "
                "Set one of: 'gte', 'lte'."
            )
        if self.view_filter not in {"none", "current", "historical"}:
            raise ValueError(f"view_filter={self.view_filter!r} is not supported")
        self.advanced.validate()

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
    def resolved_indexing_artifacts_root(self) -> Path:
        return self.resolve_path(self.indexing_artifacts_root)

    def to_indexing_config(self) -> IndexingConfig:
        sparse_stopwords_lang = (
            "it"
            if self.advanced.hybrid.query_analyzer in {"it_default", "it_legal"}
            else "none"
        )
        return IndexingConfig(
            dataset_dir=self.dataset_dir,
            qdrant_path=self.qdrant_path,
            artifacts_root=self.indexing_artifacts_root,
            collection_name=self.collection_name,
            embedding_provider=self.embedding_provider,
            utopia_base_url=self.utopia_base_url,
            utopia_embed_api_mode=self.utopia_embed_api_mode,
            utopia_embed_url=self.utopia_embed_url,
            embedding_model=self.embedding_model,
            embedding_api_key=self.embedding_api_key,
            embedding_batch_size=self.embedding_batch_size,
            embedding_timeout_seconds=self.embedding_timeout_seconds,
            chunking_profile=make_chunking_profile("balanced"),
            strict_validation=True,
            sparse_enabled=bool(self.advanced.hybrid.enabled),
            sparse_vector_name="bm25",
            sparse_min_token_len=2,
            sparse_stopwords_lang=sparse_stopwords_lang,
            sparse_store_artifacts=True,
            sparse_analyzer=self.advanced.hybrid.query_analyzer,
            index_contract_min_eval_coverage=self.index_contract_min_eval_coverage,
            index_contract_enforce_eval_coverage=self.enforce_index_contract_coverage,
        )
