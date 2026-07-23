from __future__ import annotations

import csv
from types import SimpleNamespace
from typing import Any

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from legal_rag.advanced_graph_rag import RERANK_PROMPT_VERSION
from legal_rag.advanced_graph_rag.retrieval import (
    GraphIndex,
    expand_with_graph,
    search_dense as search_advanced_dense,
    search_hybrid,
)
from legal_rag.oracle_context_evaluation.references import OracleReferenceResolver
from legal_rag.retrieval_evaluation import (
    PROFILES,
    QUERY_REWRITING_PROMPT_VERSION,
    RETRIEVAL_EVALUATION_SCHEMA_VERSION,
    CachedEmbedder,
    ChunkAvailabilityIndex,
    DiagnosticProfile,
    DirectExperimentCache,
    FILTER_AUDIT_SCHEMA_VERSION,
    QueryRewriteCache,
    RerankCache,
    align_rerank_scores,
    answer_overlap,
    build_collection_identity,
    build_filter_exact_control,
    build_filter_impact,
    build_filter_reference_audit,
    build_waterfall,
    evaluate_candidate_set,
    evaluate_query_rewrite,
    evaluate_with_rerank,
    generate_hyde,
    multi_query,
    paired_bootstrap_interval,
    query_rewrite_cache_path,
    rerank_cache_path,
    resolve_profile,
    resolve_question_targets,
    retrieve_direct,
    retrieve_query_variants,
    sanitize_cache_name,
    score_rerank_candidates,
    select_best_scenario,
    rewrite_query,
    to_advanced_config_recommendation,
    write_run_artifacts,
)
from legal_rag.simple_rag.models import RetrievedChunkRecord

LAW_EXPECTED = "vda:lr:2000-01-01:1"
LAW_SEED = "vda:lr:2000-01-01:2"


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "fake-embedding"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class FakeHybridEmbedder(FakeEmbedder):
    def embed_sparse_texts(self, texts: list[str]) -> list[dict[str, list[float] | list[int]]]:
        return [{"indices": [1], "values": [1.0]} for _ in texts]


class RecordingQdrantClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def get_collection(self, *, collection_name: str) -> Any:
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors={"dense": object()},
                    sparse_vectors={"sparse": object()},
                )
            )
        )

    def query_points(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(points=[])


class FakeRerankClient:
    def __init__(self) -> None:
        self.calls = 0

    def structured_chat(
        self,
        *,
        prompt: str,
        model: str,
        payload_schema: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        self.calls += 1
        assert "chunk_id: c_expected" in prompt
        assert model == "fake-model"
        assert payload_schema["type"] == "object"
        assert timeout_seconds == 30
        return {
            "structured": {
                "scores": [
                    {"chunk_id": "c_expected", "score": 2},
                    {"chunk_id": "c1", "score": 1},
                ]
            }
        }


class FakeQueryRewriteClient:
    def __init__(self) -> None:
        self.calls = 0

    def structured_chat(
        self,
        *,
        prompt: str,
        model: str,
        payload_schema: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        self.calls += 1
        assert "Question?" in prompt
        assert model == "fake-model"
        assert payload_schema["type"] == "object"
        assert timeout_seconds == 30
        if "rewritten_query" in payload_schema.get("properties", {}):
            return {"structured": {"rewritten_query": "organi azienda USL"}}
        if "hypothetical_answer" in payload_schema.get("properties", {}):
            return {"structured": {"hypothetical_answer": "Gli organi dell'azienda USL sono indicati dalla legge regionale."}}
        return {"structured": {"queries": ["organi azienda USL", "direttore generale collegio sindacale", "ente servizio sanitario organi"]}}


def _chunk_payload(
    chunk_id: str,
    *,
    law_id: str,
    article_label_norm: str,
    text: str,
    law_status: str = "current",
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "law_id": law_id,
        "article_id": f"{law_id}#art:{article_label_norm}",
        "article_label_norm": article_label_norm,
        "text": text,
        "law_title": f"Law {law_id}",
        "law_status": law_status,
        "article_status": "current",
        "index_views": ["current"] if law_status == "current" else ["historical"],
        "relation_types": [],
    }


def _chunks() -> list[dict[str, Any]]:
    return [
        _chunk_payload("c_wrong_article", law_id=LAW_EXPECTED, article_label_norm="1", text="Same law, wrong article."),
        _chunk_payload("c_seed", law_id=LAW_SEED, article_label_norm="1", text="Seed law."),
        _chunk_payload(
            "c_expected_article",
            law_id=LAW_EXPECTED,
            article_label_norm="2",
            text="Expected article.",
            law_status="past",
        ),
    ]


def _target() -> tuple[Any, ChunkAvailabilityIndex]:
    chunks = _chunks()
    resolver = OracleReferenceResolver(
        laws=[
            {"law_id": LAW_EXPECTED, "law_title": "Legge regionale 1 gennaio 2000, n. 1"},
            {"law_id": LAW_SEED, "law_title": "Legge regionale 1 gennaio 2000, n. 2"},
        ],
        articles=[
            {
                "law_id": LAW_EXPECTED,
                "article_id": f"{LAW_EXPECTED}#art:1",
                "article_label_norm": "1",
                "article_text": "Wrong article.",
            },
            {
                "law_id": LAW_EXPECTED,
                "article_id": f"{LAW_EXPECTED}#art:2",
                "article_label_norm": "2",
                "article_text": "Expected article.",
            },
            {
                "law_id": LAW_SEED,
                "article_id": f"{LAW_SEED}#art:1",
                "article_label_norm": "1",
                "article_text": "Seed article.",
            },
        ],
    )
    availability = ChunkAvailabilityIndex(chunks)
    records = [
        {
            "qid": "eval-0001",
            "level": "L1",
            "question": "Question?",
            "correct_answer": "Expected article.",
            "expected_references": ["Legge regionale 1 gennaio 2000, n. 1 - Art. 2"],
        }
    ]
    return resolve_question_targets(records, resolver=resolver, availability=availability)[0], availability


def _qdrant() -> QdrantClient:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="test_collection",
        vectors_config={"dense": qmodels.VectorParams(size=4, distance=qmodels.Distance.COSINE)},
    )
    client.upsert(
        collection_name="test_collection",
        points=[
            qmodels.PointStruct(id=1, vector={"dense": [1.0, 0.0, 0.0, 0.0]}, payload=_chunks()[0]),
            qmodels.PointStruct(id=2, vector={"dense": [0.9, 0.1, 0.0, 0.0]}, payload=_chunks()[1]),
            qmodels.PointStruct(id=3, vector={"dense": [0.0, 1.0, 0.0, 0.0]}, payload=_chunks()[2]),
        ],
        wait=True,
    )
    return client


def test_cached_embedder_reuses_dense_query_vectors() -> None:
    base = FakeEmbedder()
    cached = CachedEmbedder(base)

    assert cached.embed_texts(["same question", "same question"]) == [[1.0, 0.0, 0.0, 0.0]] * 2
    assert cached.embed_texts(["same question"]) == [[1.0, 0.0, 0.0, 0.0]]

    assert base.calls == 1
    assert cached.dense_cache_size == 1


def test_dense_exact_search_forwards_qdrant_search_params() -> None:
    client = RecordingQdrantClient()

    search_advanced_dense(
        client,  # type: ignore[arg-type]
        collection_name="collection",
        embedder=FakeEmbedder(),
        query_text="question",
        limit=10,
        static_filters={},
        exact=True,
    )

    assert client.calls[0]["search_params"].exact is True


def test_hybrid_search_applies_filter_to_both_prefetches() -> None:
    client = RecordingQdrantClient()

    search_hybrid(
        client,  # type: ignore[arg-type]
        collection_name="collection",
        embedder=FakeHybridEmbedder(),
        query_text="question",
        limit=10,
        rrf_k=30,
        static_filters={"passage_status": ["current", "partial"]},
        index_manifest={"hybrid_enabled": True},
    )

    prefetches = client.calls[0]["prefetch"]
    assert len(prefetches) == 2
    assert all(prefetch.filter is not None for prefetch in prefetches)
    assert all(
        prefetch.filter.must[0].match.any == ["current", "partial"]
        for prefetch in prefetches
    )


def test_direct_cache_key_includes_query_index_filter_and_exact() -> None:
    identity = build_collection_identity("collection", {"run_id": "run-1"})
    base = DirectExperimentCache._key(
        "mcq",
        "dense",
        10,
        None,
        "active",
        "q1",
        collection_identity=identity,
        query_text="question",
        filters={"law_status": ["current", "partial"]},
        exact=False,
    )

    assert base != DirectExperimentCache._key(
        "mcq",
        "dense",
        10,
        None,
        "active",
        "q1",
        collection_identity=identity,
        query_text="different",
        filters={"law_status": ["current", "partial"]},
        exact=False,
    )
    assert base != DirectExperimentCache._key(
        "mcq",
        "dense",
        10,
        None,
        "active",
        "q1",
        collection_identity=identity,
        query_text="question",
        filters={"law_status": ["current", "partial"]},
        exact=True,
    )
    assert identity != build_collection_identity("collection", {"run_id": "run-2"})


def test_resolve_question_targets_supports_mcq_question_stem() -> None:
    chunks = _chunks()
    resolver = OracleReferenceResolver(
        laws=[{"law_id": LAW_EXPECTED, "law_title": "Legge regionale 1 gennaio 2000, n. 1"}],
        articles=[
            {
                "law_id": LAW_EXPECTED,
                "article_id": f"{LAW_EXPECTED}#art:2",
                "article_label_norm": "2",
                "article_text": "Expected article.",
            }
        ],
    )

    targets = resolve_question_targets(
        [
            {
                "qid": "eval-0001",
                "level": "L1",
                "question_stem": "MCQ stem?",
                "correct_answer": "Expected article.",
                "expected_references": ["Legge regionale 1 gennaio 2000, n. 1 - Art. 2"],
            }
        ],
        resolver=resolver,
        availability=ChunkAvailabilityIndex(chunks),
        question_key="question_stem",
    )

    assert targets[0].question == "MCQ stem?"
    assert targets[0].expected_article_ids == [f"{LAW_EXPECTED}#art:2"]


def test_evaluate_candidate_set_distinguishes_law_hit_from_article_hit_and_graph_recovery() -> None:
    target, availability = _target()
    client = _qdrant()
    embedder = CachedEmbedder(FakeEmbedder())
    retrieved = retrieve_direct(
        client=client,
        collection_name="test_collection",
        embedder=embedder,
        query_text=target.question,
        limit=2,
        retrieval_mode="dense",
        static_filters={},
        rrf_k=60,
        index_manifest={},
    )
    graph = GraphIndex(
        edges=[
            {
                "src_law_id": LAW_SEED,
                "dst_law_id": LAW_EXPECTED,
                "dst_article_label_norm": "2",
                "relation_type": "REFERENCES",
                "confidence": 0.9,
            }
        ],
        chunks=_chunks(),
    )
    expanded, _ = expand_with_graph(
        client,
        collection_name="test_collection",
        graph=graph,
        seeds=retrieved[:2],
        relation_types=["REFERENCES"],
        static_filters={},
        max_chunks_per_law=1,
        embedder=embedder,
        query_text=target.question,
        max_chunks_total=1,
        min_edge_confidence=0.45,
    )

    row = evaluate_candidate_set(
        target=target,
        retrieved=retrieved,
        expanded=expanded,
        availability=availability,
        retrieval_mode="dense",
        top_k=2,
        rrf_k=None,
        filter_name="none",
        metadata_filters={},
        graph_expansion_enabled=True,
        graph_expansion_seed_k=2,
        max_chunks_per_expanded_law=1,
        min_edge_confidence=0.45,
    )

    assert row.direct_law_hit is True
    assert row.rrf_k is None
    assert row.direct_article_hit is False
    assert row.law_only_false_positive is True
    assert row.post_article_hit is True
    assert row.graph_incremental_hit is True
    assert row.expanded_expected_article_hits == 1
    assert row.expansion_noise_ratio == 0.0


def test_filter_excluded_marks_expected_article_removed_by_metadata_filter() -> None:
    target, availability = _target()

    row = evaluate_candidate_set(
        target=target,
        retrieved=[],
        expanded=[],
        availability=availability,
        retrieval_mode="dense",
        top_k=5,
        filter_name="current_law",
        metadata_filters={"law_status": "current"},
    )

    assert row.expected_article_chunk_count == 1
    assert row.expected_article_filtered_chunk_count == 0
    assert row.filter_excluded is True


def test_filter_reference_audit_distinguishes_partial_from_full_exclusion() -> None:
    target, _ = _target()
    article_id = target.expected_article_ids[0]
    chunks = [
        {
            "chunk_id": "current",
            "law_id": LAW_EXPECTED,
            "article_id": article_id,
            "passage_id": f"{article_id}#p:c1",
            "law_status": "current",
            "article_status": "partial",
            "passage_status": "current",
            "content_availability": "substantive",
            "index_views": ["historical", "current", "not_explicitly_past"],
            "status_event_ids": [],
            "status_rule_ids": ["passage-default-current"],
        },
        {
            "chunk_id": "past",
            "law_id": LAW_EXPECTED,
            "article_id": article_id,
            "passage_id": f"{article_id}#p:c2",
            "law_status": "current",
            "article_status": "partial",
            "passage_status": "past",
            "content_availability": "substantive",
            "index_views": ["historical"],
            "status_event_ids": ["event-1"],
            "status_rule_ids": ["passage-repeal"],
        },
    ]
    filters = {
        "none": {},
        "passage_active": {"passage_status": ["current", "partial"]},
        "current_view": {"index_views": "current"},
    }

    audit = build_filter_reference_audit(
        targets_by_dataset={"mcq": [target], "no_hint": [target]},
        availability=ChunkAvailabilityIndex(chunks),
        filter_names=list(filters),
        filter_variants=filters,
        collection_identity="identity",
    )

    assert len(audit) == 3
    assert audit.loc[audit["filter_name"] == "none", "coverage_status"].item() == "fully_eligible"
    active = audit[audit["filter_name"] == "passage_active"].iloc[0]
    assert active["coverage_status"] == "partially_eligible"
    assert active["active_target_chunks"] == 1
    assert active["retained_active_target_chunks"] == 1
    assert active["datasets"] == ["mcq", "no_hint"]


def test_filter_impact_uses_paired_bootstrap_and_separate_verdict_axes() -> None:
    direct = pd.DataFrame(
        [
            {
                **_direct_row("q1", hit=True, mode="hybrid", filter_name="none"),
                "dataset": "no_hint",
                "metadata_filters": {},
                "exact": False,
            },
            {
                **_direct_row(
                    "q1",
                    hit=False,
                    mode="hybrid",
                    filter_name="index_views_current",
                ),
                "dataset": "no_hint",
                "metadata_filters": {"passage_status": ["current", "partial"]},
                "exact": False,
            },
        ]
    )
    reference_audit = pd.DataFrame(
        [
            {
                "qid": "q1",
                "datasets": ["no_hint"],
                "filter_name": filter_name,
                "coverage_status": "fully_eligible",
                "active_target_chunks": 1,
                "retained_active_target_chunks": 1,
                "unknown_status_excluded": False,
                "expected_reference_validity": "current",
                "supporting_passage_retained": True,
            }
            for filter_name in ("none", "index_views_current")
        ]
    )

    impact = build_filter_impact(direct, reference_audit, resamples=100, seed=42)
    active = impact[impact["filter_name"] == "index_views_current"].iloc[0]

    assert active["article_success_delta_pp"] == -100.0
    assert active["losses"] == 1
    assert active["active_slice_safety"] == "safe"
    assert active["retrieval_effect"] == "harmful"
    assert active["article_success_ci_level"] == 0.975
    assert bool(active["bootstrap_supported"]) is True


def test_paired_bootstrap_is_deterministic() -> None:
    first = paired_bootstrap_interval(
        [1, 0, 1, 1],
        [0, 0, 1, 0],
        resamples=200,
        seed=42,
    )
    second = paired_bootstrap_interval(
        [1, 0, 1, 1],
        [0, 0, 1, 0],
        resamples=200,
        seed=42,
    )

    assert first == second
    assert first[0] == 0.5


def test_exact_control_reports_chunk_overlap_and_metric_delta() -> None:
    ann = {
        **_direct_row("q1", hit=False),
        "exact": False,
        "retrieved_chunk_ids": ["a", "b"],
    }
    exact = {
        **_direct_row("q1", hit=True),
        "exact": True,
        "retrieved_chunk_ids": ["a", "c"],
    }

    control = build_filter_exact_control(pd.DataFrame([ann, exact]))

    assert len(control) == 1
    assert control.iloc[0]["mean_chunk_overlap"] == 0.5
    assert control.iloc[0]["article_success_delta_pp"] == 100.0


def test_rerank_cache_round_trip(tmp_path) -> None:
    cache = RerankCache(tmp_path / "rerank_cache.jsonl")
    key = RerankCache.make_key(
        question="Quali sono gli organi?",
        candidate_chunk_ids=["c1", "c2", "c3"],
        model="SLURM.gpt-oss:120b",
        prompt_version=RERANK_PROMPT_VERSION,
    )
    assert cache.get(key) is None

    cache.set(key, [{"chunk_id": "c1", "score": 2}, {"chunk_id": "c2", "score": 0}])

    reopened = RerankCache(tmp_path / "rerank_cache.jsonl")
    entries = reopened.get(key)
    assert entries == [{"chunk_id": "c1", "score": 2}, {"chunk_id": "c2", "score": 0}]
    assert len(reopened) == 1


def test_rerank_cache_key_changes_with_prompt_version() -> None:
    base = {
        "question": "Quali sono gli organi?",
        "candidate_chunk_ids": ["c1", "c2", "c3"],
        "model": "SLURM.gpt-oss:120b",
    }

    assert RerankCache.make_key(**base, prompt_version="rerank-v1") != RerankCache.make_key(
        **base,
        prompt_version="rerank-v2",
    )


def test_rerank_cache_path_sanitizes_model_name(tmp_path) -> None:
    assert sanitize_cache_name("SLURM.gpt-oss:120b") == "slurm.gpt-oss_120b"
    assert rerank_cache_path(tmp_path, model="SLURM.gpt-oss:120b") == tmp_path / "rerank" / "slurm.gpt-oss_120b.jsonl"


def test_score_rerank_candidates_uses_cache_on_second_call(tmp_path) -> None:
    client = FakeRerankClient()
    cache = RerankCache(tmp_path / "rerank.jsonl")
    candidates = [
        RetrievedChunkRecord(chunk_id="c1", score=0.0, text="...", payload={"chunk_id": "c1"}),
        RetrievedChunkRecord(chunk_id="c_expected", score=0.0, text="...", payload={"chunk_id": "c_expected"}),
    ]

    first_scores, first_hit = score_rerank_candidates(
        llm_client=client,
        question="Question?",
        candidates=candidates,
        model="fake-model",
        timeout_seconds=30,
        cache=cache,
        prompt_version=RERANK_PROMPT_VERSION,
    )
    second_scores, second_hit = score_rerank_candidates(
        llm_client=client,
        question="Question?",
        candidates=candidates,
        model="fake-model",
        timeout_seconds=30,
        cache=cache,
        prompt_version=RERANK_PROMPT_VERSION,
    )

    assert first_scores == [1, 2]
    assert second_scores == [1, 2]
    assert first_hit is False
    assert second_hit is True
    assert client.calls == 1


def test_align_rerank_scores_defaults_missing_candidates_to_zero() -> None:
    candidates = [
        RetrievedChunkRecord(chunk_id="c1", score=0.0, text="...", payload={"chunk_id": "c1"}),
        RetrievedChunkRecord(chunk_id="c2", score=0.0, text="...", payload={"chunk_id": "c2"}),
    ]

    assert align_rerank_scores(candidates=candidates, score_entries=[{"chunk_id": "c2", "score": 2}]) == [0, 2]


def test_query_rewrite_cache_round_trip(tmp_path) -> None:
    cache = QueryRewriteCache(tmp_path / "rewrite.jsonl")
    key = QueryRewriteCache.make_key(
        question="Quali sono gli organi?",
        strategy="rewrite",
        model="SLURM.gpt-oss:120b",
        prompt_version=QUERY_REWRITING_PROMPT_VERSION,
    )
    assert cache.get(key) is None

    cache.set(key, ["organi azienda USL"])

    reopened = QueryRewriteCache(tmp_path / "rewrite.jsonl")
    assert reopened.get(key) == ["organi azienda USL"]
    assert len(reopened) == 1


def test_query_rewrite_cache_path_uses_strategy_model_and_prompt_version(tmp_path) -> None:
    assert query_rewrite_cache_path(
        tmp_path,
        strategy="multi_query",
        model="SLURM.gpt-oss:120b",
        prompt_version="query-rewriting-v1",
    ) == tmp_path / "query_rewriting" / "multi_query__slurm.gpt-oss_120b__query-rewriting-v1.jsonl"


def test_rewrite_query_uses_cache_on_second_call(tmp_path) -> None:
    client = FakeQueryRewriteClient()
    cache = QueryRewriteCache(tmp_path / "rewrite.jsonl")

    first = rewrite_query(
        llm_client=client,
        question="Question?",
        model="fake-model",
        timeout_seconds=30,
        cache=cache,
    )
    second = rewrite_query(
        llm_client=client,
        question="Question?",
        model="fake-model",
        timeout_seconds=30,
        cache=cache,
    )

    assert first.queries == ["organi azienda USL"]
    assert second.queries == ["organi azienda USL"]
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert client.calls == 1


def test_generate_hyde_returns_one_hypothetical_document(tmp_path) -> None:
    result = generate_hyde(
        llm_client=FakeQueryRewriteClient(),
        question="Question?",
        model="fake-model",
        timeout_seconds=30,
        cache=QueryRewriteCache(tmp_path / "hyde.jsonl"),
    )

    assert result.strategy == "hyde"
    assert len(result.queries) == 1
    assert "azienda USL" in result.queries[0]


def test_multi_query_returns_exactly_three_non_empty_queries(tmp_path) -> None:
    result = multi_query(
        llm_client=FakeQueryRewriteClient(),
        question="Question?",
        model="fake-model",
        timeout_seconds=30,
        cache=QueryRewriteCache(tmp_path / "multi.jsonl"),
        n=3,
    )

    assert result.strategy == "multi_query"
    assert len(result.queries) == 3
    assert all(query.strip() for query in result.queries)


def test_retrieve_query_variants_deduplicates_candidate_set() -> None:
    client = _qdrant()
    embedder = CachedEmbedder(FakeEmbedder())

    candidates = retrieve_query_variants(
        client=client,
        collection_name="test_collection",
        embedder=embedder,
        query_texts=["Question?", "Question?"],
        limit=2,
        retrieval_mode="dense",
        static_filters={},
        rrf_k=60,
        index_manifest={},
    )

    assert [chunk.chunk_id for chunk in candidates] == ["c_wrong_article", "c_seed"]
    assert embedder.dense_cache_size == 1


def test_evaluate_query_rewrite_records_rewritten_candidate_metrics() -> None:
    target, _ = _target()
    expected_article_id = f"{LAW_EXPECTED}#art:2"
    retrieved = [
        RetrievedChunkRecord(
            chunk_id="c_expected",
            score=0.0,
            text="...",
            payload={"chunk_id": "c_expected", "law_id": LAW_EXPECTED, "article_id": expected_article_id},
        )
    ]

    row = evaluate_query_rewrite(
        target=target,
        retrieved=retrieved,
        retrieval_mode="hybrid",
        top_k=5,
        rrf_k=30,
        filter_name="none",
        metadata_filters={},
        strategy="rewrite",
        rewritten_queries=["organi azienda USL"],
        query_rewriting_model="fake-model",
        query_rewriting_prompt_version=QUERY_REWRITING_PROMPT_VERSION,
        cache_hit=False,
    )

    assert row.direct_article_hit is True
    assert row.strategy == "rewrite"
    assert row.transformed_query_count == 1
    assert row.rrf_k == 30


def test_evaluate_with_rerank_promotes_expected_article() -> None:
    target, _ = _target()
    expected_article_id = f"{LAW_EXPECTED}#art:2"
    candidates = [
        RetrievedChunkRecord(
            chunk_id=f"c{i}",
            score=0.0,
            text="...",
            payload={
                "chunk_id": f"c{i}",
                "law_id": LAW_EXPECTED,
                "article_id": f"{LAW_EXPECTED}#art:1",
            },
        )
        for i in range(4)
    ] + [
        RetrievedChunkRecord(
            chunk_id="c_expected",
            score=0.0,
            text="...",
            payload={
                "chunk_id": "c_expected",
                "law_id": LAW_EXPECTED,
                "article_id": expected_article_id,
            },
        )
    ]
    scores = [0, 0, 0, 0, 2]

    row = evaluate_with_rerank(
        target=target,
        candidates=candidates,
        scores=scores,
        rerank_input_k=5,
        rerank_output_k=3,
        retrieval_mode="dense",
        top_k=5,
        rrf_k=None,
        filter_name="none",
        metadata_filters={},
        base_scenario="test",
        rerank_model="fake-model",
        cache_hit=False,
    )

    assert row.pre_rerank_article_hit is True
    assert row.pre_rerank_first_article_rank == 5
    assert row.reranked_article_hit is True
    assert row.reranked_first_article_rank == 1
    assert row.rerank_recovered_article is False
    assert row.rerank_demoted_article is False


def test_answer_overlap_returns_jaccard_on_long_tokens() -> None:
    correct = "organi azienda direttore generale"
    chunk_match = "Il direttore generale e il collegio sindacale sono organi"
    chunk_unrelated = "La programmazione si attua tramite atti deliberativi"

    assert 0.0 < answer_overlap(correct, chunk_match) <= 1.0
    assert answer_overlap(correct, chunk_unrelated) == 0.0
    assert answer_overlap("", "anything") == 0.0


def test_write_run_artifacts_creates_csvs_and_manifest(tmp_path) -> None:
    output_dir = tmp_path / "run_alpha"
    scenarios = [
        {
            "scenario_name": "Baseline",
            "dataset": "mcq",
            "stage": "direct",
            "article_hit_pct": 22.0,
            "law_hit_pct": 41.0,
            "config": {"top_k": 10},
        }
    ]
    sweep_direct = [{"qid": "eval-0001", "top_k": 10, "direct_article_hit": True}]
    sweep_graph = [{"qid": "eval-0001", "graph_expansion_seed_k": 3, "post_article_hit": True}]
    sweep_rerank: list[dict] = []
    sweep_query_rewriting: list[dict] = []
    manifest = {
        "run_name": "test",
        "n_questions": 1,
        "source_hashes": {"chunks": "abc"},
    }

    target = write_run_artifacts(
        output_dir,
        scenarios=scenarios,
        sweep_direct=sweep_direct,
        sweep_graph=sweep_graph,
        sweep_rerank=sweep_rerank,
        sweep_query_rewriting=sweep_query_rewriting,
        filter_reference_audit=[{"qid": "eval-0001", "coverage_status": "fully_eligible"}],
        filter_exclusions=[],
        filter_impact=[{"filter_name": "none", "article_success_delta_pp": 0.0}],
        filter_exact_control=[],
        manifest=manifest,
    )

    assert target == output_dir
    assert (output_dir / "scenarios.csv").exists()
    assert (output_dir / "sweep_direct.csv").exists()
    assert (output_dir / "sweep_graph.csv").exists()
    assert (output_dir / "sweep_rerank.csv").exists()
    assert (output_dir / "sweep_query_rewriting.csv").exists()
    assert (output_dir / "filter_reference_audit.csv").exists()
    assert (output_dir / "filter_exclusions.csv").exists()
    assert (output_dir / "filter_impact.csv").exists()
    assert (output_dir / "filter_exact_control.csv").exists()
    with (output_dir / "sweep_rerank.csv").open(encoding="utf-8") as handle:
        assert "rerank_model" in next(csv.reader(handle))
    with (output_dir / "sweep_query_rewriting.csv").open(encoding="utf-8") as handle:
        assert "strategy" in next(csv.reader(handle))
    manifest_text = (output_dir / "manifest.json").read_text(encoding="utf-8")
    assert RETRIEVAL_EVALUATION_SCHEMA_VERSION in manifest_text
    assert FILTER_AUDIT_SCHEMA_VERSION in manifest_text
    assert "filter_reference_audit.csv" in manifest_text
    assert "created_at" in manifest_text


def test_resolve_profile_returns_light_by_default() -> None:
    profile = resolve_profile()
    assert isinstance(profile, DiagnosticProfile)
    assert profile.name == "light"
    assert "light" in PROFILES
    assert "full" in PROFILES


def test_resolve_profile_rejects_unknown_name() -> None:
    import pytest

    with pytest.raises(ValueError):
        resolve_profile("nonexistent")


def _direct_row(qid: str, *, hit: bool, top_k: int = 10, mode: str = "dense", filter_name: str = "none") -> dict[str, Any]:
    return {
        "qid": qid,
        "level": "L1",
        "question": "Q?",
        "retrieval_mode": mode,
        "top_k": top_k,
        "rrf_k": 60 if mode == "hybrid" else None,
        "filter_name": filter_name,
        "metadata_filters": {},
        "dataset": "mcq",
        "direct_article_hit": hit,
        "direct_law_hit": hit,
        "direct_article_mrr": 1.0 if hit else 0.0,
        "direct_all_expected_articles_hit": hit,
        "post_article_hit": hit,
        "post_law_hit": hit,
        "post_article_mrr": 1.0 if hit else 0.0,
        "filter_excluded": False,
        "law_only_false_positive": False,
        "graph_incremental_hit": False,
        "direct_best_expected_article_answer_overlap": None,
    }


def test_build_waterfall_produces_required_scenario_columns() -> None:
    direct_df = pd.DataFrame(
        [_direct_row(f"q{i}", hit=True if i < 3 else False, top_k=10) for i in range(5)]
        + [_direct_row(f"q{i}", hit=True if i < 4 else False, top_k=50) for i in range(5)]
    )
    from legal_rag.retrieval_evaluation import summarize_direct

    direct_summary = summarize_direct(direct_df)
    scenarios = build_waterfall(
        datasets=["mcq"],
        profile=resolve_profile("light"),
        direct_df=direct_df,
        direct_summary_df=direct_summary,
        graph_df=pd.DataFrame(),
        graph_summary_df=pd.DataFrame(),
        best_graph_by_dataset_df=pd.DataFrame(),
        rerank_df=pd.DataFrame(),
        query_rewrite_df=pd.DataFrame(),
        hybrid_available=False,
        graph_enabled=False,
        rerank_enabled=False,
        query_rewriting_enabled=False,
    )
    required = {
        "scenario_name",
        "dataset",
        "stage",
        "article_hit_pct",
        "law_hit_pct",
        "article_mrr",
        "n_questions",
        "n_filter_excluded",
        "config",
        "delta_vs_baseline",
        "experiment_name",
        "status",
        "skip_reason",
    }
    assert required <= set(scenarios.columns)
    baseline = scenarios[scenarios["experiment_name"] == "dense_baseline_top10"].iloc[0]
    assert baseline["status"] == "run"
    assert baseline["article_hit_pct"] == 60.0


def test_select_best_scenario_returns_highest_hit() -> None:
    scenarios = pd.DataFrame(
        [
            {
                "experiment_name": "dense_baseline_top10",
                "scenario_name": "Baseline",
                "dataset": "mcq",
                "stage": "direct",
                "status": "run",
                "article_hit_pct": 22.0,
                "law_hit_pct": 40.0,
                "article_mrr": 0.2,
                "n_questions": 100,
                "n_filter_excluded": 0,
                "config": {},
                "delta_vs_baseline": None,
                "skip_reason": "",
            },
            {
                "experiment_name": "dense_topk_curve",
                "scenario_name": "Best dense",
                "dataset": "mcq",
                "stage": "direct",
                "status": "run",
                "article_hit_pct": 45.0,
                "law_hit_pct": 70.0,
                "article_mrr": 0.4,
                "n_questions": 100,
                "n_filter_excluded": 0,
                "config": {"top_k": 50},
                "delta_vs_baseline": 23.0,
                "skip_reason": "",
            },
        ]
    )
    best = select_best_scenario(scenarios, dataset="mcq")
    assert best is not None
    assert best["experiment_name"] == "dense_topk_curve"
    assert best["article_hit_pct"] == 45.0


def test_to_advanced_config_recommendation_returns_none_when_empty() -> None:
    result = to_advanced_config_recommendation(
        scenarios_df=pd.DataFrame(),
        datasets=[],
        profile=resolve_profile("light"),
        advanced_rag_defaults={"top_k": 10, "rrf_k": 60},
    )
    assert result is None


def test_candidate_metrics_returns_article_hit_at_k_for_default_k_values() -> None:
    from legal_rag.retrieval_evaluation import ARTICLE_HIT_K_VALUES, candidate_metrics

    expected_article = f"{LAW_EXPECTED}#art:2"
    payloads = [
        {"law_id": LAW_SEED, "article_id": f"{LAW_SEED}#art:1"},  # rank 1 - wrong
        {"law_id": LAW_SEED, "article_id": f"{LAW_SEED}#art:2"},  # rank 2 - wrong
        {"law_id": LAW_SEED, "article_id": f"{LAW_SEED}#art:3"},  # rank 3 - wrong
        {"law_id": LAW_SEED, "article_id": f"{LAW_SEED}#art:4"},  # rank 4 - wrong
        {"law_id": LAW_SEED, "article_id": f"{LAW_SEED}#art:5"},  # rank 5 - wrong
        {"law_id": LAW_SEED, "article_id": f"{LAW_SEED}#art:6"},  # rank 6 - wrong
        {"law_id": LAW_EXPECTED, "article_id": expected_article},  # rank 7 - HIT
    ]
    chunks = [
        RetrievedChunkRecord(chunk_id=f"c{i}", score=1.0 - i * 0.01, text="", payload=payload)
        for i, payload in enumerate(payloads)
    ]
    metrics = candidate_metrics(
        chunks,
        expected_law_ids=[LAW_EXPECTED],
        expected_article_ids=[expected_article],
    )
    assert metrics.article_hit is True
    assert metrics.first_article_rank == 7
    # Default ARTICLE_HIT_K_VALUES = (5, 10, 20). Expected article is at rank 7:
    # below 5 (False), at/below 10 (True), at/below 20 (True).
    assert set(int(k) for k in metrics.article_hit_at_k) == set(ARTICLE_HIT_K_VALUES)
    assert metrics.article_hit_at_k["5"] is False
    assert metrics.article_hit_at_k["10"] is True
    assert metrics.article_hit_at_k["20"] is True


def test_candidate_metrics_article_hit_at_k_is_false_when_no_hit() -> None:
    from legal_rag.retrieval_evaluation import candidate_metrics

    expected_article = f"{LAW_EXPECTED}#art:99"
    chunks = [
        RetrievedChunkRecord(
            chunk_id=f"c{i}",
            score=1.0,
            text="",
            payload={"law_id": LAW_SEED, "article_id": f"{LAW_SEED}#art:{i}"},
        )
        for i in range(3)
    ]
    metrics = candidate_metrics(
        chunks,
        expected_law_ids=[LAW_EXPECTED],
        expected_article_ids=[expected_article],
        k_values=(5, 10),
    )
    assert metrics.article_hit is False
    assert metrics.article_hit_at_k == {"5": False, "10": False}


def test_candidate_metrics_reports_first_hit_and_mrr_for_multiple_articles() -> None:
    from legal_rag.retrieval_evaluation import candidate_metrics

    expected_articles = [f"{LAW_EXPECTED}#art:2", f"{LAW_EXPECTED}#art:3"]
    chunks = [
        RetrievedChunkRecord(
            chunk_id="wrong",
            score=1.0,
            text="",
            payload={"law_id": LAW_SEED, "article_id": f"{LAW_SEED}#art:1"},
        ),
        RetrievedChunkRecord(
            chunk_id="first_expected",
            score=0.9,
            text="",
            payload={"law_id": LAW_EXPECTED, "article_id": expected_articles[0]},
        ),
        RetrievedChunkRecord(
            chunk_id="duplicate_expected",
            score=0.8,
            text="",
            payload={"law_id": LAW_EXPECTED, "article_id": expected_articles[0]},
        ),
        RetrievedChunkRecord(
            chunk_id="second_expected",
            score=0.7,
            text="",
            payload={"law_id": LAW_EXPECTED, "article_id": expected_articles[1]},
        ),
    ]

    metrics = candidate_metrics(
        chunks,
        expected_law_ids=[LAW_EXPECTED],
        expected_article_ids=expected_articles,
    )

    assert metrics.article_hit is True
    assert metrics.law_hit is True
    assert metrics.first_article_rank == 2
    assert metrics.article_mrr == 0.5
