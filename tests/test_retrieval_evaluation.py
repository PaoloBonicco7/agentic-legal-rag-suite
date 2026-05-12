from __future__ import annotations

import csv
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from legal_rag.advanced_graph_rag.retrieval import GraphIndex, expand_with_graph
from legal_rag.oracle_context_evaluation.references import OracleReferenceResolver
from legal_rag.retrieval_evaluation import (
    RETRIEVAL_EVALUATION_SCHEMA_VERSION,
    CachedEmbedder,
    ChunkAvailabilityIndex,
    RerankCache,
    answer_overlap,
    evaluate_candidate_set,
    evaluate_with_rerank,
    resolve_question_targets,
    retrieve_direct,
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
        filter_name="none",
        metadata_filters={},
        graph_expansion_enabled=True,
        graph_expansion_seed_k=2,
        max_chunks_per_expanded_law=1,
        min_edge_confidence=0.45,
    )

    assert row.direct_law_hit is True
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


def test_rerank_cache_round_trip(tmp_path) -> None:
    cache = RerankCache(tmp_path / "rerank_cache.jsonl")
    key = RerankCache.make_key(
        question="Quali sono gli organi?",
        candidate_chunk_ids=["c1", "c2", "c3"],
        model="SLURM.gpt-oss:120b",
    )
    assert cache.get(key) is None

    cache.set(key, [{"chunk_id": "c1", "score": 2}, {"chunk_id": "c2", "score": 0}])

    reopened = RerankCache(tmp_path / "rerank_cache.jsonl")
    entries = reopened.get(key)
    assert entries == [{"chunk_id": "c1", "score": 2}, {"chunk_id": "c2", "score": 0}]
    assert len(reopened) == 1


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
        manifest=manifest,
    )

    assert target == output_dir
    assert (output_dir / "scenarios.csv").exists()
    assert (output_dir / "sweep_direct.csv").exists()
    assert (output_dir / "sweep_graph.csv").exists()
    assert (output_dir / "sweep_rerank.csv").exists()
    with (output_dir / "sweep_rerank.csv").open(encoding="utf-8") as handle:
        assert "rerank_model" in next(csv.reader(handle))
    manifest_text = (output_dir / "manifest.json").read_text(encoding="utf-8")
    assert RETRIEVAL_EVALUATION_SCHEMA_VERSION in manifest_text
    assert "created_at" in manifest_text
