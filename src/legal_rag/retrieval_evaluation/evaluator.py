"""Evaluation helpers for retrieval-only RAG experiments."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from legal_rag.advanced_graph_rag.retrieval import search_dense, search_hybrid
from legal_rag.indexing.embeddings import SupportsEmbedding
from legal_rag.oracle_context_evaluation.io import (
    now_utc,
    prepare_tmp_output_dir,
    replace_output_dir,
    sha256_text,
    write_json,
)
from legal_rag.oracle_context_evaluation.references import OracleReferenceResolver, split_reference_values
from legal_rag.simple_rag.models import RetrievedChunkRecord

from .models import (
    RETRIEVAL_EVALUATION_SCHEMA_VERSION,
    CandidateMetrics,
    QuestionTarget,
    ReferenceTarget,
    RerankEvaluationRow,
    RetrievalEvaluationRow,
    RetrievalScenarioSummary,
)


class CachedEmbedder:
    """Cache query embeddings across repeated diagnostic sweeps."""

    def __init__(self, embedder: SupportsEmbedding) -> None:
        self._embedder = embedder
        self._dense_cache: dict[str, list[float]] = {}
        self._sparse_cache: dict[str, Any] = {}

    @property
    def model_name(self) -> str:
        """Return the wrapped embedder model identifier."""
        return self._embedder.model_name

    @property
    def dense_cache_size(self) -> int:
        """Return the number of cached dense query embeddings."""
        return len(self._dense_cache)

    @property
    def sparse_cache_size(self) -> int:
        """Return the number of cached sparse query embeddings."""
        return len(self._sparse_cache)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts, reusing cached vectors for repeated queries."""
        cleaned = [str(text or "") for text in texts]
        missing = _unique(text for text in cleaned if text not in self._dense_cache)
        if missing:
            vectors = self._embedder.embed_texts(missing)
            for text, vector in zip(missing, vectors):
                self._dense_cache[text] = vector
        return [self._dense_cache[text] for text in cleaned]

    def embed_sparse_texts(self, texts: list[str]) -> list[Any]:
        """Embed sparse vectors when the wrapped embedder exposes them."""
        method = getattr(self._embedder, "embed_sparse_texts", None) or getattr(
            self._embedder,
            "sparse_embed_texts",
            None,
        )
        if not callable(method):
            raise RuntimeError("Wrapped embedder does not expose sparse embeddings")
        cleaned = [str(text or "") for text in texts]
        missing = _unique(text for text in cleaned if text not in self._sparse_cache)
        if missing:
            vectors = method(missing)
            for text, vector in zip(missing, vectors):
                self._sparse_cache[text] = vector
        return [self._sparse_cache[text] for text in cleaned]


class ChunkAvailabilityIndex:
    """Small in-memory index for expected chunk coverage diagnostics."""

    def __init__(self, chunks: Sequence[dict[str, Any]]) -> None:
        self._chunks_by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._chunks_by_law: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chunk in chunks:
            article_id = str(chunk.get("article_id") or "")
            law_id = str(chunk.get("law_id") or "")
            if article_id:
                self._chunks_by_article[article_id].append(dict(chunk))
            if law_id:
                self._chunks_by_law[law_id].append(dict(chunk))

    def article_chunk_count(self, article_ids: Sequence[str], *, filters: dict[str, Any] | None = None) -> int:
        """Count chunks for the selected articles, optionally after metadata filters."""
        return self._count(self._chunks_by_article, article_ids, filters=filters)

    def law_chunk_count(self, law_ids: Sequence[str], *, filters: dict[str, Any] | None = None) -> int:
        """Count chunks for the selected laws, optionally after metadata filters."""
        return self._count(self._chunks_by_law, law_ids, filters=filters)

    def _count(
        self,
        mapping: dict[str, list[dict[str, Any]]],
        keys: Sequence[str],
        *,
        filters: dict[str, Any] | None,
    ) -> int:
        selected = []
        for key in _unique(keys):
            selected.extend(mapping.get(key, []))
        if not filters:
            return len(selected)
        return sum(1 for chunk in selected if payload_matches_filters(chunk, filters))


def resolve_question_targets(
    records: Sequence[dict[str, Any]],
    *,
    resolver: OracleReferenceResolver,
    availability: ChunkAvailabilityIndex,
    question_key: str = "question",
) -> list[QuestionTarget]:
    """Resolve expected references for retrieval evaluation records."""
    targets: list[QuestionTarget] = []
    for record in records:
        references: list[ReferenceTarget] = []
        for reference_text in split_reference_values([str(value) for value in record.get("expected_references", [])]):
            resolved = resolver.resolve_reference(reference_text)
            references.append(
                ReferenceTarget(
                    reference_text=reference_text,
                    law_id=resolved.law_id,
                    article_id=resolved.article_id,
                    article_label_norm=resolved.article_label_norm,
                )
            )
        expected_law_ids = _unique(reference.law_id for reference in references)
        expected_article_ids = _unique(reference.article_id for reference in references)
        targets.append(
            QuestionTarget(
                qid=str(record.get("qid") or ""),
                level=str(record.get("level") or ""),
                question=str(record.get(question_key) or ""),
                correct_answer=str(record.get("correct_answer") or ""),
                references=references,
                expected_law_ids=expected_law_ids,
                expected_article_ids=expected_article_ids,
                expected_law_chunk_count=availability.law_chunk_count(expected_law_ids),
                expected_article_chunk_count=availability.article_chunk_count(expected_article_ids),
            )
        )
    return targets


def retrieve_direct(
    *,
    client: QdrantClient,
    collection_name: str,
    embedder: SupportsEmbedding,
    query_text: str,
    limit: int,
    retrieval_mode: str,
    static_filters: dict[str, Any],
    rrf_k: int,
    index_manifest: dict[str, Any],
) -> list[RetrievedChunkRecord]:
    """Run dense or hybrid retrieval using the advanced RAG helpers."""
    if retrieval_mode == "hybrid":
        return search_hybrid(
            client,
            collection_name=collection_name,
            embedder=embedder,
            query_text=query_text,
            limit=limit,
            rrf_k=rrf_k,
            static_filters=static_filters,
            index_manifest=index_manifest,
        )
    if retrieval_mode == "dense":
        return search_dense(
            client,
            collection_name=collection_name,
            embedder=embedder,
            query_text=query_text,
            limit=limit,
            static_filters=static_filters,
        )
    raise ValueError(f"Unsupported retrieval_mode: {retrieval_mode!r}")


def evaluate_candidate_set(
    *,
    target: QuestionTarget,
    retrieved: Sequence[RetrievedChunkRecord],
    expanded: Sequence[RetrievedChunkRecord] | None,
    availability: ChunkAvailabilityIndex,
    retrieval_mode: str,
    top_k: int,
    filter_name: str,
    metadata_filters: dict[str, Any],
    graph_expansion_enabled: bool = False,
    graph_expansion_seed_k: int | None = None,
    max_chunks_per_expanded_law: int | None = None,
    min_edge_confidence: float | None = None,
) -> RetrievalEvaluationRow:
    """Evaluate direct and post-expansion candidates for one question."""
    expanded_chunks = list(expanded or [])
    retrieved_chunks = list(retrieved)
    candidates = dedupe_chunks([*retrieved_chunks, *expanded_chunks])
    direct = candidate_metrics(
        retrieved_chunks,
        expected_law_ids=target.expected_law_ids,
        expected_article_ids=target.expected_article_ids,
    )
    post = candidate_metrics(
        candidates,
        expected_law_ids=target.expected_law_ids,
        expected_article_ids=target.expected_article_ids,
    )
    expanded_expected_articles = {
        str(chunk.payload.get("article_id") or "")
        for chunk in expanded_chunks
        if str(chunk.payload.get("article_id") or "") in set(target.expected_article_ids)
    }
    expansion_noise_ratio = None
    if expanded_chunks:
        noisy = sum(
            1
            for chunk in expanded_chunks
            if str(chunk.payload.get("article_id") or "") not in set(target.expected_article_ids)
        )
        expansion_noise_ratio = noisy / len(expanded_chunks)
    filtered_expected_count = availability.article_chunk_count(target.expected_article_ids, filters=metadata_filters)
    return RetrievalEvaluationRow(
        qid=target.qid,
        level=target.level,
        question=target.question,
        retrieval_mode=retrieval_mode,  # type: ignore[arg-type]
        top_k=top_k,
        filter_name=filter_name,
        metadata_filters=dict(metadata_filters),
        graph_expansion_enabled=graph_expansion_enabled,
        graph_expansion_seed_k=graph_expansion_seed_k,
        max_chunks_per_expanded_law=max_chunks_per_expanded_law,
        min_edge_confidence=min_edge_confidence,
        expected_law_ids=target.expected_law_ids,
        expected_article_ids=target.expected_article_ids,
        expected_law_chunk_count=target.expected_law_chunk_count,
        expected_article_chunk_count=target.expected_article_chunk_count,
        expected_article_filtered_chunk_count=filtered_expected_count,
        filter_excluded=filtered_expected_count == 0,
        retrieved_count=len(retrieved_chunks),
        expanded_count=len(expanded_chunks),
        candidate_count=len(candidates),
        direct_law_hit=direct.law_hit,
        direct_article_hit=direct.article_hit,
        direct_all_expected_articles_hit=direct.all_expected_articles_hit,
        direct_first_law_rank=direct.first_law_rank,
        direct_first_article_rank=direct.first_article_rank,
        direct_article_mrr=direct.article_mrr,
        law_only_false_positive=direct.law_only_false_positive,
        post_law_hit=post.law_hit,
        post_article_hit=post.article_hit,
        post_all_expected_articles_hit=post.all_expected_articles_hit,
        post_first_law_rank=post.first_law_rank,
        post_first_article_rank=post.first_article_rank,
        post_article_mrr=post.article_mrr,
        graph_incremental_hit=not direct.article_hit and post.article_hit,
        expanded_expected_article_hits=len(expanded_expected_articles),
        expansion_noise_ratio=expansion_noise_ratio,
        retrieved_chunk_ids=[chunk.chunk_id for chunk in retrieved_chunks],
        expanded_chunk_ids=[chunk.chunk_id for chunk in expanded_chunks],
    )


def candidate_metrics(
    chunks: Sequence[RetrievedChunkRecord],
    *,
    expected_law_ids: Sequence[str],
    expected_article_ids: Sequence[str],
) -> CandidateMetrics:
    """Compute hit/rank metrics for an ordered chunk list."""
    expected_laws = set(expected_law_ids)
    expected_articles = set(expected_article_ids)
    first_law_rank = None
    first_article_rank = None
    seen_articles: set[str] = set()
    for index, chunk in enumerate(chunks, start=1):
        law_id = str(chunk.payload.get("law_id") or "")
        article_id = str(chunk.payload.get("article_id") or "")
        if article_id:
            seen_articles.add(article_id)
        if first_law_rank is None and law_id in expected_laws:
            first_law_rank = index
        if first_article_rank is None and article_id in expected_articles:
            first_article_rank = index
    law_hit = first_law_rank is not None
    article_hit = first_article_rank is not None
    return CandidateMetrics(
        law_hit=law_hit,
        article_hit=article_hit,
        all_expected_articles_hit=bool(expected_articles) and expected_articles.issubset(seen_articles),
        first_law_rank=first_law_rank,
        first_article_rank=first_article_rank,
        article_mrr=(1.0 / first_article_rank) if first_article_rank else 0.0,
        law_only_false_positive=law_hit and not article_hit,
    )


def dedupe_chunks(chunks: Iterable[RetrievedChunkRecord]) -> list[RetrievedChunkRecord]:
    """Deduplicate chunks while preserving first occurrence order."""
    out: list[RetrievedChunkRecord] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        out.append(chunk)
        seen.add(chunk.chunk_id)
    return out


def payload_matches_filters(payload: dict[str, Any], filters: dict[str, Any]) -> bool:
    """Return whether a payload satisfies simple exact-match metadata filters."""
    for key, expected in filters.items():
        if expected is None:
            continue
        actual = payload.get(key)
        expected_values = list(expected) if isinstance(expected, (list, tuple, set)) else [expected]
        if isinstance(actual, list):
            if not any(value in actual for value in expected_values):
                return False
        elif actual not in expected_values:
            return False
    return True


def evaluate_with_rerank(
    *,
    target: QuestionTarget,
    candidates: Sequence[RetrievedChunkRecord],
    scores: Sequence[int],
    rerank_input_k: int,
    rerank_output_k: int,
    retrieval_mode: str,
    top_k: int,
    filter_name: str,
    metadata_filters: dict[str, Any],
    base_scenario: str,
    rerank_model: str,
    cache_hit: bool,
) -> RerankEvaluationRow:
    """Apply rerank scores to candidates, cut to top-k, and compute hit/rank metrics."""
    capped_candidates = list(candidates)[:rerank_input_k]
    aligned_scores = list(scores)[: len(capped_candidates)]
    if len(aligned_scores) < len(capped_candidates):
        aligned_scores.extend([0] * (len(capped_candidates) - len(aligned_scores)))
    pre_rerank_metrics = candidate_metrics(
        capped_candidates,
        expected_law_ids=target.expected_law_ids,
        expected_article_ids=target.expected_article_ids,
    )
    indexed = [(chunk, int(score), original_rank) for original_rank, (chunk, score) in enumerate(zip(capped_candidates, aligned_scores))]
    indexed.sort(key=lambda item: (-item[1], item[2]))
    reranked = [chunk for chunk, _, _ in indexed[:rerank_output_k]]
    reranked_scores = [score for _, score, _ in indexed[:rerank_output_k]]
    post_metrics = candidate_metrics(
        reranked,
        expected_law_ids=target.expected_law_ids,
        expected_article_ids=target.expected_article_ids,
    )
    return RerankEvaluationRow(
        qid=target.qid,
        level=target.level,
        question=target.question,
        retrieval_mode=retrieval_mode,  # type: ignore[arg-type]
        top_k=top_k,
        filter_name=filter_name,
        metadata_filters=dict(metadata_filters),
        base_scenario=base_scenario,
        rerank_model=rerank_model,
        rerank_input_k=rerank_input_k,
        rerank_output_k=rerank_output_k,
        expected_law_ids=list(target.expected_law_ids),
        expected_article_ids=list(target.expected_article_ids),
        candidate_count=len(capped_candidates),
        reranked_count=len(reranked),
        reranked_law_hit=post_metrics.law_hit,
        reranked_article_hit=post_metrics.article_hit,
        reranked_all_expected_articles_hit=post_metrics.all_expected_articles_hit,
        reranked_first_law_rank=post_metrics.first_law_rank,
        reranked_first_article_rank=post_metrics.first_article_rank,
        reranked_article_mrr=post_metrics.article_mrr,
        pre_rerank_article_hit=pre_rerank_metrics.article_hit,
        pre_rerank_first_article_rank=pre_rerank_metrics.first_article_rank,
        rerank_recovered_article=not pre_rerank_metrics.article_hit and post_metrics.article_hit,
        rerank_demoted_article=pre_rerank_metrics.article_hit and not post_metrics.article_hit,
        rerank_scores=reranked_scores,
        cache_hit=cache_hit,
        reranked_chunk_ids=[chunk.chunk_id for chunk in reranked],
    )


def summarize_scenario(
    rows: Sequence[Mapping[str, Any]],
    *,
    scenario_name: str,
    dataset: str,
    stage: str,
    config: Mapping[str, Any],
    article_hit_key: str,
    law_hit_key: str,
    article_mrr_key: str,
    filter_excluded_key: str | None = "filter_excluded",
    baseline_pct: float | None = None,
) -> RetrievalScenarioSummary:
    """Aggregate per-question metrics into a single scenario row for the waterfall."""
    n = len(rows)
    if n == 0:
        return RetrievalScenarioSummary(
            scenario_name=scenario_name,
            dataset=dataset,
            stage=stage,  # type: ignore[arg-type]
            article_hit_pct=0.0,
            law_hit_pct=0.0,
            article_mrr=0.0,
            n_questions=0,
            n_filter_excluded=0,
            config=dict(config),
            delta_vs_baseline=None if baseline_pct is None else 0.0 - baseline_pct,
        )
    article_hits = sum(1 for row in rows if bool(row.get(article_hit_key)))
    law_hits = sum(1 for row in rows if bool(row.get(law_hit_key)))
    mrr_sum = sum(float(row.get(article_mrr_key) or 0.0) for row in rows)
    excluded = sum(1 for row in rows if filter_excluded_key and bool(row.get(filter_excluded_key)))
    article_pct = article_hits / n * 100.0
    law_pct = law_hits / n * 100.0
    return RetrievalScenarioSummary(
        scenario_name=scenario_name,
        dataset=dataset,
        stage=stage,  # type: ignore[arg-type]
        article_hit_pct=article_pct,
        law_hit_pct=law_pct,
        article_mrr=mrr_sum / n,
        n_questions=n,
        n_filter_excluded=excluded,
        config=dict(config),
        delta_vs_baseline=None if baseline_pct is None else article_pct - baseline_pct,
    )


class RerankCache:
    """File-backed JSONL cache for LLM rerank scores keyed by (question, candidate set, model)."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._entries: dict[str, list[dict[str, Any]]] = {}
        if self._path.exists():
            with self._path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    text = line.strip()
                    if not text:
                        continue
                    record = json.loads(text)
                    self._entries[str(record["key"])] = list(record.get("scores") or [])

    @property
    def path(self) -> Path:
        """Return the on-disk path for this cache."""
        return self._path

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def make_key(*, question: str, candidate_chunk_ids: Sequence[str], model: str) -> str:
        """Build a stable cache key from question, candidate ids, and model identity."""
        payload = {
            "q": str(question or ""),
            "candidates": list(candidate_chunk_ids),
            "model": str(model or ""),
        }
        return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    def get(self, key: str) -> list[dict[str, Any]] | None:
        """Return cached score entries (chunk_id, score) for a key, or None."""
        if key in self._entries:
            return [dict(entry) for entry in self._entries[key]]
        return None

    def set(self, key: str, scores: Sequence[Mapping[str, Any]]) -> None:
        """Append a new entry to memory and to the JSONL file on disk."""
        normalized = [{"chunk_id": str(item["chunk_id"]), "score": int(item["score"])} for item in scores]
        self._entries[key] = normalized
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "scores": normalized}, ensure_ascii=False, sort_keys=True) + "\n")


_TOKEN_RE = re.compile(r"[a-zà-ù]+", re.IGNORECASE)


def answer_overlap(correct_answer: str, chunk_text: str, *, min_token_length: int = 4) -> float:
    """Return the Jaccard overlap between long-token sets of two texts."""
    correct_tokens = {token.lower() for token in _TOKEN_RE.findall(correct_answer or "") if len(token) >= min_token_length}
    chunk_tokens = {token.lower() for token in _TOKEN_RE.findall(chunk_text or "") if len(token) >= min_token_length}
    if not correct_tokens or not chunk_tokens:
        return 0.0
    intersection = correct_tokens & chunk_tokens
    union = correct_tokens | chunk_tokens
    return len(intersection) / len(union)


def write_run_artifacts(
    output_dir: str | Path,
    *,
    scenarios: Sequence[Mapping[str, Any]],
    sweep_direct: Sequence[Mapping[str, Any]],
    sweep_graph: Sequence[Mapping[str, Any]],
    sweep_rerank: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> Path:
    """Persist scenarios, sweep tables and manifest under output_dir atomically."""
    target = Path(output_dir)
    tmp_dir = prepare_tmp_output_dir(target)
    try:
        _write_csv(tmp_dir / "scenarios.csv", scenarios, default_fields=RetrievalScenarioSummary.model_fields)
        _write_csv(tmp_dir / "sweep_direct.csv", sweep_direct, default_fields=RetrievalEvaluationRow.model_fields)
        _write_csv(tmp_dir / "sweep_graph.csv", sweep_graph, default_fields=RetrievalEvaluationRow.model_fields)
        _write_csv(
            tmp_dir / "sweep_rerank.csv",
            sweep_rerank,
            default_fields=["dataset", *RerankEvaluationRow.model_fields],
        )
        write_json(tmp_dir / "manifest.json", _augment_manifest(manifest))
        replace_output_dir(tmp_dir, target)
    except Exception:
        if tmp_dir.exists():
            import shutil

            shutil.rmtree(tmp_dir)
        raise
    return target


def _augment_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(manifest)
    out.setdefault("schema_version", RETRIEVAL_EVALUATION_SCHEMA_VERSION)
    out.setdefault("created_at", now_utc())
    return out


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, default_fields: Iterable[str] = ()) -> None:
    import csv

    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                keys.append(str(key))
                seen.add(str(key))
    if not keys:
        keys = [str(key) for key in default_fields]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in keys})


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out
