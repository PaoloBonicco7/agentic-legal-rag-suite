"""Deterministic coverage and paired-impact audit for validity filters."""

from __future__ import annotations

import csv
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import pandas as pd
from qdrant_client import QdrantClient

from legal_rag.indexing.embeddings import SupportsEmbedding

from .evaluator import ChunkAvailabilityIndex, payload_matches_filters, write_run_artifacts
from .experiments.direct import (
    DirectExperimentCache,
    build_collection_identity,
    run_direct_experiment,
)
from .models import (
    FILTER_AUDIT_PROMPT_VERSION,
    FILTER_AUDIT_SCHEMA_VERSION,
    RETRIEVAL_EVALUATION_SCHEMA_VERSION,
    FilterExactControlRow,
    FilterImpactRow,
    FilterReferenceAuditRow,
    QuestionTarget,
)
from .profiles import (
    FILTER_AUDIT_EXACT_FILTER_NAMES,
    FILTER_VARIANTS,
    DiagnosticProfile,
    resolve_profile,
)

VIGENCY_REFERENCE_REVIEW_SCHEMA_VERSION = "vigency-reference-review-v1"
PRIMARY_FILTER_NAMES = frozenset(
    {"index_views_current", "index_views_not_explicitly_past"}
)
_ACTIVE_STATUSES = frozenset({"current", "partial"})
_VALID_STATUSES = frozenset({"current", "partial", "past", "unknown"})


@dataclass(frozen=True)
class FilterAuditResult:
    """In-memory tables and manifest for one deterministic filter audit."""

    sweep_direct: pd.DataFrame
    filter_reference_audit: pd.DataFrame
    filter_exclusions: pd.DataFrame
    filter_impact: pd.DataFrame
    filter_exact_control: pd.DataFrame
    manifest: dict[str, Any]


def load_vigency_reference_review(path: str | Path) -> list[dict[str, str]]:
    """Load and validate the corpus-only review without changing evaluation targets."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if row.get("schema_version") != VIGENCY_REFERENCE_REVIEW_SCHEMA_VERSION:
            raise ValueError(
                f"Unexpected vigency review schema: {row.get('schema_version')!r}"
            )
        key = (str(row.get("qid") or ""), str(row.get("expected_article_id") or ""))
        if not all(key):
            raise ValueError("Vigency review rows require qid and expected_article_id")
        if key in seen:
            raise ValueError(f"Duplicate vigency review row: {key}")
        seen.add(key)
    return rows


def build_filter_reference_audit(
    *,
    targets_by_dataset: Mapping[str, Sequence[QuestionTarget]],
    availability: ChunkAvailabilityIndex,
    filter_names: Sequence[str],
    filter_variants: Mapping[str, dict[str, Any]],
    collection_identity: str,
    reference_review: Sequence[Mapping[str, Any]] = (),
) -> pd.DataFrame:
    """Build one static coverage row per unique QID/article/filter."""
    review_by_target = {
        (str(row.get("qid") or ""), str(row.get("expected_article_id") or "")): dict(row)
        for row in reference_review
    }
    unique_targets: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset, targets in targets_by_dataset.items():
        for target in targets:
            for reference in target.references:
                key = (target.qid, reference.article_id)
                entry = unique_targets.setdefault(
                    key,
                    {
                        "target": target,
                        "law_id": reference.law_id,
                        "article_id": reference.article_id,
                        "reference_texts": [],
                        "datasets": set(),
                    },
                )
                entry["datasets"].add(str(dataset))
                if reference.reference_text not in entry["reference_texts"]:
                    entry["reference_texts"].append(reference.reference_text)

    rows: list[dict[str, Any]] = []
    for key in sorted(unique_targets):
        entry = unique_targets[key]
        target: QuestionTarget = entry["target"]
        article_id = str(entry["article_id"])
        chunks = availability.article_chunks([article_id])
        active_chunks = [chunk for chunk in chunks if _is_active_chunk(chunk)]
        unknown_chunks = [chunk for chunk in chunks if _has_unknown_status(chunk)]
        review = review_by_target.get(key, {})
        support_passage_id = _optional_text(review.get("answer_supporting_passage_id"))
        support_chunks = (
            availability.passage_chunks([support_passage_id])
            if support_passage_id
            else []
        )
        for filter_name in filter_names:
            if filter_name not in filter_variants:
                raise KeyError(f"Unknown filter variant: {filter_name}")
            filters = dict(filter_variants[filter_name])
            retained = [chunk for chunk in chunks if payload_matches_filters(chunk, filters)]
            retained_active = [
                chunk for chunk in active_chunks if payload_matches_filters(chunk, filters)
            ]
            retained_unknown = [
                chunk for chunk in unknown_chunks if payload_matches_filters(chunk, filters)
            ]
            support_retained = None
            if support_passage_id and support_chunks:
                support_retained = any(
                    payload_matches_filters(chunk, filters) for chunk in support_chunks
                )
            coverage_status = _coverage_status(len(chunks), len(retained))
            row = FilterReferenceAuditRow(
                qid=target.qid,
                datasets=sorted(entry["datasets"]),
                level=target.level,
                reference_text=" | ".join(entry["reference_texts"]),
                law_id=str(entry["law_id"]),
                article_id=article_id,
                filter_name=filter_name,
                metadata_filters=filters,
                collection_identity=collection_identity,
                total_target_chunks=len(chunks),
                retained_target_chunks=len(retained),
                coverage_ratio=(len(retained) / len(chunks)) if chunks else 0.0,
                coverage_status=coverage_status,
                active_target_chunks=len(active_chunks),
                retained_active_target_chunks=len(retained_active),
                active_slice=bool(active_chunks),
                unknown_status_present=bool(unknown_chunks),
                unknown_status_excluded=len(retained_unknown) < len(unknown_chunks),
                law_statuses=_payload_values(chunks, "law_status"),
                article_statuses=_payload_values(chunks, "article_status"),
                passage_statuses=_payload_values(chunks, "passage_status"),
                content_availability=_payload_values(chunks, "content_availability"),
                status_event_ids=_payload_values(chunks, "status_event_ids"),
                status_rule_ids=_payload_values(chunks, "status_rule_ids"),
                expected_reference_validity=_optional_text(
                    review.get("expected_reference_validity")
                ),
                answer_support_relation=_optional_text(review.get("answer_support_relation")),
                answer_supporting_law_id=_optional_text(
                    review.get("answer_supporting_law_id")
                ),
                answer_supporting_article_id=_optional_text(
                    review.get("answer_supporting_article_id")
                ),
                answer_supporting_passage_id=support_passage_id,
                supporting_passage_retained=support_retained,
                temporal_scope_flag=_optional_text(review.get("temporal_scope_flag")),
                review_rationale=_optional_text(review.get("rationale")),
            )
            rows.append(row.to_json_record())
    return pd.DataFrame(rows, columns=FilterReferenceAuditRow.model_fields)


def build_filter_exclusions(reference_audit_df: pd.DataFrame) -> pd.DataFrame:
    """Return both partial and full exclusions from a reference audit."""
    if reference_audit_df.empty:
        return reference_audit_df.copy()
    return reference_audit_df[
        reference_audit_df["coverage_status"] != "fully_eligible"
    ].reset_index(drop=True)


def paired_bootstrap_interval(
    filtered: Sequence[float | int | bool],
    baseline: Sequence[float | int | bool],
    *,
    resamples: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Return paired mean delta and percentile interval."""
    if len(filtered) != len(baseline):
        raise ValueError("Paired bootstrap inputs must have the same length")
    if not filtered:
        raise ValueError("Paired bootstrap inputs must not be empty")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    differences = [
        float(filtered_value) - float(baseline_value)
        for filtered_value, baseline_value in zip(filtered, baseline)
    ]
    point_delta = fmean(differences)
    rng = random.Random(seed)
    n = len(differences)
    draws = [
        sum(differences[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(resamples)
    ]
    draws.sort()
    alpha = 1.0 - confidence
    return (
        point_delta,
        _quantile(draws, alpha / 2.0),
        _quantile(draws, 1.0 - alpha / 2.0),
    )


def build_filter_impact(
    direct_df: pd.DataFrame,
    reference_audit_df: pd.DataFrame,
    *,
    resamples: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare each filter with the paired unfiltered scenario."""
    if direct_df.empty:
        return pd.DataFrame(columns=FilterImpactRow.model_fields)
    working = direct_df.copy()
    if "exact" not in working:
        working["exact"] = False
    working = working[~working["exact"].astype(bool)].copy()
    group_columns = [
        "dataset",
        "retrieval_mode",
        "top_k",
        "rrf_k",
        "filter_name",
        "exact",
    ]
    rows: list[dict[str, Any]] = []
    for key, filtered_group in working.groupby(group_columns, dropna=False, sort=True):
        dataset, retrieval_mode, top_k, rrf_k, filter_name, exact = key
        baseline_group = working[
            (working["dataset"] == dataset)
            & (working["retrieval_mode"] == retrieval_mode)
            & (working["top_k"] == top_k)
            & _nullable_equals(working["rrf_k"], rrf_k)
            & (working["filter_name"] == "none")
            & (working["exact"].astype(bool) == bool(exact))
        ]
        paired = _paired_scenario_rows(filtered_group, baseline_group)
        filtered_article = paired["filtered_article"].astype(float).tolist()
        baseline_article = paired["baseline_article"].astype(float).tolist()
        filtered_law = paired["filtered_law"].astype(float).tolist()
        baseline_law = paired["baseline_law"].astype(float).tolist()
        filtered_mrr = paired["filtered_mrr"].astype(float).tolist()
        baseline_mrr = paired["baseline_mrr"].astype(float).tolist()
        is_primary = (
            dataset == "no_hint"
            and retrieval_mode == "hybrid"
            and int(top_k) == 10
            and filter_name in PRIMARY_FILTER_NAMES
        )
        article_confidence = 0.975 if is_primary else 0.95
        article_delta, article_low, article_high = paired_bootstrap_interval(
            filtered_article,
            baseline_article,
            resamples=resamples,
            seed=seed,
            confidence=article_confidence,
        )
        mrr_delta, mrr_low, mrr_high = paired_bootstrap_interval(
            filtered_mrr,
            baseline_mrr,
            resamples=resamples,
            seed=seed + 1,
            confidence=0.95,
        )
        law_delta = fmean(filtered_law) - fmean(baseline_law)
        effect = _retrieval_effect(article_delta, mrr_delta)
        coverage = _coverage_decision(
            reference_audit_df,
            filter_name=str(filter_name),
            dataset=str(dataset),
        )
        row = FilterImpactRow(
            dataset=str(dataset),
            retrieval_mode=str(retrieval_mode),  # type: ignore[arg-type]
            top_k=int(top_k),
            rrf_k=None if pd.isna(rrf_k) else int(rrf_k),
            filter_name=str(filter_name),
            metadata_filters=_mapping_value(filtered_group.iloc[0]["metadata_filters"]),
            exact=bool(exact),
            n_questions=len(paired),
            article_success_pct=fmean(filtered_article) * 100.0,
            baseline_article_success_pct=fmean(baseline_article) * 100.0,
            article_success_delta_pp=article_delta * 100.0,
            article_success_ci_low_pp=article_low * 100.0,
            article_success_ci_high_pp=article_high * 100.0,
            article_success_ci_level=article_confidence,
            law_success_pct=fmean(filtered_law) * 100.0,
            baseline_law_success_pct=fmean(baseline_law) * 100.0,
            law_success_delta_pp=law_delta * 100.0,
            article_mrr=fmean(filtered_mrr),
            baseline_article_mrr=fmean(baseline_mrr),
            article_mrr_delta=mrr_delta,
            article_mrr_ci_low=mrr_low,
            article_mrr_ci_high=mrr_high,
            gains=sum(a > b for a, b in zip(filtered_article, baseline_article)),
            losses=sum(a < b for a, b in zip(filtered_article, baseline_article)),
            ties=sum(a == b for a, b in zip(filtered_article, baseline_article)),
            total_reference_targets=coverage["total"],
            fully_eligible_targets=coverage["fully_eligible"],
            partially_eligible_targets=coverage["partially_eligible"],
            fully_excluded_targets=coverage["fully_excluded"],
            benchmark_full_coverage=coverage["benchmark_full_coverage"],
            active_slice_safety=coverage["active_slice_safety"],
            retrieval_effect=effect,
            bootstrap_supported=_bootstrap_support(
                effect=effect,
                article_delta=article_delta,
                article_low=article_low,
                article_high=article_high,
                mrr_delta=mrr_delta,
                mrr_low=mrr_low,
                mrr_high=mrr_high,
            ),
        )
        rows.append(row.to_json_record())
    return pd.DataFrame(rows, columns=FilterImpactRow.model_fields)


def build_filter_exact_control(direct_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize paired dense ANN and exact results."""
    if direct_df.empty or "exact" not in direct_df:
        return pd.DataFrame(columns=FilterExactControlRow.model_fields)
    dense = direct_df[direct_df["retrieval_mode"] == "dense"].copy()
    exact_rows = dense[dense["exact"].astype(bool)]
    rows: list[dict[str, Any]] = []
    group_columns = ["dataset", "top_k", "filter_name"]
    for key, exact_group in exact_rows.groupby(group_columns, sort=True):
        dataset, top_k, filter_name = key
        ann_group = dense[
            (dense["dataset"] == dataset)
            & (dense["top_k"] == top_k)
            & (dense["filter_name"] == filter_name)
            & ~dense["exact"].astype(bool)
        ]
        paired = _paired_exact_rows(ann_group, exact_group)
        overlaps = [
            _chunk_overlap(ann_ids, exact_ids, top_k=int(top_k))
            for ann_ids, exact_ids in zip(paired["ann_ids"], paired["exact_ids"])
        ]
        row = FilterExactControlRow(
            dataset=str(dataset),
            top_k=int(top_k),
            filter_name=str(filter_name),
            metadata_filters=_mapping_value(exact_group.iloc[0]["metadata_filters"]),
            n_questions=len(paired),
            mean_chunk_overlap=fmean(overlaps),
            ann_article_success_pct=paired["ann_article"].astype(float).mean() * 100.0,
            exact_article_success_pct=paired["exact_article"].astype(float).mean() * 100.0,
            article_success_delta_pp=(
                paired["exact_article"].astype(float).mean()
                - paired["ann_article"].astype(float).mean()
            )
            * 100.0,
            ann_law_success_pct=paired["ann_law"].astype(float).mean() * 100.0,
            exact_law_success_pct=paired["exact_law"].astype(float).mean() * 100.0,
            law_success_delta_pp=(
                paired["exact_law"].astype(float).mean()
                - paired["ann_law"].astype(float).mean()
            )
            * 100.0,
            ann_article_mrr=paired["ann_mrr"].astype(float).mean(),
            exact_article_mrr=paired["exact_mrr"].astype(float).mean(),
            article_mrr_delta=(
                paired["exact_mrr"].astype(float).mean()
                - paired["ann_mrr"].astype(float).mean()
            ),
        )
        rows.append(row.to_json_record())
    return pd.DataFrame(rows, columns=FilterExactControlRow.model_fields)


def run_filter_audit(
    *,
    targets_by_dataset: Mapping[str, Sequence[QuestionTarget]],
    cache: DirectExperimentCache,
    qdrant_client: QdrantClient,
    collection_name: str,
    embedder: SupportsEmbedding,
    index_manifest: dict[str, Any],
    rrf_k_default: int,
    availability: ChunkAvailabilityIndex,
    reference_review: Sequence[Mapping[str, Any]] = (),
    profile: DiagnosticProfile | None = None,
    filter_variants: Mapping[str, dict[str, Any]] = FILTER_VARIANTS,
    show_progress: bool = True,
    max_workers: int = 1,
) -> FilterAuditResult:
    """Run the deterministic filter matrix, coverage audit, and exact control."""
    profile = profile or resolve_profile("filter_audit")
    unsupported = set(profile.enabled_experiments) - {"direct", "hybrid"}
    if unsupported:
        raise ValueError(f"Filter audit cannot run nondeterministic experiments: {sorted(unsupported)}")
    direct = run_direct_experiment(
        targets_by_dataset=targets_by_dataset,
        cache=cache,
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        embedder=embedder,
        index_manifest=index_manifest,
        rrf_k_default=rrf_k_default,
        availability=availability,
        modes=["dense", "hybrid"],
        filter_names=profile.filter_variants,
        filter_variants=filter_variants,
        top_k_values=profile.top_k_values,
        hybrid_top_k_values=profile.hybrid_top_k_values,
        hybrid_rrf_k_values=profile.hybrid_rrf_k_values,
        hybrid_filters_enabled=True,
        exact_values=(False,),
        show_progress=show_progress,
        max_workers=max_workers,
    )
    identity = build_collection_identity(collection_name, index_manifest)
    reference_audit = build_filter_reference_audit(
        targets_by_dataset=targets_by_dataset,
        availability=availability,
        filter_names=profile.filter_variants,
        filter_variants=filter_variants,
        collection_identity=identity,
        reference_review=reference_review,
    )
    impact = build_filter_impact(
        direct,
        reference_audit,
        resamples=profile.bootstrap_resamples,
        seed=profile.bootstrap_seed,
    )
    exact = pd.DataFrame()
    if profile.exact_control_enabled:
        exact = run_direct_experiment(
            targets_by_dataset=targets_by_dataset,
            cache=cache,
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            embedder=embedder,
            index_manifest=index_manifest,
            rrf_k_default=rrf_k_default,
            availability=availability,
            modes=["dense"],
            filter_names=profile.exact_control_filter_variants,
            filter_variants=filter_variants,
            top_k_values=profile.exact_control_top_k_values,
            exact_values=(True,),
            show_progress=show_progress,
            max_workers=max_workers,
        )
    sweep_direct = pd.concat([direct, exact], ignore_index=True) if not exact.empty else direct
    exact_control = build_filter_exact_control(sweep_direct)
    manifest = {
        "schema_version": RETRIEVAL_EVALUATION_SCHEMA_VERSION,
        "filter_audit_schema_version": FILTER_AUDIT_SCHEMA_VERSION,
        "filter_audit_prompt_version": FILTER_AUDIT_PROMPT_VERSION,
        "profile": profile.model_dump(mode="json"),
        "collection_name": collection_name,
        "collection_identity": identity,
        "index_identity": {
            "run_id": index_manifest.get("run_id"),
            "source_hash": index_manifest.get("source_hash"),
            "chunks_hash": (index_manifest.get("source_output_hashes") or {}).get("chunks"),
        },
        "row_counts": {
            "sweep_direct": len(sweep_direct),
            "filter_reference_audit": len(reference_audit),
            "filter_exclusions": int(
                (reference_audit["coverage_status"] != "fully_eligible").sum()
            )
            if not reference_audit.empty
            else 0,
            "filter_impact": len(impact),
            "filter_exact_control": len(exact_control),
        },
    }
    return FilterAuditResult(
        sweep_direct=sweep_direct,
        filter_reference_audit=reference_audit,
        filter_exclusions=build_filter_exclusions(reference_audit),
        filter_impact=impact,
        filter_exact_control=exact_control,
        manifest=manifest,
    )


def write_filter_audit_artifacts(
    output_dir: str | Path,
    result: FilterAuditResult,
    *,
    manifest: Mapping[str, Any] | None = None,
) -> Path:
    """Write a filter-audit result through the shared atomic artifact writer."""
    merged_manifest = dict(result.manifest)
    merged_manifest.update(dict(manifest or {}))
    return write_run_artifacts(
        output_dir,
        scenarios=[],
        sweep_direct=result.sweep_direct.to_dict(orient="records"),
        sweep_graph=[],
        sweep_rerank=[],
        filter_reference_audit=result.filter_reference_audit.to_dict(orient="records"),
        filter_exclusions=result.filter_exclusions.to_dict(orient="records"),
        filter_impact=result.filter_impact.to_dict(orient="records"),
        filter_exact_control=result.filter_exact_control.to_dict(orient="records"),
        manifest=merged_manifest,
    )


def _coverage_status(total: int, retained: int) -> str:
    if total > 0 and retained == total:
        return "fully_eligible"
    if retained > 0:
        return "partially_eligible"
    return "fully_excluded"


def _is_active_chunk(chunk: Mapping[str, Any]) -> bool:
    statuses = [
        str(chunk.get("law_status") or ""),
        str(chunk.get("article_status") or ""),
        str(chunk.get("passage_status") or ""),
    ]
    availability = str(chunk.get("content_availability") or "")
    return all(status in _ACTIVE_STATUSES for status in statuses) and availability in {
        "substantive",
        "unstructured",
    }


def _has_unknown_status(chunk: Mapping[str, Any]) -> bool:
    return any(
        str(chunk.get(key) or "") not in _VALID_STATUSES
        or str(chunk.get(key) or "") == "unknown"
        for key in ("law_status", "article_status", "passage_status")
    )


def _payload_values(chunks: Sequence[Mapping[str, Any]], key: str) -> list[str]:
    values: set[str] = set()
    for chunk in chunks:
        value = chunk.get(key)
        items = value if isinstance(value, (list, tuple, set)) else [value]
        values.update(str(item) for item in items if item is not None and str(item).strip())
    return sorted(values)


def _paired_scenario_rows(
    filtered_group: pd.DataFrame,
    baseline_group: pd.DataFrame,
) -> pd.DataFrame:
    filtered = filtered_group[
        ["qid", "direct_article_hit", "direct_law_hit", "direct_article_mrr"]
    ].rename(
        columns={
            "direct_article_hit": "filtered_article",
            "direct_law_hit": "filtered_law",
            "direct_article_mrr": "filtered_mrr",
        }
    )
    baseline = baseline_group[
        ["qid", "direct_article_hit", "direct_law_hit", "direct_article_mrr"]
    ].rename(
        columns={
            "direct_article_hit": "baseline_article",
            "direct_law_hit": "baseline_law",
            "direct_article_mrr": "baseline_mrr",
        }
    )
    _require_unique_qids(filtered, "filtered")
    _require_unique_qids(baseline, "baseline")
    paired = filtered.merge(baseline, on="qid", how="inner", validate="one_to_one")
    if len(paired) != len(filtered) or len(paired) != len(baseline):
        raise ValueError("Filtered and baseline scenarios must contain the same QID set")
    if paired.empty:
        raise ValueError("Cannot compare an empty retrieval scenario")
    return paired.sort_values("qid").reset_index(drop=True)


def _paired_exact_rows(ann_group: pd.DataFrame, exact_group: pd.DataFrame) -> pd.DataFrame:
    ann = ann_group[
        [
            "qid",
            "retrieved_chunk_ids",
            "direct_article_hit",
            "direct_law_hit",
            "direct_article_mrr",
        ]
    ].rename(
        columns={
            "retrieved_chunk_ids": "ann_ids",
            "direct_article_hit": "ann_article",
            "direct_law_hit": "ann_law",
            "direct_article_mrr": "ann_mrr",
        }
    )
    exact = exact_group[
        [
            "qid",
            "retrieved_chunk_ids",
            "direct_article_hit",
            "direct_law_hit",
            "direct_article_mrr",
        ]
    ].rename(
        columns={
            "retrieved_chunk_ids": "exact_ids",
            "direct_article_hit": "exact_article",
            "direct_law_hit": "exact_law",
            "direct_article_mrr": "exact_mrr",
        }
    )
    _require_unique_qids(ann, "ANN")
    _require_unique_qids(exact, "exact")
    paired = ann.merge(exact, on="qid", how="inner", validate="one_to_one")
    if len(paired) != len(ann) or len(paired) != len(exact):
        raise ValueError("ANN and exact controls must contain the same QID set")
    if paired.empty:
        raise ValueError("Cannot compare an empty exact-search control")
    return paired.sort_values("qid").reset_index(drop=True)


def _coverage_decision(
    reference_audit_df: pd.DataFrame,
    *,
    filter_name: str,
    dataset: str,
) -> dict[str, Any]:
    if reference_audit_df.empty:
        return {
            "total": 0,
            "fully_eligible": 0,
            "partially_eligible": 0,
            "fully_excluded": 0,
            "benchmark_full_coverage": False,
            "active_slice_safety": "unresolved",
        }
    subset = reference_audit_df[
        (reference_audit_df["filter_name"] == filter_name)
        & reference_audit_df["datasets"].map(lambda value: dataset in _string_list(value))
    ]
    statuses = subset["coverage_status"].value_counts()
    active_lost = (
        (subset["active_target_chunks"] > 0)
        & (subset["retained_active_target_chunks"] == 0)
    ).any()
    reviewed_active_support_lost = (
        subset["expected_reference_validity"].isin(["current", "partial"])
        & (subset["supporting_passage_retained"] == False)  # noqa: E712
    ).any()
    unknown_excluded = subset["unknown_status_excluded"].astype(bool).any()
    reviewed_unknown_support_lost = (
        (subset["expected_reference_validity"] == "unknown")
        & (subset["supporting_passage_retained"] == False)  # noqa: E712
    ).any()
    if active_lost or reviewed_active_support_lost:
        safety = "unsafe"
    elif unknown_excluded or reviewed_unknown_support_lost:
        safety = "unresolved"
    else:
        safety = "safe"
    return {
        "total": len(subset),
        "fully_eligible": int(statuses.get("fully_eligible", 0)),
        "partially_eligible": int(statuses.get("partially_eligible", 0)),
        "fully_excluded": int(statuses.get("fully_excluded", 0)),
        "benchmark_full_coverage": bool(
            len(subset) > 0 and (subset["coverage_status"] == "fully_eligible").all()
        ),
        "active_slice_safety": safety,
    }


def _retrieval_effect(article_delta: float, mrr_delta: float) -> str:
    epsilon = 1e-12
    if article_delta < -epsilon or mrr_delta < -epsilon:
        return "harmful"
    if article_delta > epsilon or mrr_delta > epsilon:
        return "beneficial"
    return "inconclusive"


def _bootstrap_support(
    *,
    effect: str,
    article_delta: float,
    article_low: float,
    article_high: float,
    mrr_delta: float,
    mrr_low: float,
    mrr_high: float,
) -> bool:
    if effect == "beneficial":
        return (article_delta > 0 and article_low > 0) or (
            mrr_delta > 0 and mrr_low > 0
        )
    if effect == "harmful":
        return (article_delta < 0 and article_high < 0) or (
            mrr_delta < 0 and mrr_high < 0
        )
    return False


def _chunk_overlap(ann_ids: Any, exact_ids: Any, *, top_k: int) -> float:
    ann = _string_list(ann_ids)[:top_k]
    exact = _string_list(exact_ids)[:top_k]
    if not exact:
        return 1.0 if not ann else 0.0
    return len(set(ann) & set(exact)) / len(set(exact))


def _mapping_value(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    return {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            parsed = json.loads(text)
            return [str(item) for item in parsed]
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return []


def _nullable_equals(series: pd.Series, value: Any) -> pd.Series:
    return series.isna() if pd.isna(value) else series == value


def _require_unique_qids(frame: pd.DataFrame, name: str) -> None:
    if frame["qid"].duplicated().any():
        raise ValueError(f"{name} scenario contains duplicate QIDs")


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
    )


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
