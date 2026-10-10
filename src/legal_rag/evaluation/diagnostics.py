"""Retrieval-only diagnostics with no generation or judge dependency."""

import json
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from pydantic import Field

from legal_rag.artifacts import code_identity, digest, file_hash, utc_now, write_json, write_jsonl
from legal_rag.contracts import AnswerResult, EngineConfig, Record
from legal_rag.retrieval import build_context

from .models import Benchmark, Reference
from .runner import TERMINAL_CODES, _emit_progress, _failure, _reject_secrets, reference_proxies

DIAGNOSTIC_SCHEMA_VERSION = "retrieval-diagnostics-v1"
PROFILES = ("dense", "sparse", "hybrid", "graph")


class DiagnosticResult(Record):
    qid: str
    level: str
    retrieval_only: bool = True
    expected_references: list[Reference]
    flags: list[str]
    answer: AnswerResult
    retrieval_proxies: dict
    timings: dict[str, float]


class DiagnosticManifest(Record):
    schema_version: str = DIAGNOSTIC_SCHEMA_VERSION
    created_at: str
    run_hash: str
    inputs: dict
    source_hashes: dict[str, str]
    original_intents: int
    selected_intents: int
    retrieval_only: bool = True
    complete: bool
    errors: int
    warnings: list[str]
    stopped_reason: str | None = None
    output_hashes: dict[str, str] = Field(default_factory=dict)


def load_diagnostics(directory: Path | str) -> list[DiagnosticResult]:
    path = Path(directory) / "records.jsonl"
    rows = [DiagnosticResult.model_validate(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len({(row.qid, row.answer.profile) for row in rows}) != len(rows):
        raise ValueError("Duplicate retrieval diagnostic QID/profile")
    if any(not row.retrieval_only or row.answer.answer or row.answer.usage for row in rows):
        raise ValueError("Retrieval diagnostics cannot contain generated answers or usage")
    return rows


def summarize_diagnostics(rows: list[DiagnosticResult], denominator: int) -> dict:
    output = {"retrieval_only": True, "denominator": denominator, "processed": len(rows),
              "retrieval_errors": sum(row.answer.status == "error" for row in rows),
              "annotation_complete_intents": sum(row.retrieval_proxies.get("annotation_complete", False) for row in rows),
              "warning": "Article matches are incomplete-reference proxies, not correctness or evidence sufficiency."}
    for metric in ("article_success", "article_mrr", "candidate_pool_article_success", "candidate_pool_article_mrr"):
        output[metric] = {}
        for cutoff in (5, 10, 100):
            values = [row.retrieval_proxies.get(metric, {}).get(str(cutoff)) for row in rows]
            available = [value for value in values if value is not None]
            output[metric][str(cutoff)] = {"fixed_denominator_value": sum(available) / denominator,
                "available_value": sum(available) / len(available) if available else None,
                "proxy_coverage": len(available) / denominator}
    context = [row.retrieval_proxies.get("context_article_success") for row in rows]
    available = [value for value in context if value is not None]
    output["context_article_success"] = {"fixed_denominator_value": sum(available) / denominator,
        "available_value": sum(available) / len(available) if available else None, "proxy_coverage": len(available) / denominator}
    stages = sorted({stage for row in rows for stage in row.timings})
    output["timings_seconds"] = {stage: {"total": sum(row.timings.get(stage, 0) for row in rows),
        "mean": sum(row.timings.get(stage, 0) for row in rows) / denominator} for stage in stages}
    output["flag_counts"] = {flag: sum(flag in row.flags for row in rows) for flag in sorted({flag for row in rows for flag in row.flags})}
    return output


def run_retrieval_diagnostics(
    benchmark: Benchmark, retriever, counter, output_dir: Path | str, *,
    profiles: tuple[str, ...] = PROFILES, config: EngineConfig | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    config = config or EngineConfig()
    if not benchmark.questions or not profiles or len(set(profiles)) != len(profiles) or any(profile not in PROFILES for profile in profiles):
        raise ValueError("Diagnostics require questions and distinct deterministic retrieval profiles")
    if benchmark.view == "current_approved":
        reviews = {review.qid: review for review in benchmark.reviews if review.human_approved}
        if any(reviews[question.qid].snapshot_id != retriever.manifest.get("snapshot_id") for question in benchmark.questions):
            raise ValueError("Approved diagnostic reviews must match the indexed corpus snapshot")
    inputs = {"benchmark_hash": benchmark.benchmark_hash, "view": benchmark.view, "qids": [q.qid for q in benchmark.questions],
        "profiles": list(profiles), "config": config.to_json_record(), "index": retriever.identity,
        "tokenizer": {"name": counter.name, "revision": counter.revision}, "code": code_identity(),
        "timing_policy": "Profiles run in declared order; the shared query-embedding cache may be warm in later profiles."}
    _reject_secrets(inputs)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Diagnostic directory already exists: {output_dir}")
    rows, stopped_reason = [], None
    total = len(benchmark.questions) * len(profiles)
    _emit_progress(progress, stage="diagnostic_start", profile=None, qid=None, processed=0, total=total, status="running")
    for profile in profiles:
        policy = config.model_copy(update={"profile": profile})
        for question in benchmark.questions:
            started, timings = perf_counter(), {}
            trace = {"retrieval_only": True, "query_rankings": [], "context_budget": policy.context_tokens,
                     "context_units": policy.context_units, "exclude_ceased": policy.exclude_ceased}
            result = AnswerResult(question=question.question, profile=profile, status="insufficient_evidence",
                snapshot_id=retriever.manifest.get("snapshot_id"), queries=[question.question])
            proxies = {}
            try:
                stage = perf_counter()
                pool = retriever.search(question.question, mode=profile if profile in {"dense", "sparse"} else "hybrid",
                    config=policy, trace=trace["query_rankings"])
                timings["search"] = perf_counter() - stage
                trace["candidate_pool"] = [{"chunk_id": candidate.chunk.chunk_id, "rank": candidate.rank, "score": candidate.score, "origin": candidate.origin} for candidate in pool]
                candidates = pool
                if profile == "graph":
                    stage = perf_counter()
                    additions, relations = retriever.expand_graph(pool, config=policy)
                    candidates = [*pool[:3], *additions, *pool[3:]][:policy.candidate_limit]
                    trace["graph_relation_ids"], trace["graph_additions"] = relations, [candidate.chunk.chunk_id for candidate in additions]
                    timings["graph"] = perf_counter() - stage
                result.candidates = [candidate.model_copy(update={"rank": index + 1}) for index, candidate in enumerate(candidates)]
                trace["selected_candidates"] = [{"chunk_id": candidate.chunk.chunk_id, "rank": candidate.rank, "score": candidate.score, "origin": candidate.origin} for candidate in result.candidates]
                stage = perf_counter()
                result.context = build_context(result.candidates, counter, max_tokens=policy.context_tokens, max_units=policy.context_units)
                timings["context"] = perf_counter() - stage
                result.retrieval_trace = trace
                proxies = reference_proxies(question, result, {candidate.chunk.chunk_id: candidate.chunk for candidate in pool})
            except Exception as exc:
                result.status, result.errors = "error", [_failure(exc)]
                trace["error"] = _failure(exc)
                if _failure(exc) in TERMINAL_CODES:
                    stopped_reason = _failure(exc)
            result.retrieval_trace = trace
            timings["total"] = perf_counter() - started
            rows.append(DiagnosticResult(qid=question.qid, level=question.level, expected_references=question.expected_references,
                flags=question.flags, answer=result, retrieval_proxies=proxies, timings=timings))
            _emit_progress(progress, stage="retrieval_record", profile=profile, qid=question.qid,
                           processed=len(rows), total=total, status=result.status)
            if stopped_reason:
                break
        if stopped_reason:
            break
    output_dir.mkdir(parents=True)
    write_jsonl(output_dir / "records.jsonl", rows)
    write_json(output_dir / "metrics.json", {profile: summarize_diagnostics([row for row in rows if row.answer.profile == profile], len(benchmark.questions)) for profile in profiles})
    errors = sum(row.answer.status == "error" for row in rows)
    manifest = DiagnosticManifest(created_at=utc_now(), run_hash=digest(json.dumps(inputs, sort_keys=True)), inputs=inputs,
        source_hashes=benchmark.source_hashes, original_intents=benchmark.original_count, selected_intents=len(benchmark.questions),
        complete=len(rows) == len(benchmark.questions) * len(profiles), errors=errors, stopped_reason=stopped_reason,
        warnings=[*benchmark.warnings, "Retrieval-only diagnostics contain no correctness scores.", inputs["timing_policy"]],
        output_hashes={filename: file_hash(output_dir / filename) for filename in ("records.jsonl", "metrics.json")})
    write_json(output_dir / "manifest.json", manifest)
    _emit_progress(progress, stage="diagnostic_complete", profile=None, qid=None, processed=len(rows), total=total,
                   status="complete" if manifest.complete else "incomplete")
    return manifest.to_json_record()
