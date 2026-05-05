from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from legal_rag.evaluation_reporting import (
    EVALUATION_REPORTING_SCHEMA_VERSION,
    EvaluationReportingConfig,
    EvaluationReportingManifest,
    classify_failure,
    run_evaluation_reporting,
)
from legal_rag.oracle_context_evaluation.io import sha256_file, write_json, write_jsonl


def _metric(rows: list[dict[str, Any]], *, score_key: str, max_score: int) -> dict[str, Any]:
    valid = [int(row[score_key]) for row in rows if row.get(score_key) is not None]
    processed = len(rows)
    judged = len(valid)
    score_sum = sum(valid)
    by_level: dict[str, dict[str, Any]] = {}
    for row in rows:
        level = row["level"]
        stats = by_level.setdefault(level, {"processed": 0, "judged": 0, "score_sum": 0, "errors": 0})
        stats["processed"] += 1
        if row.get("error"):
            stats["errors"] += 1
        if row.get(score_key) is not None:
            stats["judged"] += 1
            stats["score_sum"] += int(row[score_key])
    for stats in by_level.values():
        judged_max = max_score * stats["judged"]
        processed_max = max_score * stats["processed"]
        stats["max_score_sum"] = judged_max
        stats["accuracy"] = stats["score_sum"] / judged_max if judged_max else None
        stats["mean_score"] = stats["score_sum"] / stats["judged"] if stats["judged"] else None
        stats["coverage"] = stats["judged"] / stats["processed"] if stats["processed"] else None
        stats["strict_accuracy"] = stats["score_sum"] / processed_max if processed_max else None
    return {
        "dataset": "mcq" if score_key == "score" else "no_hint",
        "processed": processed,
        "judged": judged,
        "score_sum": score_sum,
        "max_score_sum": max_score * judged,
        "accuracy": score_sum / (max_score * judged) if judged else None,
        "mean_score": score_sum / judged if judged else None,
        "coverage": judged / processed if processed else None,
        "strict_accuracy": score_sum / (max_score * processed) if processed else None,
        "errors": sum(1 for row in rows if row.get("error")),
        "by_level": by_level,
    }


def _rows(method: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mcq_scores = {
        "no_rag": [0, 1, None],
        "simple_rag": [1, 1, 0],
        "advanced_rag": [1, 1, 1],
    }[method]
    no_hint_scores = {
        "no_rag": [0, 2, None],
        "simple_rag": [1, 2, 0],
        "advanced_rag": [2, 2, 1],
    }[method]
    levels = ["L2", "L1", "L2"]
    mcq_rows: list[dict[str, Any]] = []
    no_hint_rows: list[dict[str, Any]] = []
    for idx, (level, score) in enumerate(zip(levels, mcq_scores), start=1):
        row = {
            "qid": f"eval-{idx:04d}",
            "level": level,
            "question": f"Question {idx}?",
            "predicted_label": "A" if score is not None else None,
            "correct_label": "A",
            "score": score,
            "error": None,
        }
        if method in {"simple_rag", "advanced_rag"}:
            row.update({"retrieved_count": 1, "context_count": 1, "answer": "Answer"})
        if method == "advanced_rag" and score != 1:
            row["failure_category"] = "context_noise"
        if method == "simple_rag" and idx == 3:
            row["error"] = "empty_retrieval"
            row["answer"] = None
        if method == "no_rag" and idx == 3:
            row["error"] = "invalid_mcq_label: None"
        mcq_rows.append(row)
    for idx, (level, score) in enumerate(zip(levels, no_hint_scores), start=1):
        row = {
            "qid": f"eval-{idx:04d}",
            "level": level,
            "question": f"Question {idx}?",
            "predicted_answer": "Answer" if score is not None else "",
            "correct_answer": "Answer",
            "judge_score": score,
            "judge_explanation": "ok" if score is not None else None,
            "error": None,
        }
        if method in {"simple_rag", "advanced_rag"}:
            row.update({"retrieved_count": 1, "context_count": 1, "answer": row["predicted_answer"]})
        if method == "advanced_rag" and score not in {2, None}:
            row["failure_category"] = "retrieval_miss"
        if method == "simple_rag" and idx == 3:
            row["error"] = "judge_error: invalid score"
        if method == "no_rag" and idx == 3:
            row["error"] = "no_hint_structured_error: RuntimeError"
        no_hint_rows.append(row)
    return mcq_rows, no_hint_rows


def _write_run(root: Path, method: str, evaluation_hash: str, *, omit_metric: str | None = None) -> Path:
    run_dir = {
        "no_rag": root / "baseline_runs" / "no_rag",
        "simple_rag": root / "rag_runs" / "simple",
        "advanced_rag": root / "rag_runs" / "advanced" / "default",
    }[method]
    run_dir.mkdir(parents=True)
    mcq_rows, no_hint_rows = _rows(method)
    summary = {
        "mcq": _metric(mcq_rows, score_key="score", max_score=1),
        "no_hint": _metric(no_hint_rows, score_key="judge_score", max_score=2),
    }
    if omit_metric:
        summary["mcq"].pop(omit_metric, None)
    manifest_name = {
        "no_rag": "no_rag_manifest.json",
        "simple_rag": "simple_rag_manifest.json",
        "advanced_rag": "advanced_rag_manifest.json",
    }[method]
    summary_name = {
        "no_rag": "no_rag_summary.json",
        "simple_rag": "simple_rag_summary.json",
        "advanced_rag": "advanced_rag_summary.json",
    }[method]
    write_jsonl(run_dir / "mcq_results.jsonl", mcq_rows)
    write_jsonl(run_dir / "no_hint_results.jsonl", no_hint_rows)
    write_json(run_dir / summary_name, summary)
    write_json(
        run_dir / manifest_name,
        {
            "schema_version": f"{method}-v1",
            "config": {"run_name": "default"} if method == "advanced_rag" else {},
            "source_hashes": {"evaluation_manifest": evaluation_hash},
            "summary": summary,
        },
    )
    if method == "advanced_rag":
        write_json(run_dir / "advanced_diagnostics.json", {"failure_categories": {"retrieval_miss": 1}})
    return run_dir


def _write_fixture(tmp_path: Path, *, include_advanced: bool = True) -> tuple[Path, str]:
    evaluation_dir = tmp_path / "evaluation_clean"
    evaluation_dir.mkdir()
    evaluation_manifest = evaluation_dir / "evaluation_manifest.json"
    write_json(evaluation_manifest, {"schema_version": "evaluation-dataset-v1", "level_counts": {"L2": 2, "L1": 1}})
    evaluation_hash = sha256_file(evaluation_manifest)
    _write_run(tmp_path, "no_rag", evaluation_hash)
    _write_run(tmp_path, "simple_rag", evaluation_hash)
    if include_advanced:
        _write_run(tmp_path, "advanced_rag", evaluation_hash)
    return evaluation_manifest, evaluation_hash


def _config(tmp_path: Path, evaluation_manifest: Path, *, allow_partial: bool = False) -> EvaluationReportingConfig:
    return EvaluationReportingConfig(
        no_rag_dir=str(tmp_path / "baseline_runs" / "no_rag"),
        simple_rag_dir=str(tmp_path / "rag_runs" / "simple"),
        advanced_rag_dir=str(tmp_path / "rag_runs" / "advanced" / "default"),
        evaluation_manifest_path=str(evaluation_manifest),
        output_dir=str(tmp_path / "reports"),
        allow_partial=allow_partial,
        max_examples_per_category=2,
    )


def test_evaluation_reporting_exports_contract_files(tmp_path: Path) -> None:
    evaluation_manifest, evaluation_hash = _write_fixture(tmp_path)

    manifest = run_evaluation_reporting(_config(tmp_path, evaluation_manifest))

    output_dir = tmp_path / "reports"
    assert {path.name for path in output_dir.iterdir()} == {
        "comparison_summary.json",
        "comparison_by_level.json",
        "failure_analysis.json",
        "thesis_tables.md",
        "quality_report.md",
        "report_manifest.json",
    }
    assert manifest["schema_version"] == EVALUATION_REPORTING_SCHEMA_VERSION
    assert manifest["complete"] is True
    assert manifest["evaluation_dataset_hash"] == evaluation_hash
    validated = EvaluationReportingManifest.model_validate(json.loads((output_dir / "report_manifest.json").read_text(encoding="utf-8")))
    assert validated.source_runs["advanced_rag"].run_name == "default"
    assert set(validated.output_hashes) == {
        "comparison_summary",
        "comparison_by_level",
        "failure_analysis",
        "thesis_tables",
        "quality_report",
    }


def test_missing_advanced_run_fails_by_default(tmp_path: Path) -> None:
    evaluation_manifest, _ = _write_fixture(tmp_path, include_advanced=False)

    with pytest.raises(RuntimeError, match="advanced_rag"):
        run_evaluation_reporting(_config(tmp_path, evaluation_manifest))


def test_missing_advanced_run_can_emit_partial_report(tmp_path: Path) -> None:
    evaluation_manifest, _ = _write_fixture(tmp_path, include_advanced=False)

    manifest = run_evaluation_reporting(_config(tmp_path, evaluation_manifest, allow_partial=True))

    assert manifest["complete"] is False
    assert any("advanced_rag" in issue for issue in manifest["quality_issues"])
    report = (tmp_path / "reports" / "quality_report.md").read_text(encoding="utf-8")
    assert "advanced_rag" in report
    assert "missing" in report


def test_deltas_are_separate_for_mcq_and_no_hint(tmp_path: Path) -> None:
    evaluation_manifest, _ = _write_fixture(tmp_path)

    run_evaluation_reporting(_config(tmp_path, evaluation_manifest))
    summary = json.loads((tmp_path / "reports" / "comparison_summary.json").read_text(encoding="utf-8"))

    assert summary["deltas"]["mcq"]["no_rag_vs_simple_rag"]["accuracy"] == pytest.approx(1 / 6)
    assert summary["deltas"]["no_hint"]["no_rag_vs_simple_rag"]["coverage"] == pytest.approx(1 / 3)
    assert summary["deltas"]["mcq"]["no_rag_vs_simple_rag"] != summary["deltas"]["no_hint"]["no_rag_vs_simple_rag"]


def test_by_level_is_sorted_and_tables_have_required_headers(tmp_path: Path) -> None:
    evaluation_manifest, _ = _write_fixture(tmp_path)

    run_evaluation_reporting(_config(tmp_path, evaluation_manifest))
    by_level = json.loads((tmp_path / "reports" / "comparison_by_level.json").read_text(encoding="utf-8"))
    tables = (tmp_path / "reports" / "thesis_tables.md").read_text(encoding="utf-8")

    assert by_level["levels"] == ["L1", "L2"]
    assert "## Headline MCQ accuracy" in tables
    assert "| method | processed | accuracy | coverage | strict_accuracy |" in tables
    assert "## Headline no-hint score" in tables
    assert "| method | processed | mean_score | coverage | strict_accuracy |" in tables
    assert "## Failure category breakdown" in tables


def test_metric_contract_and_row_count_mismatches_are_reported(tmp_path: Path) -> None:
    evaluation_dir = tmp_path / "evaluation_clean"
    evaluation_dir.mkdir()
    evaluation_manifest = evaluation_dir / "evaluation_manifest.json"
    write_json(evaluation_manifest, {"schema_version": "evaluation-dataset-v1"})
    evaluation_hash = sha256_file(evaluation_manifest)
    _write_run(tmp_path, "no_rag", evaluation_hash, omit_metric="coverage")
    _write_run(tmp_path, "simple_rag", evaluation_hash)
    advanced_dir = _write_run(tmp_path, "advanced_rag", evaluation_hash)
    summary_path = advanced_dir / "advanced_rag_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["mcq"]["processed"] = 99
    write_json(summary_path, summary)

    with pytest.raises(RuntimeError) as exc_info:
        run_evaluation_reporting(_config(tmp_path, evaluation_manifest))

    message = str(exc_info.value)
    assert "missing metric field coverage" in message
    assert "row count 3 != summary processed 99" in message


def test_failure_category_mapping_and_examples(tmp_path: Path) -> None:
    assert classify_failure("simple_rag", "mcq", {"score": 0, "error": "empty_retrieval", "answer": None}) == "retrieval_miss"
    assert classify_failure("simple_rag", "no_hint", {"judge_score": None, "error": "judge_error: invalid", "answer": "x"}) == "judge_error"
    assert classify_failure("no_rag", "mcq", {"score": None, "error": "invalid_mcq_label: None", "predicted_label": None}) == "generation_error"
    assert classify_failure("advanced_rag", "mcq", {"score": 0, "failure_category": "context_noise"}) == "context_noise"

    evaluation_manifest, _ = _write_fixture(tmp_path)
    run_evaluation_reporting(_config(tmp_path, evaluation_manifest))
    failures = json.loads((tmp_path / "reports" / "failure_analysis.json").read_text(encoding="utf-8"))

    assert failures["counts"]["simple_rag"]["retrieval_miss"] == 1
    assert failures["counts"]["no_rag"]["generation_error"] == 2
    assert failures["counts"]["advanced_rag"]["context_noise"] == 0
    assert failures["counts"]["advanced_rag"]["retrieval_miss"] == 1
    example = failures["examples"]["advanced_rag"]["retrieval_miss"][0]
    assert {"qid", "method", "dataset", "level", "error", "failure_category", "run_name"} <= set(example)


def test_advanced_diagnostics_counts_are_used_when_rows_lack_categories(tmp_path: Path) -> None:
    evaluation_manifest, _ = _write_fixture(tmp_path)
    advanced_dir = tmp_path / "rag_runs" / "advanced" / "default"
    for filename in ("mcq_results.jsonl", "no_hint_results.jsonl"):
        rows = [
            {key: value for key, value in json.loads(line).items() if key != "failure_category"}
            for line in (advanced_dir / filename).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        write_jsonl(advanced_dir / filename, rows)
    write_json(advanced_dir / "advanced_diagnostics.json", {"failure_category_counts": {"context_noise": 4, "unknown": 1}})

    run_evaluation_reporting(_config(tmp_path, evaluation_manifest))
    failures = json.loads((tmp_path / "reports" / "failure_analysis.json").read_text(encoding="utf-8"))

    assert failures["counts"]["advanced_rag"]["context_noise"] == 4
    assert failures["counts"]["advanced_rag"]["unknown"] == 1
