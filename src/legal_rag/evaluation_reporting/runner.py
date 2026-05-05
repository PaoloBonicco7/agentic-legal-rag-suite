"""Evaluation reporting orchestration."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from legal_rag.oracle_context_evaluation.io import (
    now_utc,
    prepare_tmp_output_dir,
    read_json,
    read_jsonl,
    replace_output_dir,
    sha256_file,
    write_json,
)
from legal_rag.oracle_context_evaluation.scoring import level_sort_key

from .models import (
    DATASETS,
    EVALUATION_REPORTING_SCHEMA_VERSION,
    FAILURE_CATEGORIES,
    METHODS,
    REQUIRED_METRIC_FIELDS,
    EvaluationReportingConfig,
    EvaluationReportingManifest,
    FailureExampleRecord,
    ReportOutputFiles,
    SafeReportingConfigRecord,
    SourceRunRecord,
)

METRIC_DELTA_FIELDS = ("accuracy", "mean_score", "strict_accuracy", "coverage")


@dataclass(frozen=True)
class RunArtifacts:
    method: str
    run_dir: Path
    manifest_path: Path
    summary_path: Path
    mcq_results_path: Path
    no_hint_results_path: Path
    diagnostics_path: Path | None
    manifest: dict[str, Any]
    summary: dict[str, Any]
    mcq_rows: list[dict[str, Any]]
    no_hint_rows: list[dict[str, Any]]
    diagnostics: dict[str, Any] | None
    source_record: SourceRunRecord


def run_evaluation_reporting(config: EvaluationReportingConfig | dict[str, Any] | None = None) -> dict[str, Any]:
    """Build all comparison report artifacts."""
    cfg = config if isinstance(config, EvaluationReportingConfig) else EvaluationReportingConfig.model_validate(config or {})
    output_dir = Path(cfg.output_dir)
    tmp_dir = prepare_tmp_output_dir(output_dir)
    files = ReportOutputFiles()
    quality_issues: list[str] = []
    try:
        evaluation_manifest_path = Path(cfg.evaluation_manifest_path)
        if not evaluation_manifest_path.exists():
            raise FileNotFoundError(f"Missing evaluation manifest: {evaluation_manifest_path}")
        evaluation_hash = sha256_file(evaluation_manifest_path)
        evaluation_manifest = read_json(evaluation_manifest_path)
        levels = _levels_from_evaluation_manifest(evaluation_manifest)

        runs = _load_runs(cfg, quality_issues)
        _validate_comparison_inputs(runs, evaluation_hash, quality_issues)
        if quality_issues and not cfg.allow_partial:
            raise RuntimeError("Evaluation reporting inputs are incompatible:\n" + "\n".join(f"- {issue}" for issue in quality_issues))

        loaded = {method: run for method, run in runs.items() if run is not None}
        if not loaded:
            raise RuntimeError("No comparable source runs were loaded")

        levels = _merge_levels(levels, loaded)
        comparison_summary = build_comparison_summary(loaded)
        comparison_by_level = build_comparison_by_level(loaded, levels)
        failure_analysis = build_failure_analysis(loaded, max_examples_per_category=cfg.max_examples_per_category)
        thesis_tables = build_thesis_tables(
            comparison_summary=comparison_summary,
            comparison_by_level=comparison_by_level,
            failure_analysis=failure_analysis,
            levels=levels,
        )
        quality_report = build_quality_report(
            config=cfg,
            complete=set(loaded) == set(METHODS),
            quality_issues=quality_issues,
            loaded=loaded,
            comparison_summary=comparison_summary,
        )

        write_json(tmp_dir / files.comparison_summary, comparison_summary)
        write_json(tmp_dir / files.comparison_by_level, comparison_by_level)
        write_json(tmp_dir / files.failure_analysis, failure_analysis)
        (tmp_dir / files.thesis_tables).write_text(thesis_tables, encoding="utf-8")
        (tmp_dir / files.quality_report).write_text(quality_report, encoding="utf-8")

        source_runs = {method: run.source_record for method, run in loaded.items()}
        for method in METHODS:
            if method not in source_runs:
                source_runs[method] = _missing_source_record(method, _run_dir(cfg, method))
        source_hashes = _source_hashes(loaded)
        output_hashes = {
            key: sha256_file(tmp_dir / filename)
            for key, filename in files.to_json_record().items()
            if key != "report_manifest"
        }
        manifest = EvaluationReportingManifest(
            schema_version=EVALUATION_REPORTING_SCHEMA_VERSION,
            created_at=now_utc(),
            config=SafeReportingConfigRecord.model_validate(cfg.model_dump(mode="json")),
            complete=set(loaded) == set(METHODS) and not quality_issues,
            evaluation_dataset_hash=evaluation_hash,
            source_runs=source_runs,
            source_hashes=source_hashes,
            quality_issues=quality_issues,
            outputs=files,
            output_hashes=output_hashes,
            manifest_hash_note="report_manifest.json is excluded from output_hashes because a file cannot contain a stable hash of itself.",
        ).to_json_record()
        write_json(tmp_dir / files.report_manifest, manifest)
        replace_output_dir(tmp_dir, output_dir)
        return manifest
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise


def _run_dir(config: EvaluationReportingConfig, method: str) -> Path:
    if method == "no_rag":
        return Path(config.no_rag_dir)
    if method == "simple_rag":
        return Path(config.simple_rag_dir)
    return Path(config.advanced_rag_dir)


def _filenames(method: str) -> tuple[str, str]:
    if method == "no_rag":
        return "no_rag_manifest.json", "no_rag_summary.json"
    if method == "simple_rag":
        return "simple_rag_manifest.json", "simple_rag_summary.json"
    return "advanced_rag_manifest.json", "advanced_rag_summary.json"


def _load_runs(config: EvaluationReportingConfig, quality_issues: list[str]) -> dict[str, RunArtifacts | None]:
    runs: dict[str, RunArtifacts | None] = {}
    for method in METHODS:
        run_dir = _run_dir(config, method)
        try:
            runs[method] = load_run_artifacts(method, run_dir)
        except FileNotFoundError as exc:
            issue = f"{method}: {exc}"
            quality_issues.append(issue)
            if not config.allow_partial:
                runs[method] = None
            else:
                runs[method] = None
        except Exception as exc:
            quality_issues.append(f"{method}: {type(exc).__name__}: {exc}")
            if not config.allow_partial:
                runs[method] = None
            else:
                runs[method] = None
    return runs


def load_run_artifacts(method: str, run_dir: Path) -> RunArtifacts:
    """Load one method run from disk."""
    manifest_name, summary_name = _filenames(method)
    manifest_path = run_dir / manifest_name
    summary_path = run_dir / summary_name
    mcq_results_path = run_dir / "mcq_results.jsonl"
    no_hint_results_path = run_dir / "no_hint_results.jsonl"
    diagnostics_path = run_dir / "advanced_diagnostics.json" if method == "advanced_rag" else None
    required = [manifest_path, summary_path, mcq_results_path, no_hint_results_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required files: " + ", ".join(missing))

    manifest = read_json(manifest_path)
    summary = read_json(summary_path)
    mcq_rows = read_jsonl(mcq_results_path)
    no_hint_rows = read_jsonl(no_hint_results_path)
    diagnostics = read_json(diagnostics_path) if diagnostics_path and diagnostics_path.exists() else None
    source = SourceRunRecord(
        method=method,  # type: ignore[arg-type]
        run_dir=str(run_dir),
        manifest_path=str(manifest_path),
        summary_path=str(summary_path),
        mcq_results_path=str(mcq_results_path),
        no_hint_results_path=str(no_hint_results_path),
        diagnostics_path=str(diagnostics_path) if diagnostics_path and diagnostics_path.exists() else None,
        manifest_hash=sha256_file(manifest_path),
        summary_hash=sha256_file(summary_path),
        mcq_results_hash=sha256_file(mcq_results_path),
        no_hint_results_hash=sha256_file(no_hint_results_path),
        diagnostics_hash=sha256_file(diagnostics_path) if diagnostics_path and diagnostics_path.exists() else None,
        run_name=_run_name(method, run_dir, manifest),
        present=True,
    )
    return RunArtifacts(
        method=method,
        run_dir=run_dir,
        manifest_path=manifest_path,
        summary_path=summary_path,
        mcq_results_path=mcq_results_path,
        no_hint_results_path=no_hint_results_path,
        diagnostics_path=diagnostics_path if diagnostics_path and diagnostics_path.exists() else None,
        manifest=manifest,
        summary=summary,
        mcq_rows=mcq_rows,
        no_hint_rows=no_hint_rows,
        diagnostics=diagnostics,
        source_record=source,
    )


def _missing_source_record(method: str, run_dir: Path) -> SourceRunRecord:
    manifest_name, summary_name = _filenames(method)
    return SourceRunRecord(
        method=method,  # type: ignore[arg-type]
        run_dir=str(run_dir),
        manifest_path=str(run_dir / manifest_name),
        summary_path=str(run_dir / summary_name),
        mcq_results_path=str(run_dir / "mcq_results.jsonl"),
        no_hint_results_path=str(run_dir / "no_hint_results.jsonl"),
        diagnostics_path=str(run_dir / "advanced_diagnostics.json") if method == "advanced_rag" else None,
        manifest_hash=None,
        summary_hash=None,
        mcq_results_hash=None,
        no_hint_results_hash=None,
        diagnostics_hash=None,
        run_name=run_dir.name if method == "advanced_rag" else None,
        present=False,
    )


def _run_name(method: str, run_dir: Path, manifest: dict[str, Any]) -> str | None:
    if method != "advanced_rag":
        return None
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    return str(config.get("run_name") or run_dir.name)


def _validate_comparison_inputs(
    runs: dict[str, RunArtifacts | None],
    evaluation_hash: str,
    quality_issues: list[str],
) -> None:
    loaded = {method: run for method, run in runs.items() if run is not None}
    for method in METHODS:
        if runs.get(method) is None:
            quality_issues.append(f"{method}: source run is not available")
    for method, run in loaded.items():
        _validate_metric_contract(method, run.summary, quality_issues)
        _validate_row_counts(method, run, quality_issues)
        manifest_hash = (run.manifest.get("source_hashes") or {}).get("evaluation_manifest")
        if manifest_hash and manifest_hash != evaluation_hash:
            quality_issues.append(
                f"{method}: evaluation_manifest hash mismatch: {manifest_hash} != {evaluation_hash}"
            )
        if not manifest_hash:
            quality_issues.append(f"{method}: missing source_hashes.evaluation_manifest")


def _validate_metric_contract(method: str, summary: dict[str, Any], quality_issues: list[str]) -> None:
    for dataset in DATASETS:
        metrics = summary.get(dataset)
        if not isinstance(metrics, dict):
            quality_issues.append(f"{method}.{dataset}: missing metrics object")
            continue
        for field in REQUIRED_METRIC_FIELDS:
            if field not in metrics:
                quality_issues.append(f"{method}.{dataset}: missing metric field {field}")
        if "by_level" in metrics and not isinstance(metrics["by_level"], dict):
            quality_issues.append(f"{method}.{dataset}: by_level must be an object")


def _validate_row_counts(method: str, run: RunArtifacts, quality_issues: list[str]) -> None:
    expected = {
        "mcq": len(run.mcq_rows),
        "no_hint": len(run.no_hint_rows),
    }
    for dataset, row_count in expected.items():
        metrics = run.summary.get(dataset)
        if not isinstance(metrics, dict):
            continue
        processed = metrics.get("processed")
        if processed != row_count:
            quality_issues.append(f"{method}.{dataset}: row count {row_count} != summary processed {processed}")


def build_comparison_summary(runs: dict[str, RunArtifacts]) -> dict[str, Any]:
    """Build global metrics and pairwise deltas."""
    methods = {method: _summary_for_run(run) for method, run in runs.items()}
    return {
        "schema_version": EVALUATION_REPORTING_SCHEMA_VERSION,
        "methods": methods,
        "deltas": {
            dataset: _pairwise_deltas({method: run.summary[dataset] for method, run in runs.items() if dataset in run.summary})
            for dataset in DATASETS
        },
    }


def _summary_for_run(run: RunArtifacts) -> dict[str, Any]:
    return {
        "run_name": run.source_record.run_name,
        "manifest_path": str(run.manifest_path),
        "mcq": run.summary.get("mcq"),
        "no_hint": run.summary.get("no_hint"),
    }


def build_comparison_by_level(runs: dict[str, RunArtifacts], levels: list[str]) -> dict[str, Any]:
    """Build by-level metrics and pairwise deltas."""
    out: dict[str, Any] = {
        "schema_version": EVALUATION_REPORTING_SCHEMA_VERSION,
        "levels": levels,
        "metrics": {},
        "deltas": {},
    }
    for dataset in DATASETS:
        out["metrics"][dataset] = {}
        for method, run in runs.items():
            by_level = (run.summary.get(dataset) or {}).get("by_level") or {}
            out["metrics"][dataset][method] = {level: by_level.get(level) for level in levels}
        out["deltas"][dataset] = {}
        for level in levels:
            values = {
                method: (((run.summary.get(dataset) or {}).get("by_level") or {}).get(level) or {})
                for method, run in runs.items()
            }
            out["deltas"][dataset][level] = _pairwise_deltas(values)
    return out


def _pairwise_deltas(metrics_by_method: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pairs = (
        ("no_rag_vs_simple_rag", "no_rag", "simple_rag"),
        ("simple_rag_vs_advanced_rag", "simple_rag", "advanced_rag"),
        ("no_rag_vs_advanced_rag", "no_rag", "advanced_rag"),
    )
    return {
        label: _metric_delta(metrics_by_method.get(left), metrics_by_method.get(right))
        for label, left, right in pairs
    }


def _metric_delta(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, float | None]:
    if left is None or right is None:
        return {field: None for field in METRIC_DELTA_FIELDS}
    return {field: _delta(right.get(field), left.get(field)) for field in METRIC_DELTA_FIELDS}


def _delta(right: Any, left: Any) -> float | None:
    if right is None or left is None:
        return None
    return float(right) - float(left)


def build_failure_analysis(runs: dict[str, RunArtifacts], *, max_examples_per_category: int) -> dict[str, Any]:
    """Build failure category counts and representative examples."""
    counts = {method: {category: 0 for category in FAILURE_CATEGORIES} for method in METHODS}
    examples = {
        method: {category: [] for category in FAILURE_CATEGORIES}
        for method in METHODS
    }
    for method, run in runs.items():
        rows_have_advanced_categories = False
        for dataset, rows in (("mcq", run.mcq_rows), ("no_hint", run.no_hint_rows)):
            for row in rows:
                if method == "advanced_rag" and row.get("failure_category"):
                    rows_have_advanced_categories = True
                category = classify_failure(method, dataset, row)
                if category is None:
                    continue
                counts[method][category] += 1
                if len(examples[method][category]) < max_examples_per_category:
                    examples[method][category].append(
                        FailureExampleRecord(
                            qid=str(row.get("qid") or ""),
                            method=method,  # type: ignore[arg-type]
                            dataset=dataset,  # type: ignore[arg-type]
                            level=str(row.get("level") or "UNKNOWN"),
                            score=_optional_int(row.get("score")),
                            judge_score=_optional_int(row.get("judge_score")),
                            error=str(row.get("error")) if row.get("error") is not None else None,
                            failure_category=category,  # type: ignore[arg-type]
                            run_name=run.source_record.run_name,
                        ).to_json_record()
                    )
        if method == "advanced_rag" and run.diagnostics and not rows_have_advanced_categories:
            for category, count in _diagnostic_failure_counts(run.diagnostics).items():
                counts[method][category] = int(count)
    return {
        "schema_version": EVALUATION_REPORTING_SCHEMA_VERSION,
        "categories": list(FAILURE_CATEGORIES),
        "counts": counts,
        "examples": examples,
    }


def classify_failure(method: str, dataset: str, row: dict[str, Any]) -> str | None:
    """Map a row to a failure category, or None when the row is successful."""
    if method == "advanced_rag" and row.get("failure_category"):
        category = str(row["failure_category"])
        return category if category in FAILURE_CATEGORIES else "unknown"
    score_key = "score" if dataset == "mcq" else "judge_score"
    max_score = 1 if dataset == "mcq" else 2
    score = row.get(score_key)
    error = str(row.get("error") or "")
    answer = str(row.get("predicted_answer") or row.get("answer") or row.get("predicted_label") or "").strip()
    if score == max_score and not error:
        return None
    if "empty_retrieval" in error:
        return "retrieval_miss"
    if "judge_error" in error:
        return "judge_error"
    if "structured_error" in error or "invalid_mcq_label" in error:
        return "generation_error"
    if not answer:
        return "abstention"
    return "unknown"


def _diagnostic_failure_counts(diagnostics: dict[str, Any]) -> dict[str, int]:
    for key in ("failure_category_counts", "failure_categories", "failure_counts"):
        value = diagnostics.get(key)
        if isinstance(value, dict):
            return {
                str(category): int(count)
                for category, count in value.items()
                if category in FAILURE_CATEGORIES and isinstance(count, int)
            }
    return {}


def build_thesis_tables(
    *,
    comparison_summary: dict[str, Any],
    comparison_by_level: dict[str, Any],
    failure_analysis: dict[str, Any],
    levels: list[str],
) -> str:
    """Render fixed-order Markdown tables for thesis drafting."""
    methods = list(METHODS)
    lines: list[str] = ["# 07 - Evaluation Reporting Tables", ""]
    lines.extend(_headline_table("Headline MCQ accuracy", methods, comparison_summary, "mcq", ["processed", "accuracy", "coverage", "strict_accuracy"]))
    lines.extend(_headline_table("Headline no-hint score", methods, comparison_summary, "no_hint", ["processed", "mean_score", "coverage", "strict_accuracy"]))
    lines.extend(_level_table("MCQ accuracy by level", methods, comparison_by_level, "mcq", "accuracy", levels))
    lines.extend(_level_table("No-hint mean score by level", methods, comparison_by_level, "no_hint", "mean_score", levels))
    lines.extend(_failure_table(methods, failure_analysis))
    return "\n".join(lines).rstrip() + "\n"


def _headline_table(title: str, methods: list[str], summary: dict[str, Any], dataset: str, columns: list[str]) -> list[str]:
    lines = [f"## {title}", "", "| method | " + " | ".join(columns) + " |", "| --- | " + " | ".join("---" for _ in columns) + " |"]
    for method in methods:
        metrics = ((summary.get("methods") or {}).get(method) or {}).get(dataset) or {}
        lines.append("| " + " | ".join([method, *[_format_value(metrics.get(column)) for column in columns]]) + " |")
    lines.append("")
    return lines


def _level_table(title: str, methods: list[str], by_level: dict[str, Any], dataset: str, metric_name: str, levels: list[str]) -> list[str]:
    lines = [f"## {title}", "", "| method | " + " | ".join(levels) + " |", "| --- | " + " | ".join("---" for _ in levels) + " |"]
    metrics = ((by_level.get("metrics") or {}).get(dataset) or {})
    for method in methods:
        method_levels = metrics.get(method) or {}
        values = [_format_value((method_levels.get(level) or {}).get(metric_name)) for level in levels]
        lines.append("| " + " | ".join([method, *values]) + " |")
    lines.append("")
    return lines


def _failure_table(methods: list[str], failure_analysis: dict[str, Any]) -> list[str]:
    categories = list(FAILURE_CATEGORIES)
    lines = [
        "## Failure category breakdown",
        "",
        "| method | " + " | ".join(categories) + " |",
        "| --- | " + " | ".join("---" for _ in categories) + " |",
    ]
    counts = failure_analysis.get("counts") or {}
    for method in methods:
        method_counts = counts.get(method) or {}
        values = [str(method_counts.get(category, 0)) for category in categories]
        lines.append("| " + " | ".join([method, *values]) + " |")
    lines.append("")
    return lines


def build_quality_report(
    *,
    config: EvaluationReportingConfig,
    complete: bool,
    quality_issues: list[str],
    loaded: dict[str, RunArtifacts],
    comparison_summary: dict[str, Any],
) -> str:
    """Render a concise human-readable quality report."""
    lines = [
        "# 07 - Evaluation Reporting Quality Report",
        "",
        "## Status",
        f"- complete={complete}",
        f"- allow_partial={config.allow_partial}",
        "",
        "## Source Runs",
    ]
    for method in METHODS:
        run = loaded.get(method)
        path = str(run.run_dir) if run else str(_run_dir(config, method))
        lines.append(f"- `{method}`: {path} ({'loaded' if run else 'missing'})")
    lines.extend(["", "## Quality Gates"])
    if quality_issues:
        for issue in quality_issues:
            lines.append(f"- FAIL: {issue}")
    else:
        lines.append("- PASS: all source run manifests exist")
        lines.append("- PASS: compared runs use the same evaluation dataset hash")
        lines.append("- PASS: summaries expose the required metric fields")
        lines.append("- PASS: row-level result counts match summary counts")
        lines.append("- PASS: deltas are computed separately for MCQ and no-hint")
    lines.extend(["", "## Headline Metrics"])
    for method in METHODS:
        method_summary = (comparison_summary.get("methods") or {}).get(method)
        if not method_summary:
            continue
        mcq = method_summary.get("mcq") or {}
        no_hint = method_summary.get("no_hint") or {}
        lines.append(
            f"- `{method}`: mcq_accuracy={_format_value(mcq.get('accuracy'))}, "
            f"no_hint_mean_score={_format_value(no_hint.get('mean_score'))}"
        )
    return "\n".join(lines) + "\n"


def _source_hashes(runs: dict[str, RunArtifacts]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for method, run in runs.items():
        for key, value in run.source_record.to_json_record().items():
            if key.endswith("_hash") and value:
                hashes[f"{method}.{key.removesuffix('_hash')}"] = str(value)
    return hashes


def _levels_from_evaluation_manifest(manifest: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for key in ("levels", "level_counts", "by_level"):
        value = manifest.get(key)
        if isinstance(value, list):
            candidates.extend(str(item) for item in value)
        elif isinstance(value, dict):
            candidates.extend(str(item) for item in value)
    return _sorted_unique_levels(candidates)


def _merge_levels(levels: list[str], runs: dict[str, RunArtifacts]) -> list[str]:
    candidates = list(levels)
    for run in runs.values():
        for dataset in DATASETS:
            by_level = (run.summary.get(dataset) or {}).get("by_level") or {}
            candidates.extend(str(level) for level in by_level)
    return _sorted_unique_levels(candidates)


def _sorted_unique_levels(levels: list[str]) -> list[str]:
    return sorted({level for level in levels if level}, key=level_sort_key)


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
