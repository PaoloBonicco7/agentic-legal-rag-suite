"""Export orchestration for the clean evaluation question dataset."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import (
    EVALUATION_SCHEMA_VERSION,
    EvaluationDatasetConfig,
    evaluation_dataset_manifest,
    evaluation_dataset_profile,
)
from .parsing import (
    build_mcq_record,
    build_no_hint_record,
    is_valid_source_row,
    read_csv_rows,
    validate_alignment,
)


def sha256_file(path: Path) -> str:
    """Compute a SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_output_dir(source_paths: list[Path], output_dir: Path) -> None:
    """Reject output paths that could overwrite source data or the project root."""
    output = output_dir.resolve()
    cwd = Path.cwd().resolve()
    for source_path in source_paths:
        source = source_path.resolve()
        if output == source:
            raise ValueError(f"Output directory must not be a source file: {output}")
        if output == source.parent:
            raise ValueError(f"Output directory must not be the source CSV directory: {output}")
        if output.is_relative_to(source.parent) and output.name == source.stem:
            raise ValueError(f"Output directory must not be derived inside a source file path: {output}")
    if output == cwd:
        raise ValueError(f"Output directory must not be the project/current directory: {output}")
    if output == Path(output.anchor):
        raise ValueError(f"Output directory must not be a filesystem root: {output}")


def _now_utc() -> str:
    """Return an ISO UTC timestamp for manifest-like artifacts."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write stable, sorted JSON for manifest-like artifacts."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write JSONL records sorted by their stable question ID."""
    with path.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda item: str(item.get("qid") or "")):
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _prepare_output_dir(output_dir: Path) -> Path:
    """Create a fresh sibling temporary directory for atomic output writes."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir.parent / f".{output_dir.name}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    return tmp_dir


def _replace_output_dir(tmp_dir: Path, output_dir: Path) -> None:
    """Replace the final output directory only after validation succeeds."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    tmp_dir.replace(output_dir)


def _build_quality(
    *,
    source_counts: dict[str, int],
    mcq_records: list[dict[str, Any]],
    no_hint_records: list[dict[str, Any]],
    expected_records: int,
    source_hashes: dict[str, str] | None = None,
    output_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compute quality gates and notebook-friendly diagnostics."""
    source_hashes = source_hashes or {}
    output_hashes = output_hashes or {}
    mcq_required = {
        "qid",
        "source_position",
        "level",
        "question_stem",
        "options",
        "correct_label",
        "correct_answer",
        "expected_references",
    }
    no_hint_required = {
        "qid",
        "source_position",
        "level",
        "question",
        "correct_answer",
        "expected_references",
        "linked_mcq_qid",
    }
    mcq_missing_fields = sum(1 for record in mcq_records if not mcq_required.issubset(record))
    no_hint_missing_fields = sum(1 for record in no_hint_records if not no_hint_required.issubset(record))
    mcq_empty_required = sum(
        1
        for record in mcq_records
        for field in mcq_required
        if record.get(field) in ("", [], {})
    )
    no_hint_empty_required = sum(
        1
        for record in no_hint_records
        for field in no_hint_required
        if record.get(field) in ("", [], {})
    )
    level_distribution = dict(Counter(str(record["level"]) for record in mcq_records))
    gates = {
        "source_files_readable": source_counts["mcq_total_rows"] > 0 and source_counts["no_hint_total_rows"] > 0,
        "expected_record_count": len(mcq_records) == expected_records and len(no_hint_records) == expected_records,
        "record_counts_match": len(mcq_records) == len(no_hint_records),
        "required_fields_present": mcq_missing_fields == 0 and no_hint_missing_fields == 0,
        "required_fields_non_empty": mcq_empty_required == 0 and no_hint_empty_required == 0,
        "stable_qids_aligned": [record["qid"] for record in mcq_records]
        == [record["linked_mcq_qid"] for record in no_hint_records],
        "level_distribution_reported": bool(level_distribution),
        "source_hashes_recorded": bool(source_hashes) and all(source_hashes.values()),
        "output_files_exist_and_hash": bool(output_hashes) and all(output_hashes.values()),
    }
    return {
        "created_at": _now_utc(),
        "counts": {
            "mcq": len(mcq_records),
            "no_hint": len(no_hint_records),
            "mcq_source_rows": source_counts["mcq_total_rows"],
            "no_hint_source_rows": source_counts["no_hint_total_rows"],
            "mcq_dropped_empty_rows": source_counts["mcq_dropped_empty_rows"],
            "no_hint_dropped_empty_rows": source_counts["no_hint_dropped_empty_rows"],
        },
        "level_distribution": level_distribution,
        "mcq_missing_fields": mcq_missing_fields,
        "no_hint_missing_fields": no_hint_missing_fields,
        "mcq_empty_required": mcq_empty_required,
        "no_hint_empty_required": no_hint_empty_required,
        "quality_gates": gates,
        "ready_for_evaluation": all(gates.values()),
    }


def _write_quality_report(path: Path, quality: dict[str, Any]) -> None:
    """Write a human-readable summary of evaluation dataset quality gates."""
    lines = [
        "# 02 - Evaluation Dataset Quality Report",
        "",
        f"- Ready for evaluation: **{quality['ready_for_evaluation']}**",
        f"- Generated at UTC: `{quality['created_at']}`",
        "",
        "## Quality Gates",
    ]
    for gate, ok in quality["quality_gates"].items():
        lines.append(f"- `{gate}`: **{ok}**")
    lines.extend(
        [
            "",
            "## Counts",
            f"- MCQ records: {quality['counts']['mcq']}",
            f"- No-hint records: {quality['counts']['no_hint']}",
            f"- MCQ source rows: {quality['counts']['mcq_source_rows']}",
            f"- No-hint source rows: {quality['counts']['no_hint_source_rows']}",
            f"- MCQ dropped empty rows: {quality['counts']['mcq_dropped_empty_rows']}",
            f"- No-hint dropped empty rows: {quality['counts']['no_hint_dropped_empty_rows']}",
            "",
            "## Level Distribution",
        ]
    )
    for level, count in sorted(quality["level_distribution"].items()):
        lines.append(f"- {level}: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_profile(
    *,
    mcq_records: list[dict[str, Any]],
    no_hint_records: list[dict[str, Any]],
    quality: dict[str, Any],
) -> dict[str, Any]:
    """Build a compact notebook-friendly profile for the clean evaluation data."""
    reference_counts = Counter(ref for record in mcq_records for ref in record["expected_references"])
    return evaluation_dataset_profile(
        {
            "counts": quality["counts"],
            "level_distribution": quality["level_distribution"],
            "reference_count": len(reference_counts),
            "sample_references": sorted(reference_counts)[:10],
            "sample_mcq_records": mcq_records[:3],
            "sample_no_hint_records": no_hint_records[:3],
            "alignment_examples": [
                {
                    "qid": mcq["qid"],
                    "mcq_stem": mcq["question_stem"],
                    "no_hint_question": no_hint["question"],
                    "correct_answer": mcq["correct_answer"],
                }
                for mcq, no_hint in list(zip(mcq_records, no_hint_records))[:3]
            ],
            "ready_for_evaluation": quality["ready_for_evaluation"],
        }
    )


def run_evaluation_dataset(config: EvaluationDatasetConfig | None = None) -> dict[str, Any]:
    """Run the complete deterministic evaluation dataset normalization pipeline."""
    cfg = config or EvaluationDatasetConfig()
    if not isinstance(cfg, EvaluationDatasetConfig):
        cfg = EvaluationDatasetConfig.model_validate(cfg)

    mcq_source = Path(cfg.mcq_source)
    no_hint_source = Path(cfg.no_hint_source)
    output_dir = Path(cfg.output_dir)
    validate_output_dir([mcq_source, no_hint_source], output_dir)

    mcq_rows_all = read_csv_rows(mcq_source)
    no_hint_rows_all = read_csv_rows(no_hint_source)
    mcq_rows = [row for row in mcq_rows_all if is_valid_source_row(row)]
    no_hint_rows = [row for row in no_hint_rows_all if is_valid_source_row(row)]
    source_counts = {
        "mcq_total_rows": len(mcq_rows_all),
        "no_hint_total_rows": len(no_hint_rows_all),
        "mcq_dropped_empty_rows": len(mcq_rows_all) - len(mcq_rows),
        "no_hint_dropped_empty_rows": len(no_hint_rows_all) - len(no_hint_rows),
    }
    if len(mcq_rows) != len(no_hint_rows):
        raise ValueError(f"MCQ/no-hint valid record counts do not match: {len(mcq_rows)} != {len(no_hint_rows)}")
    if len(mcq_rows) != cfg.expected_records:
        raise ValueError(f"Expected {cfg.expected_records} valid records, found {len(mcq_rows)}")

    mcq_records: list[dict[str, Any]] = []
    no_hint_records: list[dict[str, Any]] = []
    for position, (mcq_row, no_hint_row) in enumerate(zip(mcq_rows, no_hint_rows), start=1):
        mcq = build_mcq_record(mcq_row, position)
        no_hint = build_no_hint_record(no_hint_row, position, linked_mcq_qid=mcq["qid"])
        validate_alignment(mcq, no_hint)
        mcq_records.append(mcq)
        no_hint_records.append(no_hint)

    output_files = {
        "questions_mcq": "questions_mcq.jsonl",
        "questions_no_hint": "questions_no_hint.jsonl",
        "evaluation_manifest": "evaluation_manifest.json",
        "evaluation_profile": "evaluation_profile.json",
        "quality_report": "quality_report.md",
    }
    tmp_dir = _prepare_output_dir(output_dir)
    try:
        _write_jsonl(mcq_records, tmp_dir / output_files["questions_mcq"])
        _write_jsonl(no_hint_records, tmp_dir / output_files["questions_no_hint"])

        source_hashes = {
            "mcq_source": sha256_file(mcq_source),
            "no_hint_source": sha256_file(no_hint_source),
        }
        output_hashes = {
            "questions_mcq": sha256_file(tmp_dir / output_files["questions_mcq"]),
            "questions_no_hint": sha256_file(tmp_dir / output_files["questions_no_hint"]),
        }
        quality = _build_quality(
            source_counts=source_counts,
            mcq_records=mcq_records,
            no_hint_records=no_hint_records,
            expected_records=cfg.expected_records,
            source_hashes=source_hashes,
            output_hashes=output_hashes,
        )
        _write_quality_report(tmp_dir / output_files["quality_report"], quality)
        output_hashes["quality_report"] = sha256_file(tmp_dir / output_files["quality_report"])
        profile = _build_profile(mcq_records=mcq_records, no_hint_records=no_hint_records, quality=quality)
        _write_json(tmp_dir / output_files["evaluation_profile"], profile)
        output_hashes["evaluation_profile"] = sha256_file(tmp_dir / output_files["evaluation_profile"])

        quality = _build_quality(
            source_counts=source_counts,
            mcq_records=mcq_records,
            no_hint_records=no_hint_records,
            expected_records=cfg.expected_records,
            source_hashes=source_hashes,
            output_hashes=output_hashes,
        )
        manifest = evaluation_dataset_manifest(
            {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "created_at": _now_utc(),
                "mcq_source": str(mcq_source),
                "no_hint_source": str(no_hint_source),
                "output_dir": str(output_dir),
                "config": cfg.model_dump(mode="json"),
                "source_hashes": source_hashes,
                "counts": quality["counts"],
                "level_distribution": quality["level_distribution"],
                "quality_gates": quality["quality_gates"],
                "ready_for_evaluation": quality["ready_for_evaluation"],
                "outputs": output_files,
                "output_hashes": output_hashes,
                "manifest_hash_note": "evaluation_manifest.json is excluded from output_hashes because a file cannot contain a stable hash of itself.",
            }
        )
        _write_json(tmp_dir / output_files["evaluation_manifest"], manifest)
        final_quality = _build_quality(
            source_counts=source_counts,
            mcq_records=mcq_records,
            no_hint_records=no_hint_records,
            expected_records=cfg.expected_records,
            source_hashes=source_hashes,
            output_hashes=output_hashes,
        )
        if not final_quality["ready_for_evaluation"]:
            raise ValueError(f"Evaluation dataset quality gates failed: {final_quality['quality_gates']}")
        _replace_output_dir(tmp_dir, output_dir)
        return manifest
    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise
