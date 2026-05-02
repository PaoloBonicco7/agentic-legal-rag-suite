from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from legal_rag.evaluation_dataset import EvaluationDatasetConfig, parse_mcq_question, run_evaluation_dataset
from legal_rag.evaluation_dataset.parsing import build_mcq_record, normalize_references, validate_alignment


def _mcq_question(stem: str = "Quale risposta e corretta?") -> str:
    return "\n".join(
        [
            stem,
            "A) Opzione A",
            "B) Opzione B",
            "C) Opzione C",
            "D) Opzione D",
            "E) Opzione E",
            "F) Opzione F",
        ]
    )


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_parse_mcq_question_extracts_stem_and_options() -> None:
    stem, options = parse_mcq_question(_mcq_question("Domanda di test?"))

    assert stem == "Domanda di test?"
    assert list(options) == ["A", "B", "C", "D", "E", "F"]
    assert options["C"] == "Opzione C"


def test_build_mcq_record_maps_correct_label_to_answer_text() -> None:
    record = build_mcq_record(
        {
            "Domanda": _mcq_question(),
            "Livello": " L2 ",
            "Risposta corretta": " c ",
            "Riferimento legge per la risposta": "Legge A - Art. 1\nLegge B - Art. 2",
        },
        7,
    )

    assert record["qid"] == "eval-0007"
    assert record["correct_label"] == "C"
    assert record["correct_answer"] == "Opzione C"
    assert record["expected_references"] == ["Legge A - Art. 1", "Legge B - Art. 2"]


def test_normalize_references_keeps_multiline_references() -> None:
    assert normalize_references(" Legge A - Art. 1 \n\n Legge B   -   Art. 2 ") == [
        "Legge A - Art. 1",
        "Legge B - Art. 2",
    ]


def test_parse_mcq_question_rejects_missing_options() -> None:
    with pytest.raises(ValueError, match="Expected MCQ options"):
        parse_mcq_question("Domanda?\nA) Una\nB) Due")


def test_build_mcq_record_rejects_invalid_correct_label() -> None:
    with pytest.raises(ValueError, match="correct label"):
        build_mcq_record(
            {
                "Domanda": _mcq_question(),
                "Livello": "L1",
                "Risposta corretta": "Z",
                "Riferimento legge per la risposta": "Legge A - Art. 1",
            },
            1,
        )


def test_validate_alignment_rejects_question_mismatch() -> None:
    mcq = build_mcq_record(
        {
            "Domanda": _mcq_question("Domanda MCQ?"),
            "Livello": "L1",
            "Risposta corretta": "A",
            "Riferimento legge per la risposta": "Legge A - Art. 1",
        },
        1,
    )
    no_hint = {
        "qid": "eval-0001",
        "source_position": 1,
        "level": "L1",
        "question": "Domanda diversa?",
        "correct_answer": "Opzione A",
        "expected_references": ["Legge A - Art. 1"],
        "linked_mcq_qid": "eval-0001",
    }

    with pytest.raises(ValueError, match="do not match"):
        validate_alignment(mcq, no_hint)


def test_run_evaluation_dataset_exports_contract_files(tmp_path: Path) -> None:
    source_dir = tmp_path / "evaluation"
    output_dir = tmp_path / "evaluation_clean"
    source_dir.mkdir()
    mcq_source = source_dir / "questions.csv"
    no_hint_source = source_dir / "questions_no_hint.csv"
    mcq_fields = ["#", "Domanda", "Livello", "Risposta corretta", "Riferimento legge per la risposta"]
    no_hint_fields = ["Domanda", "Livello", "Risposta corretta", "Riferimento legge per la risposta"]
    _write_csv(
        mcq_source,
        [
            {
                "#": "1",
                "Domanda": _mcq_question("Prima domanda?"),
                "Livello": "L1",
                "Risposta corretta": "A",
                "Riferimento legge per la risposta": "Legge A - Art. 1",
            },
            {
                "#": "2",
                "Domanda": _mcq_question("Seconda domanda?"),
                "Livello": "L2",
                "Risposta corretta": "B",
                "Riferimento legge per la risposta": "Legge B - Art. 2",
            },
            {"#": "3", "Domanda": "", "Livello": "", "Risposta corretta": "", "Riferimento legge per la risposta": ""},
        ],
        mcq_fields,
    )
    _write_csv(
        no_hint_source,
        [
            {
                "Domanda": "Prima domanda?",
                "Livello": "L1",
                "Risposta corretta": "Opzione A",
                "Riferimento legge per la risposta": "Legge A - Art. 1",
            },
            {
                "Domanda": "Seconda domanda?",
                "Livello": "L2",
                "Risposta corretta": "Opzione B",
                "Riferimento legge per la risposta": "Legge B - Art. 2",
            },
        ],
        no_hint_fields,
    )

    manifest = run_evaluation_dataset(
        EvaluationDatasetConfig(
            mcq_source=str(mcq_source),
            no_hint_source=str(no_hint_source),
            output_dir=str(output_dir),
            expected_records=2,
        )
    )

    assert manifest["ready_for_evaluation"] is True
    assert {path.name for path in output_dir.iterdir()} == {
        "questions_mcq.jsonl",
        "questions_no_hint.jsonl",
        "evaluation_manifest.json",
        "evaluation_profile.json",
        "quality_report.md",
    }
    assert manifest["counts"]["mcq"] == 2
    assert manifest["counts"]["mcq_dropped_empty_rows"] == 1
    assert set(manifest["source_hashes"]) == {"mcq_source", "no_hint_source"}
    assert set(manifest["output_hashes"]) == {
        "questions_mcq",
        "questions_no_hint",
        "quality_report",
        "evaluation_profile",
    }

    mcq_records = [
        json.loads(line)
        for line in (output_dir / "questions_mcq.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    no_hint_records = [
        json.loads(line)
        for line in (output_dir / "questions_no_hint.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert set(mcq_records[0]) == {
        "qid",
        "source_position",
        "level",
        "question_stem",
        "options",
        "correct_label",
        "correct_answer",
        "expected_references",
    }
    assert no_hint_records[0]["linked_mcq_qid"] == mcq_records[0]["qid"]


def test_real_evaluation_dataset_contract(tmp_path: Path) -> None:
    output_dir = tmp_path / "evaluation_clean"

    manifest = run_evaluation_dataset(EvaluationDatasetConfig(output_dir=str(output_dir)))

    assert manifest["ready_for_evaluation"] is True
    assert manifest["counts"]["mcq"] == 100
    assert manifest["counts"]["no_hint"] == 100
    assert manifest["counts"]["mcq_dropped_empty_rows"] == 63
    assert manifest["level_distribution"] == {"L1": 25, "L2": 25, "L3": 25, "L4": 25}
    exported_lines = (output_dir / "questions_mcq.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(exported_lines) == 100
