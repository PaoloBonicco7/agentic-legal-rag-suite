from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from legal_rag.oracle_context_evaluation import (
    JudgeOutput,
    OracleEvaluationConfig,
    build_oracle_contexts,
    parse_reference,
    run_oracle_context_evaluation,
    score_mcq_label,
    split_reference_values,
)
from legal_rag.oracle_context_evaluation.runner import resolve_utopia_runtime
from legal_rag.oracle_context_evaluation.runner import resolve_answer_model, resolve_judge_model
from legal_rag.oracle_context_evaluation.io import write_json, write_jsonl
from legal_rag.oracle_context_evaluation.scoring import aggregate_results


class FakeStructuredClient:
    def structured_chat(
        self,
        *,
        prompt: str,
        model: str,
        payload_schema: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        properties = payload_schema.get("properties", {})
        if "answer_label" in properties:
            return {"structured": {"answer_label": "A", "short_rationale": "fake"}}
        if "answer_text" in properties:
            return {"structured": {"answer_text": "Risposta fake", "short_rationale": "fake"}}
        if "score" in properties:
            return {"structured": {"score": 2, "explanation": "Correct fake answer."}}
        raise AssertionError(f"Unexpected schema: {payload_schema}")


def _make_clean_inputs(tmp_path: Path) -> tuple[Path, Path]:
    evaluation_dir = tmp_path / "evaluation_clean"
    laws_dir = tmp_path / "laws_dataset_clean"
    evaluation_dir.mkdir()
    laws_dir.mkdir()

    mcq_records = [
        {
            "qid": "eval-0001",
            "source_position": 1,
            "level": "L1",
            "question_stem": "Prima domanda?",
            "options": {"A": "Corretta", "B": "Errata", "C": "Errata", "D": "Errata", "E": "Errata", "F": "Errata"},
            "correct_label": "A",
            "correct_answer": "Corretta",
            "expected_references": ["Legge regionale 1 gennaio 2000, n. 1 - Art. 4 bis"],
        },
        {
            "qid": "eval-0002",
            "source_position": 2,
            "level": "L2",
            "question_stem": "Seconda domanda?",
            "options": {"A": "Errata", "B": "Corretta", "C": "Errata", "D": "Errata", "E": "Errata", "F": "Errata"},
            "correct_label": "B",
            "correct_answer": "Corretta",
            "expected_references": ["Legge regionale 1 gennaio 2000, n. 1 - Art. 10 bis"],
        },
    ]
    no_hint_records = [
        {
            "qid": "eval-0001",
            "source_position": 1,
            "level": "L1",
            "question": "Prima domanda?",
            "correct_answer": "Corretta",
            "expected_references": ["Legge regionale 1 gennaio 2000, n. 1 - Art. 4 bis"],
            "linked_mcq_qid": "eval-0001",
        },
        {
            "qid": "eval-0002",
            "source_position": 2,
            "level": "L2",
            "question": "Seconda domanda?",
            "correct_answer": "Corretta",
            "expected_references": ["Legge regionale 1 gennaio 2000, n. 1 - Art. 10 bis"],
            "linked_mcq_qid": "eval-0002",
        },
    ]
    laws = [
        {
            "law_id": "vda:lr:2000-01-01:1",
            "law_title": "Legge regionale 1 gennaio 2000, n. 1 - Testo vigente",
        }
    ]
    articles = [
        {
            "law_id": "vda:lr:2000-01-01:1",
            "article_id": "vda:lr:2000-01-01:1#art:4bis",
            "article_label_norm": "4bis",
            "article_text": "Articolo 4 bis test.",
        },
        {
            "law_id": "vda:lr:2000-01-01:1",
            "article_id": "vda:lr:2000-01-01:1#art:10bis",
            "article_label_norm": "10bis",
            "article_text": "Articolo 10 bis test.",
        },
    ]
    write_jsonl(evaluation_dir / "questions_mcq.jsonl", mcq_records)
    write_jsonl(evaluation_dir / "questions_no_hint.jsonl", no_hint_records)
    write_json(evaluation_dir / "evaluation_manifest.json", {"schema_version": "evaluation-dataset-v1"})
    write_jsonl(laws_dir / "laws.jsonl", laws)
    write_jsonl(laws_dir / "articles.jsonl", articles)
    write_json(laws_dir / "manifest.json", {"schema_version": "laws-preprocessing-v1"})
    return evaluation_dir, laws_dir


def test_parse_reference_normalizes_article_suffixes() -> None:
    assert parse_reference("Legge regionale 1 gennaio 2000, n. 1 - Art. 4 bis") == (
        "vda:lr:2000-01-01:1",
        "4bis",
    )
    assert parse_reference("Legge regionale 1 gennaio 2000, n. 1 - Art. 10 bis")[1] == "10bis"
    assert parse_reference("Legge regionale 1 gennaio 2000, n. 1 - Art. 30 quater")[1] == "30quater"


def test_split_reference_values_preserves_pipe_order() -> None:
    assert split_reference_values(["Legge A - Art. 1 | Legge B - Art. 2", " Legge C - Art. 3 "]) == [
        "Legge A - Art. 1",
        "Legge B - Art. 2",
        "Legge C - Art. 3",
    ]


def test_score_mcq_label() -> None:
    assert score_mcq_label("a", "A") == (1, None)
    assert score_mcq_label("B", "A") == (0, None)
    score, error = score_mcq_label("Z", "A")
    assert score is None
    assert error and "invalid_mcq_label" in error


def test_judge_output_accepts_only_zero_to_two_with_explanation() -> None:
    assert JudgeOutput.model_validate({"score": 0, "explanation": "Wrong."}).score == 0
    assert JudgeOutput.model_validate({"score": 1, "explanation": "Partial."}).score == 1
    assert JudgeOutput.model_validate({"score": 2, "explanation": "Correct."}).score == 2
    with pytest.raises(ValueError):
        JudgeOutput.model_validate({"score": 3, "explanation": "Invalid."})
    with pytest.raises(ValueError):
        JudgeOutput.model_validate({"score": 2, "explanation": ""})


def test_run_oracle_context_evaluation_with_fake_client(tmp_path: Path) -> None:
    evaluation_dir, laws_dir = _make_clean_inputs(tmp_path)
    output_dir = tmp_path / "oracle_context"

    manifest = run_oracle_context_evaluation(
        OracleEvaluationConfig(
            evaluation_dir=str(evaluation_dir),
            laws_dir=str(laws_dir),
            output_dir=str(output_dir),
            chat_model="fake-answer",
            judge_model="fake-judge",
        ),
        client=FakeStructuredClient(),
    )

    assert manifest["counts"]["context_errors"] == 0
    assert manifest["counts"]["article_references"] == 2
    assert manifest["counts"]["resolved_article_references"] == 2
    assert {path.name for path in output_dir.iterdir()} == {
        "oracle_context_manifest.json",
        "source_truth_contexts.jsonl",
        "mcq_no_context_results.jsonl",
        "mcq_oracle_context_results.jsonl",
        "no_hint_no_context_results.jsonl",
        "no_hint_oracle_context_results.jsonl",
        "oracle_context_summary.json",
        "quality_report.md",
    }
    summary = json.loads((output_dir / "oracle_context_summary.json").read_text(encoding="utf-8"))
    assert summary["mcq_no_context"]["processed"] == 2
    assert summary["mcq_no_context"]["judged"] == 2
    assert summary["mcq_no_context"]["score_sum"] == 1
    assert summary["no_hint_no_context"]["score_sum"] == 4
    assert summary["delta_oracle_minus_no_context"]["mcq"]["accuracy"] == 0


def test_resolve_utopia_runtime_loads_env_file_without_leaking_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UTOPIA_API_KEY", raising=False)
    monkeypatch.delenv("UTOPIA_BASE_URL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "UTOPIA_API_KEY=test-secret",
                "UTOPIA_BASE_URL=https://utopia.example/api",
                "UTOPIA_CHAT_MODEL=test-model",
            ]
        ),
        encoding="utf-8",
    )

    runtime = resolve_utopia_runtime(OracleEvaluationConfig(env_file=str(env_file)))

    assert runtime["api_url"] == "https://utopia.example/ollama/api/chat"
    assert runtime["api_key"] == "test-secret"
    assert runtime["api_key_present"] is True
    assert runtime["env_file_loaded"] is True
    assert "UTOPIA_API_KEY" not in runtime["env_keys_loaded"]


def test_resolve_utopia_runtime_falls_back_to_legacy_old_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UTOPIA_API_KEY", raising=False)
    monkeypatch.delenv("UTOPIA_BASE_URL", raising=False)
    old_dir = tmp_path / "OLD"
    old_dir.mkdir()
    (old_dir / ".env").write_text(
        "UTOPIA_API_KEY=legacy-secret\nUTOPIA_BASE_URL=https://legacy.example/api\n",
        encoding="utf-8",
    )
    missing_root_env = tmp_path / ".env"

    runtime = resolve_utopia_runtime(OracleEvaluationConfig(env_file=str(missing_root_env)))

    assert runtime["api_url"] == "https://legacy.example/ollama/api/chat"
    assert runtime["api_key"] == "legacy-secret"
    assert runtime["env_file"].endswith("OLD/.env")


def test_model_resolution_uses_env_after_env_file_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UTOPIA_CHAT_MODEL", raising=False)
    monkeypatch.delenv("UTOPIA_JUDGE_MODEL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "UTOPIA_API_KEY=test-secret\nUTOPIA_CHAT_MODEL=server-answer\nUTOPIA_JUDGE_MODEL=server-judge\n",
        encoding="utf-8",
    )

    config = OracleEvaluationConfig(env_file=str(env_file))
    resolve_utopia_runtime(config)
    answer_model = resolve_answer_model(config)

    assert answer_model == "server-answer"
    assert resolve_judge_model(config, answer_model) == "server-judge"


def test_aggregate_results_reports_coverage_strict_accuracy_and_levels() -> None:
    rows = [
        {"level": "L1", "score": 1, "error": None},
        {"level": "L1", "score": None, "error": "invalid"},
        {"level": "L2", "score": 0, "error": None},
    ]
    summary = aggregate_results("mcq", rows, score_key="score", max_score_per_row=1)

    assert summary["processed"] == 3
    assert summary["judged"] == 2
    assert summary["coverage"] == pytest.approx(2 / 3)
    assert summary["strict_accuracy"] == pytest.approx(1 / 3)
    assert summary["by_level"]["L1"]["coverage"] == pytest.approx(1 / 2)


def test_real_no_hint_references_resolve_when_clean_datasets_exist() -> None:
    evaluation_dir = Path("data/evaluation_clean")
    laws_dir = Path("data/laws_dataset_clean")
    if not (evaluation_dir / "questions_no_hint.jsonl").exists() or not (laws_dir / "articles.jsonl").exists():
        pytest.skip("Clean generated datasets are not available.")

    records = [
        json.loads(line)
        for line in (evaluation_dir / "questions_no_hint.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    contexts = build_oracle_contexts(records, laws_dir)

    assert len(contexts) == 100
    assert sum(len(ctx.expected_references) for ctx in contexts) == 106
    assert sum(1 for ctx in contexts if ctx.error) == 0
    assert sum(len(ctx.resolved_references) for ctx in contexts) == 106
