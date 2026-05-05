from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import legal_rag.no_rag_baseline.runner as no_rag_runner
from legal_rag.no_rag_baseline import (
    NO_RAG_PROMPT_VERSION,
    NO_RAG_SCHEMA_VERSION,
    NoRagConfig,
    NoRagManifest,
    resolve_answer_model,
    resolve_judge_model,
    resolve_utopia_runtime,
    run_mcq,
    run_no_hint,
    run_no_rag_baseline,
    score_mcq_label,
)
from legal_rag.oracle_context_evaluation.io import write_json, write_jsonl


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


class FailingNoHintClient(FakeStructuredClient):
    def __init__(self) -> None:
        self.judge_calls = 0

    def structured_chat(
        self,
        *,
        prompt: str,
        model: str,
        payload_schema: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        properties = payload_schema.get("properties", {})
        if "answer_text" in properties:
            raise RuntimeError("answer failed")
        if "score" in properties:
            self.judge_calls += 1
        return super().structured_chat(
            prompt=prompt,
            model=model,
            payload_schema=payload_schema,
            timeout_seconds=timeout_seconds,
        )


class InvalidJudgeClient(FakeStructuredClient):
    def structured_chat(
        self,
        *,
        prompt: str,
        model: str,
        payload_schema: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        properties = payload_schema.get("properties", {})
        if "score" in properties:
            return {"structured": {"score": 3, "explanation": ""}}
        return super().structured_chat(
            prompt=prompt,
            model=model,
            payload_schema=payload_schema,
            timeout_seconds=timeout_seconds,
        )


class ObservedSlowClient:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def structured_chat(
        self,
        *,
        prompt: str,
        model: str,
        payload_schema: dict[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            time.sleep(0.02)
            return {"structured": {"answer_label": "A", "short_rationale": "fake"}}
        finally:
            with self._lock:
                self._active -= 1


def _make_clean_inputs(tmp_path: Path) -> Path:
    evaluation_dir = tmp_path / "evaluation_clean"
    evaluation_dir.mkdir()
    mcq_records = [
        {
            "qid": "eval-0001",
            "source_position": 1,
            "level": "L1",
            "question_stem": "Prima domanda?",
            "options": {"A": "Corretta", "B": "Errata", "C": "Errata", "D": "Errata", "E": "Errata", "F": "Errata"},
            "correct_label": "A",
            "correct_answer": "Corretta",
            "expected_references": ["Legge A - Art. 1"],
        },
        {
            "qid": "eval-0002",
            "source_position": 2,
            "level": "L2",
            "question_stem": "Seconda domanda?",
            "options": {"A": "Errata", "B": "Corretta", "C": "Errata", "D": "Errata", "E": "Errata", "F": "Errata"},
            "correct_label": "B",
            "correct_answer": "Corretta",
            "expected_references": ["Legge B - Art. 2"],
        },
    ]
    no_hint_records = [
        {
            "qid": "eval-0001",
            "source_position": 1,
            "level": "L1",
            "question": "Prima domanda?",
            "correct_answer": "Corretta",
            "expected_references": ["Legge A - Art. 1"],
            "linked_mcq_qid": "eval-0001",
        },
        {
            "qid": "eval-0002",
            "source_position": 2,
            "level": "L2",
            "question": "Seconda domanda?",
            "correct_answer": "Corretta",
            "expected_references": ["Legge B - Art. 2"],
            "linked_mcq_qid": "eval-0002",
        },
    ]
    write_jsonl(evaluation_dir / "questions_mcq.jsonl", mcq_records)
    write_jsonl(evaluation_dir / "questions_no_hint.jsonl", no_hint_records)
    write_json(evaluation_dir / "evaluation_manifest.json", {"schema_version": "evaluation-dataset-v1"})
    return evaluation_dir


def test_score_mcq_label_handles_valid_wrong_and_invalid_labels() -> None:
    assert score_mcq_label("a", "A") == (1, None)
    assert score_mcq_label("B", "A") == (0, None)
    score, error = score_mcq_label("Z", "A")
    assert score is None
    assert error and "invalid_mcq_label" in error


def test_no_rag_config_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        NoRagConfig.model_validate({"unexpected": "value"})


def test_run_no_rag_baseline_exports_contract_files(tmp_path: Path) -> None:
    evaluation_dir = _make_clean_inputs(tmp_path)
    output_dir = tmp_path / "no_rag"

    manifest = run_no_rag_baseline(
        NoRagConfig(
            evaluation_dir=str(evaluation_dir),
            output_dir=str(output_dir),
            api_key="secret-value",
            chat_model="fake-answer",
            judge_model="fake-judge",
            random_seed=7,
        ),
        client=FakeStructuredClient(),
    )

    assert {path.name for path in output_dir.iterdir()} == {
        "no_rag_manifest.json",
        "mcq_results.jsonl",
        "no_hint_results.jsonl",
        "no_rag_summary.json",
        "quality_report.md",
    }
    assert manifest["schema_version"] == NO_RAG_SCHEMA_VERSION
    assert manifest["prompt_version"] == NO_RAG_PROMPT_VERSION
    assert manifest["random_seed"] == 7
    assert "api_key" not in manifest["config"]
    assert manifest["config"]["api_key_present"] is True
    assert "api_key" not in manifest["connection"]
    assert manifest["connection"]["client"] == "injected"
    assert manifest["counts"] == {"mcq": 2, "no_hint": 2, "mcq_errors": 0, "no_hint_errors": 0}
    assert manifest["models"] == {"answer_model": "fake-answer", "judge_model": "fake-judge"}
    assert set(manifest["source_hashes"]) == {"questions_mcq", "questions_no_hint", "evaluation_manifest"}
    assert set(manifest["output_hashes"]) == {"mcq_results", "no_hint_results", "no_rag_summary", "quality_report"}

    manifest_from_disk = json.loads((output_dir / "no_rag_manifest.json").read_text(encoding="utf-8"))
    validated_manifest = NoRagManifest.model_validate(manifest_from_disk)
    assert validated_manifest.outputs.mcq_results == "mcq_results.jsonl"
    assert validated_manifest.counts.mcq == 2

    mcq_rows = [
        json.loads(line)
        for line in (output_dir / "mcq_results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    no_hint_rows = [
        json.loads(line)
        for line in (output_dir / "no_hint_results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert set(mcq_rows[0]) == {
        "qid",
        "level",
        "question",
        "options",
        "predicted_label",
        "correct_label",
        "score",
        "error",
    }
    assert set(no_hint_rows[0]) == {
        "qid",
        "level",
        "question",
        "predicted_answer",
        "correct_answer",
        "judge_score",
        "judge_explanation",
        "error",
    }

    summary = json.loads((output_dir / "no_rag_summary.json").read_text(encoding="utf-8"))
    assert summary["mcq"]["processed"] == 2
    assert summary["mcq"]["judged"] == 2
    assert summary["mcq"]["score_sum"] == 1
    assert summary["mcq"]["max_score_sum"] == 2
    assert summary["mcq"]["coverage"] == 1
    assert summary["mcq"]["strict_accuracy"] == 0.5
    assert summary["mcq"]["by_level"]["L2"]["score_sum"] == 0
    assert summary["no_hint"]["processed"] == 2
    assert summary["no_hint"]["judged"] == 2
    assert summary["no_hint"]["score_sum"] == 4
    assert summary["no_hint"]["max_score_sum"] == 4
    assert summary["no_hint"]["mean_score"] == 2


def test_no_hint_generation_failure_skips_judge() -> None:
    records = [
        {
            "qid": "eval-0001",
            "level": "L1",
            "question": "Domanda?",
            "correct_answer": "Corretta",
        }
    ]
    client = FailingNoHintClient()

    rows = run_no_hint(records=records, client=client, config=NoRagConfig(chat_model="fake", judge_model="judge"))

    assert client.judge_calls == 0
    assert rows[0]["predicted_answer"] is None
    assert rows[0]["judge_score"] is None
    assert rows[0]["error"] and "no_hint_structured_error" in rows[0]["error"]


def test_invalid_judge_result_is_recorded_as_error() -> None:
    records = [
        {
            "qid": "eval-0001",
            "level": "L1",
            "question": "Domanda?",
            "correct_answer": "Corretta",
        }
    ]

    rows = run_no_hint(records=records, client=InvalidJudgeClient(), config=NoRagConfig(chat_model="fake"))

    assert rows[0]["predicted_answer"] == "Risposta fake"
    assert rows[0]["judge_score"] is None
    assert rows[0]["error"] and "judge_error" in rows[0]["error"]


def test_manifest_records_env_api_key_presence_without_leaking_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    evaluation_dir = _make_clean_inputs(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "UTOPIA_API_KEY=env-secret\nUTOPIA_BASE_URL=https://utopia.example/api\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("UTOPIA_API_KEY", raising=False)
    monkeypatch.delenv("UTOPIA_BASE_URL", raising=False)
    monkeypatch.setattr(
        no_rag_runner,
        "UtopiaStructuredChatClient",
        lambda *, api_url, api_key, retry_attempts: FakeStructuredClient(),
    )

    manifest = run_no_rag_baseline(
        NoRagConfig(
            evaluation_dir=str(evaluation_dir),
            output_dir=str(tmp_path / "env_run"),
            env_file=str(env_file),
            chat_model="fake-answer",
            judge_model="fake-judge",
        )
    )

    assert manifest["config"]["api_key_present"] is True
    assert manifest["connection"]["api_key_present"] is True
    assert manifest["connection"]["client"] == "utopia"
    assert "api_key" not in manifest["connection"]
    assert "env-secret" not in json.dumps(manifest, ensure_ascii=False)


def test_selection_supports_start_benchmark_size_and_smoke(tmp_path: Path) -> None:
    evaluation_dir = _make_clean_inputs(tmp_path)

    manifest = run_no_rag_baseline(
        NoRagConfig(
            evaluation_dir=str(evaluation_dir),
            output_dir=str(tmp_path / "selected"),
            start=1,
            benchmark_size=1,
            chat_model="fake",
        ),
        client=FakeStructuredClient(),
    )
    assert manifest["counts"]["mcq"] == 1
    rows = [
        json.loads(line)
        for line in (tmp_path / "selected" / "mcq_results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["qid"] == "eval-0002"

    smoke_manifest = run_no_rag_baseline(
        NoRagConfig(
            evaluation_dir=str(evaluation_dir),
            output_dir=str(tmp_path / "smoke"),
            benchmark_size=2,
            smoke=True,
            chat_model="fake",
        ),
        client=FakeStructuredClient(),
    )
    assert smoke_manifest["counts"]["mcq"] == 1


def test_mcq_calls_are_parallelized_but_output_order_is_stable() -> None:
    records = [
        {
            "qid": f"eval-{idx:04d}",
            "level": "L1",
            "question_stem": f"Domanda {idx}?",
            "options": {"A": "Corretta", "B": "Errata", "C": "Errata", "D": "Errata", "E": "Errata", "F": "Errata"},
            "correct_label": "A",
        }
        for idx in range(1, 5)
    ]
    client = ObservedSlowClient()

    rows = run_mcq(records=records, client=client, config=NoRagConfig(chat_model="fake", max_concurrency=4))

    assert client.max_active > 1
    assert [row["qid"] for row in rows] == [record["qid"] for record in records]
    assert all(row["score"] == 1 for row in rows)


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

    runtime = resolve_utopia_runtime(NoRagConfig(env_file=str(env_file)))

    assert runtime["api_url"] == "https://utopia.example/ollama/api/chat"
    assert runtime["api_key"] == "test-secret"
    assert runtime["api_key_present"] is True
    assert runtime["env_file_loaded"] is True
    assert "UTOPIA_API_KEY" not in runtime["env_keys_loaded"]


def test_model_resolution_uses_env_after_env_file_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UTOPIA_CHAT_MODEL", raising=False)
    monkeypatch.delenv("UTOPIA_JUDGE_MODEL", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "UTOPIA_API_KEY=test-secret\nUTOPIA_CHAT_MODEL=server-answer\nUTOPIA_JUDGE_MODEL=server-judge\n",
        encoding="utf-8",
    )

    config = NoRagConfig(env_file=str(env_file))
    resolve_utopia_runtime(config)
    answer_model = resolve_answer_model(config)

    assert answer_model == "server-answer"
    assert resolve_judge_model(config, answer_model) == "server-judge"
