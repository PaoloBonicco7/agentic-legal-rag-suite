from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from legal_indexing.law_references import LawCatalog
from legal_indexing.rag_runtime.benchmark_runner import (
    persist_benchmark_artifacts,
    run_advanced_benchmark,
)


def _mcq_question(stem: str) -> str:
    return (
        f"{stem}\n"
        "A) Opzione A\n"
        "B) Opzione B\n"
        "C) Opzione C\n"
        "D) Opzione D\n"
        "E) Opzione E\n"
        "F) Opzione F"
    )


def test_benchmark_runner_is_robust_to_single_row_failures() -> None:
    mcq_rows = [
        {
            "#": "1",
            "Domanda": _mcq_question("Prima domanda"),
            "Livello": "L1",
            "Risposta corretta": "A",
            "Riferimento legge per la risposta": "Legge regionale 25 gennaio 2000, n. 5 - Art. 12",
        },
        {
            "#": "2",
            "Domanda": _mcq_question("Seconda domanda"),
            "Livello": "L2",
            "Risposta corretta": "B",
            "Riferimento legge per la risposta": "Legge regionale 8 ottobre 2019, n. 16 - Art. 4",
        },
    ]
    no_hint_rows = [
        {
            "Domanda": "Prima domanda",
            "Livello": "L1",
            "Risposta corretta": "Opzione A",
            "Riferimento legge per la risposta": "Legge regionale 25 gennaio 2000, n. 5 - Art. 12",
        },
        {
            "Domanda": "Seconda domanda",
            "Livello": "L2",
            "Risposta corretta": "Opzione B",
            "Riferimento legge per la risposta": "Legge regionale 8 ottobre 2019, n. 16 - Art. 4",
        },
    ]

    def fake_rag_runner(question: str) -> dict:
        second = "Seconda" in question
        return {
            "state": {
                "context": f"contesto per: {question}",
                "pipeline_errors": [],
                "retrieval_mode": "hybrid" if second else "dense_only",
                "dense_retrieved_count": 2,
                "sparse_retrieved_count": 1 if second else 0,
                "fusion_overlap_count": 1 if second else 0,
                    "retrieved": [
                        {
                            "chunk_id": "c1",
                            "payload": {
                                "law_id": (
                                    "vda:lr:2019-10-08:16"
                                    if second
                                    else "vda:lr:2000-01-25:5"
                            )
                        }
                    }
                ],
            },
            "context_summary": {
                "included_count": 1,
                "included_chunk_ids": ["c1"],
            },
            "rewritten_queries": ["q0", "q1"] if second else ["q0"],
            "filters_summary": {
                "metadata_heuristics": ["relation_modific"] if second else [],
                "metadata_hard_law_filter_applied": second,
                "metadata_hard_article_filter_applied": False,
            },
            "graph_expansion": {
                "enabled": second,
                "reason": "forced_relation_query" if second else "default",
                "graph_retrieved_count": 2 if second else 0,
            },
            "answer_summary": {
                "answer": "Opzione A" if "Prima" in question else "Opzione B",
                "answer_source": "fallback" if second else "model",
                "was_empty_before_guard": second,
                "needs_more_context": second,
            },
            "retrieved_preview": [{"chunk_id": "c1"}],
        }

    def fake_post_chat(**kwargs):
        prompt = str(kwargs.get("prompt") or "")
        schema = kwargs.get("payload_schema") or {}
        properties = (schema.get("properties") or {}) if isinstance(schema, dict) else {}

        if "answer_label" in properties:
            label = "A" if "Prima domanda" in prompt else "B"
            return {"structured": {"answer_label": label, "short_rationale": "ok"}}

        if "score" in properties:
            if "Seconda domanda" in prompt:
                raise RuntimeError("judge timeout")
            return {
                "structured": {
                    "score": 1,
                    "confidence": 0.9,
                    "matched_option_label": "A",
                    "is_semantically_equivalent": True,
                    "justification": "ok",
                }
            }

        raise RuntimeError("unexpected schema")

    catalog = LawCatalog(
        law_ids=("vda:lr:2000-01-25:5", "vda:lr:2019-10-08:16"),
        by_year_number={
            (2000, 5): ("vda:lr:2000-01-25:5",),
            (2019, 16): ("vda:lr:2019-10-08:16",),
        },
        by_date_number={
            ("2000-01-25", 5): "vda:lr:2000-01-25:5",
            ("2019-10-08", 16): "vda:lr:2019-10-08:16",
        },
        article_ids_by_law={},
        article_labels_by_law={},
    )

    out = run_advanced_benchmark(
        runtime=SimpleNamespace(config=None, law_catalog=catalog),
        mcq_rows=mcq_rows,
        no_hint_rows=no_hint_rows,
        positions=[0, 1],
        api_url="http://fake.local/chat",
        headers={"Authorization": "Bearer test"},
        chat_model="fake-chat",
        judge_model="fake-judge",
        timeout_sec=10,
        rag_runner=fake_rag_runner,
        post_chat_fn=fake_post_chat,
    )

    assert out["mcq_summary"]["processed"] == 2
    assert out["mcq_summary"]["judged"] == 2
    assert out["no_hint_summary"]["processed"] == 2
    assert out["no_hint_summary"]["judged"] == 1
    assert out["no_hint_summary"]["errors"] == 1
    assert "comparison_summary" in out
    assert "diagnostics" in out
    assert out["no_hint_results"][0]["rag_answer_source"] == "model"
    assert out["no_hint_results"][1]["rag_answer_source"] == "fallback"
    assert out["no_hint_results"][0]["rag_was_empty_before_guard"] is False
    assert out["no_hint_results"][1]["rag_was_empty_before_guard"] is True
    assert out["no_hint_results"][1]["rag_needs_more_context"] is True
    assert out["no_hint_results"][1]["retrieval_mode"] == "hybrid"
    assert out["no_hint_results"][1]["sparse_retrieved_count"] == 1
    assert out["no_hint_results"][0]["reference_law_hit"] is True
    assert out["no_hint_results"][0]["top1_law_match"] is True
    assert out["no_hint_results"][0]["context_precision_proxy"] == 1.0
    assert out["no_hint_results"][1]["rewrite_count"] == 2
    assert out["no_hint_results"][1]["metadata_heuristics"] == ["relation_modific"]
    assert out["no_hint_results"][1]["metadata_hard_law_filter_applied"] is True
    assert out["no_hint_results"][1]["graph_enabled"] is True
    assert out["no_hint_results"][1]["graph_reason"] == "forced_relation_query"
    assert out["no_hint_results"][1]["graph_retrieved_count"] == 2
    assert out["no_hint_results"][1]["unique_laws_retrieved"] >= 1
    assert out["no_hint_results"][1]["unique_laws_in_context"] >= 1
    assert out["no_hint_results"][1]["failure_category"] == "retrieval_miss"
    assert out["diagnostics"]["no_hint_empty_detected_count"] == 1
    assert out["diagnostics"]["no_hint_fallback_used_count"] == 1
    assert out["diagnostics"]["no_hint_empty_by_level"]["L2"]["fallback_used"] == 1
    assert out["diagnostics"]["no_hint_failure_breakdown"]["retrieval_miss"] >= 1
    assert out["diagnostics"]["no_hint_reference_hit_rate"] == 1.0
    assert "no_hint_graph_reason_counts" in out["diagnostics"]
    assert "no_hint_graph_enabled_count" in out["diagnostics"]
    assert "no_hint_graph_retrieved_positive_count" in out["diagnostics"]
    assert "no_hint_avg_rewrite_count" in out["diagnostics"]
    assert "no_hint_metadata_hard_law_filter_applied_count" in out["diagnostics"]


def test_persist_benchmark_artifacts_stores_top_level_audit_fields(tmp_path: Path) -> None:
    payload = {
        "mcq_summary": {"accuracy": 1.0},
        "no_hint_summary": {"accuracy": 0.5},
        "comparison_summary": {"global": {}},
        "diagnostics": {},
        "mcq_results": [],
        "no_hint_results": [],
    }
    paths = persist_benchmark_artifacts(
        artifacts_dir=tmp_path,
        label="advanced",
        mode="mini",
        config_payload={"pipeline_mode": "advanced", "qdrant_url": "http://127.0.0.1:6333"},
        index_contract={
            "run_id": "20260302_211459",
            "collection_points_count": 50978,
            "eval_reference_coverage": 1.0,
        },
        benchmark_payload=payload,
    )
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert data["run_id"] == "20260302_211459"
    assert data["collection_points_count"] == 50978
    assert data["eval_reference_coverage"] == 1.0
    assert data["qdrant_url_used"] == "http://127.0.0.1:6333"


def test_benchmark_runner_uses_retrieval_runner_for_mcq_when_provided() -> None:
    mcq_rows = [
        {
            "#": "1",
            "Domanda": _mcq_question("Domanda una"),
            "Livello": "L1",
            "Risposta corretta": "A",
            "Riferimento legge per la risposta": "",
        }
    ]
    no_hint_rows = [
        {
            "Domanda": "Domanda una",
            "Livello": "L1",
            "Risposta corretta": "Opzione A",
            "Riferimento legge per la risposta": "",
        }
    ]

    calls = {"rag": 0, "retrieval": 0}

    def full_rag_runner(question: str) -> dict:
        if "A)" in question:
            raise AssertionError("MCQ path must not use full rag_runner")
        calls["rag"] += 1
        return {
            "state": {
                "context": "contesto no-hint",
                "pipeline_errors": [],
                "retrieval_mode": "hybrid",
                "retrieved": [{"payload": {"law_id": "vda:lr:2000-01-25:5"}}],
            },
            "context_summary": {"included_count": 1, "included_chunk_ids": ["c1"]},
            "answer_summary": {
                "answer": "Opzione A",
                "answer_source": "model",
                "was_empty_before_guard": False,
                "needs_more_context": False,
            },
            "filters_summary": {},
            "graph_expansion": {},
            "rewritten_queries": [],
        }

    def retrieval_runner(question: str) -> dict:
        calls["retrieval"] += 1
        return {
            "state": {
                "context": "contesto mcq",
                "pipeline_errors": [],
                "retrieval_mode": "hybrid",
                "retrieved": [{"payload": {"law_id": "vda:lr:2000-01-25:5"}}],
            },
            "context_summary": {"included_count": 1, "included_chunk_ids": ["c1"]},
            "filters_summary": {},
            "graph_expansion": {},
            "rewritten_queries": [],
            "retrieved_preview": [{"chunk_id": "c1"}],
        }

    def fake_post_chat(**kwargs):
        schema = kwargs.get("payload_schema") or {}
        properties = (schema.get("properties") or {}) if isinstance(schema, dict) else {}
        if "answer_label" in properties:
            return {"structured": {"answer_label": "A", "short_rationale": "ok"}}
        return {
            "structured": {
                "score": 1,
                "confidence": 0.9,
                "matched_option_label": "A",
                "is_semantically_equivalent": True,
                "justification": "ok",
            }
        }

    out = run_advanced_benchmark(
        runtime=SimpleNamespace(config=None, law_catalog=None),
        mcq_rows=mcq_rows,
        no_hint_rows=no_hint_rows,
        positions=[0],
        api_url="http://fake.local/chat",
        headers={"Authorization": "Bearer test"},
        chat_model="fake-chat",
        judge_model="fake-judge",
        timeout_sec=10,
        rag_runner=full_rag_runner,
        rag_retrieval_runner=retrieval_runner,
        post_chat_fn=fake_post_chat,
        max_workers=1,
    )

    assert out["mcq_summary"]["processed"] == 1
    assert out["no_hint_summary"]["processed"] == 1
    assert calls["retrieval"] == 1
    assert calls["rag"] == 1


def test_benchmark_runner_checkpoint_resume_skips_completed_rows(tmp_path: Path) -> None:
    mcq_rows = [
        {
            "#": "1",
            "Domanda": _mcq_question("Prima domanda"),
            "Livello": "L1",
            "Risposta corretta": "A",
            "Riferimento legge per la risposta": "",
        },
        {
            "#": "2",
            "Domanda": _mcq_question("Seconda domanda"),
            "Livello": "L1",
            "Risposta corretta": "A",
            "Riferimento legge per la risposta": "",
        },
    ]
    no_hint_rows = [
        {"Domanda": "Prima domanda", "Livello": "L1", "Risposta corretta": "Opzione A"},
        {"Domanda": "Seconda domanda", "Livello": "L1", "Risposta corretta": "Opzione A"},
    ]
    checkpoint = tmp_path / "benchmark_checkpoint.jsonl"

    def rag_runner(question: str) -> dict:
        return {
            "state": {
                "context": f"ctx:{question}",
                "pipeline_errors": [],
                "retrieval_mode": "hybrid",
                "retrieved": [{"payload": {"law_id": "vda:lr:2000-01-25:5"}}],
            },
            "context_summary": {"included_count": 1, "included_chunk_ids": ["c1"]},
            "answer_summary": {
                "answer": "Opzione A",
                "answer_source": "model",
                "was_empty_before_guard": False,
                "needs_more_context": False,
            },
            "filters_summary": {},
            "graph_expansion": {},
            "rewritten_queries": [],
            "retrieved_preview": [{"chunk_id": "c1"}],
        }

    def fake_post_chat(**kwargs):
        schema = kwargs.get("payload_schema") or {}
        properties = (schema.get("properties") or {}) if isinstance(schema, dict) else {}
        if "answer_label" in properties:
            return {"structured": {"answer_label": "A", "short_rationale": "ok"}}
        return {
            "structured": {
                "score": 1,
                "confidence": 0.9,
                "matched_option_label": "A",
                "is_semantically_equivalent": True,
                "justification": "ok",
            }
        }

    first = run_advanced_benchmark(
        runtime=SimpleNamespace(config=None, law_catalog=None),
        mcq_rows=mcq_rows,
        no_hint_rows=no_hint_rows,
        positions=[0, 1],
        api_url="http://fake.local/chat",
        headers={"Authorization": "Bearer test"},
        chat_model="fake-chat",
        judge_model="fake-judge",
        timeout_sec=10,
        rag_runner=rag_runner,
        rag_retrieval_runner=rag_runner,
        post_chat_fn=fake_post_chat,
        max_workers=1,
        checkpoint_path=checkpoint,
        checkpoint_every=1,
        resume=False,
    )
    assert first["mcq_summary"]["processed"] == 2
    assert first["no_hint_summary"]["processed"] == 2

    def should_not_run(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("resume should skip all completed rows")

    resumed = run_advanced_benchmark(
        runtime=SimpleNamespace(config=None, law_catalog=None),
        mcq_rows=mcq_rows,
        no_hint_rows=no_hint_rows,
        positions=[0, 1],
        api_url="http://fake.local/chat",
        headers={"Authorization": "Bearer test"},
        chat_model="fake-chat",
        judge_model="fake-judge",
        timeout_sec=10,
        rag_runner=should_not_run,
        rag_retrieval_runner=should_not_run,
        post_chat_fn=should_not_run,
        max_workers=1,
        checkpoint_path=checkpoint,
        checkpoint_every=1,
        resume=True,
    )
    assert resumed["mcq_summary"]["processed"] == 2
    assert resumed["no_hint_summary"]["processed"] == 2
    assert resumed["diagnostics"]["mcq_checkpoint_loaded_rows"] == 2
    assert resumed["diagnostics"]["no_hint_checkpoint_loaded_rows"] == 2


def test_benchmark_runner_parallel_results_are_sorted_by_pos() -> None:
    mcq_rows = [
        {"#": "1", "Domanda": _mcq_question("Q0"), "Livello": "L1", "Risposta corretta": "A"},
        {"#": "2", "Domanda": _mcq_question("Q1"), "Livello": "L1", "Risposta corretta": "A"},
        {"#": "3", "Domanda": _mcq_question("Q2"), "Livello": "L1", "Risposta corretta": "A"},
    ]
    no_hint_rows = [
        {"Domanda": "Q0", "Livello": "L1", "Risposta corretta": "Opzione A"},
        {"Domanda": "Q1", "Livello": "L1", "Risposta corretta": "Opzione A"},
        {"Domanda": "Q2", "Livello": "L1", "Risposta corretta": "Opzione A"},
    ]

    def rag_runner(question: str) -> dict:
        if "Q0" in question:
            time.sleep(0.03)
        elif "Q1" in question:
            time.sleep(0.01)
        return {
            "state": {
                "context": "ctx",
                "pipeline_errors": [],
                "retrieval_mode": "hybrid",
                "retrieved": [{"payload": {"law_id": "vda:lr:2000-01-25:5"}}],
            },
            "context_summary": {"included_count": 1, "included_chunk_ids": ["c1"]},
            "answer_summary": {
                "answer": "Opzione A",
                "answer_source": "model",
                "was_empty_before_guard": False,
                "needs_more_context": False,
            },
            "filters_summary": {},
            "graph_expansion": {},
            "rewritten_queries": [],
            "retrieved_preview": [{"chunk_id": "c1"}],
        }

    def fake_post_chat(**kwargs):
        schema = kwargs.get("payload_schema") or {}
        properties = (schema.get("properties") or {}) if isinstance(schema, dict) else {}
        if "answer_label" in properties:
            return {"structured": {"answer_label": "A", "short_rationale": "ok"}}
        return {
            "structured": {
                "score": 1,
                "confidence": 0.9,
                "matched_option_label": "A",
                "is_semantically_equivalent": True,
                "justification": "ok",
            }
        }

    out = run_advanced_benchmark(
        runtime=SimpleNamespace(config=None, law_catalog=None),
        mcq_rows=mcq_rows,
        no_hint_rows=no_hint_rows,
        positions=[2, 1, 0],
        api_url="http://fake.local/chat",
        headers={"Authorization": "Bearer test"},
        chat_model="fake-chat",
        judge_model="fake-judge",
        timeout_sec=10,
        rag_runner=rag_runner,
        rag_retrieval_runner=rag_runner,
        post_chat_fn=fake_post_chat,
        max_workers=3,
    )

    assert [row["pos"] for row in out["mcq_results"]] == [0, 1, 2]
    assert [row["pos"] for row in out["no_hint_results"]] == [0, 1, 2]


def test_benchmark_resume_rejects_config_fingerprint_mismatch(tmp_path: Path) -> None:
    mcq_rows = [{"#": "1", "Domanda": _mcq_question("Q0"), "Livello": "L1", "Risposta corretta": "A"}]
    no_hint_rows = [{"Domanda": "Q0", "Livello": "L1", "Risposta corretta": "Opzione A"}]
    checkpoint = tmp_path / "benchmark_checkpoint_meta.jsonl"

    def rag_runner(question: str) -> dict:
        return {
            "state": {
                "context": f"ctx:{question}",
                "pipeline_errors": [],
                "retrieval_mode": "hybrid",
                "retrieved": [{"payload": {"law_id": "vda:lr:2000-01-25:5"}}],
            },
            "context_summary": {"included_count": 1, "included_chunk_ids": ["c1"]},
            "answer_summary": {
                "answer": "Opzione A",
                "answer_source": "model",
                "was_empty_before_guard": False,
                "needs_more_context": False,
            },
            "filters_summary": {},
            "graph_expansion": {},
            "rewritten_queries": [],
            "retrieved_preview": [{"chunk_id": "c1"}],
        }

    def fake_post_chat(**kwargs):
        schema = kwargs.get("payload_schema") or {}
        properties = (schema.get("properties") or {}) if isinstance(schema, dict) else {}
        if "answer_label" in properties:
            return {"structured": {"answer_label": "A", "short_rationale": "ok"}}
        return {
            "structured": {
                "score": 1,
                "confidence": 0.9,
                "matched_option_label": "A",
                "is_semantically_equivalent": True,
                "justification": "ok",
            }
        }

    run_advanced_benchmark(
        runtime=SimpleNamespace(config=None, law_catalog=None),
        mcq_rows=mcq_rows,
        no_hint_rows=no_hint_rows,
        positions=[0],
        api_url="http://fake.local/chat",
        headers={"Authorization": "Bearer test"},
        chat_model="chat-model-a",
        judge_model="judge-model-a",
        timeout_sec=10,
        rag_runner=rag_runner,
        rag_retrieval_runner=rag_runner,
        post_chat_fn=fake_post_chat,
        max_workers=1,
        checkpoint_path=checkpoint,
        checkpoint_every=1,
        resume=False,
    )

    with pytest.raises(RuntimeError, match="config fingerprint mismatch"):
        run_advanced_benchmark(
            runtime=SimpleNamespace(config=None, law_catalog=None),
            mcq_rows=mcq_rows,
            no_hint_rows=no_hint_rows,
            positions=[0],
            api_url="http://fake.local/chat",
            headers={"Authorization": "Bearer test"},
            chat_model="chat-model-b",
            judge_model="judge-model-a",
            timeout_sec=10,
            rag_runner=rag_runner,
            rag_retrieval_runner=rag_runner,
            post_chat_fn=fake_post_chat,
            max_workers=1,
            checkpoint_path=checkpoint,
            checkpoint_every=1,
            resume=True,
        )


def test_benchmark_resume_rejects_legacy_checkpoint_without_meta(tmp_path: Path) -> None:
    mcq_rows = [{"#": "1", "Domanda": _mcq_question("Q0"), "Livello": "L1", "Risposta corretta": "A"}]
    no_hint_rows = [{"Domanda": "Q0", "Livello": "L1", "Risposta corretta": "Opzione A"}]
    checkpoint = tmp_path / "benchmark_checkpoint_legacy.jsonl"
    legacy_row = {
        "section": "mcq",
        "pos": 0,
        "row": {
            "qid": "1",
            "level": "L1",
            "pos": 0,
            "ground_truth_label": "A",
            "predicted_label": "A",
            "score": 1,
            "is_correct": True,
            "error": None,
            "raw_response": {"answer_label": "A", "short_rationale": "ok"},
            "rag_pipeline_errors": [],
        },
    }
    checkpoint.write_text(json.dumps(legacy_row, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Legacy checkpoint without meta/fingerprint"):
        run_advanced_benchmark(
            runtime=SimpleNamespace(config=None, law_catalog=None),
            mcq_rows=mcq_rows,
            no_hint_rows=no_hint_rows,
            positions=[0],
            api_url="http://fake.local/chat",
            headers={"Authorization": "Bearer test"},
            chat_model="chat-model-a",
            judge_model="judge-model-a",
            timeout_sec=10,
            rag_runner=lambda _q: {},
            rag_retrieval_runner=lambda _q: {},
            post_chat_fn=lambda **_kwargs: {},
            max_workers=1,
            checkpoint_path=checkpoint,
            checkpoint_every=1,
            resume=True,
        )


def test_benchmark_diagnostics_split_hard_soft_and_recommended_workers() -> None:
    mcq_rows = [
        {"#": "1", "Domanda": _mcq_question("Q0"), "Livello": "L1", "Risposta corretta": "A"},
        {"#": "2", "Domanda": _mcq_question("Q1"), "Livello": "L1", "Risposta corretta": "A"},
    ]
    no_hint_rows = [
        {"Domanda": "Q0", "Livello": "L1", "Risposta corretta": "Opzione A"},
        {"Domanda": "Q1", "Livello": "L1", "Risposta corretta": "Opzione A"},
    ]

    def rag_runner(question: str) -> dict:
        if "Q0" in question:
            pipeline_errors = [
                {"stage": "generate_answer_structured", "error": "empty_answer_detected:first_pass"}
            ]
        else:
            pipeline_errors = []
        return {
            "state": {
                "context": f"ctx:{question}",
                "pipeline_errors": pipeline_errors,
                "retrieval_mode": "hybrid",
                "retrieved": [{"payload": {"law_id": "vda:lr:2000-01-25:5"}}],
            },
            "context_summary": {"included_count": 1, "included_chunk_ids": ["c1"]},
            "answer_summary": {
                "answer": "Opzione A",
                "answer_source": "model",
                "was_empty_before_guard": bool(pipeline_errors),
                "needs_more_context": False,
            },
            "filters_summary": {},
            "graph_expansion": {},
            "rewritten_queries": [],
            "retrieved_preview": [{"chunk_id": "c1"}],
        }

    def fake_post_chat(**kwargs):
        prompt = str(kwargs.get("prompt") or "")
        schema = kwargs.get("payload_schema") or {}
        properties = (schema.get("properties") or {}) if isinstance(schema, dict) else {}
        if "answer_label" in properties:
            return {"structured": {"answer_label": "A", "short_rationale": "ok"}}
        if "Q1" in prompt:
            raise RuntimeError("judge timeout")
        return {
            "structured": {
                "score": 1,
                "confidence": 0.9,
                "matched_option_label": "A",
                "is_semantically_equivalent": True,
                "justification": "ok",
            }
        }

    out = run_advanced_benchmark(
        runtime=SimpleNamespace(config=None, law_catalog=None),
        mcq_rows=mcq_rows,
        no_hint_rows=no_hint_rows,
        positions=[0, 1],
        api_url="http://fake.local/chat",
        headers={"Authorization": "Bearer test"},
        chat_model="chat-model-a",
        judge_model="judge-model-a",
        timeout_sec=10,
        rag_runner=rag_runner,
        rag_retrieval_runner=rag_runner,
        post_chat_fn=fake_post_chat,
        max_workers=4,
    )

    diag = out["diagnostics"]
    assert diag["no_hint_hard_error_rows"] == 1
    assert diag["no_hint_soft_guard_event_rows"] == 1
    assert diag["no_hint_guard_empty_first_pass_count"] == 1
    assert diag["no_hint_guard_retry_needs_more_context_count"] == 0
    assert diag["no_hint_guard_missing_valid_citation_count"] == 0
    assert diag["transient_error_rows"] >= 1
    assert diag["benchmark_recommended_max_workers_next_run"] == 3
