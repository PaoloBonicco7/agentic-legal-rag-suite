from __future__ import annotations

from types import SimpleNamespace

from legal_indexing.law_references import LawCatalog
from legal_indexing.rag_runtime.benchmark_runner import run_advanced_benchmark


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
    assert out["no_hint_results"][1]["failure_category"] == "retrieval_miss"
    assert out["diagnostics"]["no_hint_empty_detected_count"] == 1
    assert out["diagnostics"]["no_hint_fallback_used_count"] == 1
    assert out["diagnostics"]["no_hint_empty_by_level"]["L2"]["fallback_used"] == 1
    assert out["diagnostics"]["no_hint_failure_breakdown"]["retrieval_miss"] >= 1
    assert out["diagnostics"]["no_hint_reference_hit_rate"] == 1.0
