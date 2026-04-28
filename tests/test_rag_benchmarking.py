from __future__ import annotations

from legal_indexing.rag_runtime.benchmarking import (
    align_record,
    build_extended_summary,
    build_comparison_table,
    build_dataset_summary,
    categorize_row_error,
    extract_mcq_options,
    is_effective_answer,
    resolve_ollama_chat_url,
    summarize_error_heads,
)


def test_extract_mcq_options_parses_all_labels() -> None:
    question = (
        "Domanda test?\\n"
        "A) Opzione A\\n"
        "B) Opzione B\\n"
        "C) Opzione C\\n"
        "D) Opzione D\\n"
        "E) Opzione E\\n"
        "F) Opzione F\\n"
    )
    options = extract_mcq_options(question)
    assert options["A"] == "Opzione A"
    assert options["F"] == "Opzione F"
    assert len(options) == 6


def test_align_record_requires_stem_alignment() -> None:
    mcq_rows = [
        {
            "#": "1",
            "Domanda": (
                "Qual e il termine?\\n"
                "A) 10 giorni\\n"
                "B) 20 giorni\\n"
                "C) 30 giorni\\n"
                "D) 40 giorni\\n"
                "E) 50 giorni\\n"
                "F) 60 giorni\\n"
            ),
            "Livello": "L1",
            "Risposta corretta": "C",
        }
    ]
    no_hint_rows = [
        {
            "Domanda": "Qual e il termine?",
            "Livello": "L1",
            "Risposta corretta": "30 giorni",
        }
    ]
    record = align_record(0, no_hint_rows, mcq_rows)
    assert record["qid"] == "1"
    assert record["ground_truth_label_mcq"] == "C"


def test_summary_shapes_match_expected_contract() -> None:
    mcq_results = [
        {"level": "L1", "score": 1, "error": None},
        {"level": "L2", "score": 0, "error": None},
        {"level": "L2", "score": None, "error": "timeout"},
    ]
    no_hint_results = [
        {"level": "L1", "final_binary_score": 1, "error": None},
        {"level": "L2", "final_binary_score": 1, "error": None},
    ]

    mcq_summary = build_dataset_summary("MCQ", mcq_results, score_key="score")
    no_hint_summary = build_dataset_summary(
        "No-Hint + Judge",
        no_hint_results,
        score_key="final_binary_score",
    )
    comparison = build_comparison_table(mcq_summary, no_hint_summary)

    assert set(mcq_summary.keys()) == {
        "dataset",
        "processed",
        "judged",
        "score_sum",
        "accuracy",
        "errors",
        "by_level",
    }
    assert set(no_hint_summary.keys()) == set(mcq_summary.keys())
    assert "global_rows" in comparison
    assert "level_rows" in comparison
    assert comparison["level_rows"]


def test_resolve_ollama_chat_url_normalizes_base_api_suffix() -> None:
    url = resolve_ollama_chat_url("https://utopia.hpc4ai.unito.it/api")
    assert url == "https://utopia.hpc4ai.unito.it/ollama/api/chat"


def test_resolve_ollama_chat_url_from_plain_base() -> None:
    url = resolve_ollama_chat_url("https://utopia.hpc4ai.unito.it")
    assert url == "https://utopia.hpc4ai.unito.it/ollama/api/chat"


def test_resolve_ollama_chat_url_prefers_explicit_override() -> None:
    url = resolve_ollama_chat_url(
        "https://utopia.hpc4ai.unito.it/api",
        explicit_url="https://custom.endpoint.local/ollama/api/chat",
    )
    assert url == "https://custom.endpoint.local/ollama/api/chat"


def test_resolve_ollama_chat_url_rejects_invalid_urls() -> None:
    try:
        resolve_ollama_chat_url("utopia.hpc4ai.unito.it/api")
    except ValueError as exc:
        assert "http://" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid base_url")


def test_summarize_error_heads_aggregates_first_line() -> None:
    rows = [
        {"error": "HTTP 405 su /chat\\nBody: Method Not Allowed"},
        {"error": "HTTP 405 su /chat\\nBody: Method Not Allowed"},
        {"error": "TimeoutError: request timed out"},
        {"error": ""},
        {"error": None},
    ]
    summary = summarize_error_heads(rows)
    assert summary == [
        {"error_head": "HTTP 405 su /chat", "count": 2},
        {"error_head": "TimeoutError: request timed out", "count": 1},
    ]


def test_is_effective_answer_applies_min_quality_rules() -> None:
    assert not is_effective_answer("")
    assert not is_effective_answer("  ")
    assert not is_effective_answer("[VUOTA]")
    assert not is_effective_answer("breve", min_chars=8)
    assert is_effective_answer("Risposta normativa completa.", min_chars=8)


def test_categorize_row_error_empty_no_hint_is_technical_not_judged() -> None:
    row = {
        "predicted_answer": "",
        "final_binary_score": None,
        "error": None,
        "status": "answer_empty_or_invalid",
    }
    assert (
        categorize_row_error(row, score_key="final_binary_score")
        == "technical_generation_empty_answer"
    )


def test_build_extended_summary_adds_coverage_and_timing() -> None:
    results = [
        {
            "level": "L1",
            "final_binary_score": 1,
            "predicted_answer": "Risposta valida e completa.",
            "error": None,
            "t_retrieval_context_s": 0.5,
            "t_task_llm_s": 1.0,
            "t_judge_s": 0.6,
            "t_total_s": 2.1,
        },
        {
            "level": "L2",
            "final_binary_score": None,
            "predicted_answer": "",
            "error": None,
            "status": "answer_empty_or_invalid",
            "t_retrieval_context_s": 0.4,
            "t_task_llm_s": 0.8,
            "t_judge_s": None,
            "t_total_s": 1.2,
        },
    ]

    summary = build_extended_summary(
        "No-Hint + Judge",
        results,
        score_key="final_binary_score",
    )
    assert summary["processed"] == 2
    assert summary["judged"] == 1
    assert summary["accuracy"] == 1.0
    assert summary["coverage"] == 0.5
    assert summary["strict_accuracy"] == 0.5
    assert summary["empty_answer_count"] == 1
    assert summary["error_categories"]["technical_generation_empty_answer"] == 1
    assert summary["timing_summary"]["t_total_s"]["count"] == 2
