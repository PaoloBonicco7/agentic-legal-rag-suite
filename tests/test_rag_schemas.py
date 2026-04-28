from __future__ import annotations

import pytest

from legal_indexing.rag_runtime.schemas import JudgeResult, McqAnswer, NoHintAnswer, RagAnswer


def test_rag_answer_normalizes_citations_and_text() -> None:
    model = RagAnswer(
        answer="  Test answer  ",
        citations=[" c1 ", "c1", "", "c2"],
        needs_more_context=False,
    )
    assert model.answer == "Test answer"
    assert model.citations == ["c1", "c2"]


def test_rag_answer_accepts_dict_citations() -> None:
    model = RagAnswer(
        answer="ok",
        citations=[
            {"chunk_id": " law:1#rc:0 "},
            {"source": "law:2#rc:3"},
            {"id": "law:3#rc:9"},
            {"unknown": "skip"},
        ],
        needs_more_context=False,
    )
    assert model.citations == ["law:1#rc:0", "law:2#rc:3", "law:3#rc:9"]


def test_rag_answer_coerces_none_answer() -> None:
    model = RagAnswer(
        answer=None,
        citations=[{"chunk_id": "law:1#rc:0"}],
        needs_more_context=False,
    )
    assert model.answer == ""
    assert model.citations == ["law:1#rc:0"]


def test_mcq_answer_validates_label() -> None:
    model = McqAnswer(answer_label="a", short_rationale="ok")
    assert model.answer_label == "A"
    with pytest.raises(ValueError):
        McqAnswer(answer_label="Z")


def test_no_hint_answer_normalizes_text() -> None:
    model = NoHintAnswer(answer_text="  risposta test  ")
    assert model.answer_text == "risposta test"


def test_judge_result_validates_range_and_labels() -> None:
    model = JudgeResult(
        score=1,
        confidence=0.7,
        matched_option_label="none",
        is_semantically_equivalent=True,
        justification="  Match corretto ",
    )
    assert model.matched_option_label == "NONE"
    assert model.justification == "Match corretto"
    with pytest.raises(ValueError):
        JudgeResult(
            score=1,
            confidence=1.5,
            matched_option_label="A",
            is_semantically_equivalent=True,
            justification="x",
        )
