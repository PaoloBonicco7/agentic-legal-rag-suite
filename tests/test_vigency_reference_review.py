from __future__ import annotations

import csv
from pathlib import Path


REVIEW_PATH = Path("data/evaluation/vigency_reference_review.csv")
EXPECTED_QIDS = {
    "eval-0002",
    "eval-0003",
    "eval-0013",
    "eval-0014",
    "eval-0015",
    "eval-0016",
    "eval-0073",
    "eval-0075",
    "eval-0076",
    "eval-0100",
}


def _load_rows() -> list[dict[str, str]]:
    with REVIEW_PATH.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_vigency_reference_review_has_the_audited_qids() -> None:
    rows = _load_rows()

    assert len(rows) == len(EXPECTED_QIDS)
    assert {row["qid"] for row in rows} == EXPECTED_QIDS
    assert all(row["schema_version"] == "vigency-reference-review-v1" for row in rows)
    assert all(row["expected_article_id"].startswith(row["expected_law_id"] + "#art:") for row in rows)
    assert all(row["rationale"].strip() for row in rows)


def test_vigency_reference_review_keeps_qrel_findings_separate() -> None:
    rows = {row["qid"]: row for row in _load_rows()}

    assert {
        qid for qid, row in rows.items() if row["answer_support_relation"] == "supported_elsewhere"
    } == {"eval-0013", "eval-0076"}
    assert rows["eval-0013"]["answer_supporting_article_id"].endswith("#art:3")
    assert rows["eval-0076"]["answer_supporting_passage_id"].endswith("#art:3#p:c5")
    assert rows["eval-0100"]["temporal_scope_flag"] == "time_limited"
