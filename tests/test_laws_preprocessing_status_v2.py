from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from legal_rag.laws_preprocessing import StatusEventRecord, build_corpus_registry, ingest_law


def _write_law(root: Path, name: str, html: str) -> None:
    (root / name).write_text(html, encoding="utf-8")


def _modifier_law(root: Path) -> None:
    _write_law(
        root,
        "0002_LR-2-gennaio-2001-n2.html",
        "<article><h1>Legge regionale 2 gennaio 2001, n. 2</h1></article>",
    )


def _second_modifier_law(root: Path) -> None:
    _write_law(
        root,
        "0003_LR-3-gennaio-2002-n3.html",
        "<article><h1>Legge regionale 3 gennaio 2002, n. 3</h1></article>",
    )


def test_comma_cessation_does_not_make_whole_article_past(tmp_path: Path) -> None:
    _write_law(
        tmp_path,
        "0001_LR-1-gennaio-2000-n1.html",
        """<article>
<h1>Legge regionale 1 gennaio 2000, n. 1 - Testo vigente</h1>
<p><a name="articolo_1__">Art. 1</a></p>
<p>1. Comma ancora applicabile.</p>
<p>2. Comma cessato. <a href="#nota_1">(1)</a></p>
<p><a name="nota_1">(<span>1</span>)</a> Comma abrogato dalla lettera b) del comma 1 dell'art. 3 della
<a href="/detail?numero_legge=2%2F01">L.R. 2/2001</a>.</p>
</article>""",
    )
    _modifier_law(tmp_path)
    registry, _ = build_corpus_registry(tmp_path)

    ingested = ingest_law(
        registry.by_law_id["vda:lr:2000-01-01:1"],
        registry,
        chunk_size=100,
        chunk_overlap=10,
    )

    article = ingested.articles[0]
    passages = {passage["passage_label"]: passage for passage in ingested.passages}
    event = ingested.status_events[0]
    assert article["article_status"] == "partial"
    assert passages["c1"]["passage_status"] == "current"
    assert passages["c2"]["passage_status"] == "past"
    assert event["target_kind"] == "comma"
    assert event["resolution_status"] == "resolved"
    assert event["target_ids"] == [passages["c2"]["passage_id"]]
    assert event["modifying_law_ids"] == ["vda:lr:2001-01-02:2"]
    chunks = {chunk["passage_label"]: chunk for chunk in ingested.chunks}
    assert "current" in chunks["c1"]["index_views"]
    assert chunks["c2"]["index_views"] == ["historical"]


def test_anchorless_note_marker_is_split_from_previous_note(tmp_path: Path) -> None:
    _write_law(
        tmp_path,
        "0001_LR-1-gennaio-2000-n1.html",
        """<article>
<h1>Legge regionale 1 gennaio 2000, n. 1 - Testo vigente</h1>
<p><a name="articolo_1__">Art. 1</a></p>
<p>[1. Testo modificato.] <a href="#nota_1">(1)</a></p>
<p><a name="articolo_2__">Art. 2</a></p>
<p>1. Testo cessato. (1a)</p>
<p><a name="nota_1">(<span>1</span>)</a> Comma modificato dalla
<a href="/detail?numero_legge=2%2F01">L.R. 2/2001</a>.</p>
<p>(1a) Comma abrogato dalla <a href="/detail?numero_legge=2%2F01">L.R. 2/2001</a>.</p>
</article>""",
    )
    _modifier_law(tmp_path)
    registry, _ = build_corpus_registry(tmp_path)

    ingested = ingest_law(
        registry.by_law_id["vda:lr:2000-01-01:1"],
        registry,
        chunk_size=100,
        chunk_overlap=10,
    )

    notes = {note["note_anchor_name"]: note for note in ingested.notes}
    articles = {article["article_label_norm"]: article for article in ingested.articles}
    assert set(notes) == {"nota_1", "nota_1a"}
    assert notes["nota_1"]["note_kind"] == "modified"
    assert notes["nota_1a"]["note_kind"] == "abrogated"
    assert articles["1"]["article_status"] == "unknown"
    assert articles["2"]["article_status"] == "past"
    assert ingested.status_diagnostics["anchorless_note_markers"] == 1


def test_later_whole_law_cessation_overrides_exception(tmp_path: Path) -> None:
    _write_law(
        tmp_path,
        "0001_LR-1-gennaio-2000-n1.html",
        """<article>
<h1>Legge regionale 1 gennaio 2000, n. 1</h1>
<p>(Legge abrogata dalla L.R. 2/2001, ad eccezione dell'articolo 2).</p>
<p>(Poi interamente abrogata dalla L.R. 3/2002).</p>
<p><a name="articolo_1__">Art. 1</a></p><p>1. Primo testo.</p>
<p><a name="articolo_2__">Art. 2</a></p><p>1. Secondo testo.</p>
</article>""",
    )
    _modifier_law(tmp_path)
    _second_modifier_law(tmp_path)
    registry, _ = build_corpus_registry(tmp_path)

    ingested = ingest_law(
        registry.by_law_id["vda:lr:2000-01-01:1"],
        registry,
        chunk_size=100,
        chunk_overlap=10,
    )

    assert ingested.law["law_status"] == "past"
    assert {article["article_status"] for article in ingested.articles} == {"past"}
    assert [event["scope_mode"] for event in ingested.status_events] == ["all_except", "all"]


@pytest.mark.parametrize(
    "clause",
    [
        (
            "Abrogata dalla L.R. 2/2001, ad eccezione delle norme concernenti "
            "le professioni, che sono state poi abrogate dalla L.R. 3/2002."
        ),
        (
            "Abrogata, ad eccezione dell'art. 57, dalla L.R. 2/2001 "
            "ed abrogata dalla L.R. 3/2002."
        ),
        (
            "Abrogata dalla L.R. 2/2001, ad eccezione dell'art. 31, comma 2, "
            "e poi interamente abrogata dalla L.R. 3/2002."
        ),
    ],
)
def test_later_cessation_of_exceptions_makes_law_fully_past(
    tmp_path: Path,
    clause: str,
) -> None:
    _write_law(
        tmp_path,
        "0001_LR-1-gennaio-2000-n1.html",
        f"""<article>
<h1>Legge regionale 1 gennaio 2000, n. 1</h1>
<p>({clause})</p>
</article>""",
    )
    _modifier_law(tmp_path)
    _second_modifier_law(tmp_path)
    registry, _ = build_corpus_registry(tmp_path)

    ingested = ingest_law(
        registry.by_law_id["vda:lr:2000-01-01:1"],
        registry,
        chunk_size=100,
        chunk_overlap=10,
    )

    assert ingested.law["law_status"] == "past"
    assert ingested.articles[0]["article_status"] == "past"
    assert len(ingested.status_events) == 2
    later = ingested.status_events[1]
    assert later["scope_mode"] == "all"
    assert later["resolution_status"] == "resolved"
    assert later["rule_id"] == "status-law-exceptions-later-cessation-v1"
    assert later["modifying_law_ids"] == ["vda:lr:2002-01-03:3"]


def test_status_event_contract_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        StatusEventRecord.model_validate(
            {
                "status_event_id": "event-1",
                "law_id": "law-1",
                "event_type": "repeal",
                "source_kind": "note",
                "source_anchor": "nota_1",
                "source_clause": "Comma abrogato.",
                "evidence_text": "Comma abrogato.",
                "event_sequence": 0,
                "modifying_law_ids": [],
                "target_kind": "comma",
                "target_ids": ["passage-1"],
                "scope_mode": "only",
                "exception_labels": [],
                "exception_target_ids": [],
                "resolution_status": "resolved",
                "rule_id": "status-passage-whole-cessation-v1",
                "unexpected": True,
            }
        )
