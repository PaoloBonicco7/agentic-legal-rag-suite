from laws_ingestion.data_preparation.laws_graph.status import classify_law_status


def test_status_past_with_abrogation_phrase() -> None:
    res = classify_law_status(
        preamble_text="(Legge abrogata dall'art. 10 della L.R. 1 gennaio 2000, n. 1)",
        article_count=4,
        source_file="x.html",
    )
    assert res.status == "past"
    assert res.status_confidence >= 0.95
    assert res.status_evidence


def test_status_unknown_for_partial_abrogation_exception() -> None:
    res = classify_law_status(
        preamble_text="(Abrogata dall'art. 30 della L:R. 1 aprile 2004, n. 3, ad eccezione dell'articolo 40)",
        article_count=1,
        source_file="x.html",
    )
    assert res.status == "unknown"
    assert res.status_confidence >= 0.6


def test_status_index_or_empty() -> None:
    res = classify_law_status(
        preamble_text="INDICE",
        article_count=0,
        source_file="x.html",
    )
    assert res.status == "index_or_empty"


def test_status_current_for_structured_law() -> None:
    res = classify_law_status(
        preamble_text="Testo vigente.",
        article_count=2,
        source_file="x.html",
    )
    assert res.status == "current"


def test_status_uses_ingest_abrogated_signal() -> None:
    res = classify_law_status(
        preamble_text="",
        article_count=1,
        source_file="x.html",
        ingest_status="abrogated",
    )
    assert res.status == "past"
