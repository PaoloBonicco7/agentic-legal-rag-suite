from laws_ingestion.data_preparation.laws_graph.relations import normalize_edges


def test_relations_normalizer_removes_self_loops_in_clean() -> None:
    edges = [
        {
            "edge_id": "1",
            "edge_type": "REFERS_TO",
            "src_law_id": "vda:lr:2000-01-01:1",
            "src_article_id": None,
            "src_passage_id": None,
            "dst_law_id": "vda:lr:2000-01-01:1",
            "dst_article_label_norm": None,
            "evidence_text": "Legge regionale 1 gennaio 2000, n. 1",
            "confidence": 0.4,
        },
        {
            "edge_id": "2",
            "edge_type": "REFERS_TO",
            "src_law_id": "vda:lr:2000-01-01:1",
            "src_article_id": "vda:lr:2000-01-01:1#art:1",
            "src_passage_id": "vda:lr:2000-01-01:1#art:1#p:intro",
            "dst_law_id": "vda:lr:1999-01-01:2",
            "dst_article_label_norm": "3",
            "evidence_text": "Sostituisce l'art. 3 della L.R. 1 gennaio 1999, n. 2",
            "confidence": 0.9,
        },
    ]

    raw, clean, stats = normalize_edges(edges)
    assert stats["self_loops_raw"] == 1
    assert len(raw) == 2
    assert len(clean) == 1
    assert clean[0]["relation_scope"] == "passage"
    assert clean[0]["src_article_label_norm"] == "1"
    assert clean[0]["relation_type"] in {"REPLACES", "AMENDS", "MODIFIED_BY"}


def test_relations_normalizer_detects_abrogated_by_phrase() -> None:
    edges = [
        {
            "edge_id": "1",
            "edge_type": "REFERS_TO",
            "src_law_id": "vda:lr:1998-04-17:15",
            "src_article_id": None,
            "src_passage_id": None,
            "dst_law_id": "vda:lr:2007-03-29:4",
            "dst_article_label_norm": "37",
            "evidence_text": "Abrogata dall'art. 37, comma 2, della L.R.. 29 marzo 2007, n. 4",
            "confidence": 0.9,
        }
    ]

    raw, clean, _ = normalize_edges(edges)
    assert len(raw) == 1
    assert len(clean) == 1
    assert clean[0]["relation_type"] == "ABROGATED_BY"
