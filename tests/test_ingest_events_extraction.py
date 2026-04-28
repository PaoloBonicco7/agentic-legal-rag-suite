from laws_ingestion.data_preparation.laws_graph.events import extract_events


def test_extract_events_from_edges() -> None:
    edges = [
        {
            "norm_edge_id": "a",
            "relation_type": "ABROGATED_BY",
            "src_law_id": "vda:lr:1998-04-17:15",
            "dst_law_id": "vda:lr:2007-03-29:4",
            "src_article_id": None,
            "src_passage_id": None,
            "dst_article_label_norm": "37",
            "evidence": "Abrogata dall'art. 37, comma 2, della L.R. 29 marzo 2007, n. 4",
            "confidence": 0.9,
            "source_file": "x.html",
        },
        {
            "norm_edge_id": "b",
            "relation_type": "INSERTS",
            "src_law_id": "vda:lr:2011-11-14:26",
            "dst_law_id": "vda:lr:2006-03-16:6",
            "src_article_id": "vda:lr:2011-11-14:26#art:4",
            "src_passage_id": "vda:lr:2011-11-14:26#art:4#p:intro",
            "dst_article_label_norm": "7bis",
            "evidence": "Inserisce l'art. 7bis della L:R. 16 marzo 2006, n. 6",
            "confidence": 0.8,
            "source_file": "y.html",
        },
    ]

    events = extract_events(edges)
    assert len(events) == 2
    assert {e["event_type"] for e in events} == {"REPEAL", "INSERT"}
    repeal = next(e for e in events if e["event_type"] == "REPEAL")
    assert repeal["effective_date"] == "2007-03-29"
    assert repeal["event_id"]
