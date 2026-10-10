import pytest

from legal_rag.corpus.pipeline import build_corpus

from test_corpus_pipeline import Counter, snapshot


@pytest.mark.parametrize("names,ready", [
    (["273_LR-10-ottobre-1950-n1.html"], False),
    (["273_LR-10-ottobre-1950-n1.html", "1967_LR-5-settembre-1991-n44.html"], True),
])
def test_progress_reports_stage_counts_and_final_readiness(tmp_path, names, ready):
    source = snapshot(tmp_path, names)
    events = []
    manifest = build_corpus(source, tmp_path / "corpus", Counter(), 80, 120, progress=events.append)
    total = len(names)
    assert events == [
        {"stage": "preflight", "processed": 0, "total": 0, "ready": False},
        {"stage": "parsing", "processed": 0, "total": total, "ready": False},
        {"stage": "parsing", "processed": total, "total": total, "ready": False},
        {"stage": "graph_chunking", "processed": 0, "total": total, "ready": False},
        {"stage": "graph_chunking", "processed": total, "total": total, "ready": False},
        {"stage": "quality", "processed": 0, "total": 1, "ready": False},
        {"stage": "quality", "processed": 1, "total": 1, "ready": False},
        {"stage": "export", "processed": 0, "total": 9, "ready": False},
        {"stage": "export", "processed": 9, "total": 9, "ready": ready},
    ]
    assert manifest["ready_for_indexing"] is ready


def test_progress_callback_preserves_all_canonical_artifacts(tmp_path):
    source = snapshot(tmp_path, ["273_LR-10-ottobre-1950-n1.html", "2065_LR-19-agosto-1992-n42.html", "1967_LR-5-settembre-1991-n44.html"])
    silent, observed = tmp_path / "silent", tmp_path / "observed"
    original = build_corpus(source, silent, Counter(), 80, 120)
    events = []
    monitored = build_corpus(source, observed, Counter(), 80, 120, progress=events.append)
    assert events
    assert original["files"] == monitored["files"]
    for name in original["files"]:
        assert (silent / name).read_bytes() == (observed / name).read_bytes()
    assert {key: value for key, value in original.items() if key not in {"created_at", "code"}} == {
        key: value for key, value in monitored.items() if key not in {"created_at", "code"}
    }
