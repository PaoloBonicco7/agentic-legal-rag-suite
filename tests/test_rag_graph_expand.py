from __future__ import annotations

import json
from pathlib import Path

from legal_indexing.rag_runtime.graph_adapter import LegalGraphAdapter
from legal_indexing.rag_runtime.qdrant_retrieval import RetrievedChunk


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_graph_adapter_expands_law_neighbors_from_edges_and_events(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_jsonl(
        dataset_dir / "edges.jsonl",
        [
            {
                "src_law_id": "law:a",
                "dst_law_id": "law:b",
                "dst_article_label_norm": "3",
            },
            {
                "src_law_id": "law:a",
                "dst_law_id": "law:c",
                "dst_article_label_norm": "5",
            },
        ],
    )
    _write_jsonl(
        dataset_dir / "events.jsonl",
        [
            {
                "source_law_id": "law:a",
                "target_law_id": "law:d",
                "target_article_label_norm": "7",
            }
        ],
    )

    adapter = LegalGraphAdapter(dataset_dir)
    retrieved = [
        RetrievedChunk(
            chunk_id="law:a#art:1#rc:0",
            score=0.9,
            text="test",
            point_id="p1",
            payload={"law_id": "law:a", "article_id": "law:a#art:1"},
        )
    ]

    result = adapter.expand_from_retrieved(retrieved, max_related_laws=2)
    assert result.seed_chunk_ids == ("law:a#art:1#rc:0",)
    assert result.seed_law_ids == ("law:a",)
    assert result.seed_passage_ids == tuple()
    assert set(result.related_law_ids).issubset({"law:b", "law:c", "law:d"})
    assert len(result.related_law_ids) == 2
    assert result.edge_hits >= 2
    assert result.event_hits >= 1
    assert any(x.startswith("law:") and "#art:" in x for x in result.related_article_ids)
    assert result.related_chunk_ids == tuple()
