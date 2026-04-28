from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest

from legal_indexing.pipeline import run_indexing_pipeline
from legal_indexing.settings import IndexingConfig, make_chunking_profile


class FakeEmbedder:
    def __init__(self, size: int = 8) -> None:
        self._size = size

    @property
    def model_name(self) -> str:
        return "fake-test-embedder"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vec = [((digest[i % len(digest)] / 255.0) * 2.0) - 1.0 for i in range(self._size)]
            vectors.append(vec)
        return vectors


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_dataset(root: Path) -> Path:
    ds = root / "dataset"
    ds.mkdir(parents=True, exist_ok=True)

    laws = [
        {
            "law_id": "law:test",
            "law_date": "2025-01-01",
            "law_number": 1,
            "law_title": "Legge test",
            "status": "current",
        }
    ]
    articles = [
        {
            "article_id": "law:test#art:1",
            "law_id": "law:test",
            "article_label_norm": "1",
            "is_abrogated": False,
        }
    ]
    chunks = [
        {
            "chunk_id": "law:test#art:1#p:intro#chunk:0",
            "passage_id": "law:test#art:1#p:intro",
            "article_id": "law:test#art:1",
            "law_id": "law:test",
            "chunk_seq": 0,
            "text": "Titolo articolo",
            "text_for_embedding": "Titolo articolo",
            "law_date": "2025-01-01",
            "law_number": 1,
            "law_title": "Legge test",
            "law_status": "current",
            "article_label_norm": "1",
            "article_is_abrogated": False,
            "passage_label": "intro",
            "related_law_ids": [],
            "relation_types": [],
            "inbound_law_ids": [],
            "outbound_law_ids": [],
            "status_confidence": 0.9,
            "status_evidence": [],
            "index_views": ["historical", "current"],
        },
        {
            "chunk_id": "law:test#art:1#p:c1#chunk:0",
            "passage_id": "law:test#art:1#p:c1",
            "article_id": "law:test#art:1",
            "law_id": "law:test",
            "chunk_seq": 0,
            "text": "1. Questa e una disposizione normativa di prova",
            "text_for_embedding": "1. Questa e una disposizione normativa di prova",
            "law_date": "2025-01-01",
            "law_number": 1,
            "law_title": "Legge test",
            "law_status": "current",
            "article_label_norm": "1",
            "article_is_abrogated": False,
            "passage_label": "c1",
            "related_law_ids": [],
            "relation_types": [],
            "inbound_law_ids": [],
            "outbound_law_ids": [],
            "status_confidence": 0.9,
            "status_evidence": [],
            "index_views": ["historical", "current"],
        },
    ]

    manifest = {
        "schema_version": "laws-graph-pipeline-v1",
        "run_id": "test_run",
        "ready_to_embedding": True,
        "counts": {
            "laws": len(laws),
            "articles": len(articles),
            "notes": 0,
            "edges": 0,
            "events": 0,
            "chunks": len(chunks),
        },
        "hashes": {
            "chunks": "hash_test_chunks",
            "laws": "hash_laws",
            "articles": "hash_articles",
            "notes": "hash_notes",
            "edges": "hash_edges",
            "events": "hash_events",
        },
    }

    _write_jsonl(ds / "laws.jsonl", laws)
    _write_jsonl(ds / "articles.jsonl", articles)
    _write_jsonl(ds / "notes.jsonl", [])
    _write_jsonl(ds / "edges.jsonl", [])
    _write_jsonl(ds / "events.jsonl", [])
    _write_jsonl(ds / "chunks.jsonl", chunks)
    (ds / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return ds


def _read_chunks(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _write_eval_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Domanda", "Livello", "Risposta corretta", "Riferimento legge per la risposta"]
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(
                ",".join(
                    [
                        row["Domanda"],
                        row["Livello"],
                        row["Risposta corretta"],
                        f"\"{row['Riferimento legge per la risposta']}\"",
                    ]
                )
                + "\n"
            )


def test_incremental_reindexing_and_force_mode(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path)
    qdrant_path = tmp_path / "qdrant"
    artifacts_root = tmp_path / "artifacts"

    cfg = IndexingConfig(
        dataset_dir=dataset_dir,
        qdrant_path=qdrant_path,
        artifacts_root=artifacts_root,
        embedding_provider="utopia",
        embedding_model="test-model",
        embedding_api_key="unused-in-test",
        chunking_profile=make_chunking_profile("balanced", min_words_merge=2, max_words_split=30, overlap_words_split=5),
    )

    embedder = FakeEmbedder(size=8)

    run1 = run_indexing_pipeline(cfg.with_overrides(run_id="run1"), embedder=embedder)
    assert run1.total_refined_chunks > 0
    assert run1.total_embedded == run1.total_refined_chunks
    assert run1.skipped_unchanged == 0
    assert run1.sparse_enabled is True
    assert run1.sparse_artifact_path is not None
    assert run1.sparse_artifact_path.exists()

    run2 = run_indexing_pipeline(cfg.with_overrides(run_id="run2"), embedder=embedder)
    assert run2.total_embedded == 0
    assert run2.skipped_unchanged == run2.total_refined_chunks

    # Modify one chunk so content_hash changes.
    chunks_path = dataset_dir / "chunks.jsonl"
    chunks = _read_chunks(chunks_path)
    chunks[1]["text"] = "1. Questa disposizione e stata aggiornata"
    chunks[1]["text_for_embedding"] = chunks[1]["text"]
    _write_jsonl(chunks_path, chunks)

    run3 = run_indexing_pipeline(cfg.with_overrides(run_id="run3"), embedder=embedder)
    assert run3.total_embedded >= 1
    assert run3.total_embedded < run3.total_refined_chunks

    run4 = run_indexing_pipeline(
        cfg.with_overrides(run_id="run4", force_reembed=True),
        embedder=embedder,
    )
    assert run4.total_embedded == run4.total_refined_chunks


def test_index_contract_coverage_gate_blocks_mismatched_eval(tmp_path: Path) -> None:
    dataset_dir = _build_dataset(tmp_path)
    qdrant_path = tmp_path / "qdrant"
    artifacts_root = tmp_path / "artifacts"
    eval_mcq = tmp_path / "evaluation" / "questions.csv"
    eval_no_hint = tmp_path / "evaluation" / "questions_no_hint.csv"

    _write_eval_csv(
        eval_mcq,
        [
            {
                "Domanda": "D1",
                "Livello": "L1",
                "Risposta corretta": "A",
                "Riferimento legge per la risposta": "Legge regionale 1 gennaio 2024, n. 99 - Art. 1",
            }
        ],
    )
    _write_eval_csv(
        eval_no_hint,
        [
            {
                "Domanda": "D1",
                "Livello": "L1",
                "Risposta corretta": "X",
                "Riferimento legge per la risposta": "Legge regionale 1 gennaio 2024, n. 99 - Art. 1",
            }
        ],
    )

    cfg = IndexingConfig(
        dataset_dir=dataset_dir,
        qdrant_path=qdrant_path,
        artifacts_root=artifacts_root,
        embedding_provider="utopia",
        embedding_model="test-model",
        embedding_api_key="unused-in-test",
        chunking_profile=make_chunking_profile(
            "balanced",
            min_words_merge=2,
            max_words_split=30,
            overlap_words_split=5,
        ),
        index_contract_enforce_eval_coverage=True,
        index_contract_min_eval_coverage=0.95,
        eval_questions_csv=eval_mcq,
        eval_questions_no_hint_csv=eval_no_hint,
    )
    embedder = FakeEmbedder(size=8)

    with pytest.raises(RuntimeError, match="Index contract check failed"):
        run_indexing_pipeline(cfg.with_overrides(run_id="run_gate_fail"), embedder=embedder)
