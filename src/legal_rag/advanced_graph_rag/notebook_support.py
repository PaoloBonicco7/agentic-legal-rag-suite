"""Notebook-side helpers for `notebooks/06_advanced_graph_rag.ipynb`.

Keep the notebook to one-liners by collecting the demo builders, run loaders,
progress callbacks and trace-table renderer here.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from legal_rag.oracle_context_evaluation.io import sha256_file, write_json, write_jsonl

# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #


def read_json(path: Path) -> dict[str, Any]:
    """Load a JSON document."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts (empty when missing)."""
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_run_artifacts(run_dir: Path) -> dict[str, Any]:
    """Read summary, diagnostics, manifest and per-row JSONL for one advanced run."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "advanced_rag_manifest.json"
    return {
        "dir": run_dir,
        "summary": read_json(run_dir / "advanced_rag_summary.json") if (run_dir / "advanced_rag_summary.json").exists() else {},
        "diagnostics": read_json(run_dir / "advanced_diagnostics.json") if (run_dir / "advanced_diagnostics.json").exists() else {},
        "manifest": read_json(manifest_path) if manifest_path.exists() else {},
        "mcq": read_jsonl(run_dir / "mcq_results.jsonl"),
        "no_hint": read_jsonl(run_dir / "no_hint_results.jsonl"),
    }


def safe_run_slug(value: str | None) -> str:
    """Sanitize a value into a filename-friendly slug."""
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "none").strip()).strip("_").lower()
    return text[:48] or "none"


def metric(summary: Mapping[str, Any] | None, dataset: str, key: str) -> float:
    """Pull a numeric metric out of an advanced summary."""
    if not summary:
        return 0.0
    value = (summary.get(dataset) or {}).get(key, 0)
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def run_has_total_errors(summary: Mapping[str, Any] | None) -> bool:
    """Return True when every processed row hit an error (run is unusable)."""
    if not summary:
        return True
    datasets = [summary.get("mcq", {}) or {}, summary.get("no_hint", {}) or {}]
    return all(
        int(item.get("processed") or 0) > 0 and int(item.get("errors") or 0) >= int(item.get("processed") or 0)
        for item in datasets
    )


def compact_trace(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project a per-row record onto the most useful trace fields."""
    keys = [
        "qid", "predicted_label", "score", "judge_score", "retrieval_mode", "retrieved_chunk_ids",
        "graph_expanded_chunk_ids", "graph_relations_used", "reranked_chunk_ids", "rerank_scores",
        "context_chunk_ids", "context_sufficient", "reference_law_hit", "failure_category", "error",
    ]
    return {key: row.get(key) for key in keys if key in row}


def load_recommendation_config(
    retrieval_eval_runs_dir: Path,
    *,
    override_path: str | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    """Return the most recent recommended_advanced_config.json and its payload."""
    if override_path:
        path = Path(override_path)
    else:
        candidates = sorted(
            Path(retrieval_eval_runs_dir).glob("*/recommended_advanced_config.json"),
            key=lambda candidate: candidate.stat().st_mtime,
            reverse=True,
        )
        path = candidates[0] if candidates else None
    if path is None or not path.exists():
        return None, {}
    return path, json.loads(path.read_text(encoding="utf-8"))


def compare_runs(
    *run_dirs: Path,
    simple_summary: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Build a comparison table across advanced runs (and an optional simple baseline)."""
    rows: list[dict[str, Any]] = []
    if simple_summary:
        rows.append({
            "run": "simple",
            "mcq_strict_accuracy": metric(simple_summary, "mcq", "strict_accuracy"),
            "no_hint_strict_accuracy": metric(simple_summary, "no_hint", "strict_accuracy"),
            "mcq_errors": int((simple_summary.get("mcq") or {}).get("errors") or 0),
            "no_hint_errors": int((simple_summary.get("no_hint") or {}).get("errors") or 0),
        })
    for run_dir in run_dirs:
        run = load_run_artifacts(Path(run_dir))
        summary = run["summary"]
        if not summary:
            continue
        rows.append({
            "run": Path(run_dir).name,
            "mcq_strict_accuracy": metric(summary, "mcq", "strict_accuracy"),
            "no_hint_strict_accuracy": metric(summary, "no_hint", "strict_accuracy"),
            "mcq_errors": int((summary.get("mcq") or {}).get("errors") or 0),
            "no_hint_errors": int((summary.get("no_hint") or {}).get("errors") or 0),
        })
    return pd.DataFrame(rows)


def show_trace_table(row: Mapping[str, Any], *, title: str | None = None) -> Any:
    """Render a stage-by-chunk markdown trace via IPython.display.Markdown."""
    from IPython.display import Markdown  # local import — only needed in notebooks

    stage_by_chunk: dict[str, list[str]] = {}
    for chunk_id in row.get("retrieved_chunk_ids", []):
        stage_by_chunk.setdefault(chunk_id, []).append("retrieved")
    for chunk_id in row.get("graph_expanded_chunk_ids", []):
        stage_by_chunk.setdefault(chunk_id, []).append("graph")
    for chunk_id in row.get("reranked_chunk_ids", []):
        stage_by_chunk.setdefault(chunk_id, []).append("reranked")
    for chunk_id in row.get("context_chunk_ids", []):
        stage_by_chunk.setdefault(chunk_id, []).append("context")
    header = f"### {title or row.get('qid', 'trace')}"
    lines = [header, "", "| chunk_id | fasi |", "|---|---|"]
    for chunk_id, stages in stage_by_chunk.items():
        lines.append(f"| `{chunk_id}` | {', '.join(stages)} |")
    return Markdown("\n".join(lines))


# --------------------------------------------------------------------------- #
# Progress callbacks
# --------------------------------------------------------------------------- #


def make_debug_progress_callback() -> Callable[[dict[str, Any]], None]:
    """Build a simple line-by-line progress callback for the smoke debug run."""

    def callback(event: dict[str, Any]) -> None:
        name = event.get("event")
        if name == "setup_finished":
            print(f"[rag-debug] setup: {event.get('mcq')} MCQ + {event.get('no_hint')} no-hint", flush=True)
        elif name == "row_finished":
            print(f"[rag-debug] row_finished run={event.get('run')} qid={event.get('qid')} error={event.get('error')}", flush=True)
        elif name in {"run_started", "run_finished"}:
            print(f"[rag-debug] {name} run={event.get('run')} total={event.get('total')}", flush=True)

    return callback


def make_full_run_progress_callback(
    *,
    run_name: str,
    llm_provider: str,
    variant: str,
    answer_model: str,
    judge_model: str,
    every_rows: int = 5,
    max_error_samples: int = 5,
) -> Callable[[dict[str, Any]], None]:
    """Build a progress callback that prints aggregate run status with throttling."""
    lock = threading.Lock()
    state = {"completed": 0, "total": 0, "errors": 0, "first_errors": [] }

    def callback(event: dict[str, Any]) -> None:
        with lock:
            name = event.get("event")
            if name == "setup_finished":
                state.update({"completed": 0, "total": int(event.get("total") or 0), "errors": 0, "first_errors": []})
                print(f"[advanced] run_name={run_name}", flush=True)
                print(f"[advanced] llm_provider={llm_provider} variant={variant}", flush=True)
                print(f"[advanced] answer_model={answer_model} judge_model={judge_model}", flush=True)
                print(f"[advanced] start: {state['total']} domande totali ({event.get('mcq')} MCQ + {event.get('no_hint')} no-hint)", flush=True)
                return
            if name == "row_finished":
                state["completed"] += 1
                error = event.get("error")
                if error:
                    state["errors"] += 1
                    if len(state["first_errors"]) < max_error_samples:
                        snippet = str(error).replace("\n", " ")[:220]
                        state["first_errors"].append((event.get("run"), event.get("qid"), snippet))
                        print(f"[advanced][error-sample] run={event.get('run')} qid={event.get('qid')} error={snippet}", flush=True)
                completed = state["completed"]
                total = state["total"] or int(event.get("total") or 0)
                if completed == 1 or completed % every_rows == 0 or completed == total:
                    percent = (completed / total * 100) if total else 0.0
                    print(f"[advanced] {percent:5.1f}% ({completed}/{total}) last={event.get('run')} qid={event.get('qid')} errors={state['errors']}", flush=True)
                return
            if name == "run_finished":
                print(f"[advanced] completato dataset {event.get('run')} ({event.get('total')} righe)", flush=True)

    return callback


# --------------------------------------------------------------------------- #
# Demo runtime (lightweight, in-memory)
# --------------------------------------------------------------------------- #

LAW_SEED = "vda:lr:2024-01-15:10"
LAW_TARGET = "vda:lr:2024-02-10:11"
LAW_NOISE = "vda:lr:2023-12-20:9"
ANSWER_CHUNK_ID = "chunk_target_answer"

LAW_TITLES: dict[str, str] = {
    LAW_SEED: "Legge regionale 15 gennaio 2024, n. 10 - Misure per contributi comunali",
    LAW_TARGET: "Legge regionale 10 febbraio 2024, n. 11 - Disciplina attuativa dei termini",
    LAW_NOISE: "Legge regionale 20 dicembre 2023, n. 9 - Disposizioni generali",
}

DEMO_QUESTION = "Entro quale termine deve essere presentata la domanda di contributo?"
DEMO_EXPECTED_REFERENCE = "Legge regionale 10 febbraio 2024, n. 11 - Art. 3"


def _demo_chunk_payload(chunk_id: str, *, law_id: str, article_label: str, text: str) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "law_id": law_id,
        "article_id": f"{law_id}#art:{article_label}",
        "article_label_norm": article_label,
        "text": text,
        "law_title": LAW_TITLES[law_id],
        "law_status": "current",
        "article_status": "current",
        "index_views": ["current"],
        "relation_types": [],
    }


DEMO_PAYLOADS: list[dict[str, Any]] = [
    _demo_chunk_payload(
        "chunk_seed_retrieved", law_id=LAW_SEED, article_label="2",
        text="La domanda di contributo e' disciplinata dalla legge attuativa richiamata dalla presente legge.",
    ),
    _demo_chunk_payload(
        ANSWER_CHUNK_ID, law_id=LAW_TARGET, article_label="3",
        text="La domanda di contributo deve essere presentata entro trenta giorni dalla pubblicazione dell'avviso.",
    ),
    _demo_chunk_payload(
        "chunk_noise", law_id=LAW_NOISE, article_label="1",
        text="Disposizione generale non utile per il termine di presentazione della domanda.",
    ),
]


class DemoHybridEmbedder:
    """In-memory embedder yielding deterministic dense+sparse vectors for the demo."""

    @property
    def model_name(self) -> str:
        return "demo-bge-m3-like"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed_sparse_texts(self, texts: list[str]) -> list[dict[str, list[int] | list[float]]]:
        return [{"indices": [101], "values": [1.0]} for _ in texts]


class DemoStructuredClient:
    """Fake structured chat client that returns deterministic answers for the demo."""

    def structured_chat(self, *, prompt: str, model: str, payload_schema: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
        properties = payload_schema.get("properties", {})
        context_ids = re.findall(r"chunk_id: ([^\s]+)", prompt)
        has_answer_chunk = ANSWER_CHUNK_ID in context_ids
        if "queries" in properties:
            return {"structured": {"queries": [DEMO_QUESTION, "termine presentazione domanda contributo", "scadenza domanda contributo regionale"]}}
        if "rewritten_query" in properties:
            return {"structured": {"rewritten_query": "termine presentazione domanda contributo"}}
        if "hypothetical_answer" in properties:
            return {"structured": {"hypothetical_answer": "La domanda di contributo deve essere presentata entro un termine fissato dalla legge attuativa."}}
        if "scores" in properties:
            scores = [{"chunk_id": chunk_id, "score": 2 if chunk_id == ANSWER_CHUNK_ID else 0} for chunk_id in context_ids]
            return {"structured": {"scores": scores}}
        citation = ANSWER_CHUNK_ID if has_answer_chunk else (context_ids[0] if context_ids else "chunk_seed_retrieved")
        if "answer_label" in properties:
            return {"structured": {"answer_label": "A" if has_answer_chunk else "B", "citation_chunk_ids": [citation], "short_rationale": "demo"}}
        if "answer_text" in properties:
            text = "Entro trenta giorni dalla pubblicazione dell'avviso." if has_answer_chunk else "Il contesto non contiene il termine esatto."
            return {"structured": {"answer_text": text, "citation_chunk_ids": [citation], "short_rationale": "demo"}}
        if "score" in properties:
            return {"structured": {"score": 2 if "trenta giorni" in prompt else 0, "explanation": "Valutazione demo controllata."}}
        raise AssertionError(payload_schema)


def build_demo_files(root: Path) -> tuple[Path, Path, Path, Path]:
    """Write a minimal evaluation + laws + index manifest set under a tmp root."""
    root = Path(root)
    evaluation_dir = root / "evaluation_clean"
    laws_dir = root / "laws_dataset_clean"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    laws_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(evaluation_dir / "questions_mcq.jsonl", [{
        "qid": "demo-advanced-0001", "source_position": 1, "level": "L2",
        "question_stem": DEMO_QUESTION,
        "options": {
            "A": "Entro trenta giorni dalla pubblicazione dell'avviso.",
            "B": "Entro sessanta giorni dalla pubblicazione dell'avviso.",
            "C": "Entro il 31 dicembre dell'anno di riferimento.",
            "D": "Prima della deliberazione della Giunta.",
            "E": "Senza termine espresso.", "F": "Entro dieci giorni dalla domanda.",
        },
        "correct_label": "A",
        "correct_answer": "Entro trenta giorni dalla pubblicazione dell'avviso.",
        "expected_references": [DEMO_EXPECTED_REFERENCE],
    }])
    write_jsonl(evaluation_dir / "questions_no_hint.jsonl", [{
        "qid": "demo-advanced-0001", "source_position": 1, "level": "L2",
        "question": DEMO_QUESTION,
        "correct_answer": "Entro trenta giorni dalla pubblicazione dell'avviso.",
        "expected_references": [DEMO_EXPECTED_REFERENCE], "linked_mcq_qid": "demo-advanced-0001",
    }])
    write_json(evaluation_dir / "evaluation_manifest.json", {"schema_version": "evaluation-dataset-v1", "records": 1})
    write_jsonl(laws_dir / "laws.jsonl", [{"law_id": law_id, "law_title": title} for law_id, title in LAW_TITLES.items()])
    write_jsonl(laws_dir / "articles.jsonl", [
        {"law_id": LAW_SEED, "article_id": f"{LAW_SEED}#art:2", "article_label_norm": "2", "article_text": DEMO_PAYLOADS[0]["text"]},
        {"law_id": LAW_TARGET, "article_id": f"{LAW_TARGET}#art:3", "article_label_norm": "3", "article_text": DEMO_PAYLOADS[1]["text"]},
        {"law_id": LAW_NOISE, "article_id": f"{LAW_NOISE}#art:1", "article_label_norm": "1", "article_text": DEMO_PAYLOADS[2]["text"]},
    ])
    write_jsonl(laws_dir / "edges.jsonl", [{"edge_id": "edge-seed-target", "src_law_id": LAW_SEED, "dst_law_id": LAW_TARGET, "relation_type": "REFERENCES"}])
    write_jsonl(laws_dir / "chunks.jsonl", DEMO_PAYLOADS)
    write_json(laws_dir / "manifest.json", {"ready_for_indexing": True})

    index_manifest = root / "indexing_runs" / "demo" / "index_manifest.json"
    index_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json(index_manifest, {
        "schema_version": "indexing-contract-v1", "collection_name": "advanced_demo",
        "ready_for_retrieval": True, "hybrid_enabled": True,
        "embedding": {"provider": "demo", "model": "demo-bge-m3-like", "configured_model": "demo-bge-m3-like", "mode": "dense+sparse"},
        "config": {"embedding_provider": "demo", "embedding_model": "demo-bge-m3-like", "hybrid_enabled": True},
    })
    simple_manifest = root / "simple" / "simple_rag_manifest.json"
    simple_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json(simple_manifest, {"schema_version": "simple-rag-v1", "source_hashes": {
        "questions_mcq": sha256_file(evaluation_dir / "questions_mcq.jsonl"),
        "questions_no_hint": sha256_file(evaluation_dir / "questions_no_hint.jsonl"),
        "evaluation_manifest": sha256_file(evaluation_dir / "evaluation_manifest.json"),
        "index_manifest": sha256_file(index_manifest),
    }})
    return evaluation_dir, laws_dir, index_manifest, simple_manifest


def build_demo_qdrant() -> QdrantClient:
    """Build a Qdrant in-memory collection populated with the demo chunks."""
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="advanced_demo",
        vectors_config={"dense": qmodels.VectorParams(size=4, distance=qmodels.Distance.COSINE)},
        sparse_vectors_config={"sparse": qmodels.SparseVectorParams()},
    )
    vectors = [
        {"dense": [1.0, 0.0, 0.0, 0.0], "sparse": qmodels.SparseVector(indices=[101], values=[1.0])},
        {"dense": [0.2, 0.8, 0.0, 0.0], "sparse": qmodels.SparseVector(indices=[303], values=[1.0])},
        {"dense": [0.85, 0.15, 0.0, 0.0], "sparse": qmodels.SparseVector(indices=[909], values=[1.0])},
    ]
    points = [
        qmodels.PointStruct(id=idx, vector=vector, payload=payload)
        for idx, (vector, payload) in enumerate(zip(vectors, DEMO_PAYLOADS), start=1)
    ]
    client.upsert(collection_name="advanced_demo", points=points, wait=True)
    return client


def build_demo_config(demo_root: Path, **overrides: Any) -> dict[str, Any]:
    """Build the kwargs for `AdvancedRagConfig` used by the demo run."""
    evaluation_dir, laws_dir, index_manifest, simple_manifest = build_demo_files(demo_root)
    base = {
        "evaluation_dir": str(evaluation_dir),
        "laws_dir": str(laws_dir),
        "index_manifest_path": str(index_manifest),
        "simple_rag_manifest_path": str(simple_manifest),
        "output_root": str(Path(demo_root) / "advanced_runs"),
        "chat_model": "demo-answer",
        "judge_model": "demo-judge",
        "max_concurrency": 1,
        "top_k": 1,
        "graph_expansion_seed_k": 1,
        "max_chunks_per_expanded_law": 1,
        "max_expanded_chunks_total": 3,
        "min_edge_confidence": 0.45,
        "rerank_input_k": 3,
        "rerank_output_k": 2,
    }
    base.update(overrides)
    return base


__all__ = [
    "ANSWER_CHUNK_ID",
    "DEMO_EXPECTED_REFERENCE",
    "DEMO_PAYLOADS",
    "DEMO_QUESTION",
    "DemoHybridEmbedder",
    "DemoStructuredClient",
    "LAW_NOISE",
    "LAW_SEED",
    "LAW_TARGET",
    "LAW_TITLES",
    "build_demo_config",
    "build_demo_files",
    "build_demo_qdrant",
    "compact_trace",
    "compare_runs",
    "load_recommendation_config",
    "load_run_artifacts",
    "make_debug_progress_callback",
    "make_full_run_progress_callback",
    "metric",
    "read_json",
    "read_jsonl",
    "run_has_total_errors",
    "safe_run_slug",
    "show_trace_table",
]
