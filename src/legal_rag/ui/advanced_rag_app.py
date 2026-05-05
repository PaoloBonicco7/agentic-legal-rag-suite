"""Streamlit UI for inspecting one advanced graph-aware RAG question."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import streamlit as st

from legal_rag.advanced_graph_rag import InteractiveRagConfig, InteractiveRagResult, build_interactive_runtime
from legal_rag.oracle_context_evaluation.io import read_json
from legal_rag.oracle_context_evaluation.models import DEFAULT_CHAT_MODEL
from legal_rag.simple_rag.models import RetrievedChunkRecord


st.set_page_config(page_title="Advanced Graph RAG", layout="wide")


@st.cache_resource(show_spinner=False)
def _load_runtime(config_json: str):
    data = json.loads(config_json)
    return build_interactive_runtime(InteractiveRagConfig.model_validate(data))


def _json_key(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _read_json_if_exists(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    try:
        return read_json(target)
    except Exception:
        return None


def _latest_advanced_summary(root: str | Path) -> tuple[Path, dict[str, Any]] | None:
    summaries = sorted(Path(root).glob("*/advanced_rag_summary.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in summaries:
        data = _read_json_if_exists(path)
        if data:
            return path, data
    return None


def _metric_rows(label: str, summary: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not summary:
        return []
    rows: list[dict[str, Any]] = []
    for dataset in ("mcq", "no_hint"):
        metrics = summary.get(dataset)
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "run": label,
                "dataset": dataset,
                "processed": metrics.get("processed"),
                "judged": metrics.get("judged"),
                "accuracy": metrics.get("accuracy"),
                "strict_accuracy": metrics.get("strict_accuracy"),
                "coverage": metrics.get("coverage"),
                "errors": metrics.get("errors"),
            }
        )
    return rows


def _chunk_rows(chunks: list[RetrievedChunkRecord], *, rerank_scores: list[int] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks, start=1):
        payload = chunk.payload
        row = {
            "rank": idx,
            "chunk_id": chunk.chunk_id,
            "score": round(float(chunk.score), 6),
            "law_id": payload.get("law_id"),
            "article_id": payload.get("article_id"),
            "law_status": payload.get("law_status"),
            "chars": len(chunk.text),
        }
        if rerank_scores and idx <= len(rerank_scores):
            row["rerank_score"] = rerank_scores[idx - 1]
        rows.append(row)
    return rows


def _render_chunks(title: str, chunks: list[RetrievedChunkRecord], *, rerank_scores: list[int] | None = None) -> None:
    st.subheader(title)
    if not chunks:
        st.info("Nessun chunk in questo step.")
        return
    st.dataframe(_chunk_rows(chunks, rerank_scores=rerank_scores), use_container_width=True, hide_index=True)
    for idx, chunk in enumerate(chunks, start=1):
        payload = chunk.payload
        label = f"{idx}. {chunk.chunk_id}"
        with st.expander(label):
            st.caption(f"law_id={payload.get('law_id', '')} | article_id={payload.get('article_id', '')}")
            st.write(chunk.text)


def _render_result(result: InteractiveRagResult) -> None:
    with st.chat_message("user"):
        st.markdown(result.question)
    with st.chat_message("assistant"):
        if result.answer:
            st.markdown(result.answer)
        elif result.error:
            st.error(result.error)
        else:
            st.warning("Nessuna risposta generata.")

    timing = result.timing
    st.subheader("Timing singola domanda")
    st.dataframe(
        [
            {"step": "retrieval", "seconds": round(timing.retrieval_seconds, 3)},
            {"step": "graph_expansion", "seconds": round(timing.graph_expansion_seconds, 3)},
            {"step": "rerank", "seconds": round(timing.rerank_seconds, 3)},
            {"step": "context", "seconds": round(timing.context_seconds, 3)},
            {"step": "answer", "seconds": round(timing.answer_seconds, 3)},
            {"step": "total", "seconds": round(timing.total_seconds, 3)},
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Configurazione effettiva")
    st.json(
        {
            "flags": result.flags,
            "parameters": result.parameters,
            "metadata_filters": result.metadata_filters,
            "retrieval_mode": result.retrieval_mode,
            "hybrid_available": result.hybrid_available,
            "hybrid_unavailable_reason": result.hybrid_unavailable_reason,
        }
    )

    tab_retrieved, tab_graph, tab_rerank, tab_context, tab_citations = st.tabs(
        ["retrieval iniziale", "graph expansion", "rerank", "contesto finale", "citazioni"]
    )
    with tab_retrieved:
        _render_chunks("Chunk recuperati", result.retrieved)
    with tab_graph:
        if result.graph_relations_used:
            st.dataframe([item.to_json_record() for item in result.graph_relations_used], use_container_width=True, hide_index=True)
        else:
            st.info("Nessuna relazione graph usata.")
        _render_chunks("Chunk aggiunti dal grafo", result.expanded)
    with tab_rerank:
        _render_chunks("Chunk ordinati dal reranker", result.reranked, rerank_scores=result.rerank_scores)
    with tab_context:
        _render_chunks("Chunk inclusi nel contesto", result.context_chunks)
        with st.expander("Context text"):
            st.text(result.context_text)
    with tab_citations:
        if result.citations:
            st.dataframe([item.to_json_record() for item in result.citations], use_container_width=True, hide_index=True)
        else:
            st.info("Nessuna citazione prodotta.")
        if result.invalid_citation_chunk_ids:
            st.warning(f"Citazioni non valide: {result.invalid_citation_chunk_ids}")


def main() -> None:
    st.title("Advanced Graph RAG")
    st.caption("Demo locale per ispezionare retrieval, filtri, graph expansion, rerank e risposta finale.")

    with st.sidebar:
        st.header("Runtime")
        evaluation_dir = st.text_input("Evaluation dir", "data/evaluation_clean")
        laws_dir = st.text_input("Laws dir", "data/laws_dataset_clean")
        index_dir = st.text_input("Index dir", "data/indexes/qdrant")
        index_manifest_path = st.text_input("Index manifest", "data/indexing_runs/<latest>/index_manifest.json")
        collection_name = st.text_input("Collection", "legal_chunks")
        output_root = st.text_input("Advanced runs root", "data/rag_runs/advanced")
        simple_summary_path = st.text_input("Simple summary", "data/rag_runs/simple/simple_rag_summary.json")
        env_file = st.text_input("Env file", ".env")
        api_key = st.text_input("API key override", value="", type="password")
        api_url = st.text_input("Chat API URL override", value=os.getenv("UTOPIA_OLLAMA_CHAT_URL", ""))
        base_url = st.text_input("Base URL", os.getenv("UTOPIA_BASE_URL", "https://utopia.hpc4ai.unito.it/api"))
        chat_model = st.text_input("Chat model", os.getenv("UTOPIA_CHAT_MODEL", DEFAULT_CHAT_MODEL))
        judge_model = st.text_input("Rerank model", os.getenv("UTOPIA_JUDGE_MODEL", ""))
        timeout_seconds = st.number_input("Timeout seconds", min_value=1, max_value=600, value=180)
        retry_attempts = st.number_input("Retry attempts", min_value=1, max_value=5, value=1)
        if st.button("Reset cached runtime"):
            st.cache_resource.clear()
            st.rerun()

    resource_data = {
        "evaluation_dir": evaluation_dir,
        "laws_dir": laws_dir,
        "index_dir": index_dir,
        "index_manifest_path": index_manifest_path,
        "collection_name": collection_name,
        "output_root": output_root,
        "run_name": "interactive",
        "env_file": env_file or None,
        "api_key": api_key or None,
        "api_url": api_url or None,
        "base_url": base_url,
        "chat_model": chat_model,
        "judge_model": judge_model or None,
        "timeout_seconds": int(timeout_seconds),
        "retry_attempts": int(retry_attempts),
        "hybrid_enabled": False,
        "max_concurrency": 1,
    }
    try:
        runtime = _load_runtime(_json_key(resource_data))
        health = runtime.health()
    except Exception as exc:
        st.error(f"Runtime non disponibile: {type(exc).__name__}: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Retrieval")
        metadata_filters_enabled = st.toggle("Metadata filters", value=True)
        hybrid_enabled = st.toggle(
            "Hybrid retrieval",
            value=False,
            disabled=not bool(health["hybrid_available"]),
            help=health.get("hybrid_unavailable_reason") or "Dense+sparse retrieval via Qdrant RRF.",
        )
        graph_expansion_enabled = st.toggle("Graph expansion", value=True)
        rerank_enabled = st.toggle("LLM rerank", value=True)
        top_k = st.slider("Top-k", min_value=1, max_value=30, value=10)
        graph_seed_k = st.slider("Graph seed k", min_value=1, max_value=20, value=5)
        max_chunks_per_law = st.slider("Max chunk per legge espansa", min_value=1, max_value=10, value=2)
        rerank_input_k = st.slider("Rerank input k", min_value=1, max_value=50, value=20)
        rerank_output_k = st.slider("Context chunk cap", min_value=1, max_value=15, value=5)
        max_context_chars = st.slider("Max context chars", min_value=1000, max_value=32000, value=16000, step=1000)

    st.subheader("Stato runtime")
    left, right = st.columns(2)
    with left:
        st.json(health)
    with right:
        rows = []
        rows.extend(_metric_rows("simple", _read_json_if_exists(simple_summary_path)))
        latest = _latest_advanced_summary(output_root)
        if latest:
            path, summary = latest
            rows.extend(_metric_rows(path.parent.name, summary))
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("Nessuna metrica salvata trovata.")

    run_config = InteractiveRagConfig.model_validate(
        {
            **resource_data,
            "metadata_filters_enabled": metadata_filters_enabled,
            "hybrid_enabled": bool(hybrid_enabled),
            "graph_expansion_enabled": graph_expansion_enabled,
            "rerank_enabled": rerank_enabled,
            "static_filters": {"law_status": "current"} if metadata_filters_enabled else {},
            "top_k": int(top_k),
            "graph_expansion_seed_k": int(graph_seed_k),
            "max_chunks_per_expanded_law": int(max_chunks_per_law),
            "rerank_input_k": int(rerank_input_k),
            "rerank_output_k": int(rerank_output_k),
            "max_context_chars": int(max_context_chars),
        }
    )

    prompt = st.chat_input("Scrivi una domanda giuridica")
    if prompt:
        with st.status("Esecuzione advanced RAG", expanded=True) as status:
            result = runtime.answer_question(prompt, config=run_config, on_step=st.write)
            status.update(
                label="Pipeline completata con errori" if result.error else "Pipeline completata",
                state="error" if result.error else "complete",
            )
        st.session_state.last_interactive_result = result.to_json_record()

    if "last_interactive_result" in st.session_state:
        _render_result(InteractiveRagResult.model_validate(st.session_state.last_interactive_result))


if __name__ == "__main__":
    main()
