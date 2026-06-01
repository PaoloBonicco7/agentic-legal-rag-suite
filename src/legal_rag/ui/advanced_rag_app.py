"""Streamlit UI for inspecting one advanced graph-aware RAG question.

Mirrors the configuration matrix from `notebooks/06_advanced_graph_rag.ipynb`:
the default preset is the same `A4_combined_best` variant used by notebook 06,
and the other notebook variants (plus the raw 06b recommendation and a Custom
mode) are selectable from the sidebar.
"""

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

from legal_rag.advanced_graph_rag import (
    InteractiveRagConfig,
    InteractiveRagResult,
    build_interactive_runtime,
)
from legal_rag.advanced_graph_rag.notebook_support import load_recommendation_config
from legal_rag.oracle_context_evaluation.io import read_json
from legal_rag.oracle_context_evaluation.models import DEFAULT_CHAT_MODEL
from legal_rag.simple_rag.models import RetrievedChunkRecord


st.set_page_config(page_title="Advanced Graph RAG", layout="wide")


# --------------------------------------------------------------------------- #
# Variant presets — mirror notebook 06 cell `code-48608680`
# --------------------------------------------------------------------------- #

PRESETS: dict[str, dict[str, Any]] = {
    "A4_combined_best": {
        "label": "Notebook 06 default — A4 combined best",
        "description": (
            "Hybrid BGE-M3 + multi-query (n=3), cap=15, rrf_k=60. "
            "Configurazione promossa da 06b e usata di default dal notebook 06."
        ),
        "config": {
            "metadata_filters_enabled": False,
            "hybrid_enabled": True,
            "graph_expansion_enabled": False,
            "rerank_enabled": False,
            "top_k": 100,
            "rrf_k": 60,
            "rerank_output_k": 10,
            "rerank_input_k": 20,
            "max_context_chunks": 15,
            "max_context_chars": 16000,
            "query_rewriting_enabled": True,
            "query_rewriting_strategy": "multi_query",
            "query_rewriting_n": 3,
        },
    },
    "A0_simple_equiv": {
        "label": "A0 simple-equiv (hybrid only)",
        "description": "top_k=10, cap=3, no rewrite — controllo \"hybrid only\".",
        "config": {
            "metadata_filters_enabled": False,
            "hybrid_enabled": True,
            "graph_expansion_enabled": False,
            "rerank_enabled": False,
            "top_k": 10,
            "rrf_k": 60,
            "rerank_output_k": 5,
            "rerank_input_k": 20,
            "max_context_chunks": 3,
            "max_context_chars": 16000,
            "query_rewriting_enabled": False,
            "query_rewriting_strategy": "none",
            "query_rewriting_n": 3,
        },
    },
    "A1_lean_cap5": {
        "label": "A1 lean cap=5",
        "description": "Lean v1 con context tagliato a 5 (test \"context noise\").",
        "config": {
            "metadata_filters_enabled": False,
            "hybrid_enabled": True,
            "graph_expansion_enabled": False,
            "rerank_enabled": False,
            "top_k": 100,
            "rrf_k": 30,
            "rerank_output_k": 10,
            "rerank_input_k": 20,
            "max_context_chunks": 5,
            "max_context_chars": 16000,
            "query_rewriting_enabled": True,
            "query_rewriting_strategy": "multi_query",
            "query_rewriting_n": 3,
        },
    },
    "A2_lean_cap15": {
        "label": "A2 lean cap=15",
        "description": "Lean v1 con context allargato a 15 (test \"fa entrare il rank 11\").",
        "config": {
            "metadata_filters_enabled": False,
            "hybrid_enabled": True,
            "graph_expansion_enabled": False,
            "rerank_enabled": False,
            "top_k": 100,
            "rrf_k": 30,
            "rerank_output_k": 10,
            "rerank_input_k": 20,
            "max_context_chunks": 15,
            "max_context_chars": 16000,
            "query_rewriting_enabled": True,
            "query_rewriting_strategy": "multi_query",
            "query_rewriting_n": 3,
        },
    },
    "A3_rrf_with_original": {
        "label": "A3 RRF + original",
        "description": "rrf_k=60 + cap=10 — verifica del fix F1+F2 sotto budget baseline.",
        "config": {
            "metadata_filters_enabled": False,
            "hybrid_enabled": True,
            "graph_expansion_enabled": False,
            "rerank_enabled": False,
            "top_k": 100,
            "rrf_k": 60,
            "rerank_output_k": 10,
            "rerank_input_k": 20,
            "max_context_chunks": 10,
            "max_context_chars": 16000,
            "query_rewriting_enabled": True,
            "query_rewriting_strategy": "multi_query",
            "query_rewriting_n": 3,
        },
    },
    "recommended_06b": {
        "label": "06b raw recommendation",
        "description": (
            "Configurazione promossa dal payload `recommended_advanced_config.json` "
            "di 06b, senza l'aggiunta di multi-query/max_context_chunks."
        ),
        "config": {
            "metadata_filters_enabled": False,
            "hybrid_enabled": True,
            "graph_expansion_enabled": False,
            "rerank_enabled": False,
            "top_k": 100,
            "rrf_k": 30,
            "rerank_output_k": 10,
            "rerank_input_k": 20,
            "max_context_chunks": None,
            "max_context_chars": 16000,
            "query_rewriting_enabled": False,
            "query_rewriting_strategy": "none",
            "query_rewriting_n": 3,
        },
    },
    "graph_rerank_all_on": {
        "label": "Graph + rerank (all-on demo)",
        "description": "Attiva tutte le componenti avanzate per ispezione qualitativa.",
        "config": {
            "metadata_filters_enabled": True,
            "hybrid_enabled": True,
            "graph_expansion_enabled": True,
            "rerank_enabled": True,
            "top_k": 50,
            "rrf_k": 60,
            "rerank_output_k": 10,
            "rerank_input_k": 30,
            "max_context_chunks": 10,
            "max_context_chars": 16000,
            "query_rewriting_enabled": True,
            "query_rewriting_strategy": "multi_query",
            "query_rewriting_n": 3,
        },
    },
}
DEFAULT_PRESET = "A4_combined_best"
CUSTOM_PRESET_LABEL = "Custom (tweak liberamente)"


# Map AdvancedRagConfig field name -> Streamlit widget session_state key.
WIDGET_KEYS: dict[str, str] = {
    "metadata_filters_enabled": "ui_metadata_filters_enabled",
    "hybrid_enabled": "ui_hybrid_enabled",
    "graph_expansion_enabled": "ui_graph_expansion_enabled",
    "rerank_enabled": "ui_rerank_enabled",
    "top_k": "ui_top_k",
    "rrf_k": "ui_rrf_k",
    "rerank_input_k": "ui_rerank_input_k",
    "rerank_output_k": "ui_rerank_output_k",
    "max_context_chunks": "ui_max_context_chunks",
    "max_context_chars": "ui_max_context_chars",
    "query_rewriting_enabled": "ui_qr_enabled",
    "query_rewriting_strategy": "ui_qr_strategy",
    "query_rewriting_n": "ui_qr_n",
    "graph_expansion_seed_k": "ui_graph_seed_k",
    "max_chunks_per_expanded_law": "ui_max_chunks_per_law",
    "max_expanded_chunks_total": "ui_max_expanded_total",
    "min_edge_confidence": "ui_min_edge_confidence",
}

# Defaults for fields not driven by the preset table.
GRAPH_DEFAULTS = {
    "graph_expansion_seed_k": 3,
    "max_chunks_per_expanded_law": 2,
    "max_expanded_chunks_total": 15,
    "min_edge_confidence": 0.45,
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner="Connecting to Qdrant + LLM")
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
    summaries = sorted(
        Path(root).glob("*/advanced_rag_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
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


def _apply_preset_to_state(preset_name: str) -> None:
    """Snap every UI widget to the values of the chosen preset."""
    if preset_name == CUSTOM_PRESET_LABEL:
        return
    preset = PRESETS[preset_name]["config"]
    merged = {**GRAPH_DEFAULTS, **preset}
    # max_context_chunks=None means "fall back to rerank_output_k"; the slider still
    # needs a concrete int, so resolve the fallback here when seeding session_state.
    if merged.get("max_context_chunks") is None:
        merged["max_context_chunks"] = merged.get("rerank_output_k", 10)
    for cfg_key, ui_key in WIDGET_KEYS.items():
        if cfg_key in merged:
            st.session_state[ui_key] = merged[cfg_key]


def _initialize_state() -> None:
    if st.session_state.get("preset_initialized"):
        return
    _apply_preset_to_state(DEFAULT_PRESET)
    st.session_state["preset_initialized"] = True
    st.session_state.setdefault("ui_preset", DEFAULT_PRESET)


def _on_preset_change() -> None:
    preset_name = st.session_state.get("ui_preset", DEFAULT_PRESET)
    _apply_preset_to_state(preset_name)


# --------------------------------------------------------------------------- #
# Pipeline visualization (interactive cards above the chat input)
# --------------------------------------------------------------------------- #


STEP_DEFINITIONS: list[dict[str, Any]] = [
    {"key": "query_rewriting", "title": "Query rewriting", "flag": "query_rewriting_enabled"},
    {"key": "metadata", "title": "Metadata filter", "flag": "metadata_filters_enabled"},
    {"key": "retrieval", "title": "Retrieval", "flag": None, "mode_aware": True},
    {"key": "graph", "title": "Graph expansion", "flag": "graph_expansion_enabled"},
    {"key": "dedup", "title": "Dedup + fuse", "flag": None, "always_on": True},
    {"key": "rerank", "title": "LLM rerank", "flag": "rerank_enabled"},
    {"key": "context", "title": "Context build", "flag": None, "always_on": True},
    {"key": "answer", "title": "Answer + cite", "flag": None, "always_on": True},
]


def _render_pipeline_cards(
    *,
    flags: dict[str, bool],
    retrieval_mode: str,
    result: InteractiveRagResult | None,
) -> None:
    """Render one card per pipeline step. Active steps highlighted, others greyed."""
    columns = st.columns(len(STEP_DEFINITIONS))
    counts = _result_step_counts(result) if result else {}
    timings = _result_step_timings(result) if result else {}
    for column, step in zip(columns, STEP_DEFINITIONS):
        flag = step.get("flag")
        if flag is None:
            active = True
        else:
            active = bool(flags.get(flag, False))
        title = step["title"]
        if step.get("mode_aware") and active:
            title = f"Retrieval ({retrieval_mode})"
        body_lines: list[str] = []
        if active:
            count = counts.get(step["key"])
            if count is not None:
                body_lines.append(f"**{count}** chunk")
            seconds = timings.get(step["key"])
            if seconds is not None and seconds > 0:
                body_lines.append(f"{seconds:.2f}s")
            if not body_lines:
                body_lines.append("ON")
        else:
            body_lines.append("disabled")
        color = "#1f77b4" if active else "#9aa0a6"
        with column:
            st.markdown(
                f"<div style='border:1px solid {color};border-radius:8px;padding:8px 10px;"
                f"background-color:{'#e8f0fe' if active else '#f1f3f4'};text-align:center;'>"
                f"<div style='font-weight:600;color:{color};font-size:0.85rem;'>{title}</div>"
                f"<div style='font-size:0.8rem;color:#202124;margin-top:4px;'>{'<br/>'.join(body_lines)}</div>"
                "</div>",
                unsafe_allow_html=True,
            )


def _result_step_counts(result: InteractiveRagResult) -> dict[str, int]:
    return {
        "query_rewriting": len(result.effective_queries),
        "retrieval": len(result.retrieved),
        "graph": len(result.expanded),
        "dedup": len(set(c.chunk_id for c in [*result.retrieved, *result.expanded])),
        "rerank": len(result.reranked),
        "context": len(result.context_chunks),
        "answer": len(result.citations),
    }


def _result_step_timings(result: InteractiveRagResult) -> dict[str, float]:
    timing = result.timing
    return {
        "query_rewriting": float(timing.query_rewriting_seconds),
        "retrieval": float(timing.retrieval_seconds),
        "graph": float(timing.graph_expansion_seconds),
        "rerank": float(timing.rerank_seconds),
        "context": float(timing.context_seconds),
        "answer": float(timing.answer_seconds),
    }


# --------------------------------------------------------------------------- #
# Result rendering
# --------------------------------------------------------------------------- #


def _chunk_rows(
    chunks: list[RetrievedChunkRecord],
    *,
    rerank_scores: list[int] | None = None,
) -> list[dict[str, Any]]:
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


def _render_chunks(
    title: str,
    chunks: list[RetrievedChunkRecord],
    *,
    rerank_scores: list[int] | None = None,
    empty_message: str = "Nessun chunk in questo step.",
) -> None:
    st.subheader(title)
    if not chunks:
        st.info(empty_message)
        return
    st.dataframe(
        _chunk_rows(chunks, rerank_scores=rerank_scores),
        use_container_width=True,
        hide_index=True,
    )
    for idx, chunk in enumerate(chunks, start=1):
        payload = chunk.payload
        with st.expander(f"{idx}. {chunk.chunk_id}"):
            st.caption(
                f"law_id={payload.get('law_id', '')} | article_id={payload.get('article_id', '')}"
            )
            st.write(chunk.text)


def _render_result(result: InteractiveRagResult) -> None:
    with st.chat_message("user"):
        st.markdown(result.question)
    with st.chat_message("assistant"):
        if result.answer:
            st.markdown(result.answer)
            if result.context_sufficient:
                st.caption(f"context_sufficient={result.context_sufficient}")
            if result.answer_rationale:
                with st.expander("Rationale"):
                    st.write(result.answer_rationale)
        elif result.error:
            st.error(result.error)
        else:
            st.warning("Nessuna risposta generata.")

    tab_queries, tab_retrieved, tab_graph, tab_rerank, tab_context, tab_citations, tab_timing, tab_raw = st.tabs(
        [
            "Multi-query",
            "Retrieval",
            "Graph expansion",
            "Rerank",
            "Context finale",
            "Citazioni",
            "Timing + config",
            "JSON raw",
        ]
    )

    with tab_queries:
        qr = result.query_rewriting
        st.markdown(
            f"**Strategia attiva**: `{qr.get('strategy')}` — "
            f"enabled={qr.get('enabled', False)} | n={qr.get('n')}"
        )
        if qr.get("cache_path"):
            st.caption(
                f"cache: `{qr.get('cache_path')}` | size={qr.get('cache_size', 0)} | "
                f"hits={qr.get('cache_hits', 0)} | misses={qr.get('cache_misses', 0)} | "
                f"failures={qr.get('failures', 0)}"
            )
        if result.effective_queries:
            st.dataframe(
                [{"idx": i, "query": q} for i, q in enumerate(result.effective_queries)],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Nessuna variante generata.")

    with tab_retrieved:
        st.caption(f"retrieval_mode={result.retrieval_mode} | filters={result.metadata_filters}")
        _render_chunks("Chunk recuperati", result.retrieved)

    with tab_graph:
        if result.graph_relations_used:
            st.dataframe(
                [item.to_json_record() for item in result.graph_relations_used],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Nessuna relazione graph usata.")
        _render_chunks(
            "Chunk aggiunti dal grafo",
            result.expanded,
            empty_message="Graph expansion disattivata o nessun edge applicabile.",
        )

    with tab_rerank:
        _render_chunks(
            "Chunk ordinati dal reranker",
            result.reranked,
            rerank_scores=result.rerank_scores,
            empty_message="Rerank disattivato — l'ordine è quello del retrieval.",
        )

    with tab_context:
        _render_chunks("Chunk inclusi nel contesto", result.context_chunks)
        with st.expander("Context text passato al modello"):
            st.text(result.context_text)

    with tab_citations:
        if result.citations:
            st.dataframe(
                [item.to_json_record() for item in result.citations],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Nessuna citazione prodotta.")
        if result.invalid_citation_chunk_ids:
            st.warning(f"Citazioni non valide: {result.invalid_citation_chunk_ids}")

    with tab_timing:
        timing = result.timing
        st.dataframe(
            [
                {"step": "query_rewriting", "seconds": round(timing.query_rewriting_seconds, 3)},
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
        st.json(
            {
                "flags": result.flags,
                "parameters": result.parameters,
                "metadata_filters": result.metadata_filters,
                "retrieval_mode": result.retrieval_mode,
                "hybrid_available": result.hybrid_available,
                "hybrid_unavailable_reason": result.hybrid_unavailable_reason,
                "collection_name": result.collection_name,
            }
        )

    with tab_raw:
        st.json(result.to_json_record())


# --------------------------------------------------------------------------- #
# Main app
# --------------------------------------------------------------------------- #


def main() -> None:
    _initialize_state()

    st.title("Advanced Graph RAG")
    st.caption(
        "Replica interattiva del notebook 06. Default: variante `A4_combined_best` (hybrid + multi-query, "
        "cap=15, rrf=60). Cambia preset per riprodurre le altre varianti ablate in 06b."
    )

    # ----- Sidebar — Runtime (collapsed) -----
    with st.sidebar:
        with st.expander("Runtime — paths + provider", expanded=False):
            evaluation_dir = st.text_input("Evaluation dir", "data/evaluation_clean")
            laws_dir = st.text_input("Laws dir", "data/laws_dataset_clean")
            index_dir = st.text_input("Index dir", "data/indexes/qdrant_server")
            index_manifest_path = st.text_input(
                "Index manifest",
                os.getenv("LEGAL_RAG_INDEX_MANIFEST", "data/indexing_runs/<latest>/index_manifest.json"),
            )
            collection_name = st.text_input("Collection", "legal_chunks_bge_m3")
            output_root = st.text_input("Advanced runs root", "data/rag_runs/advanced")
            simple_summary_path = st.text_input(
                "Simple RAG summary", "data/rag_runs/simple/simple_rag_summary.json"
            )
            retrieval_eval_dir = st.text_input(
                "Retrieval eval runs dir", "data/retrieval_eval_runs"
            )
            env_file = st.text_input("Env file", ".env")
            api_key = st.text_input("API key override", value="", type="password")
            api_url = st.text_input(
                "Chat API URL override", value=os.getenv("UTOPIA_OLLAMA_CHAT_URL", "")
            )
            base_url = st.text_input(
                "Base URL", os.getenv("UTOPIA_BASE_URL", "https://utopia.hpc4ai.unito.it/api")
            )
            chat_model = st.text_input(
                "Chat model", os.getenv("UTOPIA_CHAT_MODEL", DEFAULT_CHAT_MODEL)
            )
            judge_model = st.text_input("Judge / rerank model", os.getenv("UTOPIA_JUDGE_MODEL", ""))
            timeout_seconds = st.number_input(
                "Timeout seconds", min_value=1, max_value=600, value=180
            )
            retry_attempts = st.number_input("Retry attempts", min_value=1, max_value=5, value=1)
            if st.button("Reset cached runtime"):
                st.cache_resource.clear()
                st.rerun()

    # ----- Sidebar — Preset + flags -----
    with st.sidebar:
        st.header("Preset")
        preset_options = [*PRESETS.keys(), CUSTOM_PRESET_LABEL]
        st.selectbox(
            "Configurazione di riferimento",
            options=preset_options,
            key="ui_preset",
            format_func=lambda name: PRESETS[name]["label"] if name in PRESETS else name,
            on_change=_on_preset_change,
        )
        active_preset = st.session_state.get("ui_preset", DEFAULT_PRESET)
        if active_preset in PRESETS:
            st.caption(PRESETS[active_preset]["description"])
        else:
            st.caption("Tutti i parametri sono regolabili a mano sotto.")

        st.header("Pipeline flags")
        st.toggle("Query rewriting (multi-query)", key=WIDGET_KEYS["query_rewriting_enabled"])
        st.toggle("Metadata filters (law_status=current)", key=WIDGET_KEYS["metadata_filters_enabled"])

        # Note: the hybrid toggle stays in sync with runtime capabilities later.
        st.toggle("Hybrid retrieval (dense + sparse)", key=WIDGET_KEYS["hybrid_enabled"])
        st.toggle("Graph expansion (hop=1)", key=WIDGET_KEYS["graph_expansion_enabled"])
        st.toggle("LLM rerank", key=WIDGET_KEYS["rerank_enabled"])

    # Build the resource dict used for the @st.cache_resource runtime.
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

    # Force-disable hybrid toggle if the collection doesn't support it.
    if not health.get("hybrid_available") and st.session_state.get(WIDGET_KEYS["hybrid_enabled"]):
        st.session_state[WIDGET_KEYS["hybrid_enabled"]] = False
        st.warning(
            f"Hybrid disattivato: {health.get('hybrid_unavailable_reason') or 'sparse non disponibile'}."
        )

    # ----- Sidebar — Retrieval params (live editors) -----
    with st.sidebar:
        with st.expander("Retrieval params", expanded=True):
            st.slider(
                "Top-k retrieval", min_value=1, max_value=200, key=WIDGET_KEYS["top_k"]
            )
            st.slider(
                "RRF k", min_value=1, max_value=200, key=WIDGET_KEYS["rrf_k"]
            )
            st.slider(
                "Context cap (max_context_chunks)",
                min_value=1,
                max_value=50,
                key=WIDGET_KEYS["max_context_chunks"],
                help="Numero massimo di chunk inclusi nel contesto. F3 del notebook 06.",
            )
            st.slider(
                "Max context chars",
                min_value=1000,
                max_value=32000,
                step=1000,
                key=WIDGET_KEYS["max_context_chars"],
            )

        with st.expander("Query rewriting", expanded=False):
            st.selectbox(
                "Strategy",
                options=["none", "rewrite", "hyde", "multi_query"],
                key=WIDGET_KEYS["query_rewriting_strategy"],
            )
            st.slider(
                "Variants n (solo multi-query)",
                min_value=1,
                max_value=5,
                key=WIDGET_KEYS["query_rewriting_n"],
            )

        with st.expander("Graph expansion", expanded=False):
            st.slider("Seed k", min_value=1, max_value=20, key=WIDGET_KEYS["graph_expansion_seed_k"])
            st.slider(
                "Max chunk per legge espansa",
                min_value=1,
                max_value=10,
                key=WIDGET_KEYS["max_chunks_per_expanded_law"],
            )
            st.slider(
                "Max chunk espansi totali",
                min_value=1,
                max_value=50,
                key=WIDGET_KEYS["max_expanded_chunks_total"],
            )
            st.slider(
                "Min edge confidence",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                key=WIDGET_KEYS["min_edge_confidence"],
            )

        with st.expander("Rerank", expanded=False):
            st.slider("Rerank input k", min_value=1, max_value=100, key=WIDGET_KEYS["rerank_input_k"])
            st.slider("Rerank output k", min_value=1, max_value=30, key=WIDGET_KEYS["rerank_output_k"])

    # ----- Main — Runtime status + recommended config from 06b -----
    st.subheader("Stato runtime")
    cols = st.columns([2, 3])
    with cols[0]:
        st.json(health, expanded=False)
    with cols[1]:
        rows = _metric_rows("simple", _read_json_if_exists(simple_summary_path))
        latest = _latest_advanced_summary(output_root)
        if latest:
            path, summary = latest
            rows.extend(_metric_rows(path.parent.name, summary))
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("Nessuna metrica salvata trovata sotto i path configurati.")

    # Show the latest 06b recommendation for comparison.
    rec_path, rec_payload = load_recommendation_config(
        Path(retrieval_eval_dir),
        override_path=os.getenv("ADVANCED_RAG_RECOMMENDATION_PATH"),
    )
    if rec_payload:
        with st.expander(f"06b recommendation — `{rec_path.name if rec_path else 'n/a'}`", expanded=False):
            st.caption(
                f"Source: `{rec_path}` | strategy promossa: "
                f"{rec_payload.get('query_rewriting_recommendation', {}).get('strategy', 'n/a')}"
            )
            st.json(rec_payload.get("recommended_advanced_config", {}))

    # ----- Build the per-question run config from widget state -----
    run_config_payload = {
        **resource_data,
        "metadata_filters_enabled": st.session_state[WIDGET_KEYS["metadata_filters_enabled"]],
        "hybrid_enabled": bool(st.session_state[WIDGET_KEYS["hybrid_enabled"]]),
        "graph_expansion_enabled": st.session_state[WIDGET_KEYS["graph_expansion_enabled"]],
        "rerank_enabled": st.session_state[WIDGET_KEYS["rerank_enabled"]],
        "static_filters": (
            {"law_status": "current"}
            if st.session_state[WIDGET_KEYS["metadata_filters_enabled"]]
            else {}
        ),
        "top_k": int(st.session_state[WIDGET_KEYS["top_k"]]),
        "rrf_k": int(st.session_state[WIDGET_KEYS["rrf_k"]]),
        "graph_expansion_seed_k": int(st.session_state[WIDGET_KEYS["graph_expansion_seed_k"]]),
        "max_chunks_per_expanded_law": int(
            st.session_state[WIDGET_KEYS["max_chunks_per_expanded_law"]]
        ),
        "max_expanded_chunks_total": int(st.session_state[WIDGET_KEYS["max_expanded_chunks_total"]]),
        "min_edge_confidence": float(st.session_state[WIDGET_KEYS["min_edge_confidence"]]),
        "rerank_input_k": int(st.session_state[WIDGET_KEYS["rerank_input_k"]]),
        "rerank_output_k": int(st.session_state[WIDGET_KEYS["rerank_output_k"]]),
        "max_context_chunks": int(st.session_state[WIDGET_KEYS["max_context_chunks"]]),
        "max_context_chars": int(st.session_state[WIDGET_KEYS["max_context_chars"]]),
        "query_rewriting_enabled": bool(st.session_state[WIDGET_KEYS["query_rewriting_enabled"]]),
        "query_rewriting_strategy": str(st.session_state[WIDGET_KEYS["query_rewriting_strategy"]]),
        "query_rewriting_n": int(st.session_state[WIDGET_KEYS["query_rewriting_n"]]),
    }
    run_config = InteractiveRagConfig.model_validate(run_config_payload)

    # ----- Pipeline visualization -----
    st.subheader("Flusso di esecuzione")
    last_result: InteractiveRagResult | None = None
    if "last_interactive_result" in st.session_state:
        last_result = InteractiveRagResult.model_validate(st.session_state.last_interactive_result)
    _render_pipeline_cards(
        flags={
            "metadata_filters_enabled": run_config.metadata_filters_enabled,
            "hybrid_enabled": run_config.hybrid_enabled,
            "graph_expansion_enabled": run_config.graph_expansion_enabled,
            "rerank_enabled": run_config.rerank_enabled,
            "query_rewriting_enabled": run_config.query_rewriting_enabled,
        },
        retrieval_mode="hybrid" if run_config.hybrid_enabled else "dense",
        result=last_result,
    )
    st.caption(
        f"Preset attivo: **{active_preset}** | top_k={run_config.top_k} · rrf_k={run_config.rrf_k} · "
        f"cap={run_config.effective_max_context_chunks} · "
        f"qr={run_config.active_query_rewriting_strategy}"
        + (f"·n={run_config.query_rewriting_n}" if run_config.active_query_rewriting_strategy == "multi_query" else "")
    )

    # ----- Chat input + execution -----
    prompt = st.chat_input("Scrivi una domanda giuridica")
    if prompt:
        with st.status("Esecuzione advanced RAG", expanded=True) as status:
            result = runtime.answer_question(prompt, config=run_config, on_step=st.write)
            status.update(
                label="Pipeline completata con errori" if result.error else "Pipeline completata",
                state="error" if result.error else "complete",
            )
        st.session_state.last_interactive_result = result.to_json_record()
        # Refresh so the pipeline cards pick up the new counts on the same page.
        st.rerun()

    if "last_interactive_result" in st.session_state:
        _render_result(
            InteractiveRagResult.model_validate(st.session_state.last_interactive_result)
        )


if __name__ == "__main__":
    main()
