"""Retrieval-only sanity check for the eval-0021 paradigmatic case.

eval-0021 ("Che cos'è il GAP?") regressed from Simple to advanced_lean_v1:
the definitional chunk (`vda:lr:2015-06-15:14#art:2#p:c1.lit_b#chunk:0`) is
retrieved at rank 12 in the advanced pipeline, but the answer prompt only
includes the top 10 → the LLM cannot answer. This script reproduces the
retrieval phase (no LLM answer, no judge) across the 5 ablation variants
and reports whether the fix candidates promote the chunk inside the context.

Run:
    .venv/bin/python scripts/retrieval_sanity_eval_0021.py

Requires Qdrant server running on 127.0.0.1:6333 and the multi-query cache
populated under data/cache/query_rewriting/ (it is, after advanced_lean_v1).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_rag.advanced_graph_rag import (  # noqa: E402
    AdvancedRagConfig,
    connect_qdrant,
    load_index_manifest,
    resolve_collection_name,
)
from legal_rag.advanced_graph_rag.retrieval import GraphIndex  # noqa: E402
from legal_rag.advanced_graph_rag.runner import (  # noqa: E402
    QueryRewriteStats,
    build_advanced_query_embedder,
    build_context,
    build_query_rewrite_cache,
    retrieve_candidates,
)
from legal_rag.oracle_context_evaluation.io import read_jsonl  # noqa: E402
from legal_rag.oracle_context_evaluation.references import OracleReferenceResolver  # noqa: E402


QID = "eval-0021"
DEFINITIONAL_CHUNK = "vda:lr:2015-06-15:14#art:2#p:c1.lit_b#chunk:0"


class _RaisingClient:
    """Raises on every call. Multi-query/rewrite must work via cache only."""

    def structured_chat(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("LLM call attempted: sanity script expects cache hits only")


def _base_config() -> AdvancedRagConfig:
    return AdvancedRagConfig(
        evaluation_dir=str(ROOT / "data/evaluation_clean"),
        laws_dir=str(ROOT / "data/laws_dataset_clean"),
        index_dir=str(ROOT / "data/indexes/qdrant_server"),
        index_manifest_path=os.environ.get(
            "LEGAL_RAG_INDEX_MANIFEST",
            str(ROOT / "data/indexing_runs/<latest>/index_manifest.json"),
        ),
        collection_name="legal_chunks_bge_m3",
        run_name="sanity_eval_0021",
        env_file=str(ROOT / ".env"),
        hybrid_enabled=True,
        graph_expansion_enabled=False,
        rerank_enabled=False,
        metadata_filters_enabled=False,
        static_filters={},
        max_concurrency=1,
        parallel_datasets_enabled=False,
        timeout_seconds=60,
        retry_attempts=1,
    )


def _variants(base: AdvancedRagConfig) -> dict[str, AdvancedRagConfig]:
    common = base.model_dump(exclude={"run_name"})
    return {
        "advanced_lean_v1 (regression)": AdvancedRagConfig(
            **{**common, "run_name": "regression",
               "top_k": 100, "rrf_k": 30, "max_context_chunks": 10,
               "query_rewriting_enabled": True, "query_rewriting_strategy": "multi_query",
               "query_rewriting_n": 3, "rerank_output_k": 10},
        ),
        "A0_simple_equiv": AdvancedRagConfig(
            **{**common, "run_name": "a0",
               "top_k": 10, "rrf_k": 60, "max_context_chunks": 3, "rerank_output_k": 10,
               "query_rewriting_enabled": False, "query_rewriting_strategy": "none"},
        ),
        "A1_hybrid_top20_no_rewrite": AdvancedRagConfig(
            **{**common, "run_name": "a1",
               "top_k": 20, "rrf_k": 60, "max_context_chunks": 5, "rerank_output_k": 10,
               "query_rewriting_enabled": False, "query_rewriting_strategy": "none"},
        ),
        "A2_single_rewrite": AdvancedRagConfig(
            **{**common, "run_name": "a2",
               "top_k": 100, "rrf_k": 60, "max_context_chunks": 5, "rerank_output_k": 10,
               "query_rewriting_enabled": True, "query_rewriting_strategy": "rewrite",
               "query_rewriting_n": 1},
        ),
        "A3_multi_with_original_rrf": AdvancedRagConfig(
            **{**common, "run_name": "a3",
               "top_k": 100, "rrf_k": 60, "max_context_chunks": 5, "rerank_output_k": 10,
               "query_rewriting_enabled": True, "query_rewriting_strategy": "multi_query",
               "query_rewriting_n": 3},
        ),
        "A4_combined_cap15": AdvancedRagConfig(
            **{**common, "run_name": "a4",
               "top_k": 100, "rrf_k": 60, "max_context_chunks": 15, "rerank_output_k": 10,
               "query_rewriting_enabled": True, "query_rewriting_strategy": "multi_query",
               "query_rewriting_n": 3},
        ),
    }


def main() -> int:
    # Load eval-0021 from both datasets
    no_hint_rows = read_jsonl(ROOT / "data/evaluation_clean/questions_no_hint.jsonl")
    record = next((r for r in no_hint_rows if r["qid"] == QID), None)
    if record is None:
        print(f"ERROR: {QID} not found in questions_no_hint.jsonl", file=sys.stderr)
        return 1

    base = _base_config()
    manifest_path, manifest = load_index_manifest(base)
    base = base.model_copy(update={"index_manifest_path": str(manifest_path)})
    collection = resolve_collection_name(base, manifest)
    qdrant = connect_qdrant(base, manifest)
    embedder = build_advanced_query_embedder(base, manifest)
    graph = GraphIndex.from_dir(base.laws_dir)
    resolver = OracleReferenceResolver.from_dir(base.laws_dir)  # noqa: F841  (kept for parity)

    print(f"Question: {record['question']!r}")
    print(f"Correct: {record['correct_answer']!r}")
    print(f"Definitional chunk: {DEFINITIONAL_CHUNK}")
    print(f"Collection: {collection} | rows in manifest: {manifest.get('chunk_count')}")
    print()

    print(f"{'variant':40s} | {'top_k':>5s} | {'cap':>3s} | {'queries':>7s} | {'def_rank':>8s} | def∈ctx")
    print("-" * 90)

    results: list[dict[str, Any]] = []
    for label, cfg in _variants(base).items():
        rewrite_cache = build_query_rewrite_cache(cfg)
        stats = QueryRewriteStats()
        try:
            trace = retrieve_candidates(
                record=record,
                question_key="question",
                llm_client=_RaisingClient(),
                qdrant_client=qdrant,
                embedder=embedder,
                collection_name=collection,
                index_manifest=manifest,
                graph=graph,
                config=cfg,
                query_rewrite_cache=rewrite_cache,
                query_rewrite_stats=stats,
            )
        except Exception as exc:
            print(f"{label:40s} | ERROR: {type(exc).__name__}: {exc}")
            continue

        retrieved_ids = [chunk.chunk_id for chunk in trace.retrieved]
        try:
            def_rank = retrieved_ids.index(DEFINITIONAL_CHUNK) + 1
            def_rank_str = str(def_rank)
        except ValueError:
            def_rank = None
            def_rank_str = "MISS"

        context_chunks, _ = build_context(
            trace.reranked,
            max_context_chunks=cfg.effective_max_context_chunks,
            max_context_chars=cfg.max_context_chars,
        )
        context_ids = [chunk.chunk_id for chunk in context_chunks]
        in_context = DEFINITIONAL_CHUNK in context_ids

        n_queries = stats.cache_hits + stats.cache_misses if cfg.query_rewriting_enabled else 1
        cache_size_label = "+orig" if cfg.query_rewriting_enabled else "-"
        n_queries_label = f"{n_queries}{cache_size_label}"

        results.append({
            "variant": label,
            "top_k": cfg.top_k,
            "max_context_chunks": cfg.effective_max_context_chunks,
            "query_count": n_queries,
            "definitional_rank_in_retrieved": def_rank,
            "definitional_in_context": in_context,
            "retrieved_count": len(retrieved_ids),
            "context_count": len(context_ids),
            "context_chunk_ids_top": context_ids,
        })

        print(
            f"{label:40s} | "
            f"{cfg.top_k:>5d} | "
            f"{cfg.effective_max_context_chunks:>3d} | "
            f"{n_queries_label:>7s} | "
            f"{def_rank_str:>8s} | "
            f"{'YES' if in_context else 'NO':>3s}"
        )

    print()
    print("Verdict:")
    regression_row = next(r for r in results if r["variant"].startswith("advanced_lean_v1"))
    for row in results:
        if row["variant"] == regression_row["variant"]:
            continue
        if row["definitional_in_context"] and not regression_row["definitional_in_context"]:
            print(f"  [PROMOTE] {row['variant']}: definitional chunk enters context "
                  f"(rank {row['definitional_rank_in_retrieved']})")
        elif row["definitional_in_context"] == regression_row["definitional_in_context"]:
            print(f"  [PARITY]  {row['variant']}: same outcome as regression baseline")
        else:
            print(f"  [REGRESS] {row['variant']}: lost the definitional chunk")

    out_path = ROOT / "data/retrieval_eval_runs/_sanity_eval_0021.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"qid": QID, "results": results}, indent=2, ensure_ascii=False))
    print(f"\nReport saved: {out_path}")

    close = getattr(qdrant, "close", None)
    if callable(close):
        close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
