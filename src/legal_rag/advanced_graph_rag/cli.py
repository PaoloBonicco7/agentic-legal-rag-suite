"""CLI for the advanced graph-aware RAG step."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from .models import AdvancedRagConfig
from .runner import run_advanced_graph_rag


def _parse_json_object(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    data = json.loads(value)
    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return data


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the advanced graph-aware RAG run."""
    parser = argparse.ArgumentParser(description="Run 06 advanced graph-aware RAG evaluation.")
    parser.add_argument("--evaluation-dir", default="data/evaluation_clean", help="Clean step 02 output directory.")
    parser.add_argument("--laws-dir", default="data/laws_dataset_clean", help="Clean step 01 output directory.")
    parser.add_argument("--index-dir", default="data/indexes/qdrant", help="Qdrant local persistent index directory.")
    parser.add_argument("--index-manifest", default="data/indexing_runs/<latest>/index_manifest.json")
    parser.add_argument("--simple-rag-manifest", default="data/rag_runs/simple/simple_rag_manifest.json")
    parser.add_argument("--output-root", default="data/rag_runs/advanced")
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--collection-name", default=None, help="Override Qdrant collection name.")
    parser.add_argument("--env-file", default=".env", help="Optional .env file with Utopia settings.")
    parser.add_argument("--api-url", default=os.getenv("UTOPIA_OLLAMA_CHAT_URL") or None, help="Explicit chat API URL.")
    parser.add_argument("--base-url", default=os.getenv("UTOPIA_BASE_URL", "https://utopia.hpc4ai.unito.it/api"))
    parser.add_argument("--api-key", default=os.getenv("UTOPIA_API_KEY") or None)
    parser.add_argument("--chat-model", default=os.getenv("UTOPIA_CHAT_MODEL", "SLURM.gpt-oss:120b"))
    parser.add_argument("--judge-model", default=os.getenv("UTOPIA_JUDGE_MODEL") or None)
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds.")
    parser.add_argument("--start", type=int, default=0, help="Zero-based row offset.")
    parser.add_argument("--benchmark-size", type=int, default=None, help="Optional max row count.")
    parser.add_argument("--smoke", action="store_true", help="Run only one row for connectivity checks.")
    parser.add_argument("--retry-attempts", type=int, default=1, help="Remote call attempts per model call.")
    parser.add_argument("--max-concurrency", type=int, default=4, help="Maximum parallel calls per run condition.")
    parser.add_argument("--no-metadata-filters", action="store_true", help="Disable static metadata filters.")
    parser.add_argument("--no-hybrid", action="store_true", help="Use dense-only retrieval.")
    parser.add_argument("--no-graph-expansion", action="store_true", help="Disable explicit graph expansion.")
    parser.add_argument("--no-rerank", action="store_true", help="Disable LLM reranking.")
    parser.add_argument("--static-filters", default='{"law_status":"current"}', help="Static Qdrant filters as JSON.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k candidates per retrieval mode.")
    parser.add_argument("--rrf-k", type=int, default=60, help="Qdrant RRF k parameter.")
    parser.add_argument("--graph-expansion-seed-k", type=int, default=5)
    parser.add_argument("--graph-expansion-relation-types", default=None, help="Comma-separated relation types.")
    parser.add_argument("--max-chunks-per-expanded-law", type=int, default=2)
    parser.add_argument("--rerank-input-k", type=int, default=20)
    parser.add_argument("--rerank-output-k", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=16000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run advanced graph-aware RAG and print a compact status payload."""
    args = _parse_args(argv)
    data: dict[str, Any] = {
        "evaluation_dir": args.evaluation_dir,
        "laws_dir": args.laws_dir,
        "index_dir": args.index_dir,
        "index_manifest_path": args.index_manifest,
        "simple_rag_manifest_path": args.simple_rag_manifest,
        "output_root": args.output_root,
        "run_name": args.run_name,
        "env_file": args.env_file,
        "api_url": args.api_url,
        "api_key": args.api_key,
        "base_url": args.base_url,
        "chat_model": args.chat_model,
        "judge_model": args.judge_model,
        "timeout_seconds": args.timeout,
        "start": args.start,
        "benchmark_size": args.benchmark_size,
        "smoke": args.smoke,
        "retry_attempts": args.retry_attempts,
        "max_concurrency": args.max_concurrency,
        "metadata_filters_enabled": not args.no_metadata_filters,
        "hybrid_enabled": not args.no_hybrid,
        "graph_expansion_enabled": not args.no_graph_expansion,
        "rerank_enabled": not args.no_rerank,
        "static_filters": _parse_json_object(args.static_filters),
        "top_k": args.top_k,
        "rrf_k": args.rrf_k,
        "graph_expansion_seed_k": args.graph_expansion_seed_k,
        "max_chunks_per_expanded_law": args.max_chunks_per_expanded_law,
        "rerank_input_k": args.rerank_input_k,
        "rerank_output_k": args.rerank_output_k,
        "max_context_chars": args.max_context_chars,
    }
    if args.collection_name:
        data["collection_name"] = args.collection_name
    if args.graph_expansion_relation_types:
        data["graph_expansion_relation_types"] = [
            item.strip()
            for item in args.graph_expansion_relation_types.split(",")
            if item.strip()
        ]
    manifest = run_advanced_graph_rag(AdvancedRagConfig.model_validate(data))
    print(
        json.dumps(
            {
                "output_dir": manifest["config"]["output_dir"],
                "collection_name": manifest["collection_name"],
                "counts": manifest["counts"],
                "summary_path": f"{manifest['config']['output_dir']}/advanced_rag_summary.json",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
