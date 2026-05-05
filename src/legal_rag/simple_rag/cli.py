"""CLI for the simple RAG baseline step."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from .models import SimpleRagConfig
from .runner import run_simple_rag


def _parse_static_filters(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    data = json.loads(value)
    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError("--static-filters must be a JSON object")
    return data


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the simple RAG run."""
    parser = argparse.ArgumentParser(description="Run 05 simple RAG baseline evaluation.")
    parser.add_argument("--evaluation-dir", default="data/evaluation_clean", help="Clean step 02 output directory.")
    parser.add_argument("--index-dir", default="data/indexes/qdrant", help="Qdrant local persistent index directory.")
    parser.add_argument("--index-manifest", default="data/indexing_runs/<latest>/index_manifest.json")
    parser.add_argument("--output", default="data/rag_runs/simple", help="Generated run output directory.")
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
    parser.add_argument("--progress", action="store_true", help="Print progress while rows complete.")
    parser.add_argument("--progress-interval", type=int, default=5, help="Progress print interval in completed rows.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks retrieved from Qdrant.")
    parser.add_argument("--max-context-chunks", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=16000)
    parser.add_argument("--static-filters", default=None, help='Static Qdrant filters as JSON, e.g. {"law_status":"current"}.')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run simple RAG and print a compact status payload."""
    args = _parse_args(argv)
    data = {
        "evaluation_dir": args.evaluation_dir,
        "index_dir": args.index_dir,
        "index_manifest_path": args.index_manifest,
        "output_dir": args.output,
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
        "show_progress": args.progress,
        "progress_interval": args.progress_interval,
        "top_k": args.top_k,
        "max_context_chunks": args.max_context_chunks,
        "max_context_chars": args.max_context_chars,
        "static_filters": _parse_static_filters(args.static_filters),
    }
    if args.collection_name:
        data["collection_name"] = args.collection_name
    manifest = run_simple_rag(SimpleRagConfig.model_validate(data))
    print(
        json.dumps(
            {
                "output_dir": args.output,
                "collection_name": manifest["collection_name"],
                "counts": manifest["counts"],
                "summary_path": f"{args.output}/simple_rag_summary.json",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
