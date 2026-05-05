"""Command line interface for the Qdrant indexing contract."""

from __future__ import annotations

import argparse
import json

from .models import IndexingConfig
from .pipeline import run_indexing_pipeline


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Qdrant index for the clean legal dataset.")
    parser.add_argument("--clean-dataset-dir", default="data/laws_dataset_clean")
    parser.add_argument("--index-dir", default="data/indexes/qdrant")
    parser.add_argument("--runs-dir", default="data/indexing_runs")
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--collection-name", default="legal_chunks")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--chunk-selection-mode", default="full", choices=("full", "sample"))
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--embedding-backend", default="local", choices=("local", "utopia"))
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--disable-hybrid", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--upload-batch-size", type=int, default=64)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--non-strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run indexing from CLI arguments and print a compact status payload."""
    args = _parse_args(argv)
    data = {
        "clean_dataset_dir": args.clean_dataset_dir,
        "index_dir": args.index_dir,
        "runs_dir": args.runs_dir,
        "collection_name": args.collection_name,
        "force_rebuild": args.force_rebuild,
        "chunk_selection_mode": args.chunk_selection_mode,
        "sample_size": args.sample_size,
        "run_id": args.run_id,
        "embedding_backend": args.embedding_backend,
        "embedding_dim": args.embedding_dim,
        "hybrid_enabled": not args.disable_hybrid,
        "batch_size": args.batch_size,
        "upload_batch_size": args.upload_batch_size,
        "env_file": args.env_file,
        "strict": not args.non_strict,
    }
    if args.qdrant_url:
        data["qdrant_url"] = args.qdrant_url
    if args.embedding_model:
        data["embedding_model"] = args.embedding_model
    manifest = run_indexing_pipeline(IndexingConfig.model_validate(data))
    print(
        json.dumps(
            {
                "ready_for_retrieval": manifest["ready_for_retrieval"],
                "collection_name": manifest["collection_name"],
                "indexed_count": manifest["indexed_count"],
                "run_id": manifest["run_id"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
