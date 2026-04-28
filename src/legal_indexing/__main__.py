from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .pipeline import run_indexing_pipeline
from .settings import IndexingConfig, make_chunking_profile


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m legal_indexing",
        description="Run the laws_dataset_clean -> Qdrant indexing pipeline.",
    )

    parser.add_argument("--dataset-dir", default="data/laws_dataset_clean")
    parser.add_argument("--qdrant-path", default="data/indexes/qdrant")
    parser.add_argument("--artifacts-root", default="data/qdrant_indexing")

    parser.add_argument("--collection-name", default=None)
    parser.add_argument("--collection-prefix", default="laws_clean")

    parser.add_argument(
        "--embedding-provider",
        default=os.getenv("EMBEDDING_PROVIDER", "utopia"),
        choices=("utopia",),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("UTOPIA_EMBED_MODEL", "SLURM.nomic-embed-text:latest"),
    )
    parser.add_argument(
        "--embedding-api-key",
        default=os.getenv("UTOPIA_API_KEY", ""),
    )
    parser.add_argument(
        "--utopia-base-url",
        default=os.getenv("UTOPIA_BASE_URL", "https://utopia.hpc4ai.unito.it/api"),
    )
    parser.add_argument(
        "--utopia-embed-api-mode",
        default=os.getenv("UTOPIA_EMBED_API_MODE", "auto"),
        choices=("auto", "openai", "ollama"),
    )
    parser.add_argument(
        "--utopia-embed-url",
        default=os.getenv("UTOPIA_EMBED_URL", "https://utopia.hpc4ai.unito.it/ollama/api/embed"),
    )
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-timeout-seconds", type=float, default=60.0)

    parser.add_argument(
        "--chunking-profile",
        default="balanced",
        choices=("balanced", "conservative", "aggressive"),
    )
    parser.add_argument("--min-words-merge", type=int, default=None)
    parser.add_argument("--max-words-split", type=int, default=None)
    parser.add_argument("--overlap-words-split", type=int, default=None)

    parser.add_argument("--subset-limit", type=int, default=None)
    parser.add_argument("--force-reembed", action="store_true")
    parser.add_argument("--no-strict-validation", action="store_true")

    parser.add_argument("--run-id", default=None)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    profile = make_chunking_profile(
        args.chunking_profile,
        min_words_merge=args.min_words_merge,
        max_words_split=args.max_words_split,
        overlap_words_split=args.overlap_words_split,
    )

    config = IndexingConfig(
        dataset_dir=Path(args.dataset_dir),
        qdrant_path=Path(args.qdrant_path),
        artifacts_root=Path(args.artifacts_root),
        collection_name=args.collection_name,
        collection_prefix=args.collection_prefix,
        embedding_provider=args.embedding_provider,
        utopia_base_url=args.utopia_base_url,
        utopia_embed_api_mode=args.utopia_embed_api_mode,
        utopia_embed_url=args.utopia_embed_url,
        embedding_model=args.embedding_model,
        embedding_api_key=args.embedding_api_key,
        embedding_batch_size=args.embedding_batch_size,
        embedding_timeout_seconds=args.embedding_timeout_seconds,
        chunking_profile=profile,
        force_reembed=bool(args.force_reembed),
        subset_limit=args.subset_limit,
        strict_validation=not bool(args.no_strict_validation),
        run_id=args.run_id,
    )

    summary = run_indexing_pipeline(config)
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
