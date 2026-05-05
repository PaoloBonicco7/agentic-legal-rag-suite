"""CLI for the no-RAG baseline step."""

from __future__ import annotations

import argparse
import json
import os

from .models import NoRagConfig
from .runner import run_no_rag_baseline


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the no-RAG baseline run."""
    parser = argparse.ArgumentParser(description="Run 04 no-RAG baseline evaluation.")
    parser.add_argument("--evaluation-dir", default="data/evaluation_clean", help="Clean step 02 output directory.")
    parser.add_argument("--output", default="data/baseline_runs/no_rag", help="Generated run output directory.")
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
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="Maximum parallel model calls per run condition.",
    )
    parser.add_argument("--random-seed", type=int, default=None, help="Seed recorded in the run manifest.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run no-RAG baseline and print a compact status payload."""
    args = _parse_args(argv)
    manifest = run_no_rag_baseline(
        NoRagConfig(
            evaluation_dir=args.evaluation_dir,
            output_dir=args.output,
            env_file=args.env_file,
            api_url=args.api_url,
            api_key=args.api_key,
            base_url=args.base_url,
            chat_model=args.chat_model,
            judge_model=args.judge_model,
            timeout_seconds=args.timeout,
            start=args.start,
            benchmark_size=args.benchmark_size,
            smoke=args.smoke,
            retry_attempts=args.retry_attempts,
            max_concurrency=args.max_concurrency,
            random_seed=args.random_seed,
        )
    )
    print(
        json.dumps(
            {
                "output_dir": args.output,
                "counts": manifest["counts"],
                "summary_path": f"{args.output}/no_rag_summary.json",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
