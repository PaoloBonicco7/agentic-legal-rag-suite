"""CLI for the oracle-context evaluation step."""

from __future__ import annotations

import argparse
import json
import os

from .models import DEFAULT_CHAT_MODEL, OracleEvaluationConfig
from .runner import run_oracle_context_evaluation


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the 02b evaluation run."""
    parser = argparse.ArgumentParser(description="Run 02b oracle-context evaluation.")
    parser.add_argument("--evaluation-dir", default="data/evaluation_clean", help="Clean step 02 output directory.")
    parser.add_argument("--laws-dir", default="data/laws_dataset_clean", help="Clean step 01 output directory.")
    parser.add_argument("--output", default="data/evaluation_runs/oracle_context", help="Generated run output directory.")
    parser.add_argument("--env-file", default=".env", help="Optional .env file with Utopia settings.")
    parser.add_argument("--api-url", default=None, help="Explicit chat API URL.")
    parser.add_argument("--base-url", default=os.getenv("UTOPIA_BASE_URL", "https://utopia.hpc4ai.unito.it/api"))
    parser.add_argument("--api-key", default=os.getenv("UTOPIA_API_KEY") or None)
    parser.add_argument("--api-mode", choices=["openai", "ollama"], default=os.getenv("UTOPIA_CHAT_API_MODE", "ollama"))
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--judge-model", default=os.getenv("UTOPIA_JUDGE_MODEL") or None)
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds.")
    parser.add_argument("--start", type=int, default=0, help="Zero-based row offset.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max row count.")
    parser.add_argument("--smoke", action="store_true", help="Run only one row for connectivity checks.")
    parser.add_argument("--retry-attempts", type=int, default=1, help="Remote call attempts per model call.")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="Maximum parallel model calls per run condition.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run oracle-context evaluation and print a compact status payload."""
    args = _parse_args(argv)
    manifest = run_oracle_context_evaluation(
        OracleEvaluationConfig(
            evaluation_dir=args.evaluation_dir,
            laws_dir=args.laws_dir,
            output_dir=args.output,
            env_file=args.env_file,
            api_url=args.api_url,
            api_key=args.api_key,
            base_url=args.base_url,
            api_mode=args.api_mode,
            chat_model=args.chat_model,
            judge_model=args.judge_model,
            timeout_seconds=args.timeout,
            start=args.start,
            limit=args.limit,
            smoke=args.smoke,
            retry_attempts=args.retry_attempts,
            max_concurrency=args.max_concurrency,
        )
    )
    print(
        json.dumps(
            {
                "output_dir": args.output,
                "counts": manifest["counts"],
                "summary_path": f"{args.output}/oracle_context_summary.json",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
