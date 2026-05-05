"""CLI for the evaluation reporting step."""

from __future__ import annotations

import argparse
import json

from .models import EvaluationReportingConfig
from .runner import run_evaluation_reporting


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for report generation."""
    parser = argparse.ArgumentParser(description="Run 07 evaluation reporting.")
    parser.add_argument("--no-rag-dir", default="data/baseline_runs/no_rag")
    parser.add_argument("--simple-rag-dir", default="data/rag_runs/simple")
    parser.add_argument("--advanced-rag-dir", default="data/rag_runs/advanced/default")
    parser.add_argument("--evaluation-manifest", default="data/evaluation_clean/evaluation_manifest.json")
    parser.add_argument("--output", default="data/reports")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--max-examples-per-category", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Generate reports and print a compact status payload."""
    args = _parse_args(argv)
    manifest = run_evaluation_reporting(
        EvaluationReportingConfig(
            no_rag_dir=args.no_rag_dir,
            simple_rag_dir=args.simple_rag_dir,
            advanced_rag_dir=args.advanced_rag_dir,
            evaluation_manifest_path=args.evaluation_manifest,
            output_dir=args.output,
            allow_partial=args.allow_partial,
            max_examples_per_category=args.max_examples_per_category,
        )
    )
    print(
        json.dumps(
            {
                "output_dir": args.output,
                "complete": manifest["complete"],
                "quality_issues": manifest["quality_issues"],
                "summary_path": f"{args.output}/comparison_summary.json",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
