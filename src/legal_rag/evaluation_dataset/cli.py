from __future__ import annotations

import argparse
import json

from .export import run_evaluation_dataset
from .models import EvaluationDatasetConfig


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the minimal configuration needed to build the clean evaluation dataset."""
    parser = argparse.ArgumentParser(description="Build the clean evaluation dataset from source CSV files.")
    parser.add_argument("--mcq-source", default="data/evaluation/questions.csv", help="Source MCQ CSV file.")
    parser.add_argument(
        "--no-hint-source",
        default="data/evaluation/questions_no_hint.csv",
        help="Source no-hint CSV file.",
    )
    parser.add_argument("--output", default="data/evaluation_clean", help="Generated clean evaluation directory.")
    parser.add_argument("--expected-records", type=int, default=100, help="Expected number of valid question pairs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run evaluation dataset normalization and print a compact status payload."""
    args = _parse_args(argv)
    manifest = run_evaluation_dataset(
        EvaluationDatasetConfig(
            mcq_source=args.mcq_source,
            no_hint_source=args.no_hint_source,
            output_dir=args.output,
            expected_records=args.expected_records,
        )
    )
    print(
        json.dumps(
            {"output_dir": args.output, "ready_for_evaluation": manifest["ready_for_evaluation"]},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
