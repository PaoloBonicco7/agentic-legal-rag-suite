from __future__ import annotations

import argparse
import json

from .export import run_laws_preprocessing
from .models import LawsPreprocessingConfig


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the minimal configuration needed to run the phase 1 pipeline."""
    parser = argparse.ArgumentParser(description="Build the clean legal dataset from HTML laws.")
    parser.add_argument("--source", default="data/laws_html", help="Source HTML corpus directory.")
    parser.add_argument("--output", default="data/laws_dataset_clean", help="Generated clean dataset directory.")
    parser.add_argument("--chunk-size", type=int, default=600, help="Maximum words per chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=80, help="Overlapping words between long chunks.")
    parser.add_argument("--strict", action="store_true", help="Fail on unexpected unattached content.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run preprocessing from CLI arguments and print a compact status payload."""
    args = _parse_args(argv)
    manifest = run_laws_preprocessing(
        LawsPreprocessingConfig(
            source_dir=args.source,
            output_dir=args.output,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            strict=args.strict,
        )
    )
    print(json.dumps({"output_dir": args.output, "ready_for_indexing": manifest["ready_for_indexing"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
