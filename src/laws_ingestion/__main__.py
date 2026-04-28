from __future__ import annotations

import argparse
import json
from pathlib import Path

from .debug import debug_law
from .export import ingest_and_write
from .qa import qa_artifacts


def _cmd_ingest(args: argparse.Namespace) -> int:
    manifest = ingest_and_write(
        html_dir=Path(args.html_dir),
        out_dir=Path(args.out_dir),
        scope=str(args.scope),
        questions_csv=(Path(args.csv) if str(args.scope) == "benchmark" else None),
        backend=args.backend,
        strict=bool(args.strict),
        max_words=int(args.max_words),
        overlap_words=int(args.overlap_words),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _cmd_qa(args: argparse.Namespace) -> int:
    report = qa_artifacts(out_dir=Path(args.out_dir), suspicious_note_chars=int(args.suspicious_note_chars))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _cmd_debug_law(args: argparse.Namespace) -> int:
    data = debug_law(
        html_dir=Path(args.html_dir),
        law_id=args.law_id,
        source_file=args.source_file,
        backend=args.backend,
        strict=bool(args.strict),
        preview_articles=int(args.preview_articles),
        preview_chars=int(args.preview_chars),
    )
    print(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="laws_ingestion")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Ingest HTML corpus into JSONL RAG-ready dataset")
    p_ing.add_argument("--html-dir", default="data/laws_html")
    p_ing.add_argument("--out-dir", default="data/laws_dataset")
    p_ing.add_argument("--scope", choices=["all", "benchmark"], default="all")
    p_ing.add_argument(
        "--csv",
        default="data/evaluation/questions.csv",
        help="Benchmark questions CSV (required for --scope benchmark)",
    )
    p_ing.add_argument("--backend", choices=["auto", "stdlib", "bs4"], default="auto")
    p_ing.add_argument("--strict", action="store_true")
    p_ing.add_argument("--max-words", type=int, default=600)
    p_ing.add_argument("--overlap-words", type=int, default=80)
    p_ing.set_defaults(func=_cmd_ingest)

    p_qa = sub.add_parser("qa", help="Run basic QA checks on an output directory")
    p_qa.add_argument("--out-dir", default="data/laws_dataset")
    p_qa.add_argument("--suspicious-note-chars", type=int, default=5000)
    p_qa.set_defaults(func=_cmd_qa)

    p_dbg = sub.add_parser("debug-law", help="Debug ingestion for a single law")
    p_dbg.add_argument("--html-dir", default="data/laws_html")
    p_dbg.add_argument("--backend", choices=["auto", "stdlib", "bs4"], default="auto")
    p_dbg.add_argument("--strict", action="store_true")
    p_dbg.add_argument("--law-id", default="")
    p_dbg.add_argument("--source-file", default="")
    p_dbg.add_argument("--preview-articles", type=int, default=10)
    p_dbg.add_argument("--preview-chars", type=int, default=500)
    p_dbg.set_defaults(func=_cmd_debug_law)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
