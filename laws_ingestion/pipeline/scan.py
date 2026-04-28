from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


_SIGNAL_PATTERNS: dict[str, re.Pattern[str]] = {
    "abrogat": re.compile(r"abrogat", re.IGNORECASE),
    "cessata_efficacia": re.compile(r"cessat\\w*\\s+efficac", re.IGNORECASE),
    "nota_anchor": re.compile(r'name="nota_', re.IGNORECASE),
    "articolo_anchor": re.compile(r'name="articolo_', re.IGNORECASE),
    "indice": re.compile(r"\\bINDICE\\b", re.IGNORECASE),
    "pipe_symbol": re.compile(r"¦"),
    "ellipsis_brackets": re.compile(r"\\[\\.\\.\\.\\]"),
    "pipe_delimited": re.compile(r"\\|"),
    "table_wrapper": re.compile(r"table-wrapper|table_wrapper", re.IGNORECASE),
    "artt_range": re.compile(r"\\bArtt?\\.\\s*\\d+\\s*\\.?-\\s*\\d+", re.IGNORECASE),
    "lr_typo": re.compile(r"L:R\\.|L\\.R\\.\\.", re.IGNORECASE),
}


@dataclass(frozen=True)
class InventoryRow:
    source_file: str
    abs_path: str
    size_bytes: int
    has_indice: bool
    has_table: bool
    has_abrogat: bool
    has_note_anchor: bool
    has_article_anchor: bool
    has_artt_range: bool
    has_lr_typo: bool


def iter_html_files(html_dir: Path) -> list[Path]:
    html_dir = Path(html_dir)
    return sorted(html_dir.glob("*.html"), key=lambda p: p.name)


def scan_inventory(html_dir: Path) -> tuple[list[dict], dict[str, int]]:
    rows: list[dict] = []
    signal_counts: dict[str, int] = {k: 0 for k in _SIGNAL_PATTERNS}

    for path in iter_html_files(html_dir):
        txt = path.read_text(encoding="utf-8", errors="replace")

        hits = {name: bool(pattern.search(txt)) for name, pattern in _SIGNAL_PATTERNS.items()}
        for name, ok in hits.items():
            if ok:
                signal_counts[name] += 1

        row = InventoryRow(
            source_file=path.name,
            abs_path=str(path.resolve()),
            size_bytes=path.stat().st_size,
            has_indice=hits["indice"],
            has_table=("<table" in txt.lower()) or hits["table_wrapper"],
            has_abrogat=hits["abrogat"],
            has_note_anchor=hits["nota_anchor"],
            has_article_anchor=hits["articolo_anchor"],
            has_artt_range=hits["artt_range"],
            has_lr_typo=hits["lr_typo"],
        )
        rows.append(row.__dict__)

    signal_counts["n_files"] = len(rows)
    return rows, signal_counts
