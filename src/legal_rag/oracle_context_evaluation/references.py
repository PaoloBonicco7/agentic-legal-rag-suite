"""Resolve evaluation source-of-truth references to clean legal articles."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from legal_rag.laws_preprocessing.common import normalize_article_label, normalize_ws
from legal_rag.laws_preprocessing.inventory import law_id_from_date_number, parse_italian_date

from .io import read_jsonl, sha256_text
from .models import OracleContextRecord, ResolvedReference

REFERENCE_RE = re.compile(
    r"^\s*(?:Legge\s+regionale|L[\.:]\s*R\.?\.?)\s+"
    r"(?P<day>\d{1,2})(?:\s*°)?\s+(?P<month>[a-zà]+)\s+(?P<year>\d{4}),\s*"
    r"n\.?\s*(?P<law_number>\d+)\s*-\s*"
    r"(?:Articolo|Art)\.?\s*(?P<article_label>[\w\s]+?)\s*$",
    re.IGNORECASE,
)


def split_reference_values(values: list[str]) -> list[str]:
    """Split dataset reference fields on pipes while preserving source order."""
    out: list[str] = []
    for value in values:
        for part in str(value or "").split("|"):
            text = normalize_ws(part)
            if text:
                out.append(text)
    return out


def parse_reference(text: str) -> tuple[str, str]:
    """Parse one dataset reference into canonical law ID and article label."""
    match = REFERENCE_RE.match(text or "")
    if not match:
        raise ValueError(f"Cannot parse reference: {text!r}")
    law_date = parse_italian_date(int(match.group("day")), match.group("month"), int(match.group("year")))
    law_id = law_id_from_date_number(law_date, int(match.group("law_number")))
    article_label_norm = normalize_article_label(match.group("article_label"))
    if not article_label_norm:
        raise ValueError(f"Cannot parse article label from reference: {text!r}")
    return law_id, article_label_norm


class OracleReferenceResolver:
    """Resolve dataset references against clean law and article JSONL files."""

    def __init__(self, laws: list[dict[str, Any]], articles: list[dict[str, Any]]) -> None:
        self._laws_by_id = {str(row["law_id"]): row for row in laws}
        self._articles_by_key = {
            (str(row["law_id"]), str(row["article_label_norm"])): row for row in articles
        }

    @classmethod
    def from_dir(cls, laws_dir: str | Path) -> "OracleReferenceResolver":
        """Load clean law and article records from a step 01 output directory."""
        root = Path(laws_dir)
        return cls(
            laws=read_jsonl(root / "laws.jsonl"),
            articles=read_jsonl(root / "articles.jsonl"),
        )

    def resolve_reference(self, reference_text: str) -> ResolvedReference:
        """Resolve one human-readable dataset reference to one article."""
        law_id, article_label_norm = parse_reference(reference_text)
        law = self._laws_by_id.get(law_id)
        if law is None:
            raise LookupError(f"Law not found for reference {reference_text!r}: {law_id}")
        article = self._articles_by_key.get((law_id, article_label_norm))
        if article is None:
            raise LookupError(
                f"Article not found for reference {reference_text!r}: {law_id} article {article_label_norm}"
            )
        return ResolvedReference(
            reference_text=reference_text,
            law_id=law_id,
            law_title=str(law.get("law_title") or ""),
            article_id=str(article.get("article_id") or ""),
            article_label_norm=article_label_norm,
            article_text=str(article.get("article_text") or ""),
        )

    def build_context_record(self, record: dict[str, Any]) -> OracleContextRecord:
        """Build the oracle context row for one evaluation record."""
        expected_references = split_reference_values([str(x) for x in record.get("expected_references", [])])
        resolved: list[ResolvedReference] = []
        error: str | None = None
        for reference_text in expected_references:
            try:
                resolved.append(self.resolve_reference(reference_text))
            except Exception as exc:
                error = f"context_resolution_error: {type(exc).__name__}: {exc}"
                break

        context_text = build_context_text(resolved) if error is None else ""
        context_article_ids = [item.article_id for item in resolved] if error is None else []
        return OracleContextRecord(
            qid=str(record.get("qid") or ""),
            level=str(record.get("level") or "UNKNOWN"),
            expected_references=expected_references,
            resolved_references=resolved if error is None else [],
            context_article_ids=context_article_ids,
            context_text=context_text,
            context_hash=sha256_text(context_text) if context_text else None,
            error=error,
        )


def build_context_text(resolved_references: list[ResolvedReference]) -> str:
    """Render minimal oracle article context for the answer model."""
    blocks: list[str] = []
    for index, ref in enumerate(resolved_references, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[{index}] {ref.law_title}",
                    f"Article: {ref.article_label_norm}",
                    ref.article_text,
                ]
            ).strip()
        )
    return "\n\n---\n\n".join(blocks)
