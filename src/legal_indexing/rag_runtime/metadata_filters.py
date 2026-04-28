from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from qdrant_client.http import models as qmodels

from legal_indexing.law_references import LawReferenceResolution

from .config import AdvancedMetadataFilteringConfig, QdrantPayloadFieldMap, ViewFilter
from .qdrant_retrieval import (
    build_article_filter,
    build_law_date_filter,
    build_law_filter,
    build_law_status_filter,
    build_relation_type_filter,
    build_view_filter,
    merge_filters,
)


_YEAR_RE = re.compile(r"\b(18|19|20)\d{2}\b")
_YEAR_RANGE_RE = re.compile(r"\b(?:dal|da)\s+((?:18|19|20)\d{2})\s+(?:al|a)\s+((?:18|19|20)\d{2})\b")
_CANONICAL_ARTICLE_ID_RE = re.compile(r"\blaw:[a-z0-9:_-]+#art:[a-z0-9._:-]+", re.IGNORECASE)
_CANONICAL_LAW_ID_RE = re.compile(r"\blaw:[a-z0-9:_-]+", re.IGNORECASE)
_ARTICLE_LABEL_RE = re.compile(r"\bart(?:icolo)?\.?\s*(\d+[a-z]?)\b", re.IGNORECASE)
_LR_REF_RE = re.compile(
    r"\b(?:l\\.r\\.?|legge\\s+regionale)\\s*(?:n\\.?\\s*)?(\\d+)\\s*/\\s*((?:18|19|20)\\d{2})\b",
    re.IGNORECASE,
)

_RELATION_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("abrog", ("ABROGATED_BY", "ABROGATES")),
    ("modific", ("AMENDS", "MODIFIED_BY")),
    ("sostitu", ("REPLACES", "REPLACED_BY")),
    ("inser", ("INSERTS", "INSERTED_BY")),
)


@dataclass(frozen=True)
class MetadataFilterDecision:
    view: ViewFilter
    law_status: str | None
    law_ids: tuple[str, ...]
    relation_types: tuple[str, ...]
    article_ids: tuple[str, ...]
    year_from: int | None
    year_to: int | None
    applied_heuristics: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "view": self.view,
            "law_status": self.law_status,
            "law_ids": list(self.law_ids),
            "relation_types": list(self.relation_types),
            "article_ids": list(self.article_ids),
            "year_from": self.year_from,
            "year_to": self.year_to,
            "applied_heuristics": list(self.applied_heuristics),
        }


def _infer_temporal(query: str) -> tuple[int | None, int | None, tuple[str, ...]]:
    hints: list[str] = []
    query_low = str(query or "").lower()

    range_match = _YEAR_RANGE_RE.search(query_low)
    if range_match:
        a = int(range_match.group(1))
        b = int(range_match.group(2))
        start = min(a, b)
        end = max(a, b)
        hints.append("year_range_from_query")
        return start, end, tuple(hints)

    years = [int(x.group(0)) for x in _YEAR_RE.finditer(query_low)]
    if years:
        year = years[0]
        hints.append("single_year_from_query")
        return year, year, tuple(hints)
    return None, None, tuple(hints)


def _infer_relation_types(query: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    query_low = str(query or "").lower()
    relations: set[str] = set()
    hints: list[str] = []
    for marker, rels in _RELATION_HINTS:
        if marker in query_low:
            relations.update(rels)
            hints.append(f"relation_{marker}")
    return tuple(sorted(relations)), tuple(hints)


def _infer_view_and_status(query: str) -> tuple[ViewFilter | None, str | None, tuple[str, ...]]:
    query_low = str(query or "").lower()
    hints: list[str] = []

    if any(k in query_low for k in ("vigente", "attuale", "in vigore", "corrente")):
        hints.append("current_view_from_query")
        return "current", "current", tuple(hints)
    if any(k in query_low for k in ("storic", "abrogat", "cessat")):
        hints.append("historical_view_from_query")
        return "historical", None, tuple(hints)
    return None, None, tuple(hints)


def _infer_reference_ids(query: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    query_text = str(query or "")
    law_ids: set[str] = set()
    article_ids: set[str] = set()
    hints: list[str] = []

    for match in _CANONICAL_ARTICLE_ID_RE.finditer(query_text):
        article_id = str(match.group(0)).strip()
        if article_id:
            article_ids.add(article_id)
            law_ids.add(article_id.split("#art:")[0])
            hints.append("article_id_from_query")

    for match in _CANONICAL_LAW_ID_RE.finditer(query_text):
        law_id = str(match.group(0)).strip()
        if law_id:
            law_ids.add(law_id)
            hints.append("law_id_from_query")

    if _ARTICLE_LABEL_RE.search(query_text):
        hints.append("article_label_hint_detected")
    if _LR_REF_RE.search(query_text):
        hints.append("lr_reference_detected_unresolved")

    return tuple(sorted(law_ids)), tuple(sorted(article_ids)), tuple(hints)


def resolve_metadata_filter_decision(
    query: str,
    *,
    config: AdvancedMetadataFilteringConfig,
    default_view: ViewFilter,
    resolved_references: LawReferenceResolution | None = None,
) -> MetadataFilterDecision:
    mode = config.mode
    view: ViewFilter = default_view
    law_status: str | None = None
    law_ids: tuple[str, ...] = tuple()
    relation_types: tuple[str, ...] = tuple()
    article_ids: tuple[str, ...] = tuple()
    year_from: int | None = None
    year_to: int | None = None
    applied_hints: list[str] = []

    if mode in {"explicit_only", "hybrid"}:
        if config.explicit_view is not None:
            view = config.explicit_view
        if config.explicit_law_status:
            law_status = config.explicit_law_status
        if config.explicit_law_ids:
            law_ids = tuple(sorted(config.explicit_law_ids))
        if config.explicit_relation_types:
            relation_types = tuple(sorted(config.explicit_relation_types))
        if config.explicit_article_ids:
            article_ids = tuple(sorted(config.explicit_article_ids))
        if config.explicit_year_from is not None:
            year_from = int(config.explicit_year_from)
        if config.explicit_year_to is not None:
            year_to = int(config.explicit_year_to)

    if mode == "hybrid" and config.enable_heuristics:
        inferred_view, inferred_status, view_hints = _infer_view_and_status(query)
        if inferred_view is not None and config.explicit_view is None:
            view = inferred_view
        if inferred_status is not None and not law_status:
            law_status = inferred_status
        applied_hints.extend(view_hints)

        inferred_relations, relation_hints = _infer_relation_types(query)
        if inferred_relations and not relation_types:
            relation_types = inferred_relations
        applied_hints.extend(relation_hints)

        inferred_law_ids, inferred_article_ids, ref_hints = _infer_reference_ids(query)
        if inferred_law_ids and not law_ids:
            law_ids = inferred_law_ids
        if inferred_article_ids and not article_ids:
            article_ids = inferred_article_ids
        applied_hints.extend(ref_hints)

        if resolved_references is not None:
            if resolved_references.law_ids and not law_ids:
                law_ids = tuple(sorted(resolved_references.law_ids))
                applied_hints.append("law_reference_resolved")
            if resolved_references.article_ids and not article_ids:
                article_ids = tuple(sorted(resolved_references.article_ids))
                applied_hints.append("article_reference_resolved")

        inferred_year_from, inferred_year_to, temporal_hints = _infer_temporal(query)
        if inferred_year_from is not None and year_from is None:
            year_from = inferred_year_from
        if inferred_year_to is not None and year_to is None:
            year_to = inferred_year_to
        applied_hints.extend(temporal_hints)

    if mode == "off":
        law_status = None
        law_ids = tuple()
        relation_types = tuple()
        article_ids = tuple()
        year_from = None
        year_to = None
        applied_hints = []

    return MetadataFilterDecision(
        view=view,
        law_status=law_status,
        law_ids=law_ids,
        relation_types=relation_types,
        article_ids=article_ids,
        year_from=year_from,
        year_to=year_to,
        applied_heuristics=tuple(applied_hints),
    )


def build_metadata_filter(
    field_map: QdrantPayloadFieldMap,
    decision: MetadataFilterDecision,
) -> qmodels.Filter | None:
    out: qmodels.Filter | None = None
    out = merge_filters(out, build_view_filter(field_map, decision.view))
    out = merge_filters(out, build_law_status_filter(field_map, decision.law_status))
    out = merge_filters(out, build_law_filter(field_map, decision.law_ids))
    out = merge_filters(out, build_relation_type_filter(field_map, decision.relation_types))
    out = merge_filters(out, build_article_filter(field_map, decision.article_ids))
    out = merge_filters(
        out,
        build_law_date_filter(
            field_map,
            year_from=decision.year_from,
            year_to=decision.year_to,
        ),
    )
    return out
