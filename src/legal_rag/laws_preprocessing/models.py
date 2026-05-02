"""Typed contracts for the clean dataset produced by phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "laws-preprocessing-v1"

ALLOWED_LAW_STATUSES = {"current", "past", "unknown", "index_or_empty"}
ALLOWED_RELATION_TYPES = {
    "REFERENCES",
    "ABROGATED_BY",
    "ABROGATES",
    "MODIFIED_BY",
    "AMENDS",
    "REPLACED_BY",
    "REPLACES",
    "INSERTED_BY",
    "INSERTS",
}

REQUIRED_CHUNK_FIELDS = {
    "chunk_id",
    "passage_id",
    "article_id",
    "law_id",
    "text",
    "text_for_embedding",
    "law_date",
    "law_number",
    "law_title",
    "law_status",
    "article_status",
    "article_label_norm",
    "passage_label",
    "structure_path",
    "source_file",
    "index_views",
    "related_law_ids",
    "inbound_law_ids",
    "outbound_law_ids",
    "relation_types",
}

LIST_CHUNK_FIELDS = {
    "index_views",
    "related_law_ids",
    "inbound_law_ids",
    "outbound_law_ids",
    "relation_types",
}

LawStatus = Literal["current", "past", "unknown", "index_or_empty"]
ArticleStatus = Literal["current", "past", "unknown"]
RelationType = Literal[
    "REFERENCES",
    "ABROGATED_BY",
    "ABROGATES",
    "MODIFIED_BY",
    "AMENDS",
    "REPLACED_BY",
    "REPLACES",
    "INSERTED_BY",
    "INSERTS",
]


class LawsPreprocessingConfig(BaseModel):
    """Runtime configuration for the deterministic preprocessing run."""

    source_dir: str = "data/laws_html"
    output_dir: str = "data/laws_dataset_clean"
    chunk_size: int = Field(default=600, gt=0)
    chunk_overlap: int = Field(default=80, ge=0)
    strict: bool = False

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_smaller_than_chunk_size(cls, value: int, info: Any) -> int:
        """Prevent invalid chunk windows before any corpus work starts."""
        chunk_size = (info.data or {}).get("chunk_size", 600)
        if value >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return value


@dataclass(frozen=True)
class Anchor:
    """Named HTML anchor extracted from a source block."""

    name: str
    text: str


@dataclass(frozen=True)
class Link:
    """HTML hyperlink extracted from a source block."""

    href: str
    text: str


@dataclass(frozen=True)
class Block:
    """Ordered HTML text block enriched with anchors and links."""

    kind: str
    text: str
    anchors: tuple[Anchor, ...]
    links: tuple[Link, ...]


@dataclass(frozen=True)
class LawFile:
    """Validated source file with the canonical law identity."""

    path: Path
    source_file: str
    law_date: date
    law_year: int
    law_number: int
    law_id: str


@dataclass
class CorpusRegistry:
    """Lookup indexes used to resolve explicit references between laws."""

    by_law_id: dict[str, LawFile]
    by_date_number: dict[tuple[date, int], str]
    by_year_number: dict[tuple[int, int], set[str]]

    def resolve_law_id(self, law_date: date, law_number: int) -> str | None:
        """Resolve an exact date-number citation to a law ID."""
        return self.by_date_number.get((law_date, int(law_number)))

    def resolve_law_id_year(self, law_year: int, law_number: int) -> str | None:
        """Resolve a year-number citation only when it is unambiguous."""
        hits = self.by_year_number.get((int(law_year), int(law_number))) or set()
        if len(hits) == 1:
            return next(iter(hits))
        return None


@dataclass(frozen=True)
class ResolvedLawRef:
    """Explicit law reference after successful registry resolution."""

    law_id: str
    extraction_method: str
    raw: str
    law_key_raw: str


@dataclass(frozen=True)
class Line:
    """Article or note line with the links observed in the source HTML."""

    text: str
    links: tuple[Link, ...]


@dataclass
class IngestedLaw:
    """All clean records generated from one source HTML law."""

    law: dict[str, Any]
    articles: list[dict[str, Any]]
    passages: list[dict[str, Any]]
    notes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    chunks: list[dict[str, Any]]
    unresolved_refs: int
    warnings: list[str]


class _Record(BaseModel):
    """Strict Pydantic base model for exported JSONL records."""

    model_config = ConfigDict(extra="forbid")

    def to_json_record(self) -> dict[str, Any]:
        """Serialize records with JSON-compatible values and explicit nulls."""
        return self.model_dump(mode="json", exclude_none=False)


class LawRecord(_Record):
    """One cleaned law-level record."""

    law_id: str
    law_type: str
    law_date: str
    law_number: int
    law_title: str
    law_status: LawStatus
    status_confidence: float
    status_evidence: list[dict[str, str]]
    source_file: str
    preamble_text: str
    links_out: list[dict[str, str]]


class ArticleRecord(_Record):
    """One cleaned article-level record."""

    article_id: str
    law_id: str
    article_label_raw: str
    article_label_norm: str
    anchor_name: str | None
    structure_path: str
    article_heading: str | None
    article_text: str
    article_status: ArticleStatus
    note_anchor_names: list[str]
    is_abrogated: bool
    abrogated_by: dict[str, Any] | None
    amended_by_law_ids: list[str]
    links_out: list[dict[str, str]]


class PassageRecord(_Record):
    """One passage extracted from an article."""

    passage_id: str
    article_id: str
    law_id: str
    passage_label: str
    passage_kind: str
    passage_text: str
    structure_path: str
    note_anchor_names: list[str]
    links_out: list[dict[str, str]]
    related_law_ids: list[str]
    relation_types: list[RelationType]


class NoteRecord(_Record):
    """One source note connected back to articles and passages."""

    note_id: str
    law_id: str
    note_anchor_name: str
    note_number: str | None
    note_kind: str
    note_text: str
    linked_article_ids: list[str]
    linked_passage_ids: list[str]
    links_out: list[dict[str, str]]


class EdgeRecord(_Record):
    """One explicit legal relation between two laws."""

    edge_id: str
    relation_type: RelationType
    src_law_id: str
    src_article_id: str | None
    src_passage_id: str | None
    dst_law_id: str
    dst_article_label_norm: str | None
    context: str
    extraction_method: str
    evidence: str
    evidence_text: str
    confidence: float
    source_file: str
    note_anchor_name: str | None
    is_self_loop: bool


class ChunkRecord(_Record):
    """One deterministic RAG chunk with indexing metadata."""

    chunk_id: str
    passage_id: str
    article_id: str
    law_id: str
    chunk_seq: int
    text: str
    text_for_embedding: str
    law_date: str
    law_number: int
    law_title: str
    law_status: LawStatus
    article_status: ArticleStatus
    article_label_norm: str
    passage_label: str
    structure_path: str
    source_file: str
    index_views: list[str]
    related_law_ids: list[str]
    inbound_law_ids: list[str]
    outbound_law_ids: list[str]
    relation_types: list[RelationType]


def law_record(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and serialize a law record."""
    return LawRecord.model_validate(data).to_json_record()


def article_record(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and serialize an article record."""
    return ArticleRecord.model_validate(data).to_json_record()


def passage_record(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and serialize a passage record."""
    return PassageRecord.model_validate(data).to_json_record()


def note_record(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and serialize a note record."""
    return NoteRecord.model_validate(data).to_json_record()


def edge_record(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and serialize an edge record."""
    return EdgeRecord.model_validate(data).to_json_record()


def chunk_record(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and serialize a chunk record."""
    return ChunkRecord.model_validate(data).to_json_record()
