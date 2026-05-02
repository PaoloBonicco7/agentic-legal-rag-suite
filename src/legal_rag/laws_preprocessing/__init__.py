"""Public API for phase 1 laws preprocessing.

This package contains only the code used to transform the raw legal HTML corpus
into the deterministic clean dataset required by the later indexing phases.
"""

from __future__ import annotations

from .common import normalize_article_label, normalize_month_name, normalize_ws
from .export import run_laws_preprocessing
from .html import parse_blocks_from_html
from .ingest import classify_law_status, ingest_law
from .inventory import (
    build_corpus_registry,
    compute_source_hash,
    law_id_from_date_number,
    parse_italian_date,
    parse_law_filename,
    sha256_file,
)
from .models import (
    ALLOWED_LAW_STATUSES,
    ALLOWED_RELATION_TYPES,
    LIST_CHUNK_FIELDS,
    REQUIRED_CHUNK_FIELDS,
    SCHEMA_VERSION,
    Anchor,
    Block,
    CorpusRegistry,
    IngestedLaw,
    LawFile,
    LawsPreprocessingConfig,
    Line,
    Link,
    ResolvedLawRef,
)

__all__ = [
    "ALLOWED_LAW_STATUSES",
    "ALLOWED_RELATION_TYPES",
    "LIST_CHUNK_FIELDS",
    "REQUIRED_CHUNK_FIELDS",
    "SCHEMA_VERSION",
    "Anchor",
    "Block",
    "CorpusRegistry",
    "IngestedLaw",
    "LawFile",
    "LawsPreprocessingConfig",
    "Line",
    "Link",
    "ResolvedLawRef",
    "build_corpus_registry",
    "classify_law_status",
    "compute_source_hash",
    "ingest_law",
    "law_id_from_date_number",
    "normalize_article_label",
    "normalize_month_name",
    "normalize_ws",
    "parse_blocks_from_html",
    "parse_italian_date",
    "parse_law_filename",
    "run_laws_preprocessing",
    "sha256_file",
]
