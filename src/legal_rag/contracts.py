"""Shared, secret-free artifact contracts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "legal-rag-v2"
Status = Literal["ceased_confirmed", "no_cessation_evidence", "unknown"]
Profile = Literal["no_rag", "dense", "sparse", "hybrid", "multiquery", "graph", "rerank", "oracle"]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def to_json_record(self) -> dict:
        return self.model_dump(mode="json")


class SourceRecord(Record):
    source_id: str
    url: str
    path: str
    sha256: str
    fetched_at: str | None = None
    encoding: str = "utf-8"
    source_updated_at: str | None = None
    representation: Literal["coordinated", "legacy"] = "coordinated"


class SourceSpan(Record):
    block_id: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)


class Block(Record):
    block_id: str
    document_id: str
    position: int
    text: str
    tag: str
    anchor: str | None = None
    links: list[dict[str, str]] = Field(default_factory=list)
    region: str = "unassigned"
    exclusion_reason: str | None = None


class Document(Record):
    document_id: str
    source_id: str
    snapshot_id: str
    source_url: str
    source_path: str
    identity: str
    title: str
    enacted_at: str | None = None
    published_at: str | None = None
    representation: str = "coordinated"
    availability: Literal["body", "unstructured", "metadata_only", "omitted"]
    status: Status = "no_cessation_evidence"
    event_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class Unit(Record):
    unit_id: str
    document_id: str
    kind: Literal["article", "paragraph", "letter", "annex", "unstructured"]
    label: str
    article_label: str | None = None
    heading: str = ""
    text: str
    parent_id: str | None = None
    spans: list[SourceSpan] = Field(default_factory=list)
    status: Status = "no_cessation_evidence"


class Event(Record):
    event_id: str
    document_id: str
    target_id: str | None = None
    kind: str
    scope: Literal["law", "article", "paragraph", "unknown"]
    evidence: str
    span: SourceSpan
    related_document_id: str | None = None
    effective_at: str | None = None
    resolution: Literal["confirmed", "unresolved"] = "unresolved"


class Relation(Record):
    relation_id: str
    source_id: str
    target_id: str | None = None
    target_reference: str = ""
    kind: Literal["contains", "references", "repeals", "amends"]
    evidence: str = ""
    span: SourceSpan | None = None
    resolution: Literal["confirmed", "unresolved"] = "unresolved"


class Chunk(Record):
    chunk_id: str
    document_id: str
    snapshot_id: str
    article_label: str | None = None
    title: str
    identity: str = ""
    heading: str = ""
    text: str
    embedding_text: str
    unit_ids: list[str]
    spans: list[SourceSpan]
    source_url: str
    status: Status = "no_cessation_evidence"
    warnings: list[str] = Field(default_factory=list)


class Candidate(Record):
    chunk: Chunk
    rank: int = Field(ge=1)
    score: float
    origin: str


class Evidence(Record):
    context_id: str
    chunk: Chunk
    rendered: str


class Context(Record):
    text: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    token_count: int = 0
    omissions: list[dict[str, str]] = Field(default_factory=list)


class Citation(Record):
    context_id: str
    quote: str
    source_url: str = ""
    chunk_id: str = ""


class Usage(Record):
    provider: str
    requested_model: str
    returned_model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    elapsed_seconds: float = 0
    cost_eur: float | None = None
    cost_basis: str = 'unknown'
    price_date: str | None = None
    ledger_entry_id: str | None = None
    raw_output: str | None = None


class AnswerPayload(Record):
    answer: str
    status: Literal["answered", "insufficient_evidence", "uncertain"]
    citations: list[Citation] = Field(default_factory=list)
    selected_option: str | None = None


class AnswerResult(Record):
    question: str
    profile: Profile
    snapshot_id: str | None = None
    answer: str = ""
    status: Literal["answered", "insufficient_evidence", "uncertain", "error"]
    citations: list[Citation] = Field(default_factory=list)
    selected_option: str | None = None
    candidates: list[Candidate] = Field(default_factory=list)
    context: Context = Field(default_factory=Context)
    queries: list[str] = Field(default_factory=list)
    usage: list[Usage] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    raw_payload: dict | None = None
    retrieval_trace: dict = Field(default_factory=dict)


class EngineConfig(Record):
    profile: Profile = "hybrid"
    candidate_limit: int = Field(default=100, gt=0)
    context_tokens: int = Field(default=4096, gt=0)
    context_units: int = Field(default=12, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    exclude_ceased: bool = True
    graph_additions: int = Field(default=2, ge=0, le=2)
    rerank_limit: int = Field(default=20, gt=0)


class ProviderConfig(Record):
    provider: Literal["utopia", "openai"] = "utopia"
    model: str = "SLURM.gpt-oss:120b"
    base_url: str = "https://utopia.hpc4ai.unito.it/api"
    timeout_seconds: int = Field(default=180, gt=0)
    max_output_tokens: int = Field(default=4096, gt=0)
    max_input_tokens: int = Field(default=131072, gt=0)
    context_window_tokens: int | None = Field(default=32768, gt=0)
    reasoning_effort: Literal["none", "low", "medium", "high"] | None = "low"
    retries: int = Field(default=2, ge=0, le=3)
    budget_eur: float = Field(default=20, gt=0)
    budget_reserve_eur: float = Field(default=4, ge=0)
    eur_per_usd: float = Field(default=1.0, gt=0)
    price_date: str | None = None
    price_source_url: str | None = None
    fx_date: str | None = None
    fx_source_url: str | None = None
    input_usd_per_million: float | None = Field(default=None, ge=0)
    output_usd_per_million: float | None = Field(default=None, ge=0)
