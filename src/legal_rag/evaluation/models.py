"""Evaluation-only contracts, kept outside generation inputs."""

from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from legal_rag.contracts import AnswerResult, Record, Usage

EVALUATION_SCHEMA_VERSION = "evaluation-v2"
JUDGE_PROMPT_VERSION = "legal-judge-v1"
Grounding = Literal["supported", "partial", "unsupported", "not_applicable"]


class Reference(Record):
    text: str
    document_id: str | None = None
    article_label: str | None = None


class Question(Record):
    qid: str = Field(min_length=1)
    question: str = Field(min_length=1)
    level: str = Field(min_length=1)
    correct_answer: str = Field(min_length=1)
    options: dict[str, str]
    correct_label: str | None
    expected_references: list[Reference] = Field(min_length=1)
    flags: list[str] = Field(default_factory=list)


class Review(Record):
    qid: str
    status: Literal["proposed", "current_supported", "current_revised", "historical_only", "unresolved"] = "proposed"
    human_approved: bool = False
    reviewer: str | None = None
    question: str | None = None
    correct_answer: str | None = None
    expected_references: list[str] | None = None
    snapshot_id: str | None = None
    note: str = ""
    source_urls: list[str] = Field(default_factory=list)
    evidence: str = ""

    @model_validator(mode="after")
    def approved_review_has_provenance(self):
        if self.human_approved and (not self.reviewer or not self.reviewer.strip() or self.status == "proposed"):
            raise ValueError("An approved review requires a reviewer and a resolved review status")
        if self.human_approved:
            if not self.note.strip() or not self.source_urls:
                raise ValueError("An approved review requires a reason and traceable primary source URLs")
            for url in self.source_urls:
                parsed = urlsplit(url)
                if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
                    raise ValueError("Approved review source URLs must use HTTP(S) without credentials")
            if self.status in {"current_revised", "historical_only"} and not self.evidence.strip():
                raise ValueError("A revised or historical review requires supporting evidence")
        return self


class Benchmark(Record):
    schema_version: str = EVALUATION_SCHEMA_VERSION
    view: Literal["legacy", "current_approved"] = "legacy"
    questions: list[Question]
    source_hashes: dict[str, str]
    benchmark_hash: str
    original_count: int
    reviews: list[Review] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class OracleEvidence(Record):
    qid: str
    snapshot_id: str
    corpus_manifest_sha256: str
    chunk_ids: list[str] = Field(min_length=1)
    human_approved: bool
    reviewer: str = Field(min_length=1)

    @model_validator(mode="after")
    def verified_unique_evidence(self):
        if not self.human_approved or not self.reviewer.strip():
            raise ValueError("Oracle evidence requires explicit human approval and a named reviewer")
        if len(set(self.chunk_ids)) != len(self.chunk_ids):
            raise ValueError("Oracle chunk IDs must be unique")
        return self


class JudgePayload(Record):
    correctness: Literal[0, 1, 2]
    explanation: str = Field(min_length=1)
    grounding: Grounding
    grounding_explanation: str = Field(min_length=1)


class EvaluationResult(Record):
    qid: str
    format: Literal["no_hint", "mcq"]
    level: str
    question: str
    correct_answer: str
    correct_label: str | None = None
    options: dict[str, str] = Field(default_factory=dict)
    expected_references: list[Reference]
    flags: list[str] = Field(default_factory=list)
    answer: AnswerResult
    generation_attempted: bool = True
    score: Literal[0, 1, 2] | None = None
    judge: JudgePayload | None = None
    judge_usage: list[Usage] = Field(default_factory=list)
    judge_error: str | None = None
    citation_checks: list[dict] = Field(default_factory=list)
    retrieval_proxies: dict = Field(default_factory=dict)
    control_judge: JudgePayload | None = None
    control_usage: list[Usage] = Field(default_factory=list)
    control_error: str | None = None


class EvaluationManifest(Record):
    schema_version: str = EVALUATION_SCHEMA_VERSION
    created_at: str
    updated_at: str | None = None
    run_hash: str
    inputs: dict
    source_hashes: dict[str, str]
    original_intents: int
    view_intents: int
    selected_intents: int
    selected_questions: list[dict] = Field(default_factory=list)
    pilot: bool
    reviewed_intents: int = 0
    view_provisional: bool = False
    warnings: list[str]
    reviews: list[dict]
    complete: bool = False
    status: Literal["running", "complete", "incomplete"] = "running"
    control_complete: bool = False
    output_hashes: dict[str, str] = Field(default_factory=dict)
    accounting_history: list[dict] = Field(default_factory=list)
    stopped_reason: str | None = None
