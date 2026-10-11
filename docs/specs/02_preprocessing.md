# 02 — Deterministic preprocessing

## Purpose

Convert one immutable acquisition snapshot into auditable legal structure and
retrieval chunks. Parsing does not call an LLM, an embedding service or a network
endpoint. Source representation is independent from legal status.

## Inputs

- Snapshot `catalog.jsonl` with shared `SourceRecord` contracts and relative paths.
- Snapshot `manifest.json` with `snapshot_id`, `complete=true`, catalogue
  SHA-256 and source count, and, when available, `acquisition_finished_at`.
  Legacy sources keep their unknown capture date.
- A pinned `TokenCounter`; defaults: target 384, maximum 512 tokens.

## Outputs

`documents.jsonl`, `blocks.jsonl`, `units.jsonl`, `events.jsonl`,
`relations.jsonl`, `nodes.jsonl`, `chunks.jsonl`, `quality.json` and
`manifest.json`. Existing output directories are never overwritten.

## Pipeline

1. Before parsing any source or creating outputs, verify the manifest's
   catalogue SHA-256 and source count, unique source IDs/paths, confined relative
   source paths and every source byte hash. Reject incomplete snapshots.
2. Extract ordered normalized HTML blocks, including links and note anchors.
   Split line breaks and nested table paragraphs without duplicating their text.
3. Identify the legal identity, thematic title and publication date. Preserve
   body, operative body quotation, editorial, note, historical quotation, index,
   heading and annex regions. Quoted replacement articles belong to the outer
   amending provision; they cannot create duplicate root articles.
4. Build articles, paragraphs, letters and annexes. A list item belongs to the
   preceding paragraph or to its article when no numbered paragraph exists.
   Plain, inline-rubric and plural article headings are structural; historical headings
   cannot reopen the body. Empty articles remain empty. Stubs are metadata only.
5. Assess status and explicit relations using specification 03.
6. Aggregate substantive siblings within an article. Repeat exact introductory
   context for list continuations when it fits. Split long source spans at word
   boundaries and, only if necessary, character boundaries. Count the complete
   embedding input, including identity, thematic title and article heading.
7. Validate and export immutable artifacts with hashes and code identity.

## Contract

`build_corpus(..., progress=None)` accepts an optional Python callback with
`stage`, `processed`, `total` and `ready` fields. Stages are `preflight`,
`parsing`, `graph_chunking`, `quality` and `export`. Preflight emits a start
event with 0/0 counters. Parsing and graph/chunking emit start, every 100
documents and completion; quality and export emit start and completion.
`ready` becomes true only after all artifacts are exported and all quality
gates pass. Callback failures propagate; monitoring does not change records,
manifest fields or processing rules. The CLI sends progress as JSONL to stderr
by default and keeps the final manifest on stdout; no extra flag is required.
The initial setup-only update left this wiring unexecuted. The 2026-10-11
offline suite and two complete 3,176-document rebuilds subsequently verified
the callback and CLI wiring; canonical artifacts were byte-identical.

Shared Pydantic models live in `contracts.py`. IDs are deterministic within a
snapshot. `SourceSpan` offsets are half-open character positions in the exported
normalized `Block.text`, not byte positions in raw HTML. A unit or chunk text is
the newline concatenation of its spans. Inline footnote markers are excluded
from normative text while their links remain in blocks. Every significant block
has a declared region; excluded blocks retain an explicit reason.
`Document.document_id` is the canonical legal identifier; `Document.identity`
is the exact normalized primary source heading without the page's trailing
"Testo vigente" label. Chunks carry that source identity in both metadata and
the complete embedding prefix. Graph joins always use canonical IDs.

Thematic titles come from the descriptive preamble, rather than the generic
"Legge regionale … — Testo vigente" page heading. Annexes are separate units;
an `Omissis` marker is preserved and cannot become normative retrieval content.
Unstructured substantive text remains available with a warning. No synthetic
article is invented for a repeal stub. Unknown status remains retrievable.

Repeated numeric law headings are metadata even when written as paragraphs or
with an incomplete date. Descriptive preamble lines before the publication line
form one thematic title; `BU`, `B.U.` and `Bollettino ufficiale` are metadata.
An unclosed quotation of a single paragraph/letter may stop at an explicitly
anchored next article, with a source-boundary warning. Article/section replacement
quotations keep their nested headings. Missing source punctuation is never added;
unresolved paragraph boundaries inside a recovered quotation require review.

## Quality Gates

- Acquisition completeness, catalogue hash/count and all source paths/hashes
  must pass preflight before parsing; malformed inputs produce no artifacts.
- Every catalog source parses to one unique document identity.
- Every nonempty block has a declared provenance region.
- All parents, spans, event targets and resolved relation targets exist.
- Span reconstruction equals exported unit and chunk text.
- No chunk contains only a heading, note marker, numbering or omitted text.
- A snapshot containing only metadata has no fake chunks and is not indexable.
- No chunk mixes confirmed cessation with an eligible sibling.
- Complete embedding inputs respect the configured maximum token count.
- Every output hash is recorded; readiness is false if a gate fails.

Structural gates are not a legal accuracy certification. Report status and
availability distributions, unresolved events, omitted annexes, duplicated
anchors and parser warnings for manual review.
Pre-existing encoding anomalies in legacy assets are preserved and reported;
new anomalies in acquired sources fail readiness.

## Notebook Role

Notebooks call `build_corpus`; they inspect block-to-unit provenance and the
quality artifact. They contain no reusable parsing logic.

## Acceptance Criteria

The full legacy corpus and acquired snapshots run through the same pipeline.
Real regression cases include LR 42/1992 (exceptions and historical quotations),
LR 1/1950 (stub), LR 44/1991 (repealed law retaining body), LR 21/2003 note 40
(two interventions in distinct clauses) and LR 33/1993 (omitted annex).
Tokenizer identity, source snapshot and implementation hashes accompany results.
