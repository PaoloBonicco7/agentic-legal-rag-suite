# 01 - Laws Preprocessing Specification

## Purpose

Transform the versioned HTML legal corpus in `data/laws_html/` into a clean, generated dataset ready for retrieval, filtering, and graph-aware RAG.

This step proves that the legal corpus can be converted into structured, explainable units. A retrieved chunk must carry law identity, article identity, passage context, legal status, provenance, and explicit legal relations.

## Inputs

- Source corpus: `data/laws_html/`.
- Preprocessing configuration: source path, output path, chunk size, chunk overlap, strictness level.

Non-source files in the corpus directory, such as `.DS_Store`, must be ignored and reported.

## Outputs

Default generated output directory: `data/laws_dataset_clean/`.

- `manifest.json`: schema version, source hash, configuration, counts, output hashes, and quality gates.
- `laws.jsonl`: one record per law.
- `articles.jsonl`: one record per article.
- `passages.jsonl`: one record per legal passage.
- `notes.jsonl`: one record per note.
- `status_events.jsonl`: resolved and unresolved cessation events used by the validity rules.
- `edges.jsonl`: explicit graph relations between laws or articles.
- `chunks.jsonl`: RAG-ready chunks with denormalized metadata.
- `status_transitions.jsonl`: optional comparison with a configured previous clean dataset.
- `quality_report.md`: human-readable validation report.
- `dataset_profile.json`: exploration summary for notebooks.

Generated files are reproducible artifacts and are not committed by default.

## Pipeline

1. Validate the source corpus.
   Why: fail early when files are missing, duplicated, outside the filename pattern, or not parseable into stable law identities.
2. Parse HTML blocks.
   Why: preserve headings, paragraphs, links, anchors, and table rows while removing raw HTML complexity.
3. Extract legal structure.
   Why: create laws, articles, passages, and notes before chunking so retrieval units follow legal structure.
4. Resolve explicit references.
   Why: link laws using hyperlinks and citation text that can be inspected and explained.
5. Build graph relations.
   Why: support graph-aware retrieval from explicit evidence, without speculative inference.
6. Extract scoped cessation events and classify validity.
   Why: whole-law, whole-article, comma, and letter cessations must not be collapsed into the same status decision.
7. Build RAG chunks.
   Why: attach enough metadata to each chunk for filtering, reranking, provenance, and citations.
8. Export and validate.
   Why: later steps should consume a clear dataset contract instead of revalidating notebook assumptions.

## Contract

Every chunk must include:

- stable IDs: `chunk_id`, `passage_id`, `article_id`, `law_id`;
- content fields: `text`, `text_for_embedding`;
- law fields: `law_date`, `law_number`, `law_title`, `law_status`;
- article and passage fields: `article_label_norm`, `article_status`, `passage_status`, `passage_label`, `structure_path`;
- provenance: `source_file`;
- validity evidence: `content_availability`, `status_event_ids`, `status_rule_ids`;
- graph/filter fields: `index_views`, `related_law_ids`, `inbound_law_ids`, `outbound_law_ids`, `relation_types`.

`LAWS_PREPROCESSING_SCHEMA_VERSION` is `laws-preprocessing-v2`.
`LEGAL_STATUS_RULES_VERSION` is `legal-status-rules-v1`.

Allowed validity statuses at law, article, and passage level are `current`, `partial`, `past`, and `unknown`.
Validity is independent from `content_availability`, whose values are `substantive`, `unstructured`, `metadata_only`, and `empty`.

Allowed status-event types are `repeal` and `expiration`. A status event must preserve a deterministic id, source clause, evidence text, event sequence, modifying-law ids, target kind and ids, `scope_mode` (`all`, `only`, or `all_except`), exceptions, resolution status, and rule id. Amendments, insertions, and replacements remain represented by notes and graph edges and do not by themselves terminate the consolidated text.

Allowed relation types are `REFERENCES`, `ABROGATED_BY`, `ABROGATES`, `MODIFIED_BY`, `AMENDS`, `REPLACED_BY`, `REPLACES`, `INSERTED_BY`, and `INSERTS`.

`index_views` must include `historical` for every chunk. This is an inclusive, unfiltered view and not a reconstruction of the law at a historical date.

`current` is included only when the law, article, and passage lineage contains only `current` or `partial`, and the content is `substantive` or `unstructured`.

`not_explicitly_past` is included when no lineage status is `past`; it deliberately retains `unknown`.

Status classification follows these deterministic rules:

- a resolved whole-law or whole-article cessation makes the target and its descendants `past`;
- `all_except` produces a `partial` parent, makes resolved non-excepted targets `past`, and leaves unresolved exceptions `unknown`;
- a later resolved whole-target cessation overrides an earlier partial exception;
- a comma or letter becomes `past` only when the source clause has one scope-compatible backlink;
- a cessation narrower than the available passage makes that passage `partial`; multiple or incompatible targets make it `unknown`;
- bracketed text without a resolved cessation event is `unknown`, never implicitly `past`;
- roll-up is `past` only under complete past coverage, `partial` for mixed active/past children, and `unknown` when unresolved ambiguity affects coverage;
- `current` means no explicit cessation was found in this corpus snapshot; it is not an external legal certification.

Graph edges must come only from explicit hyperlinks or citation text and must preserve evidence.

## Quality Gates

- At least one valid source HTML law file is found.
- Generated IDs are stable, non-empty, and duplicate-free.
- Required chunk fields are present on every chunk.
- List metadata fields are lists, not serialized strings.
- Every status-event reference resolves to an exported event.
- Duplicate note anchors, anchorless note markers, zero/multiple backlinks, and scope-kind mismatches are counted.
- Every applied passage-level cessation has one scope-compatible backlink.
- Modifying-law ids come from the decisive clause rather than unrelated references in the same note.
- Validity and content-availability distributions are reported separately.
- Clean graph edges contain no self-loops.
- Unresolved references are counted and reported.
- Every manifest output file exists and has a hash.
- `chunks.jsonl` is non-empty.
- The manifest records the schema version, status-rules version, source hash, pipeline code hash, Git revision, and dirty state.
- The manifest exposes `ready_for_indexing`.

## Notebook Role

`notebooks/01_laws_preprocessing.ipynb` should run the preprocessing step, display corpus counts, inspect representative laws/chunks/edges, show quality gates, and confirm whether the dataset is ready for indexing.

The notebook must call reusable core logic and must not contain the transformation implementation.

## Acceptance Criteria

- The full source law corpus can be transformed into `data/laws_dataset_clean/`.
- Chunks are usable by the indexing step without additional preprocessing.
- Graph edges can support graph-aware retrieval expansion.
- Metadata supports filtering by law status, index view, law identity, article identity, and relation information.
- Partial cessations do not make an entire article or law `past` without complete resolved coverage.
- Every validity decision is traceable to rules and, where applicable, scoped source evidence.
- The quality report explains whether the dataset is ready for indexing.
