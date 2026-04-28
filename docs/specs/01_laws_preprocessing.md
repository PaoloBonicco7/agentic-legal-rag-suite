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
- `edges.jsonl`: explicit graph relations between laws or articles.
- `chunks.jsonl`: RAG-ready chunks with denormalized metadata.
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
6. Classify legal status.
   Why: enable filters such as current-law-only search while preserving historical material.
7. Build RAG chunks.
   Why: attach enough metadata to each chunk for filtering, reranking, provenance, and citations.
8. Export and validate.
   Why: later steps should consume a clear dataset contract instead of revalidating notebook assumptions.

## Contract

Every chunk must include:

- stable IDs: `chunk_id`, `passage_id`, `article_id`, `law_id`;
- content fields: `text`, `text_for_embedding`;
- law fields: `law_date`, `law_number`, `law_title`, `law_status`;
- article and passage fields: `article_label_norm`, `article_status`, `passage_label`, `structure_path`;
- provenance: `source_file`;
- graph/filter fields: `index_views`, `related_law_ids`, `inbound_law_ids`, `outbound_law_ids`, `relation_types`.

Allowed law statuses are `current`, `past`, `unknown`, and `index_or_empty`.

Allowed relation types are `REFERENCES`, `ABROGATED_BY`, `ABROGATES`, `MODIFIED_BY`, `AMENDS`, `REPLACED_BY`, `REPLACES`, `INSERTED_BY`, and `INSERTS`.

`index_views` must include `historical` for every chunk and may include `current` when the law and article are suitable for current-law retrieval.

Graph edges must come only from explicit hyperlinks or citation text and must preserve evidence.

## Quality Gates

- At least one valid source HTML law file is found.
- Generated IDs are stable, non-empty, and duplicate-free.
- Required chunk fields are present on every chunk.
- List metadata fields are lists, not serialized strings.
- Clean graph edges contain no self-loops.
- Unresolved references are counted and reported.
- Every manifest output file exists and has a hash.
- `chunks.jsonl` is non-empty.
- The manifest exposes `ready_for_indexing`.

## Notebook Role

`notebooks/01_laws_preprocessing.ipynb` should run the preprocessing step, display corpus counts, inspect representative laws/chunks/edges, show quality gates, and confirm whether the dataset is ready for indexing.

The notebook must call reusable core logic and must not contain the transformation implementation.

## Acceptance Criteria

- The full source law corpus can be transformed into `data/laws_dataset_clean/`.
- Chunks are usable by the indexing step without additional preprocessing.
- Graph edges can support graph-aware retrieval expansion.
- Metadata supports filtering by law status, index view, law identity, article identity, and relation information.
- The quality report explains whether the dataset is ready for indexing.

