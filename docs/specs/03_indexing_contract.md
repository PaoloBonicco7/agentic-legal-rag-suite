# 03 - Indexing Contract Specification

## Purpose

Define how the clean legal dataset becomes a retrieval-ready index for RAG.

This step proves that generated chunks can be indexed with stable payload metadata, reproducible hashes, and filterable fields needed by simple and advanced retrieval.

## Inputs

- Clean legal dataset: `data/laws_dataset_clean/`.
- Required files from step 01: `manifest.json`, `chunks.jsonl`, `laws.jsonl`, `articles.jsonl`, `edges.jsonl`.
- Indexing configuration: index output path, collection name, embedding model identity, chunk selection mode, and rebuild policy.

## Outputs

Default generated output locations:

- retrieval index under `data/indexes/`;
- indexing artifacts under `data/indexing_runs/`.

Required artifacts:

- `index_manifest.json`: source dataset hash, embedding model identity, indexed count, skipped count, and payload field summary.
- `payload_profile.json`: coverage of filterable metadata fields.
- `index_quality_report.md`: human-readable validation report.
- optional sample retrieval report for a few fixed diagnostic queries.

## Pipeline

1. Validate the clean dataset contract.
   Why: indexing should fail before embedding when chunk metadata is incomplete.
2. Select chunks for indexing.
   Why: the PoC may support full runs and small notebook runs while preserving the same contract.
3. Build embedding input.
   Why: embeddings must use `text_for_embedding`, not raw text alone, so legal context is preserved.
4. Create stable point identities.
   Why: repeated runs should not duplicate unchanged chunks.
5. Store vectors with payload metadata.
   Why: retrieval needs filters for law status, index views, law identity, article identity, and graph fields.
6. Validate the index.
   Why: later RAG steps need confidence that filters and payload fields work.

## Contract

The index payload for every stored chunk must include:

- `chunk_id`
- `passage_id`
- `article_id`
- `law_id`
- `text`
- `law_date`
- `law_number`
- `law_title`
- `law_status`
- `article_status`
- `article_label_norm`
- `passage_label`
- `structure_path`
- `source_file`
- `index_views`
- `related_law_ids`
- `inbound_law_ids`
- `outbound_law_ids`
- `relation_types`
- `content_hash`

Filterable fields must include at least `law_id`, `law_status`, `index_views`, `article_id`, and `relation_types`.

The index manifest must record the source dataset hash and the embedding model identity used for the run.

## Quality Gates

- Source clean dataset has `ready_for_indexing`.
- Every indexed point has required payload fields.
- Indexed count matches selected chunk count.
- Duplicate `chunk_id` values are rejected.
- Filterable fields are present and queryable.
- Embedding model identity is recorded.
- Index manifest links back to the exact clean dataset source hash.

## Notebook Role

`notebooks/03_indexing_contract.ipynb` should validate the clean dataset, run a small indexing job, show payload examples, test filters, and run a few diagnostic retrieval queries.

The notebook should explain the payload fields because they are the bridge between preprocessing and RAG behavior.

## Acceptance Criteria

- Clean chunks can be indexed reproducibly.
- Payload metadata supports simple and advanced retrieval filters.
- The index can be traced back to the exact generated clean dataset.
- The notebook can demonstrate that retrieval and filters work before any answer generation is evaluated.

