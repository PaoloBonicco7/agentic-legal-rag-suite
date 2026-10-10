# 01 — Source snapshots

## Purpose
Freeze Italian coordinated legislation from the official Council database. Observation time is not legal effective time.

## Inputs
Official paginated catalogue, detail pages, or preserved local legacy HTML.

## Outputs
Immutable catalog.jsonl, raw bytes, snapshot manifest with hashes, acquisition window and completeness; resumable checkpoints while acquisition is pending.

## Pipeline
Enumerate every page, deduplicate source IDs, download every detail sequentially, normalize encoding without modifying raw bytes, enumerate the catalogue again, and promote only when IDs match and all details have valid law identities. Re-fetch existing laws because amendments change old acts. Start requests at least one second apart; retry twice; persist progress after each page/detail.

## Contract
SourceRecord stores URL, relative byte path, digest, encoding and observation date. Editorial update time remains null unless supplied. Legacy imports record unknown acquisition time and do not claim current completeness. Completed snapshots cannot be overwritten. Missing pages and HTTP errors do not imply repeal.

## Quality Gates
All expected pages; unique source IDs; validated law details; identical initial/final catalogues; raw hashes verified; no replacement/control-character decoding failures. Failed snapshots cannot be presented as current.

## Notebook Role
Show the manifest, changed-source counts and one decoded document; acquisition logic remains in the module.

## Acceptance Criteria
Interrupted acquisition resumes without dropped records. Completed snapshots are immutable. Legacy import is reproducible and labeled. Fixtures cover pagination, charset mismatch, failed details and catalogue drift.
