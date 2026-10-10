# 09 — Benchmark reviews and replica

## Purpose
Keep historical benchmark performance distinct from correctness against the acquired coordinated source.

## Inputs
The original 100 intents, source-backed review proposals, saved answers and the selected snapshot.

## Outputs
Independent review JSONL, human-approved benchmark view and new evaluation or rejudge runs. Portable exports accept `create_bundle(..., review_files=())` and preserve each supplied raw JSONL under `source/reviews/<sha256>.jsonl`, without normalizing its bytes.

## Pipeline
Annotate reference errors, temporal mismatch and proposed supporting evidence without changing original gold. A named human reviewer approves each change and records the relevant snapshot. If only gold changes, rejudge frozen answers in a new run. Changed questions, choices, source/policy or generation configuration require new generation. Replicate all compared treatments on the same accepted view.

## Contract
Reviews are separate artifacts; `human_approved` defaults false. No agent may approve on the user's behalf. Approved rows require a named reviewer, a non-empty `note` explaining the decision and traceable primary `source_urls` using HTTP(S). `current_revised` and `historical_only` additionally require non-empty supporting `evidence`; `current_supported` can use a note identifying verified indexed evidence. URL validation checks structure, never certifies source authority or the proposed gold. Unverified automatic proposals may remain pending with no citations or evidence until inspected. The approved-current view must match the corpus snapshot. Original, eligible and selected intent counts remain distinct. The reviewed dataset is exploratory, because the questions were used during development.

## Quality Gates
Source identity and review provenance; unchanged questions for rejudge; physical frozen corpus snapshot/manifest and every artifact hash verified before judging; no double charging of saved generation; same view and question selection across treatments; missing judgments visible. Bundle export, verification and restore require every declared source CSV and raw review to match the run's `source_hashes`; a missing review or recomposed outer checksum cannot bypass this gate. Credential fields are forbidden in packaged raw reviews.

## Notebook Role
Inspect review rows and disagreements; do not edit source CSVs or implement scoring.

## Acceptance Criteria
Original CSVs remain unchanged. Every accepted correction is attributable. Legacy and revised results are never silently merged. A partially reviewed current view cannot support a final adoption decision.
