# 06 — Evaluation

## Purpose

Compare no-RAG, dense, sparse, hybrid and isolated ablations on the same questions.
The primary result is the no-hint fully-correct rate against the unchanged legacy
gold. The 100 paired MCQ/no-hint records represent 100 intents, not 200 independent
observations. This previously explored benchmark is not an unseen test.

## Inputs

`load_benchmark(source_dir)` validates the two original CSV files, their six MCQ
options, levels, answers and stems. Source CSV files are never modified. A separate
JSONL review may propose corrections; only rows explicitly marked
`human_approved=true`, with a named reviewer, an explicit reason and primary source
URLs, enter the `current_approved` view. Revised or historical decisions also
require recorded supporting evidence; validators do not certify the proposed gold.
An approved historical/unresolved row is omitted from that view, never deleted.
Approved current reviews must name the exact corpus snapshot used by the run.

Oracle is a diagnostic ceiling, not an ordinary retrieval treatment. Its separate
human-approved JSONL maps QIDs to existing chunk IDs and records `snapshot_id`,
`corpus_manifest_sha256` and a named `reviewer`. The runner resolves those IDs in
the frozen corpus and passes only chunks to the engine, never gold answers or
reference annotations. Unapproved, duplicate or mismatched evidence is rejected.

## Outputs

`run_evaluation(benchmark, engine, judge, output_dir, profile=..., identity=...)`
exports `manifest.json`, append-only `answers.jsonl` and `results.jsonl`, and
`metrics.json`. Resume requires exactly the same benchmark, selection, format,
profile, configuration and source-code identity. Attempted generations remain
frozen; generations blocked before dispatch may be retried. Missing judge/control
calls are retried without repeating a saved generation. Existing output directories require
explicit resume. Duplicate completed QIDs and malformed checkpoints are rejected.

`run_rejudge(benchmark, judge, source_run_dir, output_dir, identity=...)` creates
a separate corrected-gold view of frozen answers. It requires identical question
text, generation configuration, corpus, prompts, filters and budgets. A changed
question requires a new generation run. The origin manifest hash is recorded;
generation usage remains attributable to the origin and is not charged again.
Before any judge call, the supplied corpus must be ready and match the frozen
snapshot/manifest identity; every recorded corpus artifact hash is verified.

## Pipeline

1. Freeze selected QIDs and hash source questions/configuration before calls.
2. Give the engine only the question and optional MCQ options. Gold answers and
   expected references never enter retrieval, rewriting or generation.
3. Score MCQ labels exactly. Score open answers with a frozen 0/1/2 rubric:
   wrong/contradictory, partial without contradiction, fully correct.
4. Assess grounding separately using the answer and the actual context. Check
   citation IDs and quoted substrings deterministically; semantic support remains
   a judge assessment. Grounding is not applicable to no-RAG.
5. Treat generation failure as score zero; missing judge scores remain null.
   A run with missing judgments cannot enter a complete comparison.
   Budget/price/credential failures before dispatch are not-attempted generations,
   with missing scores. An unverified model context window is likewise a terminal
   configuration error before dispatch, with no model attempt or zero score.
   Terminal provider/budget failures save that checkpoint and
   stop new calls; unvisited QIDs remain missing, never fabricated zeros. An HTTP
   403 from a dispatched generation is one failed generation before stopping.
6. Export per-question answer, candidates, packed evidence, quotes, judge reasons,
   usage, errors and reference/temporal flags. Flags never change legacy scoring.

## Contract

Primary metric: `fully_correct_rate = count(score=2) / selected_intents` for
no-hint and exact accuracy for MCQ. Secondary metrics include ordinal mean,
score distribution, generation/judge coverage, citation validity and separate
grounding. Report fixed-denominator and available-score means explicitly.
Article Success and MRR at 5/10/100 are reference proxies, not evidence sufficiency.
Reference `annotation_complete` records parsing of all supplied references;
it never establishes exhaustive relevance judgments. MCQ scoring remains missing
when a revised gold has no unique match among the unchanged original choices.
The original candidate pool and final selection are reported separately when a
rerank or graph transformation changes their composition.
Unknown statuses remain visible; references to confirmed cessation receive a flag.

Comparisons require identical QIDs, benchmark/view, generator, judge, answer/judge
prompts, corpus, embedding, filters, tokenizer and candidate/context budgets.
Only the treatment/profile may differ. Paired percentile bootstrap uses seed
20261010; sensitivity resamples law groups. Confidence intervals describe this
explored sample, not unseen-query performance. At ties retain the simpler method.
Operational promotion requires increased fully-correct rate, no new structured
generation failures, no deterioration in citation validity and complete judgments.
It also requires the full legacy selection, or a fully human-reviewed current
view including explicit decisions for all original intents. Partial reviews keep
the current view provisional. Original/view/selected denominators remain distinct.
Scientific provider identities exclude accounting metadata; every start/resume
records separate prices, budget and ledger metadata in `accounting_history`.
Unknown returned model identities on successful outputs block a controlled
comparison. This gate uses the last Usage producing the successful answer/judge
payload; preceding failed retries or rewrites remain available in costs/raw-output
diagnostics and are not misclassified as successful model outputs. Resolved model
IDs can differ from requested aliases when actual identities agree across runs.

An optional second judge checks 20 stable, level-stratified QIDs. Its provider/model
pair must differ from the primary judge to authorize operational promotion;
changes to timeout or other client settings do not count as a distinct judge. Disagreement is
reported without treating either judge as human truth or silently changing scores.
If the control reverses the winner, promotion remains provisional.
The control sample delta is compared with the full-run winner, while the primary
judge delta on that same sample is reported separately.
Promotion requires the complete planned control sample: 20 QIDs, or all included
QIDs when the reviewed view is smaller. MCQ and pilot subsets cannot be promoted.

## Quality Gates

- CSV hashes, selection and identities are recorded; originals remain intact.
  Portable bundles verify those hashes against the packaged source CSVs and
  preserve each declared raw review JSONL by its hash; missing reviews are rejected.
- Every selected intent has one final result; nullable scores preserve missingness.
- Approved reviews are separate, attributable and refer to existing QIDs.
- Comparison rejects incompatible models, datasets or context budgets.
- Resume never regenerates a successful answer after judge failure.
- Empty answers, invalid labels, unknown/ceased references and invalid quotes are
  inspectable in focused fake-client tests.
- No API calls are made by unit tests or report generation.

## Notebook Role

Optional notebooks call these Python APIs and inspect saved artifacts. They define
no scoring or orchestration logic. Reports are the reusable presentation layer.

## Acceptance Criteria

Fake-client tests verify aligned inputs, failed/missing scores, exact MCQ labels,
separate correctness/grounding, checkpoint integrity, oracle approval and corpus
hashes, snapshot-bound reviews and corrected-gold rejudging without regeneration.
All tests run without credentials, remote model calls or source CSV changes.
