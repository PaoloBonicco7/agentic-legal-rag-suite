# Expanded pilot — Luna generation, Terra judgment

## Frozen protocol

The authorized pilot contains 20 original no-hint intents, five per level.
Retain the previous four-case panel and add four QIDs per level using a single
`Random(20261013)`, levels and eligible QID populations sorted, with the retained
QID removed before sampling. Selection precedes model calls; no resampling based
on scores, reference quality or legal status. The benchmark and gold are unchanged.

| Level | Selected QIDs (prefix `eval-`) |
| --- | --- |
| L1 | 0025, 0053, 0061, 0077, 0081 |
| L2 | 0002, 0018, 0046, 0066, 0094 |
| L3 | 0007, 0011, 0059, 0083, 0095 |
| L4 | 0004, 0016, 0036, 0080, 0100 |

Generate fresh answers for all 20 cases with `gpt-6-luna`. Judge correctness and
grounding with `gpt-5.6-terra`; neither old generations nor Luna grades are reused.
Seven separate profiles: No-RAG, Dense, Sparse, Hybrid, Multi-query, Graph and
Rerank. The three ablations start from Hybrid; no combination of treatments.
There is no additional control judge, MCQ batch, oracle or enrichment call.
Terra's successful live structured-output preflight identified the requested model.

Freeze the same corpus/index as the four-case validation:
`data/corpora/validated_20261011`, `data/indexes/validated-bge_20261011`,
snapshot `vda-efe319281d52e13a`. Use the same policy, prompts (`answer-v3`,
`rewrite-v2`, `legal-judge-v2`), BGE tokenizer, 100-candidate and 4096-token /
12-unit final budgets. Graph is one hop/two additions; rerank uses the first 20;
multi-query uses the original and two variants. Embeddings remain local.

Both models use low reasoning, 4096 maximum output tokens, 120-second timeout
and zero retries. No provider/model fallback. Generation failures remain zeros;
missing judgments remain missing. Completed checkpoints are immutable. If a
technical interruption occurs, resume only matching identities; do not repeat
saved generations. Timings are descriptive because later profiles share a warm
query-embedding cache. No intermediate score changes the remaining configuration.

## Budget and sources

Maximum 321 new API attempts: one Terra schema preflight, up to 140 generations,
140 judgments, 20 rewrites and 20 reranks. The shared cumulative ledger limit is
5 EUR including earlier work, with 0.50 EUR reserve. The preceding ledger had
115 entries / 0.068710875 EUR. The model clients check reservations before each
dispatch; a terminal budget error stops new calls and preserves missing scores.

[Official pricing](https://developers.openai.com/api/docs/pricing), verified
2026-10-11, lists short-context standard prices of 0.10/0.50 USD per million
input/output tokens for Luna and 2/12 for Terra. The recorded input ceilings
include 1.25x cache writes: 0.125 and 2.50. Conservative EUR/USD conversion is
1.0, covering the existing sourced ECB reference. Configured input bounds are
below the 272K long-context threshold. Terra's context and structured Responses
capabilities are verified in its [model page](https://developers.openai.com/api/docs/models/gpt-5.6-terra).
Repricing the previous token distribution gives approximately 1.95 EUR for
20 cases; Terra output/reasoning consumption may differ. Amounts are conservative
estimates/reservations, not invoices.

## Interpretation and artifacts

This is an expanded, previously explored legacy sample covering 15 reference-law
groups, not an unseen benchmark. QID 0016 has historical references and a confirmed
cessation; QID 0100 concerns a time-limited measure. Preserve these cases and
explain temporal/source mismatches. Source identity warnings and short-gold
ambiguities remain visible; no agent approval changes the gold.

Compare all profiles on the same 20 questions with paired bootstrap intervals
and law-group sensitivity. Per-level results are descriptive, with five-case
denominators only after complete execution. A different generator/judge model
reduces self-judgment but does not establish human agreement or fulfill the
independent-control/full-benchmark promotion gates. Keep correctness, grounding,
context proxies and citation validity separate.

Frozen selection/configuration and logs: `data/diagnostics/terra20_20261011`.
Evaluation runs: `data/runs/terra20_20261011/<profile>`.
All seven evaluation runs and 80 retrieval-only records completed. Previous
pilot artifacts remain unchanged. The offline report is generated from these
saved artifacts at `data/reports/terra20_20261011/index.html`, with separate
automatic audit comments in `docs/results/09_case_comments.jsonl`.

## Measured results

Every profile attempted the same 20 questions. All 140 end-to-end scores are
available: 129 successful generations received Terra judgments, while 11
generation/citation failures remain operational zeros. There were no missing
judgments, retries or provider/model substitutions. Accepted citations are
formally validated against source text actually shown in the context; this
does not establish current legal applicability or human agreement.

| Profile | Fully correct | Rate | Score 0 / 1 / 2 | Generation errors | Correct and judged supported |
| --- | --- | --- | --- | --- | --- |
| No-RAG | 9/20 | 45% | 8 / 3 / 9 | 0 | Not applicable |
| Dense | 16/20 | 80% | 4 / 0 / 16 | 2 | 16/20 |
| Sparse | 14/20 | 70% | 5 / 1 / 14 | 4 | 12/20 |
| Hybrid | 17/20 | 85% | 2 / 1 / 17 | 1 | 16/20 |
| Multi-query | 15/20 | 75% | 4 / 1 / 15 | 1 | 15/20 |
| Graph | 17/20 | 85% | 3 / 0 / 17 | 2 | 17/20 |
| Rerank | 17/20 | 85% | 3 / 0 / 17 | 1 | 15/20 |

| Profile | L1 / 5 | L2 / 5 | L3 / 5 | L4 / 5 |
| --- | --- | --- | --- | --- |
| No-RAG | 4 | 2 | 0 | 3 |
| Dense | 5 | 3 | 4 | 4 |
| Sparse | 4 | 3 | 4 | 3 |
| Hybrid | 5 | 4 | 4 | 4 |
| Multi-query | 5 | 3 | 4 | 3 |
| Graph | 5 | 4 | 4 | 4 |
| Rerank | 5 | 3 | 4 | 5 |

The paired bootstrap uses 10,000 resamples and seed 20261010. Differences below
are percentage points in fully-correct rate, including operational failures.
All 21 pairwise comparisons pass artifact/scientific compatibility checks;
none passes the full-study and independent-control promotion gates.

| Comparison | Delta | QID bootstrap IC95% | Law-group sensitivity IC95% |
| --- | --- | --- | --- |
| Dense minus No-RAG | +35 | +10 to +60 | +11.1 to +57.1 |
| Hybrid minus No-RAG | +40 | +15 to +65 | +16.7 to +63.2 |
| Hybrid minus Dense | +5 | 0 to +15 | 0 to +17.4 |
| Multi-query minus Hybrid | -10 | -30 to +10 | -28.6 to +10 |
| Graph minus Hybrid | 0 | 0 to 0 | 0 to 0 |
| Rerank minus Hybrid | 0 | -15 to +15 | -15 to +15 |

Graph and Hybrid have identical fully-correct outcomes on all 20 QIDs; their
ordinal scores and errors differ. The degenerate observed bootstrap interval
is not evidence of equivalence on new questions. Rerank improves one case and
degrades another. Hybrid improves one fully-correct outcome over Dense. These
small, explored comparisons do not establish a robust winner among retrieval
profiles, and multiple descriptive comparisons are not confirmatory tests.

The expected article appears in the final context for Dense 18/20, Sparse
15/20 and Hybrid/Multi-query/Graph/Rerank 17/20. Hybrid's original candidate
pool contains it in 18/20 cases; reranking changes order/selection, not pool
recall. These are legacy-reference proxies: unjudged alternative evidence is
not automatically irrelevant, and an expected article can be historical.

## Case audit and interpretation

- **0018:** all six RAG profiles retrieve the expected provision but replace
  source `allegata` with `allegate` inside a quotation. Exact-source validation
  rejects the generation. The original payload and context are retained; no
  repair or new generation is folded into this experiment.
- **0016:** the legacy reference law is confirmed ceased. Several successful
  answers use an amending law quoting an earlier formulation of that law.
  A Terra score of 2 and exact quotation do not establish an available benefit
  at the snapshot date. This needs temporal/source review before a current-law
  benchmark claim.
- **0095:** the expected article equates accommodations for tourist-tax
  purposes, but does not itself establish regional collection or an increase
  in regional revenue. Terra penalizes the RAG responses against the affirmative
  short gold. Record a review proposal rather than silently correcting the gold.
- **0046:** Dense makes a separate, source-contradicted claim about an authority.
  Multi-query/Rerank are penalized for source-supported exceptions that also
  appear in responses graded 2. This is a primary-judge/gold consistency concern.
- **0066, 0083 and 0100:** provisions contain explicit past or limited periods.
  **0059** concerns 2025–2027, which includes the snapshot year. Keep their
  actual scopes distinct; no blanket temporal exclusion is inferred.
- Some health-law contexts contain a source identity conflict between 2002
  and 2000, although the panel's expected references have no conflicting
  headings. Exact quotations do not reconcile the identities; keep the
  document warning and require source review.

Automatic comments are interpretation, not approved gold revisions or human
legal judgments. They expose source spans and judge disagreements alongside
unchanged scores. No corpus, index, status rule, prompt or Python implementation
was changed during the frozen run. Source SHA-256 remains
`3cc5bb20e001962f46fd65f35386fb19a35c3ea2ac8b04c3f17e453536656b8a`;
run code identity is the clean protocol commit `1ac1014`.

The positive RAG versus No-RAG result warrants further investigation. Keep
Hybrid as the provisional experimental reference because the additional
treatments did not increase its aggregate fully-correct rate. Before a larger
paid study, address quotation copying in a separately versioned experiment
and obtain human review of the identified gold/temporal/judge cases. Do not
retrospectively change this pilot or claim adoption under the planned gates.

## Measured cost and completeness

The new work made **310 API calls**: one Terra schema preflight, 140 Luna answer
attempts, 129 Terra judgments, 20 Luna rewrites and 20 Luna reranks. Local BGE
query embeddings and 80 retrieval diagnostic records made no paid embedding
or extra chat call.

The shared ledger increased by **1.89508825 EUR**; cumulative accounting is
**1.963799125 EUR / 425 entries**, including earlier experiments. All new calls
settled; the earlier uncertain reservation remains unchanged. These are
conservative price-based estimates, not OpenAI invoices. The 5 EUR cumulative
cap and 0.50 EUR reserve were respected. No additional control judge, Utopia,
Sol, MCQ or enrichment batch was executed.

## Artifact verification

The final static audit passed all 13 checks with no unresolved issues. It
reconciles selection, input/output hashes, 140 result records, 80 retrieval
records, requested/returned models, 242 accepted source quotations, budgets,
metrics, 21 comparisons and accounting. The report contains 3,449 HTML pages;
10,566 local links resolve, resources are offline and generated scripts match
the expected template. Twenty automatic comments persist outside the report.
This checks saved artifacts; browser pixel review and human legal review have
not been performed. Existing software validation is 313 offline tests, with no
Python source change in this expansion.
