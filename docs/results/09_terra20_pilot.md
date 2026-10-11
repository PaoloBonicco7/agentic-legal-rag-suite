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
The final report will be generated from saved artifacts at
`data/reports/terra20_20261011/index.html`; at protocol freeze, runs and report
are not yet complete. Previous pilot artifacts remain unchanged.
