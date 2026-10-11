# 11 — Evidence-driven retrieval optimization

**Status: planned, not implemented.** The [strategy review](../results/10_strategy_review.md)
records measurements of the existing engine. This specification defines the next
changes; it does not claim that a combined Advanced profile has been validated.

## Purpose

Find the simplest tested strategy that improves supported, fully correct answers
over Dense RAG. Optimize the evidence actually shown to the generator, not just
the presence of a reference article in 100 candidates. A global optimum or a
large improvement is not guaranteed by this finite, explored benchmark.

The present operational reference is Hybrid. Graph, multi-query and reranking
remain separate experiments. Combining them is conditional on measured benefit.

## Inputs

- Frozen `validated_20261011` corpus/index and unchanged original question CSVs.
- The twenty-intent `terra20_20261011` answer/judgment checkpoints and the
  `strategy_review_20261011/full100` retrieval-only run.
- Source-inspected case evidence, separate human-reviewed relevance judgments,
  source-identity decisions and temporal/gold review proposals.
- One frozen configuration per comparison: BGE revision, cessation policy,
  candidate limit 100, context 4096 BGE tokens/12 units, generator/judge identities,
  response protocol, rule/prompt versions and code identity.

Review proposals do not change gold, denominators, corpus identities or current-law
eligibility automatically. Questions and source dates remain distinct.

## Outputs

- Versioned rule/configuration decisions with observed examples and rejection reasons.
- Per-case traces of explicit law resolution, branches, graph-edge selection,
  candidate promotion, reranking, source spans, context omissions and citations.
- Separate retrieval-only and generated-answer comparisons, with paired deltas,
  uncertainty, coverage, failures, timings, usage and conservative costs.
- Offline reports containing six source-inspected before/after cases, regressions
  and missing conditions. Their selection is illustrative, not a second metric sample.
- A frozen combined candidate only after its individual components pass their gates.

## Pipeline

### A. Establish trustworthy evidence and citations

1. Review source-identity conflicts, especially the LR 5/2000 body also returned
   under LR 5/2002. Keep original bytes and both provenances. Merge retrieval
   aliases only after the normative identity, source representation and article
   are verified. Identical text in different acts alone is insufficient.
2. Distinguish an amending act's quoted provision from that provision's target
   identity and temporal scope. A non-ceased host does not certify that a quoted
   benefit is currently applicable (Q0016). Unknown targets remain explicit.
   Full historical reconstruction is outside this step.
3. Add source-segment identifiers to the rendered evidence. The generator selects
   IDs; the backend materializes quotations from exact displayed spans. Segment
   boundaries follow legal units and preserve introductions/conditions. Count
   markers and metadata in the same 4096-token budget for every RAG profile.
4. Reject unknown, hidden or header-only segment IDs. Do not repair quotations
   with fuzzy matching or substitute words. Keep the old raw output and failure
   history. Formal quotation validity and semantic claim support remain separate.
5. Diagnose generation omissions independently: Q0036 already has relevant
   requirements in the prompt. If a common answer-prompt change is needed to
   preserve actors, conditions, exceptions and timing, test it on frozen contexts
   separately from retrieval, then freeze the same prompt for all new treatments.
   Do not add unbudgeted repair calls or include reference-answer checklists.

The citation protocol changes all RAG treatments together. Regenerate their
baselines under that protocol; do not attribute the removal of Q0018's
`allegate`/`allegata` failure to better retrieval. Source corrections that affect
chunks/payloads require a new corpus/index identity. Context-only transformations
need no document re-embedding when indexed text and identities are unchanged.

### B. Isolate small retrieval changes

| Order | Change | Why and bounded test |
|---|---|---|
| B1 | Explicit law identity | Resolve identifiers from question text against the catalog. If unambiguous, add a scoped Hybrid branch while preserving global retrieval. Q0057 asks for LR 14/2024 but Dense first returns LR 15/2024; Q0068 misses the requested article entirely. Compare with ordinary Hybrid before adding other levers. |
| B2 | Evidence completion | Preserve introductions, required conditions and exceptions; join source-linked adjacent units when necessary. Q0056 shows revocation but misses suspension prerequisites. Evaluate completion separately from ranking and canonical deduplication. |
| B3 | Local graph selection | Expand references whose evidence occurs in the selected source spans, not all references in the seed document. Prefer operative references over editorial amendment history; rank targets for the question instead of relation-ID ordering. Promote an already-retrieved target when useful. Keep one hop and at most two additions/promotions. |
| B4 | Metadata-aware reranking | Supply identity, title/rubric and status with each passage. Reorder the scored prefix while retaining the untouched tail. Start with 20; test 40 only if evidence outside the prefix matters. Original pool and final ranked candidates must remain independently visible. |
| B5 | Evidence-oriented multi-query | Retain the original question and at most two distinct variants addressing relevant missing aspects, such as requirements and exceptions. Preserve named acts, actors, negations, dates and amounts. Reject invented references. Test changed variants separately from fusion changes. |

B1 initially handles only one explicitly cited, unambiguously resolved regional
act. Unknown/multiple acts, inconsistent dates/numbers or another jurisdiction
use global retrieval only, with a reason in the trace. Resolve deterministically;
a unique catalog match does not repair a conflicting body identity.

This is an additional retrieval branch, not a global law filter: related acts and
exceptions must remain recoverable. Apply the same cessation filter in both
dense/sparse prefetches of each Hybrid branch. Global and scoped branches each
return at most 100; final fusion returns at most 100. Freeze outer fusion/ties
before testing, and record the extra search work. An article qualifier may guide
priority only when attached unambiguously to the resolved act, never globally
filter other articles. Keeping global search does not guarantee that external
exceptions survive packing: test their actual appearance in the context.

B2 deduplicates only verified equivalent source spans/canonical representations,
retains every provenance and never merges distinct legal acts merely by text hash.
Do not impose a universal two-chunk-per-law cap: one answer may need several
conditions from one law. Complete blocks must fit the budget; omissions retain reasons.

B3 tests reference locality, relevance and promotion separately before combining
them. A referenced exception can matter despite low lexical similarity; avoid
an unvalidated hard similarity cutoff. Q0007's agricultural additions and Q0056's
displaced sanction article are mandatory regression cases. The experiment tests
existing proven edges; it needs no new graph database or complete version ontology.

B4 may first use the existing LLM reranker. A local
[BGE reranker](https://huggingface.co/BAAI/bge-reranker-v2-m3) is an optional later
cost experiment, after batch-1 memory/latency checks on the 8 GB M1. Do not load
multiple large models or run concurrent index writes. Multilingual support does
not establish improvement on these questions.

B5's already-executed replay of original/variant weights 1:1:1, 2:1:1 and 4:1:1
found no context-article gain. Therefore increased original weight is **not adopted**.
If retested with new variants, freeze the formula: external linear weighted RRF
is `sum(weight / (60 + zero_based_rank))`. This differs from Qdrant's native
weighted-RRF formula; use neither interchangeably nor a changed default `k`.
Native dense/sparse fusion initially stays unweighted with `k=60`.

### C. Assemble and verify one candidate

1. Start with Hybrid plus the common source/citation corrections.
2. Measure each B intervention separately. Test at most two individually useful
   retrieval interventions in the first combination; extend only with evidence.
   Query expansion precedes search; graph expansion precedes final reranking;
   context completion and citation materialization use the same shared builder.
3. Compare the candidate with each contributing single component and remove one
   component at a time. Interactions can negate gains; negative components remain
   disabled. An adaptive query router is deferred until fixed policies work.
4. Freeze the selected candidate before the next answer batch. Apply the same
   code, prompts, source policy and context budget to all new baselines.

### D. Execute tests in increasing cost order

| Stage | Scope | Completion gate |
|---|---|---|
| D0 — offline behavior | Real source fixtures plus synthetic contract tests. | Correct identity/scope, exact citations, no gold access and no lost candidate tail. |
| D1 — local evidence | All 100 explored intents; baseline and one changed policy at a time. Reuse saved rankings where exact replay is possible. | Report gains **and** losses at pool/final/context stages; manually inspect decisive clauses for curated cases. No answer accuracy claim. |
| D2 — bounded answer pilot | The same fixed 20 QIDs, No-RAG/Dense/Hybrid/one candidate; Luna generator and Terra primary judge. | Up to 80 generations + 80 judgments; optional rewrite/rerank up to 20 calls each. No grid of all combinations. No retries by default. |
| D3 — finalist validation | No-RAG/Dense/Hybrid/frozen candidate on the legacy 100 and, separately, a fully approved current view. | Complete paired scores and independent judge control on a fixed stratified sample; reviewed disagreements and source/gold issues. |
| D4 — generalization | New questions and human-verified evidence frozen before further tuning. | Report separately; the remaining old 80 intents are not an unseen test. |

A separate paid smoke, if needed, adds calls and is not silently reused across
different selection identities. Approved oracle contexts may diagnose generation
limits on disputed cases; they are a controlled reference, not a theoretical ceiling
or deployable retrieval strategy. An oracle run is separate from the four treatments.

No new paid call is authorized by rendering this plan. For a subsequent authorized
D2 batch, propose an **incremental 2 EUR ceiling**, retaining the cumulative
5 EUR cap and 0.50 EUR reserve used by the last pilot. Freeze current prices,
exchange-rate source, exact input estimates and maximum output before dispatch.
The ledger currently accounts for 1.963799125 EUR. At a budget limit, retain
checkpoints and label missing results. D3/control/oracle may need a separately
agreed allocation; do not assume the full experiment fits the remaining budget.

## Contract

Keep one Python engine. Proposed shared additions belong in existing Pydantic v2
contracts (`extra="forbid"`, JSON-mode exports): source segment/span identity,
resolved question-act identity, versioned strategy configuration and trace decisions.
Do not create a second retrieval stack, notebook logic or provider-specific context.
No runtime resolver, graph selector, rewrite or reranker receives gold/qrels.

The orchestrator owns contracts and integration. After interfaces freeze, delegate
source/context/citation work, retrieval policies, and evaluation/reporting to
separate file owners. Each handoff includes changed files, rule/prompt versions,
focused tests and measured limits. Embedding/index/paid batches stay sequential.

## Quality Gates

- Identity fixtures: exact act/date/number, unknown or ambiguous act, multiple
  acts, conflicting body identity, historical target, external-law exception,
  and explicit law mentions without gold access.
- Context fixtures: introduction plus list, exception in an adjacent unit,
  verified alias with multiple provenances, identical text in different laws,
  and complete rendering within the unchanged budget.
- Graph fixtures: mention outside a seed span rejected, operative local edge
  retained, unresolved target not invented, useful target already in pool
  promoted, target cessation policy unchanged, hash order cannot decide relevance.
- Reranker fixtures: missing/duplicate scores rejected; metadata supplied;
  prefix 20 reordered with the remaining 80 preserved; exact tie policy recorded.
- Citation fixtures: segment IDs refer only to displayed body spans; Unicode and
  whitespace offsets resolve exactly; unknown IDs fail; quotes cannot be fabricated.
- Query fixtures: original always retained, duplicate variants removed, invented
  act/date/amount or changed negation rejected; numerical RRF formula checked.
- Evaluation/report fixtures: common identity across new baselines; incompatible
  old/new runs remain descriptive; missing judgments stay missing. Unjudged chunks
  are not false positives. A correct article with missing decisive comma is not
  full evidence coverage. Generation failures remain end-to-end failures.

Correctness and support remain separate. Human reviewers adjudicate Q0012/Q0016/
Q0046/Q0095/Q0100 and source identity; agents only propose. Freeze rubric changes
before rejudging all relevant profiles, rather than selectively rescuing a winner.

## Notebook Role

Display saved before/after rankings, source spans, omissions and paired results.
Call existing modules; do not implement strategies, change gold or dispatch paid
calls implicitly. Offline HTML remains the primary detailed presentation artifact.

## Acceptance Criteria

Technical gates must pass before answer testing. A pilot may choose a provisional
candidate but cannot authorize promotion. Final adoption follows the full-view
and independent-control requirements of [step 06](06_evaluation.md), with no
known source/exclusion/citation regression and acceptable measured cost/latency.

As a practical research target, seek at least **+10 percentage points over Dense**;
this is an objective, not a predicted result. Report paired IC95% and law-group
sensitivity; call a gain convincing only when reviewed evidence and intervals
support it. Keep a simpler treatment at ties, and report uncertain gains as
exploratory. Finding six attractive examples cannot override the complete results.

Literature motivates hypotheses, not local effect sizes:
[DMQR-RAG](https://arxiv.org/abs/2411.13154) for diverse query variation,
[SAT-Graph RAG](https://journals.sagepub.com/doi/10.3233/FAIA251598) for scoped
structural/temporal evidence, and [Lost in the Middle](https://arxiv.org/abs/2307.03172)
for testing evidence placement. None demonstrates that this combined policy
improves Luna on Valle d'Aosta law; see the bounded [source table](../notes/01_literature.md).
