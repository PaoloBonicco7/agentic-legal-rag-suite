# 03 — Conservative status and explicit graph

## Purpose

Attach inspectable cessation evidence and a small explicit graph to the current
source snapshot. Avoid false exclusions and speculative graph edges.

## Inputs

Documents, ordered blocks, units and the snapshot acquisition date from step 02.
No model-based interpretation or historical version reconstruction is used.

## Outputs

Shared `Event` and `Relation` records and graph nodes in JSONL. Document/unit
status is propagated to retrieval chunks. Events retain the decisive source
clause, source span, target scope, optional effect date and resolution.

## Pipeline

1. Read standalone editorial whole-law repeal formulas and operative note
   clauses. Separate sentences and semicolon clauses without splitting Italian
   citation abbreviations or dates.
2. Bind article/paragraph notes only through unique, scope-compatible body
   backlinks. Plural article headings may explicitly identify several targets.
   Duplicate anchors and ambiguous backlinks remain unresolved.
3. Confirm cessation only for explicit unconditional formulas. Exceptions,
   saved effects, negation, constitutional rulings affecting prior formulations,
   conditional effects and unresolved dates never trigger automatic exclusion.
4. Compare an explicit effect date with acquisition date. Future or undated
   relative effects remain unresolved. Observation date is never an effect date.
5. Propagate confirmed law/article cessation to descendants. Unknown scope
   cannot cancel a confirmed whole-target cessation. No roll-up of several
   ceased children certifies the entire law as ceased.
6. Export containment and clause-local reference/intervention relations.
   Resolve cited laws against the corpus identity registry; resolve a cited
   article only when it exists uniquely. Missing targets remain explicit.

## Contract

Statuses are `ceased_confirmed`, `no_cessation_evidence` and `unknown`.
`no_cessation_evidence` means no confirmed cessation in this source snapshot;
it does not certify legal applicability. "Testo vigente", source URL version V
and repertoire membership are never legal status evidence. Legacy provenance
cannot establish current freshness. Retrieval defaults to retaining unknowns.

Relation kinds are `contains`, `references`, `repeals` and `amends`.
Containment links document/article/paragraph/list hierarchy. Intervention direction
follows the clause: a passive "abrogato dalla LR X" makes X the repealing source;
an active "abroga la LR X" makes the current provision the source. Historical
quotations yield references only, never current intervention or cessation facts.
Each cited target gets its own clause evidence; note 40 of LR 21/2003 must not
make LR 75/1990 the repealing authority for the preceding sentence.

## Quality Gates

- Confirmed cessation always has an existing scope-compatible target and span.
- Only confirmed cessation can exclude normative chunks.
- Negated, future, exceptional, conflicting and constitutional effects abstain.
- Resolved edges point to exported nodes; unresolved edges preserve citation text.
- No self-loop or unsupported intervention edge is emitted.
- Graph export is dependency free; unresolved counts are reported separately.

## Notebook Role

Inspect event evidence, ambiguous cases and graph neighbors. Graph expansion is
an ablation consuming these artifacts, not a reason to infer additional facts.

## Acceptance Criteria

Real and synthetic regressions demonstrate no law-wide exclusion from an
article-only note, no historical heading reopening, safe exception handling and
clause-local relation assignment. Human review concentrates on unresolved
candidate exclusions and benchmark evidence; a temporal knowledge graph and
historical as-of answering remain future work.
