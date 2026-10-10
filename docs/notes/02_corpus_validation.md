# Corpus validation

## Frozen legacy result

Snapshot: `legacy-f72aee26394b281a`. Capture date remains unknown; importing the
files on 2026-10-10 does not make their legal contents current.

Reference artifacts: `data/corpora/legacy-verified`; independent repeat:
`data/corpora/legacy-repeat`. Both use the pinned BGE-M3 tokenizer, target 384
and maximum 512 tokens, counting the complete embedding input. Neither run
loads embedding weights or calls a language model.
These frozen results use `deterministic-parser-1`, `explicit-graph-1` and
`contextual-chunking-1`; the later current-snapshot validation is documented
separately in `03_current_snapshot.md`.

| Artifact or diagnostic | Count |
| --- | ---: |
| Documents | 3,145 |
| Ordered source blocks | 161,504 |
| Units | 71,407 |
| Articles / paragraphs / letters | 15,833 / 36,469 / 18,524 |
| Annexes / unstructured units | 134 / 447 |
| Cessation events / unresolved | 3,409 / 748 |
| Relations / unresolved | 111,660 / 2,167 |
| Graph nodes | 74,552 |
| Retrieval chunks | 19,770 |
| Blocks with an explicit exclusion reason | 39,575 |
| Documents with omitted annexes | 66 |
| Preserved legacy encoding anomalies | 2 |

Document availability: 1,240 body, 1,875 metadata-only, 30 unstructured.
Document status: 1,916 confirmed cessation, 1,201 no cessation evidence,
28 unknown. Chunk status: 777 confirmed cessation, 18,530 no cessation
evidence, 463 unknown. Unknowns remain eligible under the default policy.

The exclusion-reason count covers explicit structural reasons, such as headings
and omission markers. Other non-normative regions remain separately visible in
the quality report; it is not a count of every block omitted from retrieval.
Median chunk length is 113 whitespace words; 103 chunks contain at most five
words. Word counts describe the result, not the tokenizer budget.

## Verification

- All 17 structural quality gates pass, including exact span reconstruction,
  scope-compatible confirmed targets and complete embedding budgets.
- A separate tokenizer pass with special tokens enabled measured a maximum
  complete input of 510 tokens across all 19,770 chunks; none exceeds 512.
- All seven principal JSONL files and `quality.json` have identical SHA-256
  hashes across independent builds. Manifests intentionally include different
  execution timestamps and code identities.
- Measured build durations were 68.70 s and 83.84 s on the development machine.
  These are observations, not performance guarantees.
- At this validation, the focused corpus regression suite contained 32 passing
  tests. It exercises
  real laws, adversarial scope/date cases, source hash rejection, immutable
  outputs and exact introductory context in split lists.
- Twenty distributed official full pages already acquired on 2026-10-10 and
  twenty additional pages containing normative bodies were parsed in memory.
  Header identity, source hashes, source spans and navigation exclusion passed.
  These samples do not establish acquisition completeness; the full current
  snapshot requires its independent acquisition gates.

## Regression evidence

| Case | Required behavior |
| --- | --- |
| LR 1/1950, source 273 | Thematic title retained; repeal stub has no fabricated article or chunk. |
| LR 42/1992, source 2065 | Whole-law exceptions abstain; articles 2 and 5 remain available; historical article 4 cannot reopen the body. |
| LR 44/1991, source 1967 | Explicit editorial whole-law repeal propagates even when the source retains substantive text. |
| LR 21/2003, source 2711, note 40 | LR 21/2011 is the repealing authority; the following reference to LR 75/1990 belongs to its own amendment clause. |
| LR 33/1993, source 2144 | Omitted annex is separate from article 25 and yields no normative chunk. |
| Ordinal day and source 1506 | `1°` dates parse; the primary heading determines identity; a contradictory secondary heading is reported. |
| Quoted replacement article | Nested article numbering belongs to the outer amending provision, rather than becoming a duplicate root article. |
| Inline rubric, numeric sublists and financial values | Rubrics remain headings; thousands separators and nested `1)` items cannot create spurious numbered paragraphs. |
| Negation, saved effects, future dates, duplicate notes and constitutional rulings | Ambiguous evidence never triggers an automatic exclusion. |

## Remaining limits

The report lists 507 duplicate unit labels, 41 duplicate note anchors, 17
conflicting secondary identity headings and six omitted article headings.
Duplicates receive explicit disambiguated IDs and warnings; ambiguous note
bindings cannot confirm cessation. No operative body quotation remained open.
The two pre-existing C1 anomalies are retained in raw legacy assets and reported;
new encoding anomalies in acquired sources fail readiness.

These gates validate reproducibility and structural integrity. They do not
certify legal applicability, recall of every cessation formula or retrieval
accuracy. Strict rules intentionally abstain on partial, conditional and
unresolved effects. A stratified human annotation sample and reviewed benchmark
evidence are still required to measure semantic accuracy. Historical as-of
reconstruction, inferred legal relations and automatic adoption of LLM proposals
are outside this implementation.
