# Current snapshot validation

## Source and reproducibility

Official source: `https://www.consiglio.vda.it`, coordinated details requested
with `versione=V`. Snapshot `vda-efe319281d52e13a` was acquired between
2026-10-10 12:22:15 and 13:26:27 UTC. The acquisition manifest records
`complete=true` and no errors. Offline inspection of both 318-page catalogue
passes finds the same 3,176 source IDs. This establishes completeness against
that observed catalogue, not legal completeness or an official certification.

The catalogue hash, unique source IDs/paths, exact raw-file set and all 3,176
raw SHA-256 hashes were checked before preprocessing. Source decoding uses
ISO-8859-1 for 3,174 records and Windows-1252 for two records. The latter are
sources 7743 and 9321, whose legacy decoding previously exposed C1 characters.
The current decoded corpus has zero reported encoding anomalies; raw bytes
remain unchanged.

Definitive artifacts: `data/corpora/current-final`; independent repeat:
`data/corpora/current-final-repeat`. Both use parser 3, conservative status 1,
explicit graph 2 and contextual chunking 2. All seven JSONL files and
`quality.json` have identical SHA-256 hashes across the two builds. Manifests
retain their own timestamps and code identities. Measured durations: 70.94 s
and 83.21 s. Only the cached pinned BGE-M3 tokenizer was loaded, without model
weights, embeddings or language-model calls.

`snapshot_date` equals the recorded acquisition finish time. Every document
has a parsed enactment date, a nonempty title and its primary source heading
as human-readable identity, with the display label "Testo vigente" removed.
Canonical legal and graph IDs remain stable and separate from that heading.
Twenty-eight publication dates remain unextracted. A separate tokenizer pass
with special tokens enabled measured a maximum of 510 tokens across all
21,322 complete embedding inputs; none exceeds 512.

## Corpus result

| Artifact or diagnostic | Count |
| --- | ---: |
| Documents / ordered blocks | 3,176 / 171,722 |
| Units / graph nodes | 76,912 / 80,088 |
| Articles / paragraphs / letters | 16,785 / 39,402 / 20,168 |
| Annexes / unstructured units | 136 / 421 |
| Cessation events / unresolved | 3,511 / 762 |
| Relations / unresolved | 119,630 / 2,265 |
| Retrieval chunks | 21,322 |
| Blocks with an explicit exclusion reason | 42,115 |
| Documents with omitted annexes | 66 |

Document availability: 1,271 body, 1,878 metadata-only, 27 unstructured.
Document status: 1,936 confirmed cessation, 1,219 no cessation evidence,
21 unknown. Chunk status: 1,083 confirmed cessation, 19,770 no cessation
evidence, 469 unknown. The default current policy excludes confirmed cessation
and retains unknowns. Absence of detected cessation does not certify validity.

All 17 software quality gates pass, including exact unit/chunk spans,
scope-compatible confirmed event targets, resolved parents and complete token
budgets. Remaining warnings include 541 duplicate unit labels, 44 duplicate
note anchors, 17 conflicting secondary headings and nine omitted articles.
These are explicit diagnostics, not a measurement of legal parsing accuracy.

## Regression fixes and remaining review

Full-current inspection exposed repeated numeric identity paragraphs being
used as thematic titles, multiline preambles being treated as operative text,
and a missing closing quote swallowing later root articles. New raw fixtures
cover sources 11541, 11621 and 11824. At frozen-corpus validation, the focused
corpus suite had 36 passing tests and the UI suite had eight. The original LR 42/1992, repeal stub,
LR 44/1991, note 40 of LR 21/2003 and omitted-annex cases remain covered.

Source 11824, LR 10/2026, contains an unclosed replacement quote in article 6.
A narrowly scoped rule recovers the next explicit anchored root article and
records `source_quote_boundary_recovered:117`. Articles 7–11 survive as root
articles, and genuinely quoted nested articles remain within their amending
provision. No punctuation is invented. Paragraph interpretation within the
unclosed article 6 quote still requires human review; exact text preservation
does not prove its semantic segmentation.

The source contains four misspelled identity paragraphs beginning "egge
regionale" (1704, 1957, 2054, 2246), retained in thematic metadata. Six ceased
sources have publication abbreviations without the expected separator (1574,
1575, 1576, 1690, 3282, 5545), and source 8741 has a doubled opening parenthesis
before its publication line. Those seven lines remain body-classified; source
8741 is eligible. These recognition limits are disclosed rather than silently
rewriting source text. They require review before treating title or publication
classification as a semantic gold standard.

Status and relation rules intentionally abstain on unresolved, conditional or
partial evidence. A stratified human sample is still needed to measure false
cessation, segmentation and reference-resolution errors. No answer-quality
improvement is established by corpus counts or structural gates.

## Comparison with legacy identity

The official snapshot contains all 3,145 legacy source IDs plus 31 new IDs;
none is missing. All 3,145 shared raw-byte hashes differ. Current sources are
full official pages with their original encodings, while legacy sources are
UTF-8 article fragments. Those byte differences cannot be interpreted as
3,145 changed laws without a separate normalized-text comparison.

The new IDs are 11501, 11502, 11521, 11541, 11542, 11543, 11561, 11581, 11601,
11621, 11641, 11642, 11661, 11681, 11701, 11702, 11721, 11741, 11761, 11781,
11782, 11801, 11821, 11822, 11823, 11824, 11841, 11861, 11862, 11881, 11901.
The latest observed enactment date is 2026-10-02. This describes the acquired
records, not a guarantee that every legal effect has been incorporated.

Earlier `current` and `current-verified` artifacts remain diagnostic snapshots;
they are not the definitive corpus. Historical as-of reconstruction is outside
this implementation.
