# 07 — Offline reporting

## Purpose

Make every result inspectable without running an LLM, opening an index or relying
on remote assets. Distinguish operational completeness, benchmark correctness,
formal citation validity and judged grounding.

## Inputs

`build_report(corpus_dir: Path | None, run_dirs: list[Path], output_dir: Path)`
returns the generated `index.html`. It can render a corpus-only report before
evaluation. Existing output directories are rejected rather than overwritten.

Additional `diagnostic_dirs=()` accepts retrieval-only runs and shows their proxies
under separate labels, without generation-correctness metrics.
Optional `comments_path=None` reads a separate JSONL of `ReportComment` records:
`comment_id`, `author`, `text` and optional `qid`, `run_hash`, `document_id`.
`load_comments` validates unique IDs; `write_comments` creates a durable file or
appends explicitly, never editing an HTML report.

## Outputs

Outputs are a homepage, one experiment page per run, one detail page per QID/run,
and a comparative page per QID/format across supplied runs. Retrieval-only inputs
also create a per-QID comparison of profiles, with links to saved contexts.
A corpus page is included when provided, together with JSON comparison artifacts. CSS is local and
all source/model text is escaped. No external CDN, analytics or LLM calls are used.

## Pipeline

1. Load corpus/evaluation manifests, checkpoints and metrics; check identities,
   file hashes, result counts and completeness before presenting comparisons.
2. Homepage: metrics, paired deltas/intervals, promotion gates, operational errors,
   costs, denominators, temporal flags and a clear explored-benchmark statement.
   Snapshot IDs and dates are visible on the homepage, corpus and individual or
   comparative QA/retrieval pages, including pending cases.
3. Experiment: effective identities and configuration, distributions, missing
   judgments, costs with conservative accounting bases, annotations and links to
   all selected QIDs, including unvisited cases and saved unjudged checkpoints.
4. QID: question/gold, answer and status, correctness and grounding rationales,
   actual packed context, source snippets, exact quote validation, candidate ranks,
   query variants, original candidate pool/final selection trace, omissions, usage
   and errors.
   Comparative QID pages place gold, revisions, answer, score/status and judge
   reasons side by side. Missing checkpoints and unselected cases remain distinct.
   Existing compatibility and promotion gates qualify QA comparisons; incompatible
   or incomplete runs support a descriptive display only. Retrieval comparisons
   show Article Success/MRR, context proxies, timings and context links; they never
   fabricate an answer score. Homepage comparison tables are filterable.
5. Corpus: snapshot provenance, parsing/status counts and inspectable document
   evidence, extracted events and graph relations. `no_cessation_evidence` never
   means legally certified current. Artifact integrity warnings are visible even
   when only one evaluation run is supplied.

## Contract

Incompatible or incomplete runs remain inspectable but are excluded from complete
paired comparisons. Gold proposals and approved corrections are shown with their
reviewer/status; historical and approved-current metrics are never merged.
Incomplete fully-correct/ordinal rates on the planned denominator are explicitly
lower bounds. Recorded costs may be estimates or reservations; the persistent
provider ledger is authoritative. Usage-record counts are not HTTP-attempt counts.
The snapshot date comes from the supplied corpus only when both snapshot ID and
corpus manifest hash match the run or diagnostic identity. Otherwise the page
states that the date is unavailable; a new corpus date never labels older answers.
Interpretive comments remain separate from results and gold reviews. Reports
record only their source path and checksum in the manifest and escape their text.
The homepage shows every comment with its scope; QID pages require the matching
QID and, when specified, the exact run hash. Corpus pages show matching document
comments without a run binding. Rendering never modifies the comment source.

## Quality Gates

- Reports work offline, including a corpus-only run and failed/incomplete runs.
- HTML cannot execute injected source text or unsafe source URLs.
- Every displayed score resolves to a QID and original checkpoint.
- JSON numbers and HTML metrics agree; denominators are explicit.
- Reporting does not overwrite inputs or silently choose a latest run.
- Comments persist independently, cannot cross a run binding and execute no HTML.
- Focused tests cover escaping, missing scores, incompatible identities and links.

## Notebook Role

Optional notebooks display the generated report or load its JSON. They do not
redefine metrics or rerun evaluation. They may call `write_comments` to preserve
interpretive notes separately before rebuilding a report.

## Acceptance Criteria

Focused tests cover corpus-only and retrieval-only reports, HTML/script escaping,
unsafe links, secret redaction, missing scores and artifact-integrity warnings.
Reports expose saved evidence at each stage and remain usable entirely offline.
