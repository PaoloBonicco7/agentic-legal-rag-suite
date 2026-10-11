# Rebuild delivery

## Software and source

The isolated branch is `feature/legal-rag-rebuild`. The original checkout was
preserved and a private source/research archive was verified before replacing
the implementation. Original evaluation CSVs are byte-identical; no gold was
corrected or approved by an agent.

The package provides acquisition, deterministic preprocessing, conservative
status evidence, a JSONL graph, dense/sparse/hybrid retrieval, bounded graph
expansion, multi-query, LLM reranking, structured answers, citation validation,
checkpointed evaluation, paired comparisons, offline reports, a stateless UI
and thin notebooks. OpenAI requests use one persistent budget ledger with a
20 EUR ceiling and a 4 EUR validation reserve.

The current snapshot is `vda-efe319281d52e13a`, acquired on 2026-10-10 between
12:22:15 and 13:26:27 UTC. Both 318-page catalogue passes contain the same
3,176 source IDs. Every acquired file hash was verified. Catalogue agreement
does not establish an atomic view or certify legal applicability.

`data/corpora/current-final` contains 3,176 documents, 76,912 units, 119,630
relations and 21,322 chunks. All 17 structural gates pass; complete embedding
inputs have at most 510 tokens. Seven JSONL files and `quality.json` are
byte-identical to an independent repeat. Semantic annotation accuracy still
requires human sampling. See the [source audit](../notes/03_current_snapshot.md)
for observed warnings and limits.

Manual inspection also identified compound references that need separate
article targets or may attach an article to the wrong law. The
[reference review](../notes/07_reference_review.md) records exact source evidence
and IDs. Graph expansion remains experimental until those cases are reviewed;
structural gates alone do not establish correct links.

## Execution status

The initial 2026-10-10 delivery was restricted to setup and documentation,
as requested after the provider preflight. No complete answer benchmark or
retrieval ablation was run at that time.
Before the execution stop, the implementation subsequently committed as
`b92c857` passed 246 offline software tests, including Streamlit AppTest, local
Qdrant integration and portable-archive checks. Tests use controlled fixtures;
they do not establish answer quality on the real benchmark. Progress callbacks,
CLI wiring and the explanatory HTML were added afterwards without executing
tests, compilation, rendering or pipeline runs. The newly authorized work on
2026-10-11 passed 251 offline tests, including that wiring, and completed the
[small Luna pilot](07_luna_pilot.md). See the [manual workflow](../notes/06_manual_workflow.md).

The offline corpus report is `data/reports/current-final/index.html`. It records
the definitive corpus and separately stored audit comments. Its generator
uses no model or new retrieval. Browser visual verification was unavailable;
static checks and their scope are recorded in
`data/diagnostics/current_report_static_qa.json`: 3,178 HTML pages, 9,533 local
links and all recorded input/output hashes passed before the later stop.

The already-running BGE-M3 job completed successfully in
`data/indexes/current-bge`: 21,322 points, 1,024 dense dimensions and native
sparse vectors, with `ready_for_retrieval=true`. It used MPS/float16, inference
batch 8, upload groups of 64 and no cached vectors. The measured job duration
was 3,874.72 seconds; this is local execution evidence, not a retrieval-quality
result. Every payload, ID, dimension and count was reconciled by the build.
The record is `data/diagnostics/current_index_completed.json`. Its manifest
retains the actual build-time commit/dirty/source-hash identity; later code
edits do not retroactively rewrite it. Older partial indexes and parser outputs
are diagnostic artifacts, not the delivered index.
The [local execution note](../notes/04_local_embedding.md) explains the measured
hardware calibration and its limits.

Utopia's live model-inspection preflight returned HTTP 403 on 2026-10-10; the
configured endpoint, requested 32,768-token window and model are recorded in
`data/diagnostics/utopia_final_preflight.json`. Model identity/capabilities
therefore remain unverified. OpenAI was unavailable at the initial delivery;
the user subsequently configured the private key and authorized limited Luna
tests. The 2026-10-11 pilot made 46 calls for a conservative 0.02172325 EUR.
No-RAG, Dense and Hybrid each achieved 5/8 fully correct answers: no aggregate
improvement was observed. Full Advanced ablations and independent judge
validation remain unexecuted.

No new phase was launched during the setup-only restriction. The later pilot
was explicitly authorized. The portable bundle and restore commands are ready;
no final bundle or restoration was run in this pilot. The previously
verified private backup preserves the original research checkout.

## Remaining empirical work

First address the citation-format and evidence-selection failures diagnosed
by the pilot, and validate judge behavior without changing frozen scores.
Verify Utopia model identity and context capacity if using that provider,
refresh sourced OpenAI prices/exchange rates before further paid batches,
and run the full 100-intent
No-RAG/Dense/Sparse/Hybrid baseline with common judges and frozen settings.
Then inspect errors, seek human approval for any benchmark changes and measure
individual ablations before combinations. Graph, reranking and LLM enrichment
are implemented experiments, not adopted accuracy improvements.

The [numbered specs](../specs/README.md) define the commands and acceptance gates.
The [evaluation protocol](06_experiment_protocol.md) separates software behavior from
scientific evidence. Missing judgments, provider failures and negative results
remain visible; they cannot justify a final promotion decision.
