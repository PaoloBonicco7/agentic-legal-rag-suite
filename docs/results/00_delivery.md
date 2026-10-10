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

## Execution status

The current BGE-M3 index is being built in `data/indexes/current-bge` with
MPS/float16, inference batch 8 and upload groups of 64. Its ready flag remains
false until every payload, ID, dimension and count is reconciled. Older partial
indexes and parser outputs are diagnostic artifacts, not the delivered index.
The [local execution note](../notes/04_local_embedding.md) explains the measured
hardware calibration and its limits.

Utopia's live model-inspection preflight returned HTTP 403 on 2026-10-10; the
configured endpoint, requested 32,768-token window and model are recorded in
`data/diagnostics/utopia_final_preflight.json`. Model identity/capabilities
therefore remain unverified. No OpenAI runtime key is available. No paid calls
have been made, and no answer-quality improvement has been measured.

## Remaining empirical work

After access is restored, verify Utopia model identity and context capacity,
refresh the sourced OpenAI prices/exchange rate, and run the full 100-intent
No-RAG/Dense/Sparse/Hybrid baseline with common judges and frozen settings.
Then inspect errors, seek human approval for any benchmark changes and measure
individual ablations before combinations. Graph, reranking and LLM enrichment
are implemented experiments, not adopted accuracy improvements.

The [numbered specs](../specs/README.md) define the commands and acceptance gates.
The [evaluation protocol](06_experiment_protocol.md) separates software behavior from
scientific evidence. Missing judgments, provider failures and negative results
remain visible; they cannot justify a final promotion decision.
