# Rebuild delivery

## Delivered software and sources

The isolated branch is `feature/legal-rag-rebuild`. The original checkout and
verified private source/research backup remain preserved. Original evaluation
CSVs and gold are unchanged; no agent approved a benchmark revision.

The package implements acquisition, deterministic preprocessing, conservative
status evidence, a JSONL graph, local dense/sparse/hybrid retrieval, bounded
multi-query/graph/reranking ablations, structured answers, citation validation,
checkpointed evaluation, paired comparisons, offline HTML, a stateless UI and
thin notebooks. CLI, UI and notebooks call the same Python modules.

Snapshot `vda-efe319281d52e13a` was acquired on 2026-10-10, between 12:22:15 and
13:26:27 UTC. Two 318-page catalogue passes agreed on 3,176 source IDs; source
file hashes were verified. This records provenance, not an atomic legal view
or certified current applicability.

The delivered corpus is `data/corpora/validated_20261011`: 3,176 documents,
76,912 units, 21,322 chunks and 119,398 relations. All 17 structural gates pass;
eight canonical files are byte-identical across two builds. Texts, chunks and
status are unchanged from the previous corpus. Compound-reference corrections
were applied; 4,010 destinations remain unresolved and are visible. Graph
traversal uses the three documented relation types; correct
structure does not establish semantic correctness. See the
[source audit](../notes/03_current_snapshot.md) and
[reference review](../notes/07_reference_review.md).

The new index, `data/indexes/validated-bge_20261011`, contains 21,322 points with
1,024-dimensional dense and native sparse BGE-M3 vectors. All document vectors
were reused from the verified local cache; no new document embedding inference
or paid embedding call was required. Payloads and corpus identity were reconciled.

## Verified execution

The current implementation passed **313 offline tests**. Controlled tests verify
software contracts, including provider failures, local Qdrant, UI, checkpoints
and archive validation; they do not establish legal answer quality.

The bounded Luna pilot used four original no-hint intents, one per difficulty
level and three laws, with seven separate profiles. Generator and judge were
both `gpt-6-luna`, with no retry or independent control judge.

| Profile | Fully correct / four intents |
| --- | --- |
| No-RAG | 2/4 |
| Dense | 3/4 |
| Sparse | 3/4 |
| Hybrid | 4/4 |
| Multi-query | 2/4 |
| Graph | 3/4 |
| Rerank | 3/4 |

One No-RAG provider failure and one Graph quotation-of-header failure remain
operational zeros in the fixed denominator. Judge inconsistencies and a
health-law source identity conflict limit interpretation. Four explored cases
cannot support promotion or an unseen-test claim. The
[validation note](08_pipeline_validation.md) records exact cases, metrics and
limits; the [earlier eight-intent pilot](07_luna_pilot.md) remains unchanged.

Additional checks remain separate: bounded retrieval diagnostics produced
16 records for four questions and four profiles without LLM calls; an MCQ
smoke on one QID achieved 1/1 for No-RAG, Dense and Hybrid without an LLM judge.
CLI and UI smoke checks completed. Enrichment produced four evidence-backed
proposals, with **zero adopted facts** and no corpus mutation.

The new work made **69 calls**, accounted at **0.046987625 EUR**. Including the
first pilot, the ledger records **115 calls and 0.068710875 EUR**: 114 settled
entries and one uncertain reservation. These are conservative estimates and
reservations, not invoices. The bounded authorization used a cumulative
0.50 EUR ceiling and 0.10 EUR reserve; the normal 20 EUR/4 EUR defaults were
not the pilot limits. No Sol or Utopia model call and no paid BGE embedding
was used in this validation. Utopia's earlier live preflight returned HTTP 403;
its model identity and capacity still need verification if that provider is used.

## Review artifacts and remaining work

The main report is `data/reports/validated_20261011/index.html`; MCQ results are
separate at `data/reports/validated_20261011_mcq/index.html`. Reports read frozen
artifacts and durable audit comments without retrieval or model calls.
All three notebooks were executed through Jupyter with build/paid switches
left false and zero API calls; executed copies and HTML previews are under
`data/notebooks/validated_20261011`. Browser pixel review has **not** been done.

Portable packaging and restoration passed with 7,795 verified files and no
provenance warnings. Rebuilding the main report from restored source reproduced
all 21 comparisons without API calls. The frozen delivery archive is
`data/bundles/validated_20261011.tar.gz`; its restored copy is under
`data/restored/validated_20261011`. It includes source, benchmark, snapshot,
corpus, seven no-hint runs, three MCQ runs and retrieval diagnostics. Model
weights, indexes, credentials and the private provider ledger are excluded.
Report links, checksums and offline resources passed static QA; this does not
replace the outstanding browser pixel review. The original private research
backup remains available.

Next empirical work requires human review of gold, source identity and judge
behavior, then the full 100-intent comparison with an independent judge and
frozen settings. Verify Utopia access only if needed, and refresh sourced
prices/exchange rates before another authorized paid batch. Graph, reranking,
multi-query and enrichment remain experiments, not adopted improvements.
Follow the [manual workflow](../notes/06_manual_workflow.md),
[evaluation protocol](06_experiment_protocol.md) and [numbered specs](../specs/README.md).
