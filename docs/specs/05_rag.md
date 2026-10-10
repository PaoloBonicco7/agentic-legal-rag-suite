# 05 — Shared retrieval and generation

## Purpose
Compare stateless no-RAG, dense, sparse, hybrid, multi-query, graph, reranking
and verified oracle context through one answer contract.

## Inputs
Question, optional MCQ choices, immutable index, EngineConfig and chat client.
The engine accepts optional snapshot_id and snapshot_date provenance supplied
after corpus/index reconciliation; existing Python callers inherit the ID from
their retriever and keep an unknown date rather than inventing one.

## Outputs
AnswerResult containing candidates, actual context, query
variants, snapshot ID/acquisition date, usage, exact source quotations and exposed errors.

## Pipeline
- Dense/sparse retrieve 100 candidates. Hybrid applies the same policy to both
  prefetches and returns 100 through explicit RRF k=60, zero-based ranks.
- Multi-query retains the original and exactly two distinct variants, then
  fuses the three hybrid lists to 100 using the same RRF convention.
- Graph experiments follow confirmed article references for one hop, adding
  at most two chunks. Unresolved edges never imply arbitrary law expansion.
  Verified graph artifacts are loaded once into source-to-relation and
  unit-to-chunk adjacency maps. Queries inspect only seed adjacencies, retaining
  the same relation-ID/chunk-ID order and cessation policy.
- Reranking scores the first 20 candidates, compares the same candidate pool
  and does not add candidates. All profiles use the same final context budget.
- Context packs complete chunks, deduplicated by ID and fully covered source
  spans, up to 4096 pinned BGE
  tokens and 12 units. Headers count; omitted chunks are recorded, never cut.
  Each header shows the full law identity (document_id when absent), title,
  article, source URL and normative status. A chunk carrying
  source_quote_boundary_recovered, including its colon-delimited source suffix,
  also shows that specific parsing uncertainty;
  unrelated document or duplicate-unit warnings are not injected into context.
  Identity and uncertainty text count toward the same complete-context budget.
  Partial overlaps retain the complete chunk; repeated list introductions
  preserve meaning and count toward the same budget.
- Citations must name an included context ID and quote an exact nonempty
  substring of its source text. Source URLs and chunk IDs come from the corpus.
- Oracle accepts only supplied verified source chunks. No gold answer enters
  retrieval or the answer prompt. No-RAG has no source citations.
  Oracle uses the same cessation policy; excluded evidence is traced and an
  empty eligible context causes abstention.

## Contract
Clients return a validated Pydantic object and Usage. Utopia uses structured
Ollama output; OpenAI uses SDK Responses parsing with storage disabled.
answer-v2 includes the corpus acquisition date in RAG prompts, explicitly
distinguishing acquisition provenance from certified legal applicability.
No-RAG preserves the same snapshot metadata in its output but does not receive
that date, corpus text or other new evidence in its model prompt.
Requested and returned model identities are distinct; a missing returned ID is
recorded as unknown. A known Utopia returned model different from the request
is a terminal model_mismatch, preserving its usage and model output. OpenAI
alias-to-snapshot resolution remains recorded rather than rejected.
Candidate-pool, selected-candidate and displayed-context
stages remain separate in the trace. RAG with no context returns insufficient
evidence without generating from model knowledge.
Utopia requires an explicit positive context window (operational default 32768)
and always sends num_ctx. A null window raises context_unverified before chat
dispatch. Benchmark preflight verifies that the request does not exceed the
inspected model capacity; absent capacity blocks the benchmark. Full prompt,
schema, protocol allowance and maximum output must fit the requested window.
No automatic maximum-capacity selection or configuration mutation occurs;
the gateway's actual handling of num_ctx still requires a verified live smoke.
Utopia receives explicit think settings. OpenAI receives
the requested reasoning effort, including none; a null effort deliberately
selects the provider default and never claims verified thinking settings.
Usage.raw_output retains the original model text from successful and unsuccessful
HTTP200 responses, including whitespace. Parsed output remains separate.
HTTP error bodies and authorization headers are never exported.
The stateless UI uses the same factory as the CLI, including timeout and output
limits. Snapshot identity and acquisition dates are visible before submission;
the corpus path can be derived from the index manifest. Model weights are
loaded only after a question is submitted.

## Quality Gates
Retries are bounded, only transient transport errors are retried, and 403 is
terminal. No provider fallback exists. Model refusals and truncated responses
remain errors. Full prompt, schema and protocol overhead have a conservative
UTF-8 byte token bound checked before dispatch. The input bound plus maximum
output must also fit the explicitly configured
model context window when known; an input limit alone is not a verified total
capacity. Utopia show/tags inspection exports only identity, digest,
capabilities and declared model context capacity. It cannot certify a service
context limit; unavailable metadata stays unknown and is recorded explicitly.
OpenAI requires prices verified today, FX dated within four days, source URLs and a persistent ledger;
every attempt reserves a conservative bound before dispatch. The experiment
cap is 16 EUR plus a separately usable 4 EUR validation reserve within 20 EUR.
An uncertain billed request keeps its full reservation. Cost estimates use the
configured ceilings, including cache-write rates; they are not billing receipts.
Any observed cost-bound violation blocks later requests across process restarts.
Scientific provider identity excludes accounting dates, prices and ledger paths,
so refreshing a quote does not invalidate a checkpoint. Separate accounting
metadata and Usage link estimates/reservations to dated ledger entries. An
uncertain retry is included in the reported conservative reservation cost.
Optional propose_enrichment returns review-only facts with unique exact source
quotations and deterministic spans. Unsupported proposals are recorded as
rejected; neither accepted proposals nor LLM descriptions change status,
relations or retrieval filters automatically.

## Notebook Role
Thin notebooks call RagEngine and display the same trace and answer artifacts.

## Acceptance Criteria
Test exact RRF, dense/sparse policy consistency, unknown retention, graph
target resolution, complete-context budgets, invalid quotations, oracle
validation, refusal/403/retries and persistent budget caps without paid calls.
New profiles are promoted only after paired measurements at fixed context
budget; graph construction/experimentation is required, benefit is not assumed.

Implementation references for these acceptance criteria:
[RRF](https://doi.org/10.1145/1571941.1572114),
[DMQR-RAG](https://arxiv.org/abs/2411.13154),
[OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
