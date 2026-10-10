# Decision record

| Problem | Choice and reason | Alternative / limit | Verification |
|---|---|---|---|
| Divergent baselines | One engine with explicit profiles and a shared context builder. | Separate pipelines hide changed prompts and budgets. | Controlled-comparison contract rejects mismatched identities. |
| Fragile legal preprocessing | Ordered HTML blocks, regions, fine units and exact spans; aggregate meaning for retrieval. | LLM extraction is review-only and cannot repair missing text. | Real HTML fixtures, complete block accounting and exact source spans. |
| Uncertain cessation | Exclude only resolved explicit cessation; preserve unknown cases. | “Testo vigente” and missing repeal evidence do not establish applicability. | Total/partial/conditional/future/prior-wording tests. |
| Small, multilingual corpus | Pinned BGE-M3 dense/native sparse in local Qdrant. | BM25 is optional for other dense providers; no extra IDF for BGE. | Dimensions, finite values, payload/hash reconciliation, exact native RRF. |
| Context loss | Select complete chunks with identity, headings and list introductions. | Fine structure remains available but is not always a retrieval item. | Full rendering token budget and candidates-to-context diagnosis. |
| Graph complexity | JSONL nodes/edges, explicit article references, one hop, at most two additions. | No separate graph service or full historical version reconstruction. | Local relation evidence and resolved destinations; benefit measured separately. |
| Model variability | Explicit provider/model/config, structured outputs, refusals and failures recorded. | No automatic fallback; deterministic software does not make LLM outputs identical. | Mocked provider contracts, frozen outputs, live preflight when access exists. |
| Benchmark contamination | Original gold stays frozen; approved revisions form a separate view. | Reused intents support exploratory results, not an untouched test set. | CSV hashes, review authors and approval contracts. |
| Spending and secrets | Runtime keys separate; shared locked ledger reserves every paid attempt. | Cost ceilings are estimates, including uncertain reservations, not receipts. | Persistent 20 EUR cap, price/FX freshness and credential-free exports. |
| Agent coordination | Shared contracts owned by the orchestrator; module-specific ownership. | No concurrent index writers or model-loading jobs on the M1. | Focused tests, integration tests, explicit artifact hashes. |

The previous source/data archive was verified before replacing implementation in the isolated branch. The original checkout and historical outputs remain recoverable. The historical `data/evaluation_runs/oracle_context/oracle_context_manifest.json` contains a credential field. It is untracked and no commits for that path were found in the local Git refs checked on 2026-10-10. This does not establish whether an archive or file was shared externally. The file remains private in the original checkout and must be sanitized before sharing; external exposure and rotation need account-level evidence. New portable bundles reject structured credential fields and exclude `.env`.

Architecture is kept direct: Python, Pydantic, lxml, requests, Qdrant/OpenAI SDKs and optional FlagEmbedding/Streamlit. Notebooks contain calls and displays only. Each step emits versioned artifacts and a manifest; source identities, snapshot identity and content hashes are separate. New dependencies require a concrete research reason.
