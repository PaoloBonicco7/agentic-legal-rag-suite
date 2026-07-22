# Inventario degli output in `data`

Data revisione: 2026-06-14.

Questa nota fotografa lo stato locale della cartella `data/`, separando sorgenti, output
riproducibili, run di riferimento e candidati di pulizia. La revisione è basata su:

- specifiche in `docs/specs/`;
- manifest e summary JSON presenti sotto `data/`;
- dimensioni rilevate con `du`;
- note risultato in `docs/results/`.

## Sintesi

`data/` pesa circa **8.3 GB**. Il peso è quasi tutto in:

- `data/indexes/`: **4.2 GB**, indici Qdrant locali/server;
- `data/retrieval_eval_runs/`: **3.7 GB**, diagnostics retrieval-only e storico di sweep;
- `data/laws_dataset_clean/`: **289 MB**, dataset pulito generato dai sorgenti HTML.

Le run end-to-end finali sono leggere (`data/rag_runs/` pesa circa **10 MB**) e vanno tenute:
costano poco e documentano direttamente il confronto no-RAG / simple RAG / advanced RAG.

La run end-to-end piu recente e rilevante per la tesi è:

`data/rag_runs/advanced/full_100__answer_slurm_gpt_oss_120b__judge_slurm_gpt_oss_120b__a4_combined_best_v2/`

- creata il `2026-05-26T12:59:37Z`;
- schema `advanced-graph-rag-v3`;
- collection `legal_chunks_bge_m3`;
- configurazione promossa: hybrid + multi-query;
- MCQ accuracy `0.84`;
- no-hint `mean_score=1.26`, accuracy `0.63`.

La run diagnostics di riferimento è:

`data/retrieval_eval_runs/diagnostic_full_utopia_throttled__20260525T091921Z/`

- creata il `2026-05-25T12:06:15Z`;
- schema `retrieval-evaluation-v4`;
- collection `legal_chunks_bge_m3`;
- usata da `docs/results/06b_retrieval_diagnostics.md` per motivare hybrid + multi-query.


### Run usate nel confronto di tesi

| Path | Stato | Metriche principali |
|---|---|---|
| `data/baseline_runs/no_rag/` | tenere | MCQ `0.81`; no-hint `mean_score=0.97`, accuracy `0.485`. |
| `data/rag_runs/simple/` | tenere | MCQ `0.79`; no-hint `mean_score=1.18`, accuracy `0.59`. |
| `data/rag_runs/advanced/full_100__answer_slurm_gpt_oss_120b__judge_slurm_gpt_oss_120b__a4_combined_best_v2/` | tenere | MCQ `0.84`; no-hint `mean_score=1.26`, accuracy `0.63`. |
| `data/evaluation_runs/oracle_context/` | tenere | Ceiling controllato: MCQ oracle `0.99`; no-hint oracle `mean_score=1.78`, accuracy `0.89`. |

Queste run sono referenziate da `docs/results/00_overview.md`, `docs/results/04_no_rag.md`,
`docs/results/05_simple_rag.md` e `docs/results/06_advanced_rag.md`.

### Indice attivo

| Path | Dimensione | Stato | Motivo |
|---|---:|---|---|
| `data/indexing_runs/bge_m3_full_20260523_095655/` | 23 KB | tenere | Manifest della full run BGE-M3 usata dalle run recenti. |
| `data/indexing_runs/bge_m3_full_20260523_095655_progress.jsonl` | 390 KB | tenere o archiviare | Log progressivo della stessa full run; utile per audit, non richiesto a runtime. |
| `data/indexes/qdrant_server/` | 2.1 GB | tenere | Storage Qdrant server della collection `legal_chunks_bge_m3`, 76.467 point, dense 1024 + sparse. |

Il manifest attivo registra:

- embedding `BAAI/bge-m3`;
- hybrid abilitato;
- Qdrant server `data/indexes/qdrant_server`;
- `selected=indexed=collection_points=76467`;
- `failure_count=0`;
- `ready_for_retrieval=true`.

## Miglioramenti documentati

I miglioramenti principali sono gia riassunti in `docs/results/00_overview.md` e
`docs/results/06b_retrieval_diagnostics.md`.

### Re-index BGE-M3

Il passaggio dal vecchio indice dense-only `legal_chunks` al nuovo `legal_chunks_bge_m3` è il salto
retrieval piu importante:

| Scenario | article_hit |
|---|---:|
| Utopia/Nomic dense@10 | 45.0% |
| Utopia/Nomic dense@100 | 66.0% |
| BGE-M3 dense@10 | 73.0% |
| BGE-M3 dense@100 | 88.0% |
| BGE-M3 hybrid@100 | 89.0% |

Il confronto storico usa anche la run:

`data/retrieval_eval_runs/default__20260511T180530Z/`

Questa directory pesa circa **2.3 GB** ed è la candidata piu grande per archiviazione esterna, ma
non va rimossa alla cieca perché supporta il confronto "vecchio indice vs BGE-M3" citato in 06b.

### Hybrid retrieval

Hybrid dense + sparse via RRF è promosso perché porta `article_hit` da `73.0%` (`dense@10`) a
`89.0%` (`hybrid@100`), con MRR migliore del dense puro.

### Query rewriting

`multi_query` con `n=3` è promosso perché sul pilot porta l'hybrid da `83.3%` a `93.3%`
`article_hit`. La run advanced finale usa questa leva con cache già popolata:

- `query_rewriting.enabled=true`;
- `strategy=multi_query`;
- `cache_hits=200`;
- `failures=0`.

### Leve scartate

Le run diagnostics documentano anche cosa non promuovere:

- `metadata_filters_enabled`: scartato perché `law_status=current` esclude domande valide;
- `graph_expansion_enabled`: scartato per rumore elevato (`expansion_noise_ratio` circa 99.7%);
- `rerank_enabled`: scartato perché declassa articoli corretti e perde recall.

## Storico utile ma non indispensabile alla run piu recente

Questi dati sono riproducibili e non servono per eseguire la pipeline piu recente, ma possono essere
utili come audit trail o per spiegare l'evoluzione della tesi.

### Indexing runs storiche

`data/indexing_runs/` pesa solo **1.4 MB**, quindi non conviene pulirla per recuperare spazio.

Da tenere come storico leggero:

- `20260505_084836/`: full vecchio indice `legal_chunks`, Utopia/Nomic dense-only;
- `20260511_172154/`: altra full run vecchio indice `legal_chunks`;
- `20260512_212818/`: full BGE-M3 locale;
- `notebook_demo_*`: demo minime da 3 chunk, utili solo per traccia notebook.

### RAG ablation e smoke

`data/rag_runs/advanced/` contiene run vecchie e ablation:

| Run | Stato consigliato |
|---|---|
| `full_100_dense_graph_rerank/` | storico vecchia pipeline graph+rerank su `legal_chunks`. |
| `full_100__...__full_run_20260506_new_models_01/` | storico vecchia pipeline. |
| `full_100__...__advanced_lean_v1/` | storico ablation vicina alla run finale, ma non headline. |
| `smoke_dense_graph_rerank/` | cancellabile se non serve debug notebook. |
| `_debug_utopia_rag_flow/` | cancellabile se non serve debug. |
| `.full_100_b1_b2_b3_parallel.tmp/` | cancellabile; directory temporanea vuota. |

Visto che l'intera cartella `data/rag_runs/` pesa solo circa **10 MB**, la priorita di pulizia qui
è bassa.

### Baseline smoke

`data/baseline_runs/no_rag_smoke/` pesa circa **20 KB**. È cancellabile se si vuole mantenere solo
la run completa, ma non dà benefici di spazio.

### Cache

`data/cache/` pesa circa **1.4 MB**:

- query rewriting: 105 righe cache dirette + 100 righe nella sottocartella `query_rewriting/`;
- rerank: 267 righe.

Le cache sono piccole e aiutano a evitare chiamate LLM ripetute. Conviene tenerle finché si lavora
sui notebook 06/06b. Sono cancellabili solo se si accetta di rigenerarle.


### Indici Qdrant locali

`data/indexes/qdrant/` pesa circa **2.1 GB** e contiene:

- `collection/legal_chunks_bge_m3/` (~991 MB);
- `collection/legal_chunks_bge_m3_sample/` (~15 MB);
- `collections/legal_chunks/` (~690 MB);
- `collections/legal_chunks_sample/` (~404 MB).

L'indice documentato per la pipeline recente è `data/indexes/qdrant_server/collections/legal_chunks_bge_m3/`.
Tuttavia alcuni config di run/notebook hanno ancora default o path locali verso `data/indexes/qdrant`.
Quindi `data/indexes/qdrant/` è un buon candidato di archiviazione, ma non va cancellato prima di:

1. verificare che i notebook 03, 05, 06 e 06b puntino esplicitamente a `qdrant_server` o possano
   rigenerare l'indice;
2. decidere se mantenere il vecchio indice `legal_chunks` come audit locale;
3. salvare i manifest delle run che lo usano.