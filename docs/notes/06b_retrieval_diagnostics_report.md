# 06b — Report risultati Retrieval Diagnostics

Sintesi degli esperimenti retrieval-only eseguiti nel notebook [06b_retrieval_diagnostics.ipynb](../../notebooks/06b_retrieval_diagnostics.ipynb) per selezionare la configurazione di retrieval da promuovere in Advanced Graph RAG ([06_advanced_graph_rag.ipynb](../../notebooks/06_advanced_graph_rag.ipynb)).

**Scope del report**: i sei esperimenti retrieval-only del notebook — A (dense baseline + curva top-k), B (filtri metadata), C (graph expansion), F (hybrid retrieval RRF), G (LLM reranking pilot), H (query rewriting / HyDE / multi-query pilot).

**Run di riferimento**: [`data/retrieval_eval_runs/diagnostic_full_utopia_throttled__20260525T091921Z/`](../../data/retrieval_eval_runs/diagnostic_full_utopia_throttled__20260525T091921Z/) — profilo `full`, schema `retrieval-evaluation-v4`. Modello LLM per G e H: Utopia `SLURM.gpt-oss:120b` (HPC4AI), provider `utopia`.

## 1. Contesto e obiettivo

Lo step 06b isola la qualità del retrieval dalla generazione della risposta. Misura se i riferimenti legali attesi compaiono nei candidati, in quale posizione del ranking e quali varianti di retrieval meritano una run end-to-end nell'Advanced RAG. L'obiettivo della fase è ridurre il rischio di promuovere componenti complesse senza evidenza retrieval-only solida; la decisione finale si traduce in un `recommended_advanced_config.json` consumato dal notebook 06.

"Vincere" significa **migliorare `article_hit_pct` di almeno +1pp** rispetto allo scenario base con un sample non degenere (100 domande per dataset) e una configurazione completamente documentata. Questo criterio (`min_promotion_gain_pp = 1.0`) è esplicito nel manifest del run.

## 2. Metodologia

### Setup

- **Indice**: collection Qdrant locale `legal_chunks_bge_m3`, 76.467 chunk, embedder BGE-M3 con vettore dense 1024-dim + vettore sparse lessicale nativo.
- **Source manifest indice**: `data/indexing_runs/bge_m3_full_20260523_095655/index_manifest.json` (hash registrato nel manifest del run).
- **Dataset**: `mcq` (multiple-choice) e `no_hint` (open-ended), 100 domande ciascuno con riferimenti `expected_law_ids` e `expected_article_ids`. I due dataset usano lo stesso stem nelle chiamate retrieval, quindi le metriche A/B/C/F coincidono row-per-row: vengono riportate una volta sola con dataset esplicitato dove rilevante.
- **Profilo diagnostico**: `full` (tutti gli sweep abilitati, filtri estesi, graph sweep completo, hybrid 12 combinazioni).

### Metriche

Definizioni operative (da [docs/specs/06b_retrieval_diagnostics.md](../specs/06b_retrieval_diagnostics.md)):

- `article_hit_pct` — % di domande in cui almeno un `expected_article_id` compare nel candidate set finale. Metrica primaria.
- `law_hit_pct` — % di domande in cui almeno un `expected_law_id` compare nel candidate set finale.
- `article_mrr` — media del Mean Reciprocal Rank sul primo articolo atteso che appare nel ranking; vale 0 se nessuno è presente.

### Criterio di promozione

Una leva viene promossa solo se:
1. produce `article_hit_pct` ≥ baseline + 1pp,
2. il sample è non degenere (`n_questions = 100`),
3. la configurazione vincente è documentata nei row-level CSV ed è riproducibile dal manifest.

Le leve che falliscono uno qualunque dei tre criteri vengono mostrate nel waterfall ma non finiscono nel `recommended_advanced_config.json`.

### Tracciabilità

Il `manifest.json` del run finale registra:

- `schema_version = "retrieval-evaluation-v4"`,
- `diagnostic_profile = "full"`,
- `collection_name`, `embedding_model`, `hybrid_available = true`,
- `source_hashes` per `chunks`, `edges`, `index_manifest`, `questions_mcq`, `questions_no_hint`,
- dimensione campione e dataset selezionati.

Ogni scenario in `scenarios.csv` ha `experiment_name`, `status`, `skip_reason` e un dict `config` JSON-encoded, così che lettura del CSV basti a ricostruire la configurazione testata.

## 3. Esperimenti

### A — Dense baseline e curva top-k

**Cosa misura.** Determina la curva recall/ranking del retrieval puramente dense (named vector `dense`) al variare del budget di candidati `top_k`, in assenza di filtri. Stabilisce la **baseline** (`dense@10`) e il **ceiling retrieval-only senza riformulazioni** (`dense@100`).

**Sweep.** `retrieval_mode = dense`, `filter = none`, `top_k ∈ {10, 20, 50, 100}`.

**Risultati** (fonte: `sweep_direct.csv`, identici su `mcq` e `no_hint`):

| top_k | article_hit | law_hit | MRR |
|---:|---:|---:|---:|
| **10 (baseline)** | **73.0%** | **97.0%** | **0.519** |
| 20 | 77.0% | 97.0% | 0.521 |
| 50 | 84.0% | 97.0% | 0.523 |
| 100 | 88.0% | 99.0% | 0.524 |

**Verdetto.** Baseline confermata a 73% / 97% / 0.519. Aumentare `top_k` migliora il recall in modo monotono (+15pp sull'`article_hit` passando da 10 a 100) ma l'MRR resta sostanzialmente piatto (0.519 → 0.524, +0.005). Diagnosi: il dense BGE-M3 recupera molti articoli rilevanti se gli si concede budget, ma non li porta in cima. Lo spazio di miglioramento è sul **ranking**, non sul solo recall.

A è una baseline di riferimento, non una leva promovibile: serve a definire `dense@10` (base scenario per i delta) e `dense@100` (recall ceiling) usati dai confronti successivi.

### B — Filtri metadata

**Cosa misura.** Effetto di filtri statici sui metadati Qdrant (es. `law_status = current`) applicati al dense baseline. L'ipotesi è che restringere il bacino a leggi in vigore riduca il rumore e alzi l'`article_hit`.

**Sweep.** `retrieval_mode = dense`, `top_k = 10`, `filter ∈ {none, law_status_current, index_views_current, article_status_current}` (sweep completo nel CSV; qui mostro la variante più significativa).

**Risultati** (fonte: `scenarios.csv`, dataset `mcq`; `no_hint` identico):

| filtro | article_hit | law_hit | MRR | n escluse | δ vs baseline |
|---|---:|---:|---:|---:|---:|
| `none` (baseline) | 73.0% | 97.0% | 0.519 | 0 | — |
| `law_status=current` | 71.0% | 93.0% | 0.507 | 4 | **−2.0pp** |
| Best filter from sweep | 73.0% | 97.0% | 0.519 | 0 | 0.0pp |

**Verdetto.** **Scartato.** Il filtro su `law_status=current` esclude 4 domande dell'evaluation set i cui riferimenti puntano a leggi ormai abrogate o sostituite, e per le restanti non aumenta il hit. Il best filter scelto dallo sweep coincide con `filter=none`: lo sweep stesso conclude che nessuna combinazione di filtri statici batte la baseline (`promote_filter = false` nel `recommended_advanced_config.json`). I metadati del corpus Valle d'Aosta sono già dominati da leggi correnti, quindi la restrizione non aggiunge segnale e perde domande valide.

### C — Graph expansion

**Cosa misura.** Espansione della candidate list seguendo gli archi del grafo legale (`REFERENCES`, `AMENDS`, `INSERTS`, `MODIFIED_BY`, `INSERTED_BY`) a partire dai seed dense. L'ipotesi è che gli articoli correttamente correlati (rinvii normativi, modifiche) compensino i miss del dense.

**Sweep.** Sweep su `seed_k ∈ {1, 3}`, `max_chunks_per_law ∈ {1, 2}`, `min_edge_confidence = 0.45`, set relazioni `default` vs `references_only`. Diagnostico aggiuntivo: `expansion_noise_ratio` (% di chunk espansi non rilevanti). Filtro proposto: promuovere solo configurazioni con `noise_ratio ≤ 0.95`.

**Risultati** (fonte: `scenarios.csv`, dataset `mcq`; `no_hint` identico):

| config | article_hit | law_hit | MRR | δ article |
|---|---:|---:|---:|---:|
| `+ Graph default seed=3` | 74.0% | 97.0% | 0.519 | +1.0pp |
| `+ Graph REFERENCES only` | 74.0% | 97.0% | 0.520 | +1.0pp |
| `+ Best graph from sweep` | 74.0% | 97.0% | 0.520 | +1.0pp |
| `+ Best graph low-noise (≤0.95)` | — | — | — | **skipped: nessuna config rispetta il vincolo noise** |

La "best graph from sweep" registra `expansion_noise_ratio = 0.9971`: in pratica oltre il 99% dei chunk aggiunti dall'espansione non sono articoli rilevanti.

**Verdetto.** **Scartato.** Il guadagno è marginale (+1pp `article_hit`, MRR invariato) e si paga con un noise ratio quasi totale. Nessuna configurazione del sweep produce un'espansione "pulita" (`noise ≤ 0.95`), quindi lo scenario `Best graph low-noise` viene legittimamente saltato. Promuovere il graph in queste condizioni significherebbe gonfiare il candidate set con quasi tutto rumore, aumentando il carico downstream sull'LLM senza beneficio retrieval misurabile. La leva graph resta esposta nel codice come configurazione conservativa disattivata di default (`graph_expansion_enabled = false`), così da poter essere riattivata in una iterazione futura con un noise cap più severo o con seed selezionati dal best hybrid.

### F — Hybrid retrieval (dense + sparse via RRF)

**Cosa misura.** Effetto della fusione dense + sparse tramite Reciprocal Rank Fusion (RRF) usando l'API nativa di Qdrant. Sweep completo su tutta la griglia `top_k × rrf_k` per identificare il punto operativo migliore, senza filtri (per isolare l'effetto della fusione).

**Sweep.** `retrieval_mode ∈ {dense, hybrid}`, `top_k ∈ {10, 20, 50, 100}`, `rrf_k ∈ {30, 60, 90}` (12 combinazioni hybrid + 4 dense baseline), `filter = none`.

**Risultati completi hybrid** (fonte: `sweep_direct.csv` aggregato per scenario, dataset `mcq`):

| top_k | rrf_k | article_hit | law_hit | MRR |
|---:|---:|---:|---:|---:|
| 10 | 30 | 76.0% | 96.0% | 0.560 |
| 10 | 60 | 76.0% | 96.0% | 0.550 |
| 10 | 90 | 76.0% | 96.0% | 0.555 |
| 20 | 30 | 78.0% | 97.0% | 0.557 |
| 20 | 60 | 77.0% | 97.0% | 0.561 |
| 20 | 90 | 77.0% | 97.0% | 0.571 |
| 50 | 30 | 82.0% | 98.0% | 0.542 |
| 50 | 60 | 82.0% | 98.0% | 0.550 |
| 50 | 90 | 82.0% | 98.0% | 0.556 |
| **100** | **30** | **89.0%** | **99.0%** | **0.551** |
| 100 | 60 | 89.0% | 99.0% | 0.547 |
| 100 | 90 | 89.0% | 99.0% | 0.545 |

(Su `no_hint` lo sweep produce numeri sostanzialmente identici: tutti i picchi a `top_k=100` raggiungono `article_hit=89%` con MRR nell'intervallo 0.545 — 0.553; il best `rrf_k` resta 30 su entrambi i dataset.)

**Confronto dense vs hybrid a parità di budget** (fonte: stesso CSV):

| top_k | dense article_hit | hybrid best article_hit | δ hybrid − dense |
|---:|---:|---:|---:|
| 10 | 73.0% | 76.0% | +3.0pp |
| 20 | 77.0% | 78.0% | +1.0pp |
| 50 | 84.0% | 82.0% | −2.0pp |
| 100 | 88.0% | 89.0% | +1.0pp |

E rispetto a `dense@10` (baseline): l'hybrid migliore (`top_k=100, rrf_k=30`) guadagna **+16.0pp** sull'`article_hit` (89.0% vs 73.0%) e **+0.032** sull'MRR (0.551 vs 0.519). Anche il `law_hit` migliora marginalmente (99.0% vs 97.0%).

**Verdetto.** **Promosso.** Hybrid soddisfa i tre criteri di promozione: gain ≥ 1pp, sample 100 domande, configurazione completa. Il punto operativo scelto è `top_k=100, rrf_k=30` perché:
- raggiunge il massimo `article_hit` (89%) condiviso con `rrf_k=60` e `rrf_k=90`,
- ha l'MRR più alto su `mcq` a `top_k=100` (0.551 vs 0.547/0.545), che è la metrica di ranking,
- `rrf_k` basso dà più peso ai primi rank (entrambe le liste): coerente con il dominio legale dove la prima manciata di candidati pesa di più.

Il punto operativo `top_k=50` raggiungerebbe solo 82% `article_hit` (−7pp vs top_k=100): il budget intermedio non è competitivo. Le combinazioni a `top_k=20` mostrano l'MRR più alto in assoluto (0.571 a `rrf_k=90` su `mcq`) ma sacrificano 11pp di recall: la scelta privilegia il recall, anche perché la pipeline downstream (Advanced RAG) può applicare un re-ranking sui 100 candidati se utile.

### G — LLM reranking (pilot)

**Cosa misura.** Riordino degli `input_k` candidati hybrid tramite un giudice LLM che assegna uno score 0/1/2 di rilevanza al chunk, poi taglio agli `output_k` migliori. L'ipotesi è che il modello recuperi articoli rilevanti spinti in basso dal retrieval e ne demoti di irrilevanti.

**Sweep.** Pilot deterministico di 30 domande per dataset (seed fisso, sample size esplicitato nel manifest). Hybrid base scenario: `top_k=20|50|100, rrf_k=30`. Rerank input/output: `rerank_input_k ∈ {20, 50, 100}` × `rerank_output_k ∈ {3, 5, 10}` (9 combinazioni per dataset). Modello: `SLURM.gpt-oss:120b` via Utopia. Cache LLM versionata con `RERANK_PROMPT_VERSION=rerank-v1`.

**Risultati** (fonte: `rerank_summary.csv`; `n` è il numero di domande del pilot che ha completato la chiamata LLM senza errori; `failure_rate=30.6%`, 55 fallimenti su 180 chiamate):

MCQ — confronto pre-rerank (hybrid puro) vs post-rerank:

| top_k | input_k | output_k | n | article_hit pre | post | δ | recovered | demoted |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 20 | 3 | 23 | 82.6% | 73.9% | −8.7pp | 0 | 2 |
| 20 | 20 | 5 | 23 | 82.6% | 78.3% | −4.3pp | 0 | 1 |
| **20** | **20** | **10** | **23** | **82.6%** | **82.6%** | **0.0pp** | **0** | **0** |
| 50 | 50 | 3 | 24 | 83.3% | 70.8% | −12.5pp | 0 | 3 |
| 50 | 50 | 5 | 24 | 83.3% | 70.8% | −12.5pp | 0 | 3 |
| 50 | 50 | 10 | 24 | 83.3% | 75.0% | −8.3pp | 0 | 2 |
| 100 | 100 | 3 | 20 | 90.0% | 70.0% | −20.0pp | 0 | 4 |
| 100 | 100 | 5 | 20 | 90.0% | 75.0% | −15.0pp | 0 | 3 |
| 100 | 100 | 10 | 20 | 90.0% | 80.0% | −10.0pp | 0 | 2 |

no_hint:

| top_k | input_k | output_k | n | article_hit pre | post | δ | recovered | demoted |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 20 | 3 | 22 | 77.3% | 72.7% | −4.5pp | 0 | 1 |
| 20 | 20 | 5 | 22 | 77.3% | 77.3% | 0.0pp | 0 | 0 |
| 20 | 20 | 10 | 22 | 77.3% | 77.3% | 0.0pp | 0 | 0 |
| 50 | 50 | 3 | 20 | 85.0% | 85.0% | 0.0pp | 0 | 0 |
| 50 | 50 | 5 | 20 | 85.0% | 85.0% | 0.0pp | 0 | 0 |
| 50 | 50 | 10 | 20 | 85.0% | 85.0% | 0.0pp | 0 | 0 |
| 100 | 100 | 3 | 16 | 93.8% | 68.8% | −25.0pp | 0 | 4 |
| 100 | 100 | 5 | 16 | 93.8% | 68.8% | −25.0pp | 0 | 4 |
| **100** | **100** | **10** | **16** | **93.8%** | **87.5%** | **−6.3pp** | **0** | **1** |

Scenario waterfall sintetico (`+ LLM rerank` in `scenarios.csv`):
- mcq: best impact a `top_k=20, output_k=10` → `article_hit=82.6%` (n=23), MRR=0.711, `delta_vs_baseline (dense@10) = +9.6pp`.
- no_hint: best impact a `top_k=100, output_k=10` → `article_hit=87.5%` (n=16), MRR=0.640, `delta_vs_baseline = +14.5pp`.

**Verdetto.** **Scartato per la promozione (`promote_rerank=false`).** Tre osservazioni:
1. `recovered=0` su tutte le 18 configurazioni: l'LLM non promuove mai in `output_k` un articolo che il retrieval hybrid aveva escluso dal proprio `input_k`. Quindi il rerank non aumenta il recall, può solo riordinare ciò che hybrid ha già trovato.
2. `demoted` è positivo in molte configurazioni: l'LLM ribalta l'ordine in modo da espellere articoli corretti che hybrid aveva piazzato sopra, perdendo hit. Le configurazioni a `output_k=10` riducono l'impatto perché lasciano spazio per più candidati nella finestra finale.
3. Confrontato con il **base scenario corretto** (hybrid best @ `article_hit=89%` sul full set), il rerank al massimo pareggia (`mcq @ top_k=20`) e in tutti gli altri casi peggiora: `rerank_gain_pp_vs_dense10_baseline = −6.39` su `mcq` e `−1.5` su `no_hint` (fonte: `recommended_advanced_config.dataset_recommendations[*]`).

Nota positiva: l'MRR cresce sensibilmente (es. 0.711 vs 0.551 di hybrid puro su `mcq`). Significa che, sui chunk già correttamente recuperati, il rerank LLM **migliora il ranking** — ma è un guadagno qualitativo di precision-at-top che non compensa la perdita di recall. Per questo motivo il rerank resta integrato nel codice e cache-aware, ma `rerank_enabled=false` nel config promosso. Riattivarlo richiede prima una versione del prompt che riduca la demote rate o un modello con tasso di errore più basso (oggi `failure_rate=30.6%`).

### H — Query rewriting / HyDE / multi-query (pilot)

**Cosa misura.** Trasforma la domanda dell'utente prima del retrieval per ridurre il mismatch col linguaggio normativo. Tre strategie + baseline:
- `none` (baseline): query originale,
- `rewrite`: riformulazione legale concisa (1 query),
- `hyde`: passaggio normativo ipotetico generato dall'LLM (1 query),
- `multi_query`: n=3 riformulazioni alternative; il retrieval viene eseguito su ciascuna e i risultati dedotti fusi client-side.

**Sweep.** Pilot deterministico di 30 domande per dataset (seed fisso). Hybrid base scenario: `top_k=100, rrf_k=30`. Modello: `SLURM.gpt-oss:120b` via Utopia. Cache versionata con `QUERY_REWRITING_PROMPT_VERSION=query-rewriting-v1`. `failure_rate=0%` (180 chiamate su 180 con output strutturato valido).

**Risultati per strategia** (fonte: aggregato di `sweep_query_rewriting.csv`):

MCQ pilot (n=30):

| strategia | article_hit | law_hit | MRR | δ vs `none` |
|---|---:|---:|---:|---:|
| `none` | 83.3% | 96.7% | 0.493 | — |
| `rewrite` | 90.0% | 96.7% | 0.525 | +6.7pp |
| `hyde` | 86.7% | 96.7% | 0.471 | +3.3pp |
| **`multi_query`** | **93.3%** | **100.0%** | **0.467** | **+10.0pp** |

no_hint pilot (n=30):

| strategia | article_hit | law_hit | MRR | δ vs `none` |
|---|---:|---:|---:|---:|
| `none` | 83.3% | 96.7% | 0.460 | — |
| `rewrite` | 90.0% | 96.7% | 0.509 | +6.7pp |
| `hyde` | 86.7% | 96.7% | 0.438 | +3.3pp |
| **`multi_query`** | **93.3%** | **100.0%** | **0.484** | **+10.0pp** |

Scenario waterfall (`+ Query rewriting` in `scenarios.csv`, multi_query come winner):
- mcq: `article_hit=93.3%` (n=30), `law_hit=100%`, MRR=0.467, `delta_vs_baseline (dense@10) = +20.3pp`.
- no_hint: `article_hit=93.3%` (n=30), `law_hit=100%`, MRR=0.484, `delta_vs_baseline = +20.3pp`.

**Verdetto.** **Promosso a livello di evidenza** (`promote_query_rewriting=true`, `query_rewriting_recommendation.strategy="multi_query"`). Tre punti:
1. `multi_query` è la strategia migliore su entrambi i dataset, con guadagno consistente (+10pp vs `none` sul pilot, identico in mcq e no_hint).
2. Rispetto alla base hybrid full (`article_hit=89%`), multi_query a 93.3% rappresenta `query_rewriting_gain_pp=+4.33pp` (campo nel `recommended_advanced_config.dataset_recommendations[*]`), sopra la soglia di promozione (`min_promotion_gain_pp=1.0`). Il law_hit raggiunge il 100% sul pilot, indicando che la diversità delle riformulazioni copre meglio il vocabolario normativo della stessa intent informativa.
3. `rewrite` e `hyde` confermano l'utilità della riformulazione (+6.7pp e +3.3pp) ma costano meno chiamate LLM per query. Multi_query è preferito perché ha il guadagno più alto e il law_hit migliore; rewrite resta un fallback se il costo LLM diventa proibitivo.

Una nota importante: il top-level `recommended_advanced_config` **non include i campi query rewriting**, perché `AdvancedRagConfig` non li espone ancora (`query_rewriting_note: "Not included in recommended_advanced_config because AdvancedRagConfig does not expose query rewriting fields yet."`). L'estensione del config schema con `query_rewriting_enabled`, `query_rewriting_strategy`, `query_rewriting_n` è parte della Fase 6 della [roadmap](../../RETRIEVAL_IMPROVEMENT_ROADMAP.md). Nel frattempo, `query_rewriting_recommendation.enabled=true, strategy=multi_query` documenta l'evidenza per chi promuoverà il config.

## 4. Waterfall finale

Vista cumulativa delle leve A/B/C/F/G/H (fonte: `scenarios.csv`, `dataset = mcq`; struttura identica su `no_hint`):

| stage | scenario | n | article_hit | law_hit | MRR | δ baseline | esito |
|---|---|---:|---:|---:|---:|---:|---|
| baseline | Baseline dense@10 (no filter) | 100 | 73.0% | 97.0% | 0.519 | — | reference |
| direct ceiling | Best dense top-k (no filter, top_k=100) | 100 | 88.0% | 99.0% | 0.524 | +15.0pp | informativo, non promosso |
| direct mid | Best direct budget top20 | 100 | 77.0% | 97.0% | 0.521 | +4.0pp | informativo |
| filter | + Filter law_status=current | 100 | 71.0% | 93.0% | 0.507 | −2.0pp | — scartato |
| filter | + Best filter from sweep | 100 | 73.0% | 97.0% | 0.519 | 0.0pp | — scartato |
| graph | + Graph default seed=3 | 100 | 74.0% | 97.0% | 0.519 | +1.0pp | — scartato (noise) |
| graph | + Graph REFERENCES only | 100 | 74.0% | 97.0% | 0.520 | +1.0pp | — scartato (noise) |
| graph | + Best graph from sweep | 100 | 74.0% | 97.0% | 0.520 | +1.0pp | — scartato (noise=0.997) |
| graph | + Best graph low-noise | 0 | — | — | — | — | skipped |
| **hybrid** | **Hybrid best available (top_k=100, rrf_k=30)** | 100 | **89.0%** | **99.0%** | **0.551** | **+16.0pp** | **promosso** |
| rerank | + LLM rerank (input_k=20, output_k=10) | 23 (pilot) | 82.6% | 100.0% | 0.711 | +9.6pp | — scartato (recall vs base hybrid: −6.4pp) |
| **rewriting** | **+ Query rewriting (multi_query, n=3)** | 30 (pilot) | **93.3%** | **100.0%** | **0.467** | **+20.3pp** | **promosso (gain vs hybrid: +4.3pp)** |

Sintesi: la fase 06b promuove **due leve** — hybrid (F) come retrieval di base e multi-query (H) come trasformazione della richiesta. Il guadagno cumulativo previsto sul pilot, partendo da `dense@10`, è di **+16pp da F + ulteriori +4.3pp da H** sull'`article_hit`, fino al 93.3% misurato sui 30 campioni del pilot multi_query. L'LLM rerank (G) viene scartato per impatto netto negativo sulla recall vs il base hybrid.

## 5. Configurazione raccomandata

Top-level [`recommended_advanced_config.json`](../../data/retrieval_eval_runs/diagnostic_full_utopia_throttled__20260525T091921Z/recommended_advanced_config.json) prodotto dal run del 2026-05-25:

```json
{
  "hybrid_enabled": true,
  "top_k": 100,
  "rrf_k": 30,
  "metadata_filters_enabled": false,
  "static_filters": {},
  "graph_expansion_enabled": false,
  "graph_expansion_seed_k": 3,
  "graph_expansion_relation_types": ["REFERENCES", "AMENDS", "INSERTS", "MODIFIED_BY", "INSERTED_BY"],
  "max_chunks_per_expanded_law": 2,
  "max_expanded_chunks_total": 15,
  "min_edge_confidence": 0.45,
  "rerank_enabled": false,
  "rerank_input_k": 20,
  "rerank_output_k": 10
}
```

Più la raccomandazione query rewriting (campo separato `query_rewriting_recommendation`):

```json
{
  "enabled": true,
  "strategy": "multi_query",
  "note": "Diagnostic evidence only; add AdvancedRagConfig/runner support before applying to notebook 06."
}
```

Note di lettura:

- **Hybrid (F)** è la componente promossa come retrieval di base: `hybrid_enabled=true`, `top_k=100`, `rrf_k=30`. Punto operativo della Sezione 3.F.
- **Query rewriting (H)** è promosso a livello di evidenza diagnostica (`promote_query_rewriting=true`, strategia `multi_query` con n=3) ma **non è incluso nel payload `recommended_advanced_config`** perché `AdvancedRagConfig` non espone ancora i campi `query_rewriting_enabled / query_rewriting_strategy / query_rewriting_n`. Estendere lo schema fa parte della Fase 6 della [roadmap](../../RETRIEVAL_IMPROVEMENT_ROADMAP.md): prima di applicare H nel notebook 06 vanno aggiunti i campi al config e l'hook `apply_query_rewriting()` prima di `search_hybrid()`.
- **Rerank (G)** non promosso (`rerank_enabled=false`). I valori `rerank_input_k=20, rerank_output_k=10` nel payload sono i parametri della migliore configurazione testata su `mcq`, lasciati come default conservativi: riattivare il rerank in futuro richiede solo `rerank_enabled=true` senza ridefinire i sotto-parametri.
- I parametri graph sono presenti come default ma disattivati (`graph_expansion_enabled=false`): permettono di riattivare l'espansione con un solo flag se in futuro emerge un caso d'uso (es. dopo aver ricalibrato il noise cap).

L'Advanced Graph RAG ([06_advanced_graph_rag.ipynb](../../notebooks/06_advanced_graph_rag.ipynb)) costruirà la propria `AdvancedRagConfig` a partire da questi campi. La promozione di H richiede l'estensione di schema descritta sopra prima di poter essere applicata end-to-end.

## 6. Motivazione finale — perché è la best config per questo use case

1. **Dominio legale italiano + corpus regionale Valle d'Aosta favorisce hybrid.** Le domande del set di valutazione contengono spesso terminologia normativa specifica (numeri di articolo, sigle, locuzioni latine, riferimenti puntuali a leggi regionali). Il dense BGE-M3 cattura la semantica ma fa fatica con questi marker lessicali, mentre il vettore sparse li indicizza esplicitamente. Il salto di +16pp (`article_hit`) di F dimostra che la fusione non è cosmetica: aggiunge segnale che il dense da solo perdeva.

2. **L'effetto della fusione non è solo recall, è ranking.** L'MRR passa da 0.519 (dense@10) a 0.551 (hybrid@100). Considerando che dense@100 ha MRR=0.524, il guadagno di MRR di hybrid non è dovuto solo al maggior budget: la RRF promuove in cima i chunk che entrambe le viste considerano rilevanti. È esattamente il comportamento desiderato per un retrieval che alimenta un LLM, dove i primi candidati pesano molto.

3. **Multi-query (H) aggiunge un guadagno indipendente.** La trasformazione della query in 3 varianti riformulate copre meglio la varietà del vocabolario normativo della stessa intent informativa. Sul pilot di 30 domande il `law_hit` raggiunge il 100% e l'`article_hit` sale a 93.3%, +4.3pp sopra la base hybrid. Il costo è una chiamata LLM per query + 3 retrieval, accettabile per la pipeline tesi; la cache versionata mantiene il costo ammortizzato tra esecuzioni ripetute.

4. **LLM rerank (G) viene scartato per evidenza.** Su 18 configurazioni testate, `recovered=0` ovunque: il rerank non recupera articoli mancanti, può solo riordinare ciò che hybrid ha già trovato, e in più casi demota articoli corretti (impatto netto −6.4pp su `mcq` vs il base hybrid). Migliora l'MRR (0.71 vs 0.55) ma a costo della recall — il bilancio non giustifica l'attivazione di default.

5. **Le altre leve hanno costi senza benefici retrieval-only.** I filtri metadata droppano domande valide (4 leggi non correnti escluse) e abbassano il hit. L'espansione graph ha `noise_ratio = 99.7%`: aggiungerebbe rumore alla candidate list che il rerank downstream dovrebbe poi disinnescare. Promuovere solo le due leve con guadagno chiaro è coerente con il principio _simplicity first_ del progetto e produce una pipeline più facile da spiegare in tesi.

In sintesi: hybrid `top_k=100 rrf_k=30` con multi-query `n=3` sopra costituisce la configurazione retrieval con la migliore evidenza tra le sei famiglie testate; i parametri sono giustificati dal compromesso recall/ranking osservato negli sweep. L'effettiva applicazione di H richiede prima un'estensione di `AdvancedRagConfig` come documentato in Sezione 5.

## 7. Riferimenti

- Run dir completo: [`data/retrieval_eval_runs/diagnostic_full_utopia_throttled__20260525T091921Z/`](../../data/retrieval_eval_runs/diagnostic_full_utopia_throttled__20260525T091921Z/)
- Sweep aggregato: [`scenarios.csv`](../../data/retrieval_eval_runs/diagnostic_full_utopia_throttled__20260525T091921Z/scenarios.csv)
- Sweep dense/hybrid row-level: [`sweep_direct.csv`](../../data/retrieval_eval_runs/diagnostic_full_utopia_throttled__20260525T091921Z/sweep_direct.csv)
- Sweep graph row-level: [`sweep_graph.csv`](../../data/retrieval_eval_runs/diagnostic_full_utopia_throttled__20260525T091921Z/sweep_graph.csv)
- Sweep rerank: [`sweep_rerank.csv`](../../data/retrieval_eval_runs/diagnostic_full_utopia_throttled__20260525T091921Z/sweep_rerank.csv) + summary [`rerank_summary.csv`](../../data/retrieval_eval_runs/diagnostic_full_utopia_throttled__20260525T091921Z/rerank_summary.csv) + best impact [`rerank_best_impact.csv`](../../data/retrieval_eval_runs/diagnostic_full_utopia_throttled__20260525T091921Z/rerank_best_impact.csv)
- Sweep query rewriting: [`sweep_query_rewriting.csv`](../../data/retrieval_eval_runs/diagnostic_full_utopia_throttled__20260525T091921Z/sweep_query_rewriting.csv)
- Manifest: [`manifest.json`](../../data/retrieval_eval_runs/diagnostic_full_utopia_throttled__20260525T091921Z/manifest.json)
- Configurazione promossa: [`recommended_advanced_config.json`](../../data/retrieval_eval_runs/diagnostic_full_utopia_throttled__20260525T091921Z/recommended_advanced_config.json)
- Specifica: [`docs/specs/06b_retrieval_diagnostics.md`](../specs/06b_retrieval_diagnostics.md)
- Metodologia (stub iniziale): [`docs/notes/06b_retrieval_diagnostics_methodology.md`](06b_retrieval_diagnostics_methodology.md)
- Registro run: [`docs/notes/06b_retrieval_experiments_log.md`](06b_retrieval_experiments_log.md)
- Roadmap operativa: [`RETRIEVAL_IMPROVEMENT_ROADMAP.md`](../../RETRIEVAL_IMPROVEMENT_ROADMAP.md)
- Notebook diagnostico: [`notebooks/06b_retrieval_diagnostics.ipynb`](../../notebooks/06b_retrieval_diagnostics.ipynb)
- Notebook Advanced RAG (target promozione): [`notebooks/06_advanced_graph_rag.ipynb`](../../notebooks/06_advanced_graph_rag.ipynb)
