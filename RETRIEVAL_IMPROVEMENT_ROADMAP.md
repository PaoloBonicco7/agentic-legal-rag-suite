# Roadmap operativa per migliorare il Retrieval Legal RAG

> **Status generale**: Fase 0 done, Fase 1 done, Fase 2 next, Fase 3 integrata ma non eseguita, Fase 4 integrata ma non eseguita  
> **Ultimo aggiornamento**: 2026-05-13  
> **Scope**: tesi PoC, retrieval diagnostics e promozione in Advanced Graph RAG  
> **Uso previsto**: documento vivo per agenti autonomi. Ogni agente prende una fase, la completa, aggiorna questo file e passa il testimone.

## 1. Contesto comune

### Obiettivo della tesi

Dimostrare empiricamente che un Advanced RAG con retrieval migliorato supera Simple RAG su domande legali (corpus leggi regionali Valle d'Aosta).

Il banco di prova retrieval-only è `notebooks/06b_retrieval_diagnostics.ipynb`. La configurazione migliore prodotta da 06b viene poi promossa in `notebooks/06_advanced_graph_rag.ipynb`.

**Target end-to-end (Fase 6)**: `reference_article_hit` ≥ Simple RAG + 20pp, `reference_law_hit` ≥ +15pp, MCQ accuracy ≥ +10pp, no_hint judge score ≥ +5pp. Se non raggiunto, documentare risultato e diagnosi in `docs/results/06_advanced_graph_rag.md` invece di forzare ulteriori complessità.

### Baseline misurata

Run di riferimento: `data/retrieval_eval_runs/default__20260511T180530Z/`. I valori sono identici per MCQ e no_hint perché il retrieval usa solo lo stem della domanda.

| scenario | article_hit | law_hit | MRR | stato |
|---|---:|---:|---:|---|
| Baseline dense@10 | 45.0% | 72.0% | 0.288 | run |
| Best dense top-k=100 | 66.0% | 87.0% | 0.295 | run |
| + Filter law_status=current | 45.0% | 72.0% | 0.288 | run |
| + Graph default seed=3 | 45.0% | 74.0% | 0.288 | run |
| + Best graph from sweep | 66.0% | 88.0% | 0.295 | run, degenerato su top_k=100 |
| Hybrid retrieval | n/a | n/a | n/a | skipped, indice dense-only |

Diagnosi:

- Recall ceiling `66%@100`: 34 domande su 100 non hanno l'articolo corretto nei top-100.
- `MRR = 0.29` con `recall@100 = 66%`: spazio per reranking.
- Filtri metadata non migliorano la baseline (dataset già dominato da leggi correnti).
- Graph expansion attuale aggiunge molto rumore: non è la prima leva.

### Componenti già presenti

- `src/legal_rag/indexing/embeddings.py`: backend locale BGE-M3 con dense 1024 e sparse `lexical_weights`.
- `src/legal_rag/indexing/qdrant_store.py`: named vectors `dense` e `sparse`, payload indexes, Qdrant file mode.
- `src/legal_rag/advanced_graph_rag/retrieval.py`: `search_dense()`, `search_hybrid()` con Qdrant Query API e RRF, `expand_with_graph()`.
- `src/legal_rag/retrieval_evaluation/evaluator.py`: `retrieve_direct()`, `evaluate_candidate_set()`, `evaluate_with_rerank()`, `RerankCache()`, `summarize_scenario()`, `write_run_artifacts()`.
- `src/legal_rag/advanced_graph_rag/prompts.py`: prompt rerank con score in `{0, 1, 2}`.
- `src/legal_rag/oracle_context_evaluation/llm.py`: `UtopiaStructuredChatClient`.

### Convenzioni artifact

- Run diagnostiche: `data/retrieval_eval_runs/<run_name>__<YYYYMMDDTHHMMSSZ>/`
- Run Advanced RAG: `data/rag_runs/advanced/<run_name>/`
- Index Qdrant locale: `data/indexes/qdrant`
- Collection storica: `legal_chunks` (non sovrascrivere)
- Collection BGE-M3: `legal_chunks_bge_m3`
- Cache rerank: `data/cache/rerank/<model>.jsonl`
- Cache rewriting: `data/cache/query_rewriting/<strategy>__<model>__<prompt_version>.jsonl`

Non sovrascrivere run precedenti: ogni run è evidenza metodologica per la tesi. Le cartelle `data/cache/` e `data/retrieval_eval_runs/` devono essere in `.gitignore`.

### Versioning di schema e prompt

Reference pattern: `src/legal_rag/oracle_context_evaluation/`.

- `<STEP>_SCHEMA_VERSION`: cambia al cambio del contratto Pydantic dei record output.
- `<STEP>_PROMPT_VERSION`: cambia al cambio del testo o della struttura di un prompt.

Regole: bump semver locale (`v1` → `v2`, mai riusare la stessa stringa); le versioni vanno nel `manifest.json` e nella chiave delle cache (`RerankCache`, `QueryRewriteCache`) così un cambio prompt invalida la cache senza cancellare file; ri-esportare le costanti dall'`__init__.py` del package.

### Schema entry experiment log

Ogni entry in `docs/notes/06b_retrieval_experiments_log.md` segue questo template fisso:

```markdown
## <run_name>__<YYYYMMDD> — <titolo sintetico>

- **Data**: 2026-MM-DD
- **Run dir**: data/retrieval_eval_runs/<run_name>__<...>
- **Cambio rispetto al precedente**: <una riga>
- **Configurazione chiave**: embedding=<model>, retrieval_mode=<dense|hybrid>, top_k=<n>, rerank=<on|off>, query_rewriting=<strategy>
- **Metriche** (MCQ / no_hint):
  - article_hit@<k>: X% / Y%
  - article_hit@5 (post-rerank, se applicabile): X% / Y%
  - law_hit@<k>: X% / Y%
  - MRR: X.XXX / X.XXX
- **Costo LLM**: <n_calls totali>, <cache_hit_rate>%
- **Conclusione**: <tecnica utile/inutile e perché>
- **Decisione successiva**: <prossima leva da provare>
```

## 2. Protocollo per agenti

Un agente lavora su una sola fase alla volta. Se una fase è già `in_progress` da un altro agente, fermarsi e chiedere.

### Prima di iniziare

1. Leggere questo file e `AGENTS.md`.
2. Leggere la spec dello step coinvolto in `docs/specs/`.
3. Controllare `git status --short` e lo stato della fase nel tracker.

### Durante la fase

- Toccare solo i file indicati, salvo necessità tecnica documentata nel handoff.
- Aggiornare spec, note e result nello stesso passo logico del codice.
- Usare pilot prima di full run per qualunque chiamata Utopia costosa.
- Registrare ogni run rilevante in `docs/notes/06b_retrieval_experiments_log.md`.
- Non cancellare cache, run o index esistenti.

### A fine fase

Aggiornare nella fase: `Status` (`todo`/`in_progress`/`blocked`/`done`), `Owner`, `Started`, `Completed`, checklist task, sezione Handoff. Replicare lo stato nel tracker (Sezione 3).

Formato handoff:

```markdown
#### Handoff
- **Completato**: ...
- **File modificati**: ...
- **Artifact prodotti**: ...
- **Verifica**: ...
- **Problemi aperti**: ...
- **Prossimo agente**: partire da ...
```

## 3. Tracker fasi

Legenda **Complessità**: 🟢 LOW (cambi locali, mostly doc/config) · 🟡 MED (integrazione con codice esistente, sweep) · 🟠 HIGH (nuovo codice + test + integrazioni multiple).
Legenda **Costo LLM**: nessuno · medio (≤ ~2k chiamate con cache attiva) · alto (≥ ~2k chiamate, oppure run end-to-end).

| Fase | Titolo | Status | Owner | Started | Completed | Dipende da | Complessità | Tempo | Costo LLM | Output principale |
|---:|---|---|---|---|---|---|---|---|---|---|
| 0 | Setup tracciabilità | done | Codex | 2026-05-12 | 2026-05-12 | nessuna | 🟢 LOW | 30-45 min | nessuno | log esperimenti + spec 06b |
| 1 | Re-index BGE-M3 | done | Codex | 2026-05-12 | 2026-05-13 | 0 | 🟡 MED | manuale | nessuno | collection `legal_chunks_bge_m3` |
| 2 | Hybrid retrieval | todo |  |  |  | 1 | 🟡 MED | 1-2 h | nessuno | sweep hybrid F |
| 3 | LLM reranking | in_progress | Codex | 2026-05-13 |  | 2 | 🟠 HIGH | 2-4 h | medio | sweep rerank G |
| 4 | Query rewriting / HyDE | in_progress | Codex | 2026-05-13 |  | 2, 3 | 🟠 HIGH | 3-5 h | medio | modulo rewriting + sweep H |
| 5 | Waterfall finale | todo |  |  |  | 2, 3, 4 | 🟢 LOW | 30-60 min | nessuno | `best_config.json` |
| 6 | Promozione Advanced RAG | todo |  |  |  | 5 | 🟠 HIGH | 3-5 h | alto | run end-to-end e report |

## 4. Fase 0 - Setup tracciabilità

**Status**: done  
**Owner**: Codex  
**Started**: 2026-05-12  
**Completed**: 2026-05-12

### Obiettivo

Creare la documentazione minima che permette alle fasi successive di registrare esperimenti, scelte e risultati.

### File da leggere

- `docs/specs/README.md`
- `docs/notes/README.md`
- `docs/specs/05_simple_rag.md`
- `docs/notes/06_advanced_graph_rag_methodology.md`
- `data/retrieval_eval_runs/default__20260511T180530Z/scenarios.csv`

### Task

- [x] Creare `docs/notes/06b_retrieval_experiments_log.md`.
- [x] Inserire la prima entry retroattiva per `default__20260511T180530Z`.
- [x] Creare `docs/specs/06b_retrieval_diagnostics.md` in inglese con struttura standard.
- [x] Creare `docs/notes/06b_retrieval_diagnostics_methodology.md` in italiano con stub iniziale.
- [x] Aggiornare `docs/specs/README.md` con `06b_retrieval_diagnostics.md`.
- [x] Aggiornare `docs/notes/README.md` con i due nuovi file.

### Verifica

- I tre file nuovi esistono.
- I README linkano i file corretti.
- Le metriche baseline sono copiate da `scenarios.csv`, non riscritte a memoria.

### Handoff

- **Completato**: creata la tracciabilità iniziale per le run retrieval-only; registrata la baseline retroattiva `default__20260511T180530Z`; aggiunta la spec 06b e lo stub metodologico.
- **File modificati**: `docs/specs/06b_retrieval_diagnostics.md`, `docs/notes/06b_retrieval_experiments_log.md`, `docs/notes/06b_retrieval_diagnostics_methodology.md`, `docs/specs/README.md`, `docs/notes/README.md`, `RETRIEVAL_IMPROVEMENT_ROADMAP.md`.
- **Artifact prodotti**: nessun artifact runtime; solo documentazione versionabile.
- **Verifica**: metriche baseline copiate da `data/retrieval_eval_runs/default__20260511T180530Z/scenarios.csv`; README aggiornati con i nuovi link.
- **Problemi aperti**: la run hybrid resta skipped finché Fase 1 non produce una collection con sparse vectors.
- **Prossimo agente**: Fase 1.

## 5. Fase 1 - Re-index BGE-M3

**Status**: done  
**Owner**: Codex  
**Started**: 2026-05-12  
**Completed**: 2026-05-13

### Obiettivo

Costruire una collection Qdrant parallela `legal_chunks_bge_m3` con dense vectors BGE-M3 1024-dim e sparse vectors native. Questa fase sblocca hybrid retrieval.

### File da leggere

- `src/legal_rag/indexing/models.py`
- `src/legal_rag/indexing/cli.py`
- `src/legal_rag/indexing/embeddings.py`
- `src/legal_rag/indexing/qdrant_store.py`
- `docs/specs/03_indexing_contract.md`
- `docs/notes/03_indexing_contract_methodology.md`
- `docs/results/03_indexing_contract.md`
- `docs/notes/06b_retrieval_experiments_log.md`

### Comando base

Hybrid è attivo di default. Non usare `--hybrid-enabled`.

```bash
PYTHONPATH=src .venv/bin/python -m legal_rag.indexing \
  --embedding-backend local \
  --embedding-model BAAI/bge-m3 \
  --collection-name legal_chunks_bge_m3 \
  --force-rebuild
```

Smoke test sample, se serve:

```bash
PYTHONPATH=src .venv/bin/python -m legal_rag.indexing \
  --embedding-backend local \
  --embedding-model BAAI/bge-m3 \
  --collection-name legal_chunks_bge_m3_sample \
  --chunk-selection-mode sample \
  --sample-size 2000 \
  --force-rebuild
```

### Task

- [x] Verificare che `pyproject.toml` contenga `FlagEmbedding`, `sentence-transformers`, `qdrant-client[fastembed]`. Sono già dichiarati: l'agente NON deve aggiungerli, solo confermare.
- [x] Verificare che il venv li abbia installati (`.venv/bin/python -c "import FlagEmbedding, sentence_transformers"`); se manca, eseguire `uv sync`.
- [x] Preparare il notebook 03 per full indexing BGE-M3 monitorabile e protetta da flag esplicito.
- [x] Verificare che `force_rebuild=True` sia confinato a `collection_name=legal_chunks_bge_m3` e non tocchi `legal_chunks`.
- [x] Eseguire sample indexing se la macchina è lenta o la GPU non è disponibile.
- [x] Eseguire full indexing su `legal_chunks_bge_m3`.
- [x] Leggere il nuovo `data/indexing_runs/<run_id>/index_manifest.json`.
- [x] Verificare collection count contro `chunks.jsonl`.
- [x] Verificare che la collection esponga named vector `dense` e sparse vector `sparse`.
- [x] Eseguire una query smoke con `search_dense()` e `search_hybrid()`.

### Documentazione da aggiornare

- `docs/specs/03_indexing_contract.md`: chiarire collection parallele e sparse vectors.
- `docs/notes/03_indexing_contract_methodology.md`: scelta BGE-M3 vs nomic.
- `docs/results/03_indexing_contract.md`: blocco run BGE-M3 con count, vector size, tempi, manifest.
- `docs/notes/06b_retrieval_experiments_log.md`: entry "indice BGE-M3 disponibile".

### Verifica

- `hybrid_enabled = true` nel manifest.
- `embedding_model = "BAAI/bge-m3"` o resolved equivalent.
- Dense vector size 1024, sparse vector presente.
- `failure_count = 0` (se il manifest espone il campo) e `collection_points_count` coincide con il numero di chunk selezionati.

### Stop & ask

- Installazione o import di `FlagEmbedding` fallisce.
- Qdrant locale non apre la collection dopo indexing.

### Handoff

- **Completato**: predisposto setup notebook per re-index BGE-M3 locale in collection parallela `legal_chunks_bge_m3`; aggiunti eventi di progresso runtime alla pipeline; completata la full indexing manuale; verificati manifest, count, dense vector e sparse vector; corretta la risoluzione del manifest nel notebook per allineare progress log e run future.
- **File modificati**: `notebooks/03_indexing_contract.ipynb`, `src/legal_rag/indexing/pipeline.py`, `tests/test_indexing_contract.py`, `docs/specs/03_indexing_contract.md`, `docs/notes/03_indexing_contract_methodology.md`, `docs/results/03_indexing_contract.md`, `docs/notes/06b_retrieval_experiments_log.md`, `.gitignore`, `RETRIEVAL_IMPROVEMENT_ROADMAP.md`.
- **Artifact prodotti**: `data/indexing_runs/20260512_212818/index_manifest.json`; `data/indexing_runs/bge_m3_full_20260512_212742_progress.jsonl`; collection Qdrant locale `legal_chunks_bge_m3`; modello `BAAI/bge-m3` scaricato/cache locale; collection sample parziale `legal_chunks_bge_m3_sample` a 1216 punti, non usata come artifact di fase.
- **Verifica**: `ready_for_retrieval=true`; `selected_count=indexed_count=collection_points_count=76467`; `failure_count=0`; embedding `BAAI/bge-m3`, `hybrid_enabled=true`, vector size `1024`; Qdrant espone named vector `dense` e sparse vector `sparse`; smoke retrieval dense e hybrid restituiscono 3 risultati ciascuno.
- **Problemi aperti**: Qdrant local mode segnala warning prestazionale per collection oltre 20.000 punti; per la tesi PoC resta accettabile, ma gli sweep lunghi vanno monitorati.
- **Prossimo agente**: Fase 2, puntare `notebooks/06b_retrieval_diagnostics.ipynb` a `data/indexing_runs/20260512_212818/index_manifest.json` e riattivare l'esperimento hybrid.

## 6. Fase 2 - Esperimento F: Hybrid retrieval

**Status**: todo  
**Owner**:  
**Started**:  
**Completed**:

### Obiettivo

Misurare se dense + sparse via RRF migliora `article_hit@k` rispetto al dense-only.

### Nota tecnica

RRF è il default corretto: dense similarity e sparse lexical score non sono comparabili direttamente. Non usare somme pesate lineari senza valutazione dedicata.

### File da leggere

- `notebooks/06b_retrieval_diagnostics.ipynb`
- `src/legal_rag/advanced_graph_rag/retrieval.py`
- `src/legal_rag/retrieval_evaluation/evaluator.py`
- nuovo `index_manifest.json` della Fase 1
- `docs/specs/06b_retrieval_diagnostics.md`
- `docs/notes/06b_retrieval_diagnostics_methodology.md`

### Task

- [ ] Puntare `CONFIG.index_manifest_path` del notebook 06b al manifest BGE-M3.
- [ ] Verificare che `hybrid_available = True` nella cella preflight.
- [ ] Riutilizzare o convertire l'esperimento esistente `hybrid_if_available` (evitare celle duplicate).
- [ ] Estendere lo sweep a `top_k in {10, 20, 50, 100}` e `rrf_k in {30, 60, 90}`.
- [ ] Usare `retrieve_direct(... retrieval_mode="hybrid" ...)` o `search_hybrid()` con `rrf_k`.
- [ ] Aggregare con `evaluate_candidate_set()` e `summarize_scenario()`.
- [ ] Salvare righe in `sweep_direct.csv` e scenario in `scenarios.csv`.
- [ ] Aggiornare waterfall con best hybrid.

### Documentazione da aggiornare

- `docs/specs/06b_retrieval_diagnostics.md`: aggiungere Esperimento F.
- `docs/notes/06b_retrieval_diagnostics_methodology.md`: sezione "Hybrid retrieval".
- `docs/notes/06b_retrieval_experiments_log.md`: entry `hybrid_sweep__<date>`.

### Verifica

- Esperimento F `status=run`, `rows > 0`.
- Best hybrid confrontato con best dense BGE-M3 e con baseline storica.
- Se hybrid non migliora, documentare il risultato negativo invece di forzare parametri.

### Stop & ask

- `hybrid_available` resta `False`.
- Qdrant local mode fallisce su `query_points(prefetch=...)`.
- Sweep oltre 2 ore senza output intermedio.

### Handoff

- **Completato**:
- **File modificati**:
- **Artifact prodotti**:
- **Verifica**:
- **Problemi aperti**:
- **Prossimo agente**: Fase 3.

## 7. Fase 3 - Esperimento G: LLM reranking

**Status**: in_progress  
**Owner**: Codex  
**Started**: 2026-05-13  
**Completed**:

### Obiettivo

Misurare il guadagno del reranker LLM sugli output del best hybrid.

### File da leggere

- `src/legal_rag/retrieval_evaluation/evaluator.py`
- `src/legal_rag/advanced_graph_rag/prompts.py`
- `src/legal_rag/oracle_context_evaluation/llm.py`
- `src/legal_rag/advanced_graph_rag/runner.py`
- `notebooks/06b_retrieval_diagnostics.ipynb`
- ultima entry `hybrid_sweep` nel log esperimenti

### Nota tecnica

`evaluate_with_rerank()` applica score pre-calcolati: NON chiama il modello. La chiamata LLM va costruita esplicitamente con `build_rerank_prompt()` + `UtopiaStructuredChatClient.structured_chat()`, gli score vanno persistiti in `RerankCache`, e solo poi passati all'evaluator.

La chiave cache deve includere `model` + `RERANK_PROMPT_VERSION` + lista ordinata di `candidate_chunk_ids`. La cache memorizza score grezzi `{chunk_id: int}`, NON l'ordinamento finale: cambi di `rerank_output_k` non invalidano la cache.

### Task

- [x] Preparare `data/cache/rerank/<sanitized_model>.jsonl` e assicurarsi che `data/cache/` sia in `.gitignore`.
- [x] Identificare/definire `RERANK_PROMPT_VERSION` in `src/legal_rag/advanced_graph_rag/prompts.py` (se non esiste, aggiungerlo) e includerlo nella chiave cache.
- [x] Usare `build_rerank_prompt(question, candidates)` per generare il prompt.
- [x] Usare `UtopiaStructuredChatClient.structured_chat()` con `RerankOutput.model_json_schema()`.
- [x] Salvare in `RerankCache` score `{chunk_id, score}` e applicarli con `evaluate_with_rerank()`.
- [ ] Pilot su 30 domande per dataset con seed fisso, sweep `rerank_input_k in {20, 50, 100}` × `rerank_output_k in {3, 5, 10}`.
- [ ] Scegliere winner su `article_hit@5`, con input più piccolo a parità di risultato.
- [ ] Full run solo sul winner.
- [x] Aggiornare waterfall con `+ LLM rerank`.

### Documentazione da aggiornare

- `docs/specs/06b_retrieval_diagnostics.md`: aggiungere Esperimento G e `sweep_rerank.csv`.
- `docs/notes/06b_retrieval_diagnostics_methodology.md`: sezione "LLM reranking".
- `docs/notes/06b_retrieval_experiments_log.md`: entry pilot e full.

### Verifica

- Seconda esecuzione pilot usa cache con hit rate circa 100%.
- Numero chiamate Utopia registrato.
- `article_hit@5` reranked confrontato con direct@5.
- Structured-output failures sotto soglia accettabile.

### Stop & ask

- Pilot supera 1500 chiamate previste.
- Structured-output failures oltre 5%.
- Si vuole cambiare `judge_model` rispetto al default `resolved_judge_model`.

### Handoff

- **Completato**: integrata la chiamata rerank LLM cache-aware in `retrieval_evaluation`; aggiunta `RERANK_PROMPT_VERSION`; aggiornata la chiave cache con prompt version, modello e lista ordinata candidati; aggiunto supporto notebook per pilot G, `sweep_rerank.csv`, manifest cache/failure e scenario `+ LLM rerank`.
- **File modificati**: `src/legal_rag/advanced_graph_rag/prompts.py`, `src/legal_rag/advanced_graph_rag/__init__.py`, `src/legal_rag/retrieval_evaluation/models.py`, `src/legal_rag/retrieval_evaluation/evaluator.py`, `src/legal_rag/retrieval_evaluation/__init__.py`, `tests/test_retrieval_evaluation.py`, `notebooks/06b_retrieval_diagnostics.ipynb`, `docs/specs/06b_retrieval_diagnostics.md`, `docs/notes/06b_retrieval_diagnostics_methodology.md`, `docs/notes/06b_retrieval_experiments_log.md`, `RETRIEVAL_IMPROVEMENT_ROADMAP.md`.
- **Artifact prodotti**: nessuna run retrieval; nessuna chiamata Utopia; nessuna cache runtime creata.
- **Verifica**: `pytest tests/test_retrieval_evaluation.py` verde (`12 passed`); notebook JSON e celle Python parse OK.
- **Problemi aperti**: Fase 2 risulta ancora `todo` nel tracker; il pilot G va eseguito solo dopo avere un best hybrid reale. Restano da verificare seconda esecuzione con cache hit circa 100%, winner su `article_hit@5` e full run winner.
- **Prossimo agente**: completare Fase 2 se non già eseguita, poi attivare `RUN_RERANK=True` nel notebook 06b per il pilot G.

## 8. Fase 4 - Esperimento H: Query rewriting / HyDE / Multi-query

**Status**: in_progress  
**Owner**: Codex  
**Started**: 2026-05-13  
**Completed**:

### Obiettivo

Ridurre il mismatch tra domanda naturale e linguaggio normativo tramite rewrite, HyDE o multi-query.

### File da leggere

- `src/legal_rag/oracle_context_evaluation/llm.py`
- `src/legal_rag/oracle_context_evaluation/prompts.py`
- `src/legal_rag/retrieval_evaluation/evaluator.py`
- `src/legal_rag/retrieval_evaluation/__init__.py`
- `tests/test_retrieval_evaluation.py`
- `notebooks/06b_retrieval_diagnostics.ipynb`

### Task codice

- [x] Creare `src/legal_rag/retrieval_evaluation/query_rewriting_prompts.py` con `QUERY_REWRITING_PROMPT_VERSION = "query-rewriting-v1"` e prompt per `rewrite`, `hyde`, `multi_query`.
- [x] Creare `src/legal_rag/retrieval_evaluation/query_rewriting.py` con modelli Pydantic `extra="forbid"` e le funzioni `rewrite_query()`, `generate_hyde()`, `multi_query()`.
- [x] Implementare `QueryRewriteCache` JSONL — chiave `SHA256(question | strategy | model | QUERY_REWRITING_PROMPT_VERSION)`, path `data/cache/query_rewriting/<strategy>__<sanitized_model>__<prompt_version>.jsonl`.
- [x] Esportare API + costante versione in `src/legal_rag/retrieval_evaluation/__init__.py`.

### Task notebook

- [ ] Pilot 30 domande per dataset confrontando `none`, `rewrite`, `hyde`, `multi_query`.
- [ ] Retrieval usando best hybrid e winner rerank della Fase 3.
- [ ] Mostrare tre trasformazioni qualitative reali.
- [ ] Full run solo sulla strategia winner.
- [x] Aggiornare waterfall.

### Test

- [x] Cache round-trip.
- [x] Due chiamate uguali con cache producono una sola chiamata client mock.
- [x] `multi_query(n=3)` ritorna esattamente tre stringhe non vuote.
- [x] Test di integrazione minimale con retrieval candidate set, senza chiamate remote.

### Documentazione da aggiornare

- [x] `docs/specs/06b_retrieval_diagnostics.md`: aggiungere Esperimento H.
- [x] `docs/notes/06b_retrieval_diagnostics_methodology.md`: sezione "Query rewriting / HyDE".
- [x] `docs/specs/06_advanced_graph_rag.md`: preparare flag query rewriting.
- [x] `docs/notes/06b_retrieval_experiments_log.md`: entry per strategia testata.

### Verifica

- `pytest tests/test_retrieval_evaluation.py` verde.
- Cache riusabile senza nuove chiamate.
- Winner scelto con evidenza; se nessuna strategia migliora, impostare `strategy="none"` e documentare.

### Stop & ask

- Structured-output failures oltre 5%.
- Multi-query costa più del previsto nel pilot.
- I prompt generano testo non legale o troppo lungo.

### Handoff

- **Completato**: integrati prompt versionati per `rewrite`, `hyde` e `multi_query`; aggiunto `QueryRewriteCache` JSONL con chiave versionata; aggiunti modelli Pydantic strict, funzioni LLM cache-aware, retrieval multi-variante deduplicato, evaluator row-level, `sweep_query_rewriting.csv`, manifest e scenario waterfall `+ Query rewriting`.
- **File modificati**: `src/legal_rag/retrieval_evaluation/query_rewriting_prompts.py`, `src/legal_rag/retrieval_evaluation/query_rewriting.py`, `src/legal_rag/retrieval_evaluation/models.py`, `src/legal_rag/retrieval_evaluation/evaluator.py`, `src/legal_rag/retrieval_evaluation/__init__.py`, `tests/test_retrieval_evaluation.py`, `notebooks/06b_retrieval_diagnostics.ipynb`, `docs/specs/06b_retrieval_diagnostics.md`, `docs/specs/06_advanced_graph_rag.md`, `docs/notes/06b_retrieval_diagnostics_methodology.md`, `docs/notes/06b_retrieval_experiments_log.md`, `RETRIEVAL_IMPROVEMENT_ROADMAP.md`.
- **Artifact prodotti**: nessuna run retrieval; nessuna chiamata Utopia; nessuna cache runtime creata.
- **Verifica**: `pytest tests/test_retrieval_evaluation.py` verde (`19 passed`); notebook JSON e celle Python parse OK.
- **Problemi aperti**: Fase 2 resta `todo`; Fase 3 e Fase 4 sono integrate ma non hanno ancora pilot reali. Il confronto con winner rerank richiede prima eseguire il pilot G.
- **Prossimo agente**: completare Fase 2, poi eseguire pilot G/H con flag espliciti prima della Fase 5.

## 9. Fase 5 - Waterfall finale e best_config.json

**Status**: todo  
**Owner**:  
**Started**:  
**Completed**:

### Obiettivo

Consolidare le evidenze di Fasi 2-4 e produrre una configurazione unica promuovibile nello step 06.

### File da leggere

- `notebooks/06b_retrieval_diagnostics.ipynb`
- `docs/notes/06b_retrieval_experiments_log.md`
- output run Fase 2, 3, 4

### Task

- [ ] Aggiornare waterfall con baseline, BGE-M3 dense, hybrid, rewriting, rerank, graph se utile.
- [ ] Disattivare nel best config qualunque leva con performance negativa.
- [ ] Scrivere `best_config.json` nella run finale + README breve nella stessa run dir.
- [ ] Creare `docs/results/06b_retrieval_diagnostics.md`.
- [ ] Aggiornare methodology con "Best config emersa".
- [ ] Aggiornare experiment log con entry finale.

### Schema minimo `best_config.json`

```json
{
  "embedding_backend": "local",
  "embedding_model": "BAAI/bge-m3",
  "index_collection_name": "legal_chunks_bge_m3",
  "index_manifest_path": "data/indexing_runs/<run_id>/index_manifest.json",
  "retrieval_mode": "hybrid",
  "top_k": 50,
  "rrf_k": 60,
  "query_rewriting": {
    "strategy": "none",
    "n": 3,
    "prompt_version": "query-rewriting-v1"
  },
  "rerank_enabled": true,
  "rerank_input_k": 50,
  "rerank_output_k": 5,
  "graph_expansion_enabled": false,
  "graph_expansion_seed_k": 3,
  "max_chunks_per_expanded_law": 2,
  "max_expanded_chunks_total": 15,
  "min_edge_confidence": 0.45,
  "relation_types": ["REFERENCES", "AMENDS", "INSERTS", "MODIFIED_BY", "INSERTED_BY"],
  "metrics": {
    "mcq": {"article_hit_at_output_k": 0.0, "law_hit_at_output_k": 0.0, "article_mrr": 0.0},
    "no_hint": {"article_hit_at_output_k": 0.0, "law_hit_at_output_k": 0.0, "article_mrr": 0.0}
  },
  "source_run": "<run_name>__<timestamp>",
  "produced_at": "<iso8601>"
}
```

I valori numerici sopra sono placeholder: l'agente li sostituisce con i risultati reali.

### Verifica

- `best_config.json` è JSON valido con tutti i campi popolati da run reali.
- Il waterfall mostra almeno una leva con guadagno positivo, oppure documenta perché non accade.

### Stop & ask

- Le leve F, G o H peggiorano tutte la baseline.
- Il best config richiede graph expansion nonostante il noise ratio resti alto.

### Handoff

- **Completato**:
- **File modificati**:
- **Artifact prodotti**:
- **Verifica**:
- **Problemi aperti**:
- **Prossimo agente**: Fase 6.

## 10. Fase 6 - Promozione in Advanced Graph RAG e run end-to-end

**Status**: todo  
**Owner**:  
**Started**:  
**Completed**:

### Obiettivo

Applicare `best_config.json` alla pipeline Advanced Graph RAG e confrontare Simple RAG, Advanced current e Advanced best.

### File da leggere

- `data/retrieval_eval_runs/<final_run>__<ts>/best_config.json`
- `src/legal_rag/advanced_graph_rag/models.py`
- `src/legal_rag/advanced_graph_rag/runner.py`
- `src/legal_rag/advanced_graph_rag/retrieval.py`
- `src/legal_rag/advanced_graph_rag/prompts.py`
- `src/legal_rag/retrieval_evaluation/query_rewriting.py`
- `notebooks/06_advanced_graph_rag.ipynb`
- `tests/test_advanced_graph_rag.py`
- `docs/specs/06_advanced_graph_rag.md`

### Decisione tecnica preliminare

- Verificare con `grep -n collection_name src/legal_rag/advanced_graph_rag/models.py` se esiste già un campo per la collection. Se sì, **riusarlo**: NON aggiungere `index_collection_name`. Se non esiste, aggiungere `collection_name: str | None = None` (un solo campo).
- `AdvancedRagConfig` espone già un `index_manifest_path`: questa è la via canonica per puntare al nuovo indice BGE-M3.

### Task codice

- [ ] Estendere `AdvancedRagConfig` con:
  - `query_rewriting_enabled: bool = False`
  - `query_rewriting_strategy: Literal["none", "rewrite", "hyde", "multi_query"] = "none"`
  - `query_rewriting_n: int = 3`
  - eventuale override della collection (un solo campo, vedi "Decisione tecnica preliminare").
- [ ] Aggiungere hook `apply_query_rewriting()` prima di `search_dense()` / `search_hybrid()`. Per `strategy="none"` l'hook è identità (no chiamate LLM, no allocazioni).
- [ ] Per `multi_query`, fondere risultati client-side con RRF semplice e deterministico (riusare il pattern di `dedupe_chunks`).
- [ ] Importare i prompt da `src/legal_rag/retrieval_evaluation/query_rewriting_prompts.py` invece di duplicarli in `advanced_graph_rag/prompts.py`.
- [ ] Bump `ADVANCED_RAG_SCHEMA_VERSION` (semver locale v→v+1).
- [ ] Bump `ADVANCED_RAG_PROMPT_VERSION` solo se cambia un prompt Advanced RAG effettivo.
- [ ] Aggiornare manifest con query rewriting config e versioning effettive.
- [ ] Aggiornare export in `src/legal_rag/advanced_graph_rag/__init__.py` se vengono esposte nuove API.

### Task notebook

- [ ] Caricare `best_config.json` e popolare `AdvancedRagConfig`.
- [ ] Eseguire dry-run su 10 domande con seed fisso.
- [ ] Eseguire run full solo dopo dry-run ok.
- [ ] Produrre tabella Simple vs Advanced current vs Advanced best.
- [ ] Salvare run in `data/rag_runs/advanced/<run_name>/`.

### Test

- [ ] `AdvancedRagConfig` accetta i nuovi flag.
- [ ] `apply_query_rewriting(strategy="none")` è identità.
- [ ] Strategy `rewrite` funziona con client mock.
- [ ] Manifest contiene schema version bump e config rewriting.
- [ ] `pytest tests/test_advanced_graph_rag.py` verde.

### Documentazione da aggiornare

- `docs/specs/06_advanced_graph_rag.md`: input, pipeline, contract, quality gates, acceptance criteria.
- `docs/notes/06_advanced_graph_rag_methodology.md`: promozione best config 06b.
- `docs/results/06_advanced_graph_rag.md`: confronto finale.
- `docs/notes/06b_retrieval_experiments_log.md`: entry di chiusura end-to-end.

### Verifica

- Dry-run 10 domande senza errori non banali.
- Full run produce `advanced_rag_summary.json`.
- Tabelle finali presenti in notebook e docs result.
- Target confrontato con Simple RAG usando stesso evaluation manifest hash.

### Stop & ask

- Non è chiaro quale run Simple RAG usare come baseline.
- Dry-run fallisce con structured-output errors, timeouts o schema mismatch.
- Target end-to-end (Sezione 1) non viene raggiunto: documentare risultato e chiedere se attivare Future Work.

### Handoff

- **Completato**:
- **File modificati**:
- **Artifact prodotti**:
- **Verifica**:
- **Problemi aperti**:
- **Prossimo agente**: chiusura o Future Work.

## 11. Future Work

Attivare solo se la Fase 6 non raggiunge i target o se serve una seconda iterazione sperimentale.

| Esperimento | Quando attivarlo | Implementazione suggerita |
|---|---|---|
| Diversity / MMR cap per legge | Candidati saturati da pochi articoli lunghi | `src/legal_rag/retrieval_evaluation/diversity.py` |
| Failure taxonomy qualitativa | Serve capire perché falliscono i casi residui | `src/legal_rag/retrieval_evaluation/failure_taxonomy.py` |
| Cross-encoder locale | Rerank LLM troppo costoso o instabile | `BAAI/bge-reranker-v2-m3` come confronto |
| Graph expansion ricalibrata | Hybrid/rewrite alzano recall e il graph può aggiungere casi residui | nuovo sweep con noise cap più severo |
| Chunking strutturale ibrido | Molti errori sono `right_law_wrong_article` | revisione step 01/03 su chunk lunghi |
