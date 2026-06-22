# 06 - Metodologia Advanced Graph RAG

## Obiettivo

Questo step misura se una pipeline RAG più ricca del baseline semplice migliora le risposte legali mantenendo ogni passaggio spiegabile. L'unità di confronto resta la stessa degli step 04 e 05: stessi dataset puliti, stesso contratto di metriche, stesso modello Utopia/Ollama e stessa indicizzazione Qdrant.

L'esperimento aggiunge cinque componenti attivabili singolarmente: filtri metadata, retrieval hybrid, espansione tramite grafo esplicito, rerank LLM e **query rewriting** (riformulazione/HyDE/multi-query). Ogni componente scrive tracce di riga, così una risposta può essere analizzata partendo dai chunk recuperati e arrivando al contesto effettivamente passato al modello.

## Modalità adottata

La pipeline parte dal manifest dello step 05 e fallisce se gli hash di evaluation e index non sono comparabili. Questo evita di confrontare advanced RAG contro una baseline semplice generata su dati diversi.

Per ogni domanda il processo è:

1. (opzionale) trasformare la query con `apply_query_rewriting` (strategie `none / rewrite / hyde / multi_query`);
2. applicare i filtri statici, se abilitati;
3. interrogare Qdrant in dense-only oppure in hybrid dense+sparse con RRF; per `multi_query` si interroga una volta per variante e si fondono i candidati client-side via dedup;
4. espandere i risultati usando solo edge reali `src_law_id -> dst_law_id` da `edges.jsonl`;
5. deduplicare i chunk, ordinare con rerank LLM se abilitato e costruire un contesto limitato;
6. generare risposta, citazioni e giudizio usando output strutturato.

L'espansione graph-aware è volutamente conservativa: usa solo hop 1, solo relation type consentiti, solo chunk presenti in `chunks.jsonl`, e applica `max_chunks_per_expanded_law` per legge target. Non inferisce relazioni mancanti e non chiede al modello di inventare collegamenti.

## Scelte di framework

La parte retrieval usa il client Qdrant diretto perché la specifica richiede controllo esplicito su named vectors, sparse vectors, `Prefetch` e `RrfQuery`. L'astrazione LangChain sarebbe utile per una chain RAG generica, ma qui nasconderebbe dettagli che devono restare misurabili nel manifest e nelle tracce.

Pydantic AI non è stato introdotto nel runtime. Il progetto usa già un client Utopia/Ollama-compatible con `format=schema`, mentre Pydantic AI e LangChain sono più adatti quando servono agenti, tool calling o orchestrazione conversazionale. In questo step il comportamento desiderato è una pipeline deterministica attorno a poche chiamate strutturate, quindi Pydantic v2 sui contratti è sufficiente e più leggibile.

## Come leggere gli output

Gli artifact stanno in `data/rag_runs/advanced/<run_name>/`.

- `advanced_rag_manifest.json` documenta configurazione effettiva, modelli, hash degli input e riferimento alla run simple RAG.
- `mcq_results.jsonl` e `no_hint_results.jsonl` contengono le tracce per domanda: filtri, retrieval mode, chunk recuperati, edge usati, punteggi rerank, chunk nel contesto e `reference_law_hit`.
- `advanced_diagnostics.json` riassume quante righe hanno usato filtri, hybrid, graph expansion, rerank e quali failure category sono emerse.
- `advanced_rag_summary.json` mantiene le metriche compatibili con gli step precedenti.
- `quality_report.md` è il riepilogo umano della run e degli errori principali.

Le ablation run si ottengono cambiando solo i flag e assegnando un `run_name` diverso: il notebook 06 percorre una scala A0–A4 (da equivalente al simple RAG fino alla configurazione promossa) in directory affiancate e confrontabili.

La configurazione **promossa** dai diagnostics [06b](../results/06b_retrieval_diagnostics.md), usata per la run riportata in [results/06](../results/06_advanced_rag.md), attiva solo `hybrid` + `multi_query`. Graph expansion, rerank e filtri metadata sono stati valutati e **non** promossi (rispettivamente: rumore quasi totale, perdita di recall, esclusione di domande valide); restano nel codice come leve di ablation, non come parte della pipeline raccomandata.

## Integrazione multi-query da 06b

L'Esperimento H del notebook diagnostico 06b ha promosso `multi_query` (n=3) come leva utile con guadagno `+4.3pp` di `article_hit` sul pilot (vedi [docs/results/06b_retrieval_diagnostics.md](../results/06b_retrieval_diagnostics.md)). L'integrazione nel runner segue tre principi:

- **Niente duplicazione**: `apply_query_rewriting` in `advanced_graph_rag/runner.py` riusa direttamente `rewrite_query / generate_hyde / multi_query` da `retrieval_evaluation/query_rewriting.py`, con la stessa `QUERY_REWRITING_PROMPT_VERSION` e lo stesso `QueryRewriteCache` JSONL.
- **Cache condivisa con 06b**: il default `query_rewriting_cache_dir = data/cache/query_rewriting` coincide con quello del diagnostico. Se modello e prompt version restano invariati tra 06b e 06, la prima full run hit la cache esistente con zero chiamate LLM aggiuntive per il rewriting.
- **Fusione client-side semplice**: per `multi_query` si esegue un `search_hybrid` per ciascuna variante e si fondono i candidati con `_dedupe_chunks` preservando l'ordine di prima apparizione, troncando a `top_k`. L'approccio è deterministico.

I contatori cache (`cache_hits / cache_misses / failures`) e i primi errori sono registrati in `manifest.query_rewriting` per audit. Strategy `"none"` resta un no-op senza overhead.

## Limiti dichiarati

L'hybrid retrieval richiede che collection e manifest dichiarino sparse vectors; se manca il supporto sparse la pipeline fallisce prima di processare le righe. La graph expansion resta a hop 1 per preservare tracciabilità e semplicità del PoC. Il rerank LLM è vincolato a score interi `0`, `1`, `2`, ma il contenuto del giudizio resta una decisione del modello: per questo i punteggi sono esportati e vanno ispezionati nelle analisi. Per il query rewriting, `multi_query` triplica le chiamate Qdrant per ogni domanda (locali, costo trascurabile) ma il rewriting LLM aggiunge una chiamata Utopia per domanda che non sia già in cache; la cache versionata da `QUERY_REWRITING_PROMPT_VERSION` invalida automaticamente i risultati al cambio prompt.
