# 06 - Metodologia Advanced Graph RAG

## Obiettivo

Questo step misura se una pipeline RAG più ricca del baseline semplice migliora le risposte legali mantenendo ogni passaggio spiegabile. L'unità di confronto resta la stessa degli step 04 e 05: stessi dataset puliti, stesso contratto di metriche, stesso modello Utopia/Ollama e stessa indicizzazione Qdrant.

L'esperimento aggiunge quattro componenti attivabili singolarmente: filtri metadata, retrieval hybrid, espansione tramite grafo esplicito e rerank LLM. Ogni componente scrive tracce di riga, così una risposta può essere analizzata partendo dai chunk recuperati e arrivando al contesto effettivamente passato al modello.

## Modalità adottata

La pipeline parte dal manifest dello step 05 e fallisce se gli hash di evaluation e index non sono comparabili. Questo evita di confrontare advanced RAG contro una baseline semplice generata su dati diversi.

Per ogni domanda il processo è:

1. applicare i filtri statici, se abilitati;
2. interrogare Qdrant in dense-only oppure in hybrid dense+sparse con RRF;
3. espandere i risultati usando solo edge reali `src_law_id -> dst_law_id` da `edges.jsonl`;
4. deduplicare i chunk, ordinare con rerank LLM se abilitato e costruire un contesto limitato;
5. generare risposta, citazioni e giudizio usando output strutturato.

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

Le ablation run si ottengono cambiando solo i flag e assegnando un `run_name` diverso. Per esempio, `all_on`, `no_hybrid`, `no_graph` e `no_rerank` producono directory affiancate e confrontabili.

## Limiti dichiarati

L'hybrid retrieval richiede che collection e manifest dichiarino sparse vectors; se manca il supporto sparse la pipeline fallisce prima di processare le righe. La graph expansion resta a hop 1 per preservare tracciabilità e semplicità del PoC. Il rerank LLM è vincolato a score interi `0`, `1`, `2`, ma il contenuto del giudizio resta una decisione del modello: per questo i punteggi sono esportati e vanno ispezionati nelle analisi.
