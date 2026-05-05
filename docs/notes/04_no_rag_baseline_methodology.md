# 04 - Metodologia del Baseline No-RAG

## Obiettivo

Questo step misura quanto il modello risponde correttamente senza usare retrieval, corpus legale, Qdrant o contesto oracle. Il risultato è il punto di confronto minimo per gli step RAG successivi: ogni miglioramento introdotto dal retrieval deve essere leggibile rispetto a questa baseline model-only.

## Modalità adottata

La pipeline usa gli stessi dataset puliti prodotti dallo step 02:

- `questions_mcq.jsonl` per le domande a scelta multipla;
- `questions_no_hint.jsonl` per le risposte aperte senza suggerimenti;
- `evaluation_manifest.json` come manifest upstream.

Prima dell'esecuzione viene verificato che MCQ e no-hint siano allineati per `qid`, `linked_mcq_qid` e testo della domanda. Questo evita di confrontare risultati su record non corrispondenti.

Il modello viene interrogato in tre passaggi:

1. risposta MCQ senza contesto;
2. risposta aperta senza contesto;
3. giudizio semantico della risposta aperta, solo se la generazione è riuscita.

Il client rimane quello Utopia/Ollama già usato dal progetto, con output strutturato tramite JSON schema e `temperature=0`. Non sono stati introdotti LangChain o Pydantic AI perché in questo step non ci sono retrieval chain, tool, agent loop o dependency injection agentica: aggiungerli avrebbe reso meno diretto il confronto scientifico.

## Contratto degli output

Gli artifact vengono scritti in modo atomico in `data/baseline_runs/no_rag/`:

- `mcq_results.jsonl`: una riga per domanda MCQ, con label predetta, label corretta, score ed eventuale errore;
- `no_hint_results.jsonl`: una riga per domanda aperta, con risposta generata, score del giudice, spiegazione ed eventuale errore;
- `no_rag_summary.json`: metriche globali e per livello;
- `quality_report.md`: riepilogo leggibile degli errori e della copertura;
- `no_rag_manifest.json`: configurazione effettiva, modelli, hash dei dataset, hash degli output e summary.

Il manifest non salva mai il valore della chiave API. Registra solo `api_key_present`, sia nella configurazione sicura sia nei metadati della connessione.

## Metriche

Le metriche mantengono lo stesso contratto dello step oracle-context:

- `processed`: record selezionati;
- `judged`: record con score valido;
- `score_sum`: somma degli score;
- `max_score_sum`: massimo ottenibile sui record giudicati;
- `accuracy`: score normalizzato sui soli record giudicati;
- `mean_score`: score medio;
- `coverage`: quota di record giudicati;
- `strict_accuracy`: score normalizzato su tutti i record processati;
- `errors`: record con errore;
- `by_level`: stesse metriche raggruppate per livello.

Per MCQ lo score massimo per riga è `1`. Per no-hint lo score massimo per riga è `2`, secondo la rubrica semantica `0-2`.

## Uso sperimentale

La modalità consigliata è:

1. eseguire uno smoke run reale per verificare credenziali e formato degli output;
2. eseguire il benchmark completo solo quando il dataset pulito è stabile;
3. confrontare `no_rag_summary.json` con gli step RAG successivi usando le stesse chiavi metriche;
4. ispezionare le righe JSONL per capire se gli errori derivano da conoscenza mancante, ambiguità della domanda o fallimenti strutturati del modello.

Il notebook associato mostra una run simulata riproducibile senza credenziali e celle opzionali per smoke/full run reale. Le metriche simulate servono solo a spiegare la forma degli artifact; le metriche scientifiche devono provenire da una run Utopia reale.
