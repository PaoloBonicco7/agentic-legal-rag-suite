# Notebook 02 - Evaluation No-RAG

## Obiettivo
`notebooks/evaluation/02_evalutation_no_rag.ipynb` misura la baseline **senza retrieval RAG** su due task:
- MCQ con output strutturato (`McqAnswer`)
- no-hint con risposta aperta + judge LLM (`NoHintAnswer` + `JudgeResult`)

## Input e prerequisiti
Input dataset:
- `data/evaluation/questions.csv`
- `data/evaluation/questions_no_hint.csv`

Prerequisiti ambiente:
- `UTOPIA_API_KEY` obbligatoria
- endpoint chat (`UTOPIA_BASE_URL`, default `.../ollama/api/chat`)
- modelli configurabili: `UTOPIA_CHAT_MODEL`, `UTOPIA_JUDGE_MODEL`

Il notebook usa utility condivise in `src/legal_indexing/rag_runtime/benchmarking.py`.

## Flusso step-by-step
1. Setup notebook e configurazione API (`API_URL`, headers, timeout).
2. Load righe valide da entrambi i CSV (`load_valid_rows`).
3. Allineamento record no-hint/MCQ per posizione (`align_record`) con fail-fast su mismatch dello stem.
4. Smoke test opzionale su una domanda (`RUN_SMOKE`).
5. Batch MCQ:
- prompt structured (`build_mcq_prompt`)
- chiamata API (`post_structured_chat`)
- validazione output (`validate_mcq_output`)
- scoring 0/1 su label.
6. Batch no-hint + judge:
- risposta aperta structured (`build_no_hint_prompt`)
- judge structured 0/1 (`build_judge_prompt`, `validate_judge_output`).
7. Aggregazioni metriche:
- `build_dataset_summary`
- `build_comparison_table`
- breakdown globale e per livello.
8. Dashboard e campionamento errori con `matplotlib`.

## Output e artifact
Output principali del notebook:
- oggetti in memoria: `mcq_results`, `no_hint_results`, summary e comparison table
- visualizzazioni inline (accuracy globale e per livello)
- stampa campioni di errore/wrong cases

Limite operativo corrente:
- il notebook **non salva artifact strutturati su file in modo nativo** (risultati principalmente in output notebook).

## Contratti con notebook precedente/successivo
- Richiede `questions_no_hint.csv` prodotto dal notebook 01.
- Condivide schema metriche e utility con notebook 05, facilitando confronto no-RAG vs RAG.

## Note operative essenziali
- In assenza di `UTOPIA_API_KEY` il notebook termina subito.
- Errori API (HTTP/network/parse) vengono contabilizzati in `errors`; solo score validi 0/1 entrano in `judged`.
- Il judge e' LLM-as-a-judge: utile come proxy sperimentale, non come ground truth normativa.

## Riferimenti codice
- Notebook: `notebooks/evaluation/02_evalutation_no_rag.ipynb`
- Utility condivise: `src/legal_indexing/rag_runtime/benchmarking.py`
- Schemi: `src/legal_indexing/rag_runtime/schemas.py`
