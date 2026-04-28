# Notebook 01 - Build Questions No Hint

## Obiettivo
`notebooks/evaluation/01_build_questions_no_hint.ipynb` trasforma il benchmark MCQ (`questions.csv`) in un dataset no-hint (`questions_no_hint.csv`) senza opzioni A-F, mantenendo risposta corretta testuale, livello e riferimento normativo.

## Input e prerequisiti
Input:
- `data/evaluation/questions.csv`

Output:
- `data/evaluation/questions_no_hint.csv`

Prerequisiti:
- notebook eseguito nel repository (bootstrap root detection)
- `pandas` disponibile

Non richiede API key o servizi esterni.

## Flusso step-by-step
1. Setup path e bootstrap notebook (`ROOT`, `INPUT_CSV`, `OUTPUT_CSV`).
2. Caricamento CSV originale con colonne `#`, `Domanda`, `Livello`, `Risposta corretta`, `Riferimento legge per la risposta`.
3. Parsing opzioni con regex multilinea `A) ... F)` dal campo `Domanda`.
4. Estrazione stem (testo domanda senza opzioni).
5. Mapping label corretta (`A-F`) in testo risposta corrispondente.
6. Normalizzazione whitespace e riferimenti legge (linee unite con separatore ` | ` quando multilinea).
7. Filtro righe valide: rimuove record senza `Risposta corretta`.
8. Scrittura CSV finale con colonne:
- `Domanda`
- `Livello`
- `Risposta corretta`
- `Riferimento legge per la risposta`

## Output e validazioni hard
Validazioni applicate nel notebook:
- shape output attesa: **100 righe**
- distribuzione livelli attesa: `L1=25, L2=25, L3=25, L4=25`
- assenza di NaN in output
- ordine colonne fissato

Queste condizioni rendono il file pronto per il notebook 02 e per i benchmark nel notebook 05.

## Contratti con notebook precedente/successivo
- Precede il notebook 02: produce `questions_no_hint.csv` allineato alla versione MCQ.
- Precede il notebook 05: il mini benchmark RAG usa lo stesso dataset no-hint.

Allineamento record verificato successivamente da `align_record(...)` in `src/legal_indexing/rag_runtime/benchmarking.py`.

## Note operative essenziali
- La regex richiede opzioni complete e ordinate da `A` a `F`; record malformati falliscono in modo esplicito.
- Il notebook assume benchmark bilanciato su 100 domande; se il dataset cambia, vanno aggiornate le assert.
- Non produce artifact run-specific aggiuntivi: l'output persistente e' solo `questions_no_hint.csv`.

## Riferimenti codice
- Notebook: `notebooks/evaluation/01_build_questions_no_hint.ipynb`
- Utility benchmark collegate: `src/legal_indexing/rag_runtime/benchmarking.py`
