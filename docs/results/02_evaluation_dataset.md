# 02 — Evaluation dataset

Normalizzazione deterministica CSV → dataset di valutazione in `src/legal_rag/evaluation_dataset/`.
Legge `data/evaluation/`, scarta le righe MCQ vuote, estrae le opzioni multiple, verifica
l'allineamento con il set no-hint e scrive `data/evaluation_clean/`. Metodologia in
[note/02](../notes/02_evaluation_dataset_methodology.md).

## Configurazione

- Sorgenti: `data/evaluation/questions.csv` (MCQ), `questions_no_hint.csv`
- Output: `data/evaluation_clean` · schema `evaluation-dataset-v1`

## Risultati osservati

- MCQ: 163 righe sorgente → 100 valide (63 righe vuote scartate)
- No-hint: 100 → 100 (0 scartate)
- Distribuzione livelli: L1 = L2 = L3 = L4 = 25
- Tutti i quality gate superati; `evaluation_manifest.json` espone `ready_for_evaluation: true`.

## Revisione corpus-only dei riferimenti di vigenza

`data/evaluation/vigency_reference_review.csv` è un companion separato, schema
`vigency-reference-review-v1`, SHA-256
`0b7ea8b967f3cc0e1a22efb6c30704dcda8b578f50d3eb2baa4f61c04b324133`.
Contiene soltanto i dieci QID esclusi da almeno uno dei filtri nel baseline storico.

- Validità del riferimento atteso: 4 `partial`, 1 `unknown`, 5 `past`.
- Relazione con la risposta: 8 `supported_by_expected`, 2 `supported_elsewhere`.
- Flag temporale: 4 `historical`, 1 `time_limited`, 5 `none`.

I due mismatch tra qrel e passaggio di supporto sono:

- `eval-0013`: la risposta è supportata dall'art. 3, comma 3 della LR 44/1991, non dal qrel
  art. 2;
- `eval-0076`: il qrel art. 5 è realmente `past`, mentre la risposta è supportata dall'art. 3,
  comma 5.

Il file non modifica domande o qrel e non entra nello scoring. Serve a distinguere errore di
annotazione della vigenza, riferimento storico e qrel non allineata al passaggio di risposta.

## Limiti noti

- I riferimenti legali restano stringhe human-readable: questo step non li parsifica in
  identificatori legge/articolo.
- L'allineamento MCQ ↔ no-hint è volutamente rigido (stesso qid, livello, intento e risposta dopo
  normalizzazione degli spazi).
- I qrel sono annotazioni di rilevanza del benchmark, non asserzioni di validità giuridica.

## Riproduzione

```bash
PYTHONPATH=src python -m legal_rag.evaluation_dataset --mcq-source data/evaluation/questions.csv --no-hint-source data/evaluation/questions_no_hint.csv --output data/evaluation_clean
```

Notebook: `notebooks/02_evaluation_dataset.ipynb`.
