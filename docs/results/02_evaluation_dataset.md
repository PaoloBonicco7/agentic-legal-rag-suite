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

## Limiti noti

- I riferimenti legali restano stringhe human-readable: questo step non li parsifica in
  identificatori legge/articolo.
- L'allineamento MCQ ↔ no-hint è volutamente rigido (stesso qid, livello, intento e risposta dopo
  normalizzazione degli spazi).

## Riproduzione

```bash
PYTHONPATH=src python -m legal_rag.evaluation_dataset --mcq-source data/evaluation/questions.csv --no-hint-source data/evaluation/questions_no_hint.csv --output data/evaluation_clean
```

Notebook: `notebooks/02_evaluation_dataset.ipynb`.
