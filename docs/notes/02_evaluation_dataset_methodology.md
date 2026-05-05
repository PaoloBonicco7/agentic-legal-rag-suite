# 02 - Metodologia Evaluation Dataset

## Obiettivo

Questo step prepara il benchmark di domande usato nelle valutazioni successive. Il punto non e creare nuove domande, ma trasformare i due CSV sorgenti in un contratto pulito, stabile e verificabile.

La metodologia separa il dato originale dagli artifact generati: i file in `data/evaluation/` restano la fonte autorevole, mentre `data/evaluation_clean/` contiene solo output rigenerabili.

## Scelta metodologica

Il dataset esiste in due forme:

- MCQ, con domanda, sei opzioni e label corretta;
- no-hint, con la stessa domanda senza opzioni e con risposta testuale.

La pipeline tratta le due forme come due viste dello stesso benchmark. Per questo ogni record riceve lo stesso `qid` stabile, basato sulla posizione valida: `eval-0001`, `eval-0002`, ecc.

L'allineamento e intenzionalmente rigido. Una coppia MCQ/no-hint viene accettata solo se coincidono livello, testo della domanda normalizzato e risposta corretta normalizzata. In questo modo le valutazioni no-RAG, simple RAG e advanced RAG partono dallo stesso intento di domanda.

## Processo

Il processo seguito e deterministico:

1. leggere i CSV sorgenti con `csv.DictReader`;
2. normalizzare header e whitespace con helper condivisi;
3. scartare solo le righe finali realmente vuote del file MCQ;
4. estrarre le opzioni MCQ dalle righe `A)`-`F)`;
5. validare che la label corretta esista tra le opzioni;
6. derivare `correct_answer` MCQ dal testo dell'opzione corretta;
7. dividere i riferimenti normativi multilinea in una lista leggibile;
8. validare l'allineamento posizione per posizione;
9. esportare JSONL, manifest, profilo e quality report.

La pipeline fallisce appena trova dati non coerenti. Questo e preferibile a esportare un benchmark ambiguo, perche gli errori nei dati di valutazione altererebbero tutte le metriche successive.

## Contratti e framework

I record esportati, la configurazione, il profilo e il manifest sono validati con Pydantic v2 e `extra="forbid"` dove rilevante. Questo mantiene esplicito il contratto consumato dagli step successivi.

La pipeline non introduce nuove dipendenze per il parsing: `csv`, `json`, `hashlib`, `pathlib` e Pydantic sono sufficienti. Il notebook usa `matplotlib` solo per rendere visibili conteggi e distribuzioni.

## Artifact prodotti

Gli output generati sono:

- `questions_mcq.jsonl`;
- `questions_no_hint.jsonl`;
- `evaluation_manifest.json`;
- `evaluation_profile.json`;
- `quality_report.md`.

Il manifest registra configurazione, hash dei sorgenti, conteggi, distribuzione livelli, quality gates e hash degli output generati. Il manifest non contiene l'hash di se stesso, perche un file non puo includere un hash stabile del proprio contenuto.

## Esito osservato

Sul dataset versionato il processo produce:

- 100 record MCQ validi;
- 100 record no-hint validi;
- 63 righe MCQ vuote scartate;
- distribuzione bilanciata: `L1=25`, `L2=25`, `L3=25`, `L4=25`;
- `ready_for_evaluation: true`.

## Limiti

I riferimenti normativi vengono mantenuti come stringhe leggibili, una per riga non vuota. Non vengono ancora trasformati in identificatori strutturati di legge o articolo.

La validazione e stretta sul testo normalizzato. Se in futuro i CSV contengono varianti equivalenti ma non identiche, sara necessario decidere esplicitamente se introdurre una mappatura controllata, invece di allentare silenziosamente i controlli.
