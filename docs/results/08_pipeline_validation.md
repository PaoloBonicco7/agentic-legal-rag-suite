# Verifica end-to-end — 11 ottobre 2026

## Protocollo congelato

Obiettivo: verificare la pipeline dopo le correzioni del primo pilot e misurare
segnali iniziali delle singole strategie Advanced. Nessuna promessa di guadagno
positivo. I risultati del [pilot precedente](07_luna_pilot.md) restano invariati.

Campione prima delle chiamate: `Random(20261012)`, popolazioni di QID ordinate,
livelli ordinati L1–L4, `sample(..., 1)` per livello sui 100 intenti originali.
QID: L1 `eval-0053`, L2 `eval-0002`, L3 `eval-0007`, L4 `eval-0004`.
Domande e gold legacy invariati; i quattro intenti non sono un test indipendente
mai esplorato. Non sono scelti per i risultati del precedente pilot.

Sette profili separati sullo stesso campione: No-RAG, Dense (RAG naive), Sparse,
Hybrid, Multi-query, Graph e Rerank. Le tre ablation Advanced partono da Hybrid:
nessuna combinazione di leve in questo test. Stesso generatore/giudice Luna,
reasoning low, massimo 4.096 output token, nessun retry o secondo modello.
Stesso corpus e indice, policy di cessazione e budget finale: 100 candidati,
4.096 token BGE, 12 blocchi; reranking sui primi 20, grafo un hop/due aggiunte,
multi-query originale e due varianti. Registrare il costo aggiuntivo delle leve.

Nuovi contratti: `answer-v3`, citazioni whitespace normalizzate solo verso un
unico span originale; `legal-judge-v2`, completezza e condizioni essenziali,
con stato della risposta visibile. Tutti i profili ricevono la stessa
indicazione di ambito Valle d'Aosta, senza
evidenze normative nel No-RAG; `rewrite-v2` vieta esiti e condizioni presunte.
Il confronto con il precedente pilot sarà descrittivo: prompt, campione e
corpus sono cambiati. Nessuna rivalutazione
retroattiva dei suoi score, né approvazione automatica del benchmark.

Limiti: massimo **72 nuove chiamate**, ledger comune cumulativo massimo
**0,50 EUR**, di cui 0,10 EUR di riserva; comprende anche il pilot precedente.
Solo `gpt-6-luna`; embedding locali riusati dalla cache verificata.
Massimo previsto per i sette profili: 28 generazioni, 28 giudizi, quattro
rewriting e quattro reranking (64 chiamate). Smoke separati: preflight,
CLI, UI, una proposta enrichment e tre generazioni MCQ (massimo altre sette).
Uno stop prima del dispatch conserva score mancanti e checkpoint.

## Verifiche richieste

- Suite offline completa dopo le correzioni di citazioni, judge, riferimenti
  composti, errori terminali, diagnostica provider e riproducibilità.
- Nuovo corpus dal medesimo snapshot congelato; tutte le quality gate, conteggi,
  differenze dal corpus precedente e ripetibilità dei JSONL.
- Nuovo indice immutabile con cache modello/testo e payload riconciliati;
  nessuna modifica all'indice del precedente pilot.
- Run comparabili dei sette profili; ranking, contesto, citazioni, errori,
  tempi, usage, costi e IC appaiati sul campione ristretto.
- Smoke reali delle interfacce e MCQ separati dalle metriche principali.
  Enrichment soltanto proposte non adottate; oracle e revisioni richiedono
  approvazione umana e sono verificati dai test di contratto, non inventati.
- Report statico offline con audit dei casi; notebook con switch disabilitati.
- Bundle e ripristino del corpus e dei run congelati, con checksum e provenienza
  del codice riconciliati; nessuna chiave, peso del modello o indice incluso.

Le misure saranno aggiunte dopo l'esecuzione. Quattro casi danno 25 punti
percentuali ciascuno: risultati esplorativi e nessuna promozione automatica.
