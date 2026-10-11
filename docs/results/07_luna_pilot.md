# Pilot Luna — 11 ottobre 2026

## Protocollo congelato prima delle chiamate

Confronto limitato a domande aperte legacy, senza alternative. Generatore e
giudice: `gpt-6-luna`, reasoning `low`, massimo 4.096 token di uscita, nessun
retry automatico. Nessun secondo modello o giudice indipendente.

Tre condizioni: `no_rag`, `dense` e `hybrid`. Hybrid rappresenta la prima
configurazione Advanced del pilot: dense+sparse con RRF, senza grafo, rewriting
o reranking. Ogni confronto usa lo stesso budget finale di 4.096 token BGE e
12 blocchi, lo stesso corpus `current-final` e il filtro conservativo corrente.
L'indice BGE-M3 locale `current-bge` viene riusato, senza reindicizzazione.

Campionamento: `Random(20261011)`, livelli L1–L4 ordinati, popolazioni ordinate
per QID, `sample(..., 2)` per livello. La scelta precede i risultati e non usa
score o revisioni gold. QID selezionati:

| Livello | QID |
| --- | --- |
| L1 | `eval-0045`, `eval-0061` |
| L2 | `eval-0042`, `eval-0062` |
| L3 | `eval-0007`, `eval-0035` |
| L4 | `eval-0008`, `eval-0048` |

Previsti al massimo 24 generazioni, 24 giudizi e un controllo strutturato API.
Tetto prudenziale del pilot: 1 EUR sul ledger comune, con 0,10 EUR di riserva.
I prezzi sono verificati l'11 ottobre sulle [fonti ufficiali OpenAI](https://developers.openai.com/api/docs/pricing):
ingresso Luna 0,10 USD/M, scrittura cache 0,125 USD/M, uscita 0,50 USD/M.
La contabilità usa l'ingresso più caro e una conversione prudente 1 EUR/USD;
il riferimento [ECB](https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/eurofxref-graph-usd.en.html)
del 9 ottobre è 1,1206 USD/EUR. Gli importi registrati sono stime conservative,
non ricevute di fatturazione. Ogni errore nel controllo iniziale interrompe il pilot.

## Limiti

Otto intenti coprono cinque leggi; ciascun caso vale 12,5 punti percentuali.
Il giudice coincide con il generatore. Gold e domande rimangono invariati;
l'assenza di flag non certifica l'applicabilità giuridica allo snapshot.
Un pilot completo consente un confronto esplorativo, mai la promozione
automatica di una strategia. L'espansione del grafo resta esclusa finché i
riferimenti composti segnalati non sono revisionati.

## Verifica software

La suite locale dell'11 ottobre ha superato tutti i 251 test senza API.
Include le aggiunte al monitoraggio precedentemente lasciate ineseguite.

## Risultati misurati

Tutti e tre i run sono completi per gli otto QID selezionati. Il generatore e
il giudice hanno restituito sempre `gpt-6-luna`. Nessun errore HTTP, timeout,
retry, giudizio mancante o cambio di modello. Tre risposte RAG sono state
respinte dal validatore delle citazioni: contano come insuccessi end-to-end,
non come giudizi mancanti.

| Profilo | Score 0 / 1 / 2 | Fully correct | Media 0–2 | Errori generativi | Chiamate | Stima EUR |
| --- | --- | --- | --- | --- | --- | --- |
| No-RAG | 2 / 1 / 5 | 5/8, 62,5% | 1,375 | 0 | 16 | 0,001994125 |
| Dense | 2 / 1 / 5 | 5/8, 62,5% | 1,375 | 1 | 15 | 0,010205250 |
| Hybrid | 3 / 0 / 5 | 5/8, 62,5% | 1,250 | 2 | 14 | 0,009482125 |

| QID | No-RAG | Dense | Hybrid |
| --- | --- | --- | --- |
| eval-0007 | 0 | 0 | 0 |
| eval-0008 | 2 | 0 | 0 |
| eval-0035 | 2 | 1 | 0 |
| eval-0042 | 2 | 2 | 2 |
| eval-0045 | 1 | 2 | 2 |
| eval-0048 | 2 | 2 | 2 |
| eval-0061 | 0 | 2 | 2 |
| eval-0062 | 2 | 2 | 2 |

Rispetto al No-RAG entrambi i RAG migliorano due intenti e ne peggiorano due.
Il delta della fully-correct rate è zero, IC95% bootstrap appaiato
[-50; +50] punti percentuali (10.000 resample, seed 20261010). La sensibilità
ai cinque gruppi di legge dà [-57,1; +44,4]. Non emerge un miglioramento
aggregato. Dense e Hybrid hanno gli stessi indicatori binari: il loro IC
degenera a zero su questo campione e non dimostra equivalenza generale.
Nessuna configurazione supera i criteri di adozione.

Per entrambi i RAG l'articolo atteso compare nel pool di 100 candidati in 8/8
casi e nel contesto in 7/8. I 19 estratti accettati sono formalmente validi;
questa quota esclude le citazioni grezze delle tre generazioni respinte.
Cinque risposte per RAG sono corrette e supportate secondo Luna. Grounding
No-RAG è non applicabile. Le proxy degli articoli non misurano la sufficienza
dei commi recuperati, né certificano il diritto corrente.

## Diagnosi dei casi

- **eval-0007:** entrambi i RAG generano 14.000 euro nel payload grezzo, ma
  sostituiscono un newline con uno spazio nella citazione. È un passaggio
  contiguo del chunk mostrato, non una combinazione di fonti distanti.
- **eval-0008:** il chunk `chunk:836d004e66de8dea15e02a82`, LR16/2019 art15
  commi1–2, contiene revoca e interessi ma manca dai 100 candidati di entrambi
  i RAG e dai due branch Hybrid. Non è uno scarto per budget. Il riferimento
  legacy art13 copre il vincolo, senza coprire questa conseguenza. Il giudice
  premia con 2 una risposta No-RAG generica/condizionale: possibile sovrastima.
  Applicabilità al caso del 2025 e revisione del riferimento richiedono controllo umano.
- **eval-0035:** Hybrid cambia `La` in `la` in una citazione dell'E10 dichiarato.
  Il gold sintetico potrebbe inoltre confondere autorizzazione di attività private
  e accreditamento per prestazioni del servizio sanitario. Nessuna correzione automatica.
- **eval-0045 e eval-0061:** miglioramenti osservati: definizione completa di
  piante officinali e Consiglio regionale al posto di Giunta. La definizione
  proviene dall'art2, mentre il riferimento CSV indica art1: proposta da revisionare.
- **eval-0062:** il No-RAG aggiunge un nulla osta regionale che la fonte mostrata
  dai RAG esclude. Il giudice assegna comunque 2 al Comune: lo stesso voto
  nasconde una contraddizione procedurale oltre il gold breve.

Prima di ampliare i batch: rendere robusta la verifica delle sole differenze
whitespace con un unico span contiguo nell'E-ID dichiarato; preservare il testo
originale della fonte. Il cambio di maiuscola richiede una decisione distinta.
Congelare una rubric che consideri condizioni, contraddizioni aggiunte e stato
della risposta; sperimentare il recupero dei commi dipendenti dalla domanda.
Questi interventi non sono applicati retroattivamente e richiederanno nuovi run.

## Costi e tracciabilità

Totale: **46 chiamate**, 24 generazioni, 21 giudizi e un preflight; 137.586 token
input e 9.050 output. Tutte le 46 voci del ledger sono `settled`.
I run contabilizzano 0,0216815 EUR; il preflight 0,00004175 EUR.
Totale prudenziale **0,02172325 EUR**, senza embedding a pagamento.

Tempi osservati per profilo: 51,25 s No-RAG, 67,31 s Dense, 80,73 s Hybrid.
Sono misure sequenziali con setup e cache differenti, escluso il setup comune
iniziale del motore: non dimostrano un confronto equo di latenza fredda.

Il primo job si è fermato dopo Dense per ispezionare il fallimento di citazione;
la continuazione ha completato soltanto Hybrid, senza rigenerare risposte o
giudizi. `completed.json` conserva lo stato intermedio, `continuation.json`
la motivazione e i checksum, `final_summary.json` il riepilogo conclusivo.

Identità congelata: commit `d566676810156e74426032ea3a7e883648222363`, clean;
source SHA-256 `dae2e43ba19d260dfb864d7418c39bb9fa513f6e94b92314dbcb42aa5a8056bf`.
La successiva modifica del reporter rende visibili gli output grezzi respinti,
senza cambiare generazione, contesti, gold o score di questi run.
I 17 test del reporter passano, inclusi quattro nuovi casi per payload grezzi,
escaping, redazione e invariabilità dei risultati.

Artefatti locali, ignorati da Git:

- `data/runs/pilot_20261011/{no_rag,dense,hybrid}`: manifest, risposte e giudizi.
- `data/diagnostics/pilot_20261011`: configurazione effettiva, log intermedi,
  accounting, riepiloghi e confronti appaiati.
- `data/annotations/pilot_20261011/comments.jsonl`: audit automatico separato.
- `data/reports/pilot_20261011/index.html`: confronto offline con dettagli per QID,
  ranking, contesto, fonti, payload grezzi e commenti. Il rendering non usa modelli.

Questo pilot verifica l'accesso reale e il motore su pochi casi. Non sostituisce
la baseline sui 100 intenti, le ablation Advanced complete o un controllo umano
e indipendente del giudice. I risultati negativi restano parte della ricerca.
