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
Risultati, token, costi e report del pilot saranno aggiunti dopo l'esecuzione;
questo protocollo non contiene risultati simulati.
