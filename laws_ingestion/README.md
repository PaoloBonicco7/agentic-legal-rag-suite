# laws_ingestion

Questa cartella contiene solo documentazione.

Il codice Python e' in:
- `src/laws_ingestion/core`
- `src/laws_ingestion/data_preparation/laws_graph`

Import consigliati:
```python
from laws_ingestion.core.ingest import ingest_law
from laws_ingestion.core.registry import build_corpus_registry
from laws_ingestion.data_preparation.laws_graph.pipeline import PipelineConfig, run_pipeline
```
