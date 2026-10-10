"""Auditable LLM proposals that cannot change corpus status or source text."""

from pathlib import Path

from legal_rag.artifacts import file_hash, new_output, read_json, read_jsonl, utc_now, write_json, write_jsonl
from legal_rag.contracts import Block, Document, Event, Record, Relation
from legal_rag.generation.enrichment import ENRICHMENT_PROMPT_VERSION, propose_enrichment as propose_document
from legal_rag.providers import ProviderError


class EnrichmentManifest(Record):
    schema_version: str = 'enrichment-v1'
    prompt_version: str = ENRICHMENT_PROMPT_VERSION
    created_at: str
    snapshot_id: str
    corpus_manifest_sha256: str
    provider: dict
    planned: int
    completed: int
    errors: int
    adopted: bool = False
    complete: bool
    files: dict[str, str]


def propose_enrichment(corpus_dir, output_dir, client, *, limit=20):
    if not 1 <= limit <= 100:
        raise ValueError('The proposal batch limit must be between 1 and 100')
    root, output = Path(corpus_dir), Path(output_dir)
    manifest = read_json(root / 'manifest.json')
    if not manifest.get('ready_for_indexing'):
        raise ValueError('Enrichment requires a verified corpus')
    for name, expected in manifest['files'].items():
        if file_hash(root / name) != expected:
            raise ValueError(f'Corpus artifact hash mismatch: {name}')
    blocks = {block.block_id: block for block in read_jsonl(root / 'blocks.jsonl', Block)}
    documents = {document.document_id: document for document in read_jsonl(root / 'documents.jsonl', Document)}
    unresolved = [event for event in read_jsonl(root / 'events.jsonl', Event) if event.resolution == 'unresolved']
    unresolved += [edge for edge in read_jsonl(root / 'relations.jsonl', Relation) if edge.resolution == 'unresolved' and edge.span]
    unique = {}
    for item in unresolved:
        unique.setdefault(item.span.block_id, []).append(item)
    selected = list(unique.items())[:limit]
    if selected:
        client.preflight()
    new_output(output)
    rows, failures = [], 0
    write_jsonl(output / 'proposals.jsonl', [])
    for block_id, items in selected:
        block = blocks[block_id]
        row = {'block_id': block.block_id, 'originals': [item.to_json_record() for item in items], 'status': 'proposed', 'adopted': False}
        try:
            proposal = propose_document(client, documents[block.document_id], [block])
            row['result'] = proposal.to_json_record()
        except (ProviderError, ValueError) as error:
            failures += 1
            row.update(status='error', error=getattr(error, 'code', type(error).__name__),
                       usage=[usage.to_json_record() for usage in getattr(error, 'usage', [])])
        rows.append(row)
        write_jsonl(output / 'proposals.jsonl', rows)
        if row['status'] == 'error' and row['error'] in {'budget_exceeded', 'budget_review_required', 'cost_bound_exceeded',
                'prices_stale', 'prices_missing', 'prices_invalid', 'http_403', 'http_401', 'http_404', 'model_mismatch', 'credentials_missing'}:
            break
    files = {'proposals.jsonl': file_hash(output / 'proposals.jsonl')}
    result = EnrichmentManifest(created_at=utc_now(), snapshot_id=manifest['snapshot_id'],
        corpus_manifest_sha256=file_hash(root / 'manifest.json'), provider=client.identity,
        planned=len(selected), completed=sum(row['status'] == 'proposed' for row in rows), errors=failures,
        complete=len(rows) == len(selected) and failures == 0, files=files)
    write_json(output / 'manifest.json', result)
    return result.to_json_record()
