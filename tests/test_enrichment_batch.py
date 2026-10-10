from types import SimpleNamespace

from legal_rag.artifacts import file_hash, write_json, write_jsonl
from legal_rag.contracts import Block, Document, Event, SourceSpan
from legal_rag.corpus import enrichment


def make_corpus(root, with_events):
    root.mkdir()
    block = Block(block_id='b', document_id='law', position=0, text='An unresolved source note.', tag='p')
    doc = Document(document_id='law', source_id='1', snapshot_id='frozen', source_url='https://example.test',
                   source_path='source.html', identity='Law', title='Title', availability='body')
    events = [Event(event_id=str(i), document_id='law', kind='repeal', evidence=block.text,
                   span=SourceSpan(block_id='b', start=0, end=len(block.text)), scope='unknown') for i in range(2)] if with_events else []
    for name, rows in [('blocks', [block]), ('documents', [doc]), ('events', events), ('relations', [])]:
        write_jsonl(root / (name + '.jsonl'), rows)
    hashes = {name: file_hash(path) for path in root.glob('*.jsonl') for name in [path.name]}
    write_json(root / 'manifest.json', {'snapshot_id': 'frozen', 'ready_for_indexing': True, 'files': hashes})
    return hashes


def test_empty_unresolved_batch_needs_no_credentials(tmp_path):
    root = tmp_path / 'corpus'
    make_corpus(root, False)
    def forbidden():
        raise AssertionError('No API preflight for an empty proposal batch')
    result = enrichment.propose_enrichment(root, tmp_path / 'proposals', SimpleNamespace(preflight=forbidden, identity={}))
    assert result['complete'] and result['planned'] == 0 and not result['adopted']


def test_multiple_unresolved_mentions_preserved_and_corpus_unchanged(tmp_path, monkeypatch):
    root = tmp_path / 'corpus'
    before = make_corpus(root, True)
    class Proposal:
        def to_json_record(self):
            return {'proposals': [], 'rejected': []}
    calls = []
    monkeypatch.setattr(enrichment, 'propose_document', lambda client, doc, blocks: calls.append(blocks) or Proposal())
    client = SimpleNamespace(preflight=lambda: {}, identity={'provider': 'fake'})
    result = enrichment.propose_enrichment(root, tmp_path / 'proposals', client)
    import json
    row = json.loads((tmp_path / 'proposals/proposals.jsonl').read_text())
    assert len(calls) == 1 and len(row['originals']) == 2
    assert result['complete'] and not row['adopted']
    assert {name: file_hash(root / name) for name in before} == before
