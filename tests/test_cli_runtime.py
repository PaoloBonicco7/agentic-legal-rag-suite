from pathlib import Path
from types import SimpleNamespace

import pytest

from legal_rag import cli
from legal_rag.artifacts import file_hash, write_json
from legal_rag.contracts import AnswerResult, EngineConfig, ProviderConfig
from legal_rag.runtime import evaluation_identity


def test_cli_returns_failure_for_saved_generation_error(tmp_path, monkeypatch):
    result = AnswerResult(question='question', profile='no_rag', status='error', errors=['http_403'])
    monkeypatch.setattr(cli, '_engine', lambda args: (SimpleNamespace(answer=lambda question: result), None))
    monkeypatch.setattr(cli, 'close_engine', lambda engine: None)
    output = tmp_path / 'answer.json'
    assert cli.main(['ask', '--profile', 'no_rag', '--output', str(output), 'question']) == 1
    assert output.is_file()


def test_ui_forwards_effective_provider_limits(monkeypatch):
    calls = []
    monkeypatch.setattr(cli.subprocess, 'run', lambda args, check: calls.append(args))
    assert cli.main(['ui', '--timeout', '23', '--max-output-tokens', '512', '--context-window-tokens', '16384',
                     '--profile', 'no_rag']) == 0
    command = calls[0]
    assert command[command.index('--timeout') + 1] == '23'
    assert command[command.index('--max-output-tokens') + 1] == '512'
    assert command[command.index('--context-window-tokens') + 1] == '16384'


@pytest.mark.parametrize('capacity', [None, 16384])
def test_controlled_utopia_requires_verified_sufficient_context_capacity(capacity):
    from legal_rag.providers.chat import ModelInspection

    metadata = ModelInspection(requested_model='SLURM.gpt-oss:120b', returned_model='SLURM.gpt-oss:120b',
                               digest='sha256:verified', capabilities=['completion'], context_window_tokens=capacity)
    client = SimpleNamespace(config=ProviderConfig(), inspect_model=lambda: metadata)
    with pytest.raises(ValueError, match='context window'):
        cli._model_metadata(client)


def test_controlled_utopia_records_requested_and_verified_capacity_separately():
    from legal_rag.providers.chat import ModelInspection

    metadata = ModelInspection(requested_model='SLURM.gpt-oss:120b', returned_model='SLURM.gpt-oss:120b',
                               digest='sha256:verified', capabilities=['completion'], context_window_tokens=131072)
    client = SimpleNamespace(config=ProviderConfig(), inspect_model=lambda: metadata)
    assert cli._model_metadata(client)['context_window_tokens'] == 131072
    assert client.config.context_window_tokens == 32768


def test_oracle_evaluation_accepts_explicit_evidence_contract():
    args = cli._parser().parse_args(['evaluate', '--profile', 'oracle', '--oracle-evidence', 'approved.jsonl', '--output', 'run'])
    assert args.oracle_evidence == Path('approved.jsonl')


def test_controlled_identity_rejects_another_or_changed_corpus(tmp_path):
    (tmp_path / 'chunks.jsonl').write_text('{}\n')
    corpus = {'snapshot_id': 'snapshot', 'ready_for_indexing': True,
              'files': {'chunks.jsonl': file_hash(tmp_path / 'chunks.jsonl')}}
    write_json(tmp_path / 'manifest.json', corpus)
    index = {'snapshot_id': 'snapshot', 'ready_for_retrieval': True, 'files': corpus['files'],
             'corpus_manifest_sha256': file_hash(tmp_path / 'manifest.json'),
             'chunks_sha256': file_hash(tmp_path / 'chunks.jsonl'), 'embedding': {'model': 'fixed'}}
    client = SimpleNamespace(identity={'model': 'fixed'})
    engine = SimpleNamespace(client=client, config=EngineConfig())
    assert evaluation_identity(engine, client, index, tmp_path)['corpus']['snapshot_id'] == 'snapshot'
    index['corpus_manifest_sha256'] = 'another-corpus'
    with pytest.raises(ValueError, match='identities differ'):
        evaluation_identity(engine, client, index, tmp_path)
    (tmp_path / 'chunks.jsonl').write_text('changed\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        evaluation_identity(engine, client, index, tmp_path)
