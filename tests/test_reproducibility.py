import io
import json
import tarfile

import pytest

from legal_rag.artifacts import digest, file_hash, write_json, write_jsonl
from legal_rag.contracts import SourceRecord
from legal_rag.reproducibility import create_bundle, restore_bundle, verify_bundle


def snapshot_fixture(path):
    path.mkdir()
    (path / 'original.html').write_bytes(b'raw bytes')
    write_jsonl(path / 'catalog.jsonl', [SourceRecord(source_id='1', url='https://example.test/1',
        path='original.html', sha256=file_hash(path / 'original.html'))])
    write_json(path / 'manifest.json', {'complete': True, 'snapshot_id': 'frozen', 'source_count': 1,
        'catalog_sha256': file_hash(path / 'catalog.jsonl')})
    return path


def recompose_bundle(archive_path, mutate):
    with tarfile.open(archive_path, 'r:gz') as archive:
        entries = {member.name: archive.extractfile(member).read() for member in archive.getmembers()}
    mutate(entries)
    manifest = json.loads(entries['bundle.json'])
    manifest['files'] = {name: digest(raw) for name, raw in entries.items() if name != 'bundle.json'}
    entries['bundle.json'] = json.dumps(manifest).encode()
    with tarfile.open(archive_path, 'w:gz') as archive:
        for name, raw in entries.items():
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))


def change_json(entries, name, **updates):
    record = json.loads(entries[name])
    record.update(updates)
    entries[name] = json.dumps(record).encode()


def corpus_and_run_fixture(source, tmp_path):
    corpus = tmp_path / 'corpus'
    corpus.mkdir()
    (corpus / 'chunks.jsonl').write_text('{}\n')
    write_json(corpus / 'manifest.json', {'snapshot_id': 'frozen', 'ready_for_indexing': True,
        'source_manifest_sha256': file_hash(source / 'manifest.json'),
        'source_catalog_sha256': file_hash(source / 'catalog.jsonl'),
        'files': {'chunks.jsonl': file_hash(corpus / 'chunks.jsonl')}})
    run = tmp_path / 'experiment'
    run.mkdir()
    (run / 'results.jsonl').write_text('{}\n')
    corpus_identity = {'snapshot_id': 'frozen', 'manifest_sha256': file_hash(corpus / 'manifest.json')}
    write_json(run / 'manifest.json', {'inputs': {'identity': {'corpus': corpus_identity}, 'index': {'corpus': corpus_identity}},
        'output_hashes': {'results.jsonl': file_hash(run / 'results.jsonl')}})
    return corpus, run


def test_verified_bundle_restores_exact_snapshot(tmp_path):
    project = tmp_path / 'project'
    (project / 'src').mkdir(parents=True)
    (project / 'src/example.py').write_text('x = 1\n')
    (project / '.env').write_text('OPENAI_API_KEY=not-for-export')
    snapshot = tmp_path / 'snapshot'
    snapshot_fixture(snapshot)
    archive = tmp_path / 'bundle.tar.gz'
    result = create_bundle(snapshot, archive, project_dir=project)
    restored = tmp_path / 'restored'
    assert restore_bundle(archive, restored)['files_verified'] == result['files']
    assert (restored / 'snapshot/original.html').read_bytes() == b'raw bytes'
    assert not (restored / 'source/.env').exists()
    assert file_hash(restored / 'source/src/example.py') == file_hash(project / 'src/example.py')
    with pytest.raises(FileExistsError):
        restore_bundle(archive, restored)


def test_export_rejects_credential_fields(tmp_path):
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    write_json(snapshot / 'manifest.json', {'complete': True, 'snapshot_id': 'frozen', 'config': {'api_key': 'secret'}})
    with pytest.raises(ValueError, match='credential'):
        create_bundle(snapshot, tmp_path / 'bundle.tar.gz', project_dir=tmp_path / 'empty')


def test_path_traversal_rejected_before_creating_destination(tmp_path):
    archive = tmp_path / 'malicious.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        info = tarfile.TarInfo('../escape')
        info.size = 1
        tar.addfile(info, io.BytesIO(b'x'))
    target = tmp_path / 'restored'
    with pytest.raises(ValueError, match='Unsafe'):
        restore_bundle(archive, target)
    assert not target.exists()


def test_tampered_checksum_rejected(tmp_path):
    archive = tmp_path / 'malicious.tar.gz'
    manifest = {'created_at': 'now', 'snapshot_id': 'frozen', 'code': {}, 'files': {'snapshot/a': 'wrong'}}
    with tarfile.open(archive, 'w:gz') as tar:
        for name, raw in [('bundle.json', json.dumps(manifest).encode()), ('snapshot/a', b'changed')]:
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    with pytest.raises(ValueError, match='checksum'):
        verify_bundle(archive)


def test_bundle_rejects_changed_source_and_corpus(tmp_path):
    source = snapshot_fixture(tmp_path / 'snapshot')
    corpus = tmp_path / 'corpus'
    corpus.mkdir()
    (corpus / 'chunks.jsonl').write_text('{}\n')
    write_json(corpus / 'manifest.json', {'snapshot_id': 'frozen', 'ready_for_indexing': True,
        'source_manifest_sha256': file_hash(source / 'manifest.json'),
        'source_catalog_sha256': file_hash(source / 'catalog.jsonl'),
        'files': {'chunks.jsonl': file_hash(corpus / 'chunks.jsonl')}})
    (corpus / 'chunks.jsonl').write_text('changed\n')
    with pytest.raises(ValueError, match='checksum'):
        create_bundle(source, tmp_path / 'bad.tar.gz', corpus_dir=corpus, project_dir=tmp_path / 'empty')
    (source / 'original.html').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='checksum'):
        create_bundle(source, tmp_path / 'bad.tar.gz', project_dir=tmp_path / 'empty')


def test_bundle_rejects_run_from_another_corpus(tmp_path):
    source = snapshot_fixture(tmp_path / 'snapshot')
    run = tmp_path / 'run'
    run.mkdir()
    write_json(run / 'manifest.json', {'inputs': {'identity': {'corpus': {'snapshot_id': 'other'}}}, 'output_hashes': {}})
    with pytest.raises(ValueError, match='Run provenance'):
        create_bundle(source, tmp_path / 'bad.tar.gz', run_dirs=[run], project_dir=tmp_path / 'empty')


def test_bundle_rejects_catalogue_source_through_symlink_directory(tmp_path):
    source = snapshot_fixture(tmp_path / 'snapshot')
    (source / 'linked').symlink_to(source, target_is_directory=True)
    write_jsonl(source / 'catalog.jsonl', [SourceRecord(source_id='1', url='https://example.test/1',
        path='linked/original.html', sha256=file_hash(source / 'original.html'))])
    write_json(source / 'manifest.json', {'complete': True, 'snapshot_id': 'frozen', 'source_count': 1,
        'catalog_sha256': file_hash(source / 'catalog.jsonl')})
    with pytest.raises(ValueError, match='regular files'):
        create_bundle(source, tmp_path / 'bad.tar.gz', project_dir=tmp_path / 'empty')


@pytest.mark.parametrize(('mutation', 'error'), [
    (lambda entries: entries.update({'snapshot/original.html': b'recomposed raw bytes'}), 'checksum'),
    (lambda entries: entries.pop('snapshot/original.html'), 'Missing packaged artifact'),
    (lambda entries: change_json(entries, 'snapshot/manifest.json', complete=False), 'complete snapshots'),
    (lambda entries: change_json(entries, 'snapshot/manifest.json', source_count=2), 'count or identifiers'),
    (lambda entries: change_json(entries, 'snapshot/manifest.json', catalog_sha256='wrong'), 'checksum'),
    (lambda entries: change_json(entries, 'snapshot/manifest.json', catalog_sha256=None), 'checksum'),
    (lambda entries: change_json(entries, 'bundle.json', snapshot_id='different'), 'snapshot identity'),
])
def test_recomposed_outer_hashes_cannot_bypass_snapshot_provenance(tmp_path, mutation, error):
    source = snapshot_fixture(tmp_path / 'snapshot')
    archive = tmp_path / 'bundle.tar.gz'
    create_bundle(source, archive, project_dir=tmp_path / 'empty')
    recompose_bundle(archive, mutation)
    with pytest.raises(ValueError, match=error):
        verify_bundle(archive)
    target = tmp_path / 'restored'
    with pytest.raises(ValueError, match=error):
        restore_bundle(archive, target)
    assert not target.exists()


@pytest.mark.parametrize(('mutation', 'error'), [
    (lambda entries: change_json(entries, 'corpus/manifest.json', source_manifest_sha256='other'), 'Corpus provenance'),
    (lambda entries: change_json(entries, 'corpus/manifest.json', source_catalog_sha256='other'), 'Corpus provenance'),
    (lambda entries: change_json(entries, 'corpus/manifest.json', snapshot_id='other'), 'another snapshot'),
    (lambda entries: entries.update({'corpus/chunks.jsonl': b'{"changed": true}\n'}), 'checksum'),
    (lambda entries: entries.update({'runs/experiment/results.jsonl': b'{"changed": true}\n'}), 'checksum'),
])
def test_recomposed_bundle_requires_corpus_and_run_internal_hashes(tmp_path, mutation, error):
    source = snapshot_fixture(tmp_path / 'snapshot')
    corpus, run = corpus_and_run_fixture(source, tmp_path)
    archive = tmp_path / 'bundle.tar.gz'
    create_bundle(source, archive, corpus_dir=corpus, run_dirs=[run], project_dir=tmp_path / 'empty')
    recompose_bundle(archive, mutation)
    with pytest.raises(ValueError, match=error):
        verify_bundle(archive)


def test_every_run_corpus_identity_must_match_packaged_corpus(tmp_path):
    source = snapshot_fixture(tmp_path / 'snapshot')
    corpus, run = corpus_and_run_fixture(source, tmp_path)
    archive = tmp_path / 'bundle.tar.gz'
    create_bundle(source, archive, corpus_dir=corpus, run_dirs=[run], project_dir=tmp_path / 'empty')

    def mismatch_index(entries):
        record = json.loads(entries['runs/experiment/manifest.json'])
        record['inputs']['index']['corpus'] = {'snapshot_id': 'different'}
        entries['runs/experiment/manifest.json'] = json.dumps(record).encode()

    recompose_bundle(archive, mismatch_index)
    with pytest.raises(ValueError, match='Run provenance'):
        verify_bundle(archive)


@pytest.mark.parametrize(('name', 'raw', 'error'), [
    ('snapshot/.env', b'UTOPIA_API_KEY=forbidden', 'Credentials'),
    ('source/config.json', b'{"nested": {"openai_api_key": "forbidden"}}', 'credential field'),
    ('snapshot/./alias', b'alias', 'Unsafe'),
    ('snapshot\\alias', b'alias', 'Unsafe'),
])
def test_recomposed_archive_rejects_credentials_and_alias_paths(tmp_path, name, raw, error):
    source = snapshot_fixture(tmp_path / 'snapshot')
    archive = tmp_path / 'bundle.tar.gz'
    create_bundle(source, archive, project_dir=tmp_path / 'empty')
    recompose_bundle(archive, lambda entries: entries.update({name: raw}))
    target = tmp_path / 'restored'
    with pytest.raises(ValueError, match=error):
        restore_bundle(archive, target)
    assert not target.exists()


def test_catalogue_paths_are_validated_inside_archive(tmp_path):
    source = snapshot_fixture(tmp_path / 'snapshot')
    archive = tmp_path / 'bundle.tar.gz'
    create_bundle(source, archive, project_dir=tmp_path / 'empty')

    def unsafe_catalogue(entries):
        record = json.loads(entries['snapshot/catalog.jsonl'])
        record['path'] = '../original.html'
        entries['snapshot/catalog.jsonl'] = json.dumps(record).encode() + b'\n'
        change_json(entries, 'snapshot/manifest.json', catalog_sha256=digest(entries['snapshot/catalog.jsonl']))

    recompose_bundle(archive, unsafe_catalogue)
    with pytest.raises(ValueError, match='Unsafe'):
        verify_bundle(archive)


def benchmark_sources_fixture(project, run, review=None):
    evaluation = project / 'data/evaluation'
    evaluation.mkdir(parents=True)
    for name in ('questions.csv', 'questions_no_hint.csv'):
        (evaluation / name).write_bytes(b'qid,question\r\n1,Original question\r\n')
    record = json.loads((run / 'manifest.json').read_bytes())
    record['source_hashes'] = {path.name: file_hash(path) for path in evaluation.iterdir()}
    if review:
        record['source_hashes']['review'] = file_hash(review)
    write_json(run / 'manifest.json', record)
    return evaluation


def test_bundle_preserves_review_bytes_by_declared_hash(tmp_path):
    source = snapshot_fixture(tmp_path / 'snapshot')
    corpus, run = corpus_and_run_fixture(source, tmp_path)
    review = tmp_path / 'human-review.jsonl'
    raw = b'{ "qid": "eval-0001", "note": "Human evidence" }\r\n\r\n'
    review.write_bytes(raw)
    project = tmp_path / 'project'
    evaluation = benchmark_sources_fixture(project, run, review)
    archive = tmp_path / 'bundle.tar.gz'
    create_bundle(source, archive, corpus_dir=corpus, run_dirs=[run], project_dir=project, review_files=[review])
    restored = tmp_path / 'restored'
    restore_bundle(archive, restored)
    assert (restored / f'source/reviews/{file_hash(review)}.jsonl').read_bytes() == raw
    for path in evaluation.iterdir():
        assert (restored / 'source/data/evaluation' / path.name).read_bytes() == path.read_bytes()


@pytest.mark.parametrize('missing', [False, True])
def test_create_bundle_rejects_mismatched_or_missing_benchmark_csv(tmp_path, missing):
    source = snapshot_fixture(tmp_path / 'snapshot')
    corpus, run = corpus_and_run_fixture(source, tmp_path)
    project = tmp_path / 'project'
    evaluation = benchmark_sources_fixture(project, run)
    path = evaluation / 'questions.csv'
    if missing:
        path.unlink()
    else:
        path.write_bytes(b'qid,question\n1,Changed question\n')
    archive = tmp_path / 'bundle.tar.gz'
    with pytest.raises(ValueError, match='regular files|checksum'):
        create_bundle(source, archive, corpus_dir=corpus, run_dirs=[run], project_dir=project)
    assert not archive.exists()


def test_create_bundle_requires_declared_raw_review_and_rejects_credentials(tmp_path):
    source = snapshot_fixture(tmp_path / 'snapshot')
    corpus, run = corpus_and_run_fixture(source, tmp_path)
    review = tmp_path / 'review.jsonl'
    review.write_bytes(b'{"qid":"eval-0001"}\n')
    project = tmp_path / 'project'
    benchmark_sources_fixture(project, run, review)
    archive = tmp_path / 'bundle.tar.gz'
    with pytest.raises(ValueError, match='Missing raw benchmark review'):
        create_bundle(source, archive, corpus_dir=corpus, run_dirs=[run], project_dir=project)
    review.write_bytes(b'{"qid":"eval-0001","api_key":"forbidden"}\n')
    with pytest.raises(ValueError, match='credential field'):
        create_bundle(source, archive, corpus_dir=corpus, run_dirs=[run], project_dir=project, review_files=[review])
    assert not archive.exists()


@pytest.mark.parametrize('artifact', ['csv_changed', 'csv_missing', 'review_changed', 'review_missing'])
def test_recomposed_bundle_cannot_hide_benchmark_or_review_mismatch(tmp_path, artifact):
    source = snapshot_fixture(tmp_path / 'snapshot')
    corpus, run = corpus_and_run_fixture(source, tmp_path)
    review = tmp_path / 'review.jsonl'
    review.write_bytes(b'{"qid":"eval-0001","note":"Human evidence"}\n')
    project = tmp_path / 'project'
    benchmark_sources_fixture(project, run, review)
    archive = tmp_path / 'bundle.tar.gz'
    create_bundle(source, archive, corpus_dir=corpus, run_dirs=[run], project_dir=project, review_files=[review])
    name = ('source/data/evaluation/questions.csv' if artifact.startswith('csv')
            else f'source/reviews/{file_hash(review)}.jsonl')

    def mutate(entries):
        if artifact.endswith('missing'):
            entries.pop(name)
        else:
            entries[name] = b'{"changed":true}\n' if artifact.startswith('review') else b'changed CSV\n'

    recompose_bundle(archive, mutate)
    with pytest.raises(ValueError, match='checksum|Missing packaged artifact'):
        verify_bundle(archive)
    target = tmp_path / 'restored'
    with pytest.raises(ValueError, match='checksum|Missing packaged artifact'):
        restore_bundle(archive, target)
    assert not target.exists()


def test_bundle_uses_one_decompression_and_closes_temporary_storage(tmp_path, monkeypatch):
    import legal_rag.reproducibility as reproducibility

    source = snapshot_fixture(tmp_path / 'snapshot')
    archive = tmp_path / 'bundle.tar.gz'
    create_bundle(source, archive, project_dir=tmp_path / 'empty')
    temporary_files, compressed_files = [], []
    original_tempfile, original_gzip_open = reproducibility.tempfile.TemporaryFile, reproducibility.gzip.open

    def tracked_tempfile(*args, **kwargs):
        handle = original_tempfile(*args, **kwargs)
        temporary_files.append(handle)
        return handle

    def tracked_gzip_open(*args, **kwargs):
        handle = original_gzip_open(*args, **kwargs)
        compressed_files.append(handle)
        return handle

    monkeypatch.setattr(reproducibility.tempfile, 'TemporaryFile', tracked_tempfile)
    monkeypatch.setattr(reproducibility.gzip, 'open', tracked_gzip_open)
    verify_bundle(archive)
    restore_bundle(archive, tmp_path / 'restored')
    corrupt = tmp_path / 'corrupt.tar.gz'
    corrupt.write_bytes(b'not gzip')
    with pytest.raises(OSError):
        verify_bundle(corrupt)
    assert len(temporary_files) == len(compressed_files) == 3
    assert all(handle.closed for handle in [*temporary_files, *compressed_files])


def test_restore_uses_verified_seekable_instance_after_source_archive_changes(tmp_path, monkeypatch):
    import legal_rag.reproducibility as reproducibility

    source = snapshot_fixture(tmp_path / 'snapshot')
    archive_path = tmp_path / 'bundle.tar.gz'
    create_bundle(source, archive_path, project_dir=tmp_path / 'empty')
    original_verify = reproducibility._verify_archive

    def verify_then_change_source(archive):
        manifest = original_verify(archive)
        assert archive.fileobj.seekable()
        archive_path.write_bytes(b'changed after verification')
        return manifest

    monkeypatch.setattr(reproducibility, '_verify_archive', verify_then_change_source)
    restored = tmp_path / 'restored'
    restore_bundle(archive_path, restored)
    assert (restored / 'snapshot/original.html').read_bytes() == b'raw bytes'


def run_code_fixture(project, run):
    source = project / 'src/legal_rag/example.py'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'value = 1\n')
    fingerprint = digest(f'src/legal_rag/example.py:{file_hash(source)}')
    record = json.loads((run / 'manifest.json').read_bytes())
    record['inputs']['code'] = {'commit': 'experimental', 'dirty': False, 'source_sha256': fingerprint}
    write_json(run / 'manifest.json', record)
    return source, fingerprint


def test_bundle_code_identity_uses_selected_project_and_archived_paths(tmp_path, monkeypatch):
    import legal_rag.reproducibility as reproducibility

    source = snapshot_fixture(tmp_path / 'snapshot')
    project = tmp_path / 'selected-project'
    (project / 'src/legal_rag/nested').mkdir(parents=True)
    first, second = project / 'src/legal_rag/a.py', project / 'src/legal_rag/nested/z.py'
    first.write_bytes(b'a = 1\n')
    second.write_bytes(b'z = 2\n')
    (project / 'src/unrelated.py').write_bytes(b'other = 3\n')
    expected = digest(f'src/legal_rag/a.py:{file_hash(first)}\nsrc/legal_rag/nested/z.py:{file_hash(second)}')
    directories = []

    def project_git(arguments, *, cwd, **kwargs):
        directories.append(cwd)
        return 'selected-commit\n' if arguments[1] == 'rev-parse' else ''

    monkeypatch.setattr(reproducibility.subprocess, 'check_output', project_git)
    archive = tmp_path / 'bundle.tar.gz'
    create_bundle(source, archive, project_dir=project)
    manifest = verify_bundle(archive)
    assert manifest.code == {'commit': 'selected-commit', 'dirty': False, 'source_sha256': expected}
    assert directories == [project, project]


def test_matching_run_code_is_verified_and_restored(tmp_path):
    source = snapshot_fixture(tmp_path / 'snapshot')
    corpus, run = corpus_and_run_fixture(source, tmp_path)
    project = tmp_path / 'project'
    code, fingerprint = run_code_fixture(project, run)
    archive = tmp_path / 'bundle.tar.gz'
    result = create_bundle(source, archive, corpus_dir=corpus, run_dirs=[run], project_dir=project)
    assert result['warnings'] == []
    manifest = verify_bundle(archive)
    assert manifest.code['source_sha256'] == fingerprint and manifest.warnings == []
    restored = tmp_path / 'restored'
    assert restore_bundle(archive, restored)['warnings'] == []
    assert (restored / 'source/src/legal_rag/example.py').read_bytes() == code.read_bytes()


def test_changed_experimental_code_rejects_bundle_before_archive_creation(tmp_path):
    source = snapshot_fixture(tmp_path / 'snapshot')
    corpus, run = corpus_and_run_fixture(source, tmp_path)
    project = tmp_path / 'project'
    code, _ = run_code_fixture(project, run)
    code.write_bytes(b'value = 2\n')
    archive = tmp_path / 'bundle.tar.gz'
    with pytest.raises(ValueError, match='correct experimental checkout'):
        create_bundle(source, archive, corpus_dir=corpus, run_dirs=[run], project_dir=project)
    assert not archive.exists() and not archive.with_suffix('.gz.partial').exists()


@pytest.mark.parametrize('mutation', ['source', 'source_and_bundle_identity', 'source_without_bundle_identity', 'run_identity'])
def test_recomposed_bundle_checks_experimental_code_during_verify_and_restore(tmp_path, mutation):
    source = snapshot_fixture(tmp_path / 'snapshot')
    corpus, run = corpus_and_run_fixture(source, tmp_path)
    project = tmp_path / 'project'
    run_code_fixture(project, run)
    archive = tmp_path / 'bundle.tar.gz'
    create_bundle(source, archive, corpus_dir=corpus, run_dirs=[run], project_dir=project)

    def alter_code(entries):
        if mutation == 'run_identity':
            record = json.loads(entries['runs/experiment/manifest.json'])
            record['inputs']['code']['source_sha256'] = digest('different experimental code')
            entries['runs/experiment/manifest.json'] = json.dumps(record).encode()
        else:
            entries['source/src/legal_rag/example.py'] = b'value = 2\n'
            if mutation != 'source':
                bundle = json.loads(entries['bundle.json'])
                if mutation == 'source_and_bundle_identity':
                    bundle['code']['source_sha256'] = digest(f"src/legal_rag/example.py:{digest(entries['source/src/legal_rag/example.py'])}")
                else:
                    bundle['code'].pop('source_sha256')
                entries['bundle.json'] = json.dumps(bundle).encode()

    recompose_bundle(archive, alter_code)
    error = 'archived source bytes' if mutation == 'source' else 'correct experimental checkout'
    with pytest.raises(ValueError, match=error):
        verify_bundle(archive)
    target = tmp_path / 'restored'
    with pytest.raises(ValueError, match=error):
        restore_bundle(archive, target)
    assert not target.exists()


def test_legacy_run_missing_code_fingerprint_remains_explicitly_unverified(tmp_path):
    source = snapshot_fixture(tmp_path / 'snapshot')
    corpus, run = corpus_and_run_fixture(source, tmp_path)
    archive = tmp_path / 'bundle.tar.gz'
    result = create_bundle(source, archive, corpus_dir=corpus, run_dirs=[run], project_dir=tmp_path / 'empty')
    assert len(result['warnings']) == 1 and 'code provenance is unverified' in result['warnings'][0]

    def remove_warning(entries):
        record = json.loads(entries['bundle.json'])
        record.pop('warnings')
        entries['bundle.json'] = json.dumps(record).encode()

    recompose_bundle(archive, remove_warning)
    manifest = verify_bundle(archive)
    assert manifest.warnings == result['warnings']
    restored = tmp_path / 'restored'
    assert restore_bundle(archive, restored)['warnings'] == result['warnings']
    assert (restored / 'runs/experiment/results.jsonl').read_bytes() == (run / 'results.jsonl').read_bytes()
