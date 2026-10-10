"""Verified portable archives of frozen data and the code that produced it."""

import gzip
import io
import json
import shutil
import tarfile
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from pydantic import Field

from legal_rag.artifacts import code_identity, digest, file_hash, new_output, utc_now
from legal_rag.contracts import Record, SourceRecord


class BundleManifest(Record):
    schema_version: str = 'bundle-v1'
    created_at: str
    snapshot_id: str
    code: dict
    files: dict[str, str]
    limitations: list[str] = Field(default_factory=lambda: ['LLM outputs are preserved; regenerated responses need not be identical.',
        'Indexes can be rebuilt; local model weights are identified by revision and are not bundled.'])


SECRET_KEYS = {'api_key', 'authorization', 'password', 'access_token', 'secret', 'utopia_api_key', 'openai_api_key'}


def _check_json(value):
    if isinstance(value, dict):
        if any(str(key).lower() in SECRET_KEYS for key in value):
            raise ValueError('A structured artifact contains a credential field; sanitize it before export')
        for nested in value.values():
            _check_json(nested)
    elif isinstance(value, list):
        for nested in value:
            _check_json(nested)


def _safe_name(name):
    relative = PurePosixPath(name)
    if (not name or name != relative.as_posix() or relative.is_absolute() or
            '..' in relative.parts or '\\' in name or relative == PurePosixPath('.')):
        raise ValueError('Unsafe artifact path')
    return relative


def _check_content(name, raw):
    path = PurePosixPath(name)
    if path.name.startswith('.env') and path.name != '.env.example':
        raise ValueError('Credentials and symlinks cannot be packaged')
    if path.suffix == '.json':
        _check_json(json.loads(raw))
    elif path.suffix == '.jsonl':
        for line in raw.decode('utf-8').splitlines():
            if line.strip():
                _check_json(json.loads(line))


def _check_file(path):
    if path.is_symlink():
        raise ValueError('Credentials and symlinks cannot be packaged')
    _check_content(path.name, path.read_bytes())


def _directory_reader(directory):
    directory = Path(directory)

    def read(name, expected=None):
        relative = _safe_name(name)
        path = directory / relative
        if (not path.is_file() or any((directory / parent).is_symlink() for parent in (relative, *relative.parents)) or
                not path.resolve().is_relative_to(directory.resolve())):
            raise ValueError('Artifact paths must identify regular files inside their dataset')
        raw = path.read_bytes()
        if expected is not None and digest(raw) != expected:
            raise ValueError(f'Artifact checksum mismatch: {name}')
        _check_content(name, raw)
        return raw

    return read


def _verify_provenance(source_read, corpus_read=None, run_reads=(), *, benchmark_read=None, review_read=None):
    def verified(read, name, expected):
        if not isinstance(expected, str) or len(expected) != 64 or any(char not in '0123456789abcdef' for char in expected):
            raise ValueError(f'Artifact checksum missing or invalid: {name}')
        return read(name, expected)

    source_raw = source_read('manifest.json')
    manifest = json.loads(source_raw)
    if manifest.get('complete') is not True:
        raise ValueError('Only complete snapshots can be packaged')
    catalog_raw = verified(source_read, 'catalog.jsonl', manifest['catalog_sha256'])
    records = [SourceRecord.model_validate(json.loads(line)) for line in catalog_raw.decode('utf-8').splitlines() if line.strip()]
    if len(records) != manifest['source_count'] or len({row.source_id for row in records}) != len(records):
        raise ValueError('Snapshot catalogue count or identifiers disagree with its manifest')
    if len({row.path for row in records}) != len(records):
        raise ValueError('Snapshot catalogue contains duplicate source paths')
    for row in records:
        verified(source_read, row.path, row.sha256)
    corpus_identity = None
    if corpus_read:
        corpus_raw = corpus_read('manifest.json')
        corpus = json.loads(corpus_raw)
        if corpus['snapshot_id'] != manifest['snapshot_id'] or not corpus.get('ready_for_indexing'):
            raise ValueError('The corpus is incomplete or belongs to another snapshot')
        if (corpus['source_manifest_sha256'] != digest(source_raw) or
                corpus['source_catalog_sha256'] != manifest['catalog_sha256']):
            raise ValueError('Corpus provenance disagrees with the packaged snapshot')
        for name, expected in corpus['files'].items():
            verified(corpus_read, name, expected)
        corpus_identity = {'snapshot_id': corpus['snapshot_id'], 'manifest_sha256': digest(corpus_raw)}
    for run_read in run_reads:
        run = json.loads(run_read('manifest.json'))
        inputs = run['inputs']
        identities = [value for value in (inputs.get('corpus'), inputs.get('identity', {}).get('corpus'),
                                           inputs.get('index', {}).get('corpus')) if value is not None]
        if corpus_identity is None or not identities or any(identity != corpus_identity for identity in identities):
            raise ValueError('Run provenance disagrees with the packaged corpus')
        for name, expected in run.get('source_hashes', {}).items():
            if name == 'review':
                if review_read is None:
                    raise ValueError('Missing raw benchmark review')
                verified(review_read, str(expected) + '.jsonl', expected)
            else:
                if _safe_name(name).suffix != '.csv':
                    raise ValueError('Unsupported benchmark source artifact')
                if benchmark_read is None:
                    raise ValueError('Missing benchmark source dataset')
                verified(benchmark_read, name, expected)
        for name, expected in run['output_hashes'].items():
            verified(run_read, name, expected)
    return manifest


def create_bundle(snapshot_dir, output_path, *, corpus_dir=None, run_dirs=(), project_dir=None, review_files=()):
    source, output = Path(snapshot_dir), Path(output_path)
    run_dirs = tuple(run_dirs)
    project = Path(project_dir) if project_dir else Path(__file__).resolve().parents[2]
    reviews = {}
    for path in map(Path, review_files):
        if path.suffix != '.jsonl':
            raise ValueError('Raw benchmark reviews must be JSONL files')
        _check_file(path)
        reviews[file_hash(path) + '.jsonl'] = path

    def read_review(name, expected=None):
        if name not in reviews:
            raise ValueError(f'Missing raw benchmark review: {name}')
        path = reviews[name]
        return _directory_reader(path.parent)(path.name, expected)

    manifest = _verify_provenance(_directory_reader(source), _directory_reader(corpus_dir) if corpus_dir else None,
                                  [_directory_reader(directory) for directory in run_dirs],
                                  benchmark_read=_directory_reader(project / 'data/evaluation'), review_read=read_review)
    if output.exists():
        raise FileExistsError(output)
    entries = {}
    def add(directory, prefix):
        for path in sorted(Path(directory).rglob('*')):
            if path.is_file() and not any(part in {'__pycache__', '.pytest_cache'} for part in path.parts) and path.suffix not in {'.pyc'}:
                name = prefix + '/' + path.relative_to(directory).as_posix()
                if name in entries:
                    raise ValueError('Duplicate archive path')
                _check_file(path)
                entries[name] = path
    add(source, 'snapshot')
    if corpus_dir:
        add(Path(corpus_dir), 'corpus')
    names = set()
    for directory in run_dirs:
        directory = Path(directory)
        if directory.name in names:
            raise ValueError('Run directory names must be distinct')
        names.add(directory.name)
        add(directory, 'runs/' + directory.name)
    for name in ('src', 'docs', 'tests', 'notebooks', 'configs', 'data/evaluation'):
        if (project / name).exists():
            add(project / name, 'source/' + name)
    for name in ('pyproject.toml', 'uv.lock', 'README.md', 'AGENTS.md', '.env.example', '.gitignore', '.gitattributes'):
        path = project / name
        if path.exists():
            _check_file(path)
            entries['source/' + name] = path
    for name, path in reviews.items():
        entries['source/reviews/' + name] = path
    bundle = BundleManifest(created_at=utc_now(), snapshot_id=manifest['snapshot_id'], code=code_identity(),
                            files={name: file_hash(path) for name, path in entries.items()})
    encoded = json.dumps(bundle.to_json_record(), ensure_ascii=False, indent=2, sort_keys=True).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.partial')
    try:
        with tarfile.open(temporary, 'w:gz') as archive:
            info = tarfile.TarInfo('bundle.json')
            info.size, info.mode = len(encoded), 0o600
            archive.addfile(info, io.BytesIO(encoded))
            for name, path in entries.items():
                info = archive.gettarinfo(str(path), arcname=name)
                info.mode = 0o600
                with path.open('rb') as handle:
                    archive.addfile(info, handle)
        verify_bundle(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {'path': str(output.resolve()), 'sha256': file_hash(output), 'files': len(entries), 'snapshot_id': manifest['snapshot_id']}


def _verify_archive(archive):
    members = archive.getmembers()
    names = [member.name for member in members]
    if len(names) != len(set(names)) or any(not member.isfile() for member in members):
        raise ValueError('Unsafe or duplicate archive members')
    for name in names:
        _safe_name(name)
    names = set(names)

    def read(name, expected=None):
        _safe_name(name)
        if name not in names:
            raise ValueError(f'Missing packaged artifact: {name}')
        with archive.extractfile(name) as handle:
            raw = handle.read()
        if expected is not None and digest(raw) != expected:
            raise ValueError(f'Bundle checksum mismatch: {name}')
        _check_content(name, raw)
        return raw

    manifest = BundleManifest.model_validate(json.loads(read('bundle.json')))
    if names != {'bundle.json', *manifest.files}:
        raise ValueError('Bundle contents disagree with its manifest')
    for name, expected in manifest.files.items():
        read(name, expected)

    def subtree(prefix):
        def nested_read(name, expected=None):
            _safe_name(name)
            return read(prefix + '/' + name, expected)
        return nested_read

    run_prefixes = set()
    for name in names:
        if name.startswith('runs/'):
            parts = PurePosixPath(name).parts
            if len(parts) < 3:
                raise ValueError('Unsafe run archive layout')
            run_prefixes.add('/'.join(parts[:2]))
    snapshot = _verify_provenance(subtree('snapshot'), subtree('corpus') if any(name.startswith('corpus/') for name in names) else None,
                                  [subtree(prefix) for prefix in sorted(run_prefixes)],
                                  benchmark_read=subtree('source/data/evaluation'), review_read=subtree('source/reviews'))
    if snapshot['snapshot_id'] != manifest.snapshot_id:
        raise ValueError('Bundle snapshot identity disagrees with its packaged snapshot')
    return manifest


@contextmanager
def _open_bundle(archive_path):
    with tempfile.TemporaryFile() as uncompressed:
        with gzip.open(archive_path, 'rb') as compressed:
            shutil.copyfileobj(compressed, uncompressed, length=1024 * 1024)
        uncompressed.seek(0)
        with tarfile.open(fileobj=uncompressed, mode='r:') as archive:
            yield archive


def verify_bundle(archive_path):
    with _open_bundle(archive_path) as archive:
        return _verify_archive(archive)


def restore_bundle(archive_path, output_dir):
    output = Path(output_dir)
    with _open_bundle(archive_path) as archive:
        manifest = _verify_archive(archive)
        new_output(output)
        try:
            for member in archive.getmembers():
                target = output / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open('wb') as destination:
                    shutil.copyfileobj(source, destination)
                target.chmod(0o600)
            for name, expected in manifest.files.items():
                if file_hash(output / name) != expected:
                    raise ValueError(f'Restored checksum mismatch: {name}')
        except Exception:
            shutil.rmtree(output)
            raise
    return {'path': str(output.resolve()), 'files_verified': len(manifest.files), 'snapshot_id': manifest.snapshot_id}
