"""Small entry points; all pipeline logic lives in importable modules."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from legal_rag.artifacts import read_json, write_json
from legal_rag.contracts import EngineConfig
from legal_rag.env import load_env_file
from legal_rag.runtime import close_engine, create_engine, evaluation_identity, provider_config

PROFILES = ('no_rag', 'dense', 'sparse', 'hybrid', 'multiquery', 'graph', 'rerank', 'oracle')


def _json(value):
    print(json.dumps(value.to_json_record() if hasattr(value, 'to_json_record') else value, ensure_ascii=False, indent=2))


def _progress(state):
    print(json.dumps(state, ensure_ascii=False), file=sys.stderr, flush=True)


def _provider_args(parser):
    parser.add_argument('--provider', choices=('utopia', 'openai'), default='utopia')
    parser.add_argument('--model')
    parser.add_argument('--pricing', type=Path, default=Path('configs/openai-pricing.json'))
    parser.add_argument('--ledger', type=Path, default=Path('data/budget/openai.json'))
    parser.add_argument('--max-output-tokens', type=int, default=4096)
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--context-window-tokens', type=int, help='Explicit provider context window; Utopia defaults to 32768.')


def _engine_args(parser):
    _provider_args(parser)
    parser.add_argument('--index', type=Path)
    parser.add_argument('--corpus', type=Path)
    parser.add_argument('--profile', choices=PROFILES, default='hybrid')
    parser.add_argument('--include-ceased', action='store_true')
    parser.add_argument('--context-tokens', type=int, default=4096)
    parser.add_argument('--context-units', type=int, default=12)
    parser.add_argument('--local-files-only', action='store_true')


def _engine(args):
    config = provider_config(args.provider, args.model, pricing_path=args.pricing,
                             max_output_tokens=args.max_output_tokens, timeout_seconds=args.timeout,
                             context_window_tokens=args.context_window_tokens)
    experiment = EngineConfig(profile=args.profile, context_tokens=args.context_tokens,
                              context_units=args.context_units, exclude_ceased=not args.include_ceased)
    return create_engine(config, experiment, index_dir=args.index, corpus_dir=args.corpus,
                         ledger_path=args.ledger, pricing_path=args.pricing, local_files_only=args.local_files_only)


def _model_metadata(client):
    if not hasattr(client, 'inspect_model'):
        return None
    inspection = client.inspect_model()
    if inspection.returned_model == 'unknown' or inspection.digest is None or 'completion' not in inspection.capabilities:
        raise ValueError('Controlled Utopia runs require verified model identity, digest and completion capability')
    requested = client.config.context_window_tokens
    if requested is None or inspection.context_window_tokens is None or requested > inspection.context_window_tokens:
        raise ValueError('Controlled Utopia runs require an explicit context window within the verified model capacity')
    return inspection.to_json_record()


def _parser():
    parser = argparse.ArgumentParser(description='Reproducible legal question answering and research artifacts.')
    parser.add_argument('--env-file', type=Path, default=Path('.env'))
    commands = parser.add_subparsers(dest='command', required=True)
    item = commands.add_parser('acquire', help='Acquire every official coordinated law into an immutable snapshot.')
    item.add_argument('--output', type=Path, required=True)
    item.add_argument('--resume', action='store_true')
    item = commands.add_parser('import-legacy', help='Preserve existing HTML as a separate, dated-unknown snapshot.')
    item.add_argument('--input', type=Path, required=True)
    item.add_argument('--output', type=Path, required=True)
    item = commands.add_parser('preprocess', help='Parse, annotate, build graph and chunk a frozen snapshot.')
    item.add_argument('--snapshot', type=Path, required=True)
    item.add_argument('--output', type=Path, required=True)
    item.add_argument('--target-tokens', type=int, default=384)
    item.add_argument('--max-tokens', type=int, default=512)
    item.add_argument('--local-files-only', action='store_true')
    item = commands.add_parser('index', help='Build one immutable local Qdrant index.')
    item.add_argument('--corpus', type=Path, required=True)
    item.add_argument('--output', type=Path, required=True)
    item.add_argument('--embedding-backend', choices=('local', 'utopia', 'openai'), default='local')
    item.add_argument('--embedding-model')
    item.add_argument('--batch-size', type=int, default=2)
    item.add_argument('--upload-batch-size', type=int, help='Group writes separately from the embedding inference batch.')
    item.add_argument('--embedding-device', choices=('cpu', 'mps'), default='cpu')
    item.add_argument('--embedding-dtype', choices=('float32', 'float16'), default='float32')
    item.add_argument('--cache', type=Path)
    item.add_argument('--bm25', action='store_true')
    item.add_argument('--local-files-only', action='store_true')
    item.add_argument('--pricing', type=Path, default=Path('configs/openai-pricing.json'))
    item.add_argument('--ledger', type=Path, default=Path('data/budget/openai.json'))
    item = commands.add_parser('ask', help='Answer one question without conversational memory.')
    _engine_args(item)
    item.add_argument('question')
    item.add_argument('--output', type=Path)
    item = commands.add_parser('evaluate', help='Evaluate frozen benchmark intents, with checkpoint resume.')
    _engine_args(item)
    item.add_argument('--benchmark', type=Path, default=Path('data/evaluation'))
    item.add_argument('--view', choices=('legacy', 'current_approved'), default='legacy')
    item.add_argument('--reviews', type=Path)
    item.add_argument('--output', type=Path, required=True)
    item.add_argument('--judge-provider', choices=('utopia', 'openai'), default='openai')
    item.add_argument('--judge-model')
    item.add_argument('--control-model', default='gpt-6-sol')
    item.add_argument('--without-control', action='store_true')
    item.add_argument('--mcq', action='store_true')
    item.add_argument('--qids', nargs='+')
    item.add_argument('--resume', action='store_true')
    item.add_argument('--oracle-evidence', type=Path)
    item = commands.add_parser('rejudge', help='Judge frozen answers against an approved benchmark revision.')
    _provider_args(item)
    item.set_defaults(provider='openai')
    item.add_argument('--source-run', type=Path, required=True)
    item.add_argument('--benchmark', type=Path, default=Path('data/evaluation'))
    item.add_argument('--reviews', type=Path, required=True)
    item.add_argument('--corpus', type=Path, required=True)
    item.add_argument('--output', type=Path, required=True)
    item.add_argument('--resume', action='store_true')
    item.add_argument('--without-control', action='store_true')
    item.add_argument('--control-model', default='gpt-6-sol')
    item = commands.add_parser('diagnose', help='Measure retrieval and context on the benchmark without LLM calls.')
    item.add_argument('--index', type=Path, required=True)
    item.add_argument('--corpus', type=Path)
    item.add_argument('--benchmark', type=Path, default=Path('data/evaluation'))
    item.add_argument('--output', type=Path, required=True)
    item.add_argument('--profiles', nargs='+', choices=('dense', 'sparse', 'hybrid', 'graph'), default=['dense', 'sparse', 'hybrid', 'graph'])
    item.add_argument('--include-ceased', action='store_true')
    item.add_argument('--local-files-only', action='store_true')
    item.add_argument('--pricing', type=Path, default=Path('configs/openai-pricing.json'))
    item.add_argument('--ledger', type=Path, default=Path('data/budget/openai.json'))
    item = commands.add_parser('report', help='Render offline HTML from artifacts; never call a model.')
    item.add_argument('--corpus', type=Path)
    item.add_argument('--runs', type=Path, nargs='*', default=[])
    item.add_argument('--diagnostics', type=Path, nargs='*', default=[])
    item.add_argument('--comments', type=Path)
    item.add_argument('--output', type=Path, required=True)
    item = commands.add_parser('compare', help='Compare compatible paired runs with bootstrap intervals.')
    item.add_argument('runs', type=Path, nargs='+')
    item.add_argument('--output', type=Path)
    item = commands.add_parser('preflight', help='Validate credentials and optionally test structured chat.')
    _provider_args(item)
    item.add_argument('--live', action='store_true')
    item.add_argument('--output', type=Path)
    item = commands.add_parser('enrich', help='Propose LLM annotations on unresolved evidence; never change corpus.')
    _provider_args(item)
    item.add_argument('--corpus', type=Path, required=True)
    item.add_argument('--output', type=Path, required=True)
    item.add_argument('--limit', type=int, default=20)
    item = commands.add_parser('ui', help='Open the stateless question-answer interface.')
    _engine_args(item)
    item = commands.add_parser('bundle', help='Package frozen source, benchmark and research artifacts with checksums.')
    item.add_argument('--snapshot', type=Path, required=True)
    item.add_argument('--corpus', type=Path)
    item.add_argument('--runs', type=Path, nargs='*', default=[])
    item.add_argument('--review-files', type=Path, nargs='*', default=[])
    item.add_argument('--output', type=Path, required=True)
    item = commands.add_parser('restore', help='Verify and restore a research bundle into a new directory.')
    item.add_argument('archive', type=Path)
    item.add_argument('--output', type=Path, required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    load_env_file(args.env_file)
    try:
        return _run(args) or 0
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 2
    except RuntimeError as error:
        print(f'{getattr(error, "code", type(error).__name__)}: operation failed; inspect the saved checkpoint.', file=sys.stderr)
        return 2


def _run(args):
    if args.command == 'acquire':
        from legal_rag.corpus.acquisition import acquire_snapshot
        def progress(state):
            catalog = 'page' in state and (state['page'] == 1 or state['page'] % 25 == 0 or state['page'] == state.get('pages_seen'))
            details = 'completed' in state and (state['completed'] == 1 or state['completed'] % 100 == 0 or state['completed'] == state.get('total'))
            if catalog or details:
                _progress(state)
        _json(acquire_snapshot(args.output, resume=args.resume, progress=progress))
    elif args.command == 'import-legacy':
        from legal_rag.corpus.acquisition import import_legacy
        _json(import_legacy(args.input, args.output))
    elif args.command == 'preprocess':
        from legal_rag.corpus.pipeline import build_corpus
        from legal_rag.tokens import BgeTokenCounter
        result = build_corpus(args.snapshot, args.output, BgeTokenCounter(local_files_only=args.local_files_only),
                              target_tokens=args.target_tokens, max_tokens=args.max_tokens, progress=_progress)
        _json(result)
        return int(not result['ready_for_indexing'])
    elif args.command == 'index':
        from legal_rag.indexing import build_index
        from legal_rag.providers.embeddings import BgeEmbedder, BM25Embedder, OpenAIEmbedder, UtopiaEmbedder
        if args.embedding_backend == 'local':
            if args.embedding_model not in (None, 'BAAI/bge-m3'):
                raise ValueError('Local embedding supports pinned BGE-M3')
            embedder = BgeEmbedder(batch_size=args.batch_size, local_files_only=args.local_files_only,
                                   device=args.embedding_device, dtype=args.embedding_dtype)
        elif args.embedding_backend == 'utopia':
            if not args.embedding_model:
                raise ValueError('An explicit Utopia embedding model is required')
            import os
            embedder = UtopiaEmbedder(args.embedding_model, base_url=os.getenv('UTOPIA_BASE_URL', 'https://utopia.hpc4ai.unito.it/api'))
        else:
            config = provider_config('openai', args.embedding_model or 'text-embedding-3-small', pricing_path=args.pricing)
            embedder = OpenAIEmbedder(config, ledger_path=args.ledger)
        if args.embedding_backend == 'local' and args.bm25:
            raise ValueError('Reference BGE-M3 uses native sparse vectors; BM25 is a separate alternative')
        sparse = BM25Embedder() if args.bm25 else None
        _json(build_index(args.corpus, args.output, embedder, sparse_embedder=sparse,
                          batch_size=args.upload_batch_size or args.batch_size, cache_dir=args.cache,
                          progress=lambda state: _progress({'stage': 'index', **state})))
    elif args.command == 'ask':
        if args.profile == 'oracle':
            raise ValueError('Oracle is diagnostic only and requires verified evidence via the Python API')
        engine, _ = _engine(args)
        try:
            result = engine.answer(args.question)
            if args.output:
                if args.output.exists():
                    raise FileExistsError(args.output)
                write_json(args.output, result)
            _json(result)
            return int(result.status == 'error')
        finally:
            close_engine(engine)
    elif args.command == 'evaluate':
        from legal_rag.evaluation import load_benchmark, run_evaluation
        from legal_rag.providers import build_chat_client
        if args.corpus is None or args.index is None:
            raise ValueError('Every controlled evaluation, including no-RAG, requires corpus and index identities')
        if args.profile == 'oracle' and args.oracle_evidence is None:
            raise ValueError('Oracle evaluation requires --oracle-evidence with approved supporting chunks')
        benchmark = load_benchmark(args.benchmark, view=args.view, review_path=args.reviews)
        judge_config = provider_config(args.judge_provider, args.judge_model, pricing_path=args.pricing)
        judge = build_chat_client(judge_config, ledger_path=args.ledger)
        judge.preflight()
        control = None
        if not args.without_control:
            control = build_chat_client(provider_config('openai', args.control_model, pricing_path=args.pricing), ledger_path=args.ledger, purpose='validation')
            control.preflight()
        engine, manifest = _engine(args)
        try:
            identity = evaluation_identity(engine, judge, manifest, args.corpus)
            for role, client in (('generator', engine.client), ('judge', judge)):
                metadata = _model_metadata(client)
                if metadata:
                    identity[role] = {**identity[role], 'verified_model': metadata}
            result = run_evaluation(benchmark, engine, judge, args.output, profile=args.profile, identity=identity,
                                  resume=args.resume, formats=('no_hint', 'mcq') if args.mcq else ('no_hint',),
                                  qids=args.qids, judge_control=control, corpus_dir=args.corpus,
                                  oracle_evidence_path=args.oracle_evidence, progress=_progress)
            _json(result)
            return int(not result['complete'])
        finally:
            close_engine(engine)
    elif args.command == 'rejudge':
        from legal_rag.evaluation import load_benchmark, run_rejudge
        from legal_rag.providers import build_chat_client
        benchmark = load_benchmark(args.benchmark, view='current_approved', review_path=args.reviews)
        judge = build_chat_client(provider_config(args.provider, args.model, pricing_path=args.pricing,
            max_output_tokens=args.max_output_tokens, timeout_seconds=args.timeout,
            context_window_tokens=args.context_window_tokens), ledger_path=args.ledger)
        judge.preflight()
        control = None
        if not args.without_control:
            control = build_chat_client(provider_config('openai', args.control_model, pricing_path=args.pricing), ledger_path=args.ledger, purpose='validation')
            control.preflight()
        identity = dict(read_json(args.source_run / 'manifest.json')['inputs']['identity'])
        identity['judge'] = judge.identity
        metadata = _model_metadata(judge)
        if metadata:
            identity['judge'] = {**identity['judge'], 'verified_model': metadata}
        result = run_rejudge(benchmark, judge, args.source_run, args.output, identity=identity,
                            resume=args.resume, judge_control=control, corpus_dir=args.corpus)
        _json(result)
        return int(not result['complete'])
    elif args.command == 'diagnose':
        from legal_rag.evaluation import load_benchmark, run_retrieval_diagnostics
        from legal_rag.indexing import open_index
        from legal_rag.retrieval import Retriever
        from legal_rag.runtime import index_embedder
        from legal_rag.tokens import BgeTokenCounter
        manifest = read_json(args.index / 'manifest.json')
        embedder, sparse = index_embedder(manifest, pricing_path=args.pricing, ledger_path=args.ledger, local_files_only=args.local_files_only)
        counter = BgeTokenCounter(local_files_only=args.local_files_only)
        client = open_index(args.index, corpus_dir=args.corpus)
        try:
            retriever = Retriever(client, embedder, sparse_embedder=sparse, corpus_dir=args.corpus, manifest=manifest)
            result = run_retrieval_diagnostics(load_benchmark(args.benchmark), retriever, counter, args.output,
                profiles=tuple(args.profiles), config=EngineConfig(exclude_ceased=not args.include_ceased), progress=_progress)
            _json(result)
            return int(not result['complete'] or result['errors'] > 0)
        finally:
            client.close()
    elif args.command == 'report':
        from legal_rag.reporting import build_report
        print(build_report(args.corpus, args.runs, args.output, diagnostic_dirs=args.diagnostics, comments_path=args.comments))
    elif args.command == 'compare':
        from legal_rag.evaluation import compare_runs
        result = compare_runs(args.runs)
        if args.output:
            if args.output.exists():
                raise FileExistsError(args.output)
            write_json(args.output, result)
        _json(result)
    elif args.command == 'preflight':
        from legal_rag.contracts import Record
        from legal_rag.providers import build_chat_client
        class Check(Record):
            ok: bool
        config = provider_config(args.provider, args.model, pricing_path=args.pricing, max_output_tokens=args.max_output_tokens,
                                 timeout_seconds=args.timeout, context_window_tokens=args.context_window_tokens)
        client = build_chat_client(config, ledger_path=args.ledger)
        from legal_rag.artifacts import utc_now
        from legal_rag.providers.errors import ProviderError
        result = {'identity': client.preflight(), 'live': args.live, 'checked_at': utc_now()}
        try:
            if args.live:
                if hasattr(client, 'inspect_model'):
                    result['model_inspection'] = client.inspect_model().to_json_record()
                payload, usage = client.complete('Return a JSON object with ok equal to true.', Check)
                result.update(payload=payload.to_json_record(), usage=usage.to_json_record())
        except ProviderError as error:
            result['error'] = error.code
            if args.output:
                write_json(args.output, result)
            _json(result)
            return 1
        if args.output:
            write_json(args.output, result)
        _json(result)
    elif args.command == 'enrich':
        from legal_rag.corpus.enrichment import propose_enrichment
        from legal_rag.providers import build_chat_client
        config = provider_config(args.provider, args.model, pricing_path=args.pricing, max_output_tokens=args.max_output_tokens,
                                 timeout_seconds=args.timeout, context_window_tokens=args.context_window_tokens)
        client = build_chat_client(config, ledger_path=args.ledger)
        result = propose_enrichment(args.corpus, args.output, client, limit=args.limit)
        _json(result)
        return int(not result['complete'])
    elif args.command == 'ui':
        path = Path(__file__).parent / 'ui' / 'app.py'
        forwarded = ['--provider', args.provider, '--profile', args.profile, '--pricing', str(args.pricing), '--ledger', str(args.ledger),
                     '--context-tokens', str(args.context_tokens), '--context-units', str(args.context_units),
                     '--timeout', str(args.timeout), '--max-output-tokens', str(args.max_output_tokens)]
        for name in ('index', 'corpus', 'model'):
            if getattr(args, name) is not None:
                forwarded += ['--' + name, str(getattr(args, name))]
        if args.context_window_tokens is not None:
            forwarded += ['--context-window-tokens', str(args.context_window_tokens)]
        if args.include_ceased:
            forwarded += ['--include-ceased']
        if args.local_files_only:
            forwarded += ['--local-files-only']
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', str(path), '--', *forwarded], check=True)
    elif args.command == 'bundle':
        from legal_rag.reproducibility import create_bundle
        _json(create_bundle(args.snapshot, args.output, corpus_dir=args.corpus, run_dirs=args.runs,
                            review_files=args.review_files))
    elif args.command == 'restore':
        from legal_rag.reproducibility import restore_bundle
        _json(restore_bundle(args.archive, args.output))
