"""Shared construction for CLI and the stateless question interface."""

from datetime import date
from pathlib import Path

from pydantic import Field

from legal_rag.artifacts import file_hash, read_json
from legal_rag.contracts import EngineConfig, ProviderConfig, Record
from legal_rag.tokens import BgeTokenCounter


class TokenPrice(Record):
    input_usd_per_million: float = Field(ge=0)
    output_usd_per_million: float = Field(ge=0)
    context_window_tokens: int | None = Field(default=None, gt=0)


class PricingQuote(Record):
    verified_at: str
    price_source_url: str
    fx_date: str
    fx_source_url: str
    ecb_usd_per_eur: float = Field(gt=0)
    eur_per_usd: float = Field(gt=0)
    conversion_note: str
    models: dict[str, TokenPrice]


def provider_config(provider='utopia', model=None, *, pricing_path=None, max_output_tokens=4096, timeout_seconds=180,
                    context_window_tokens=None):
    import os

    values = {'provider': provider, 'model': model or ('SLURM.gpt-oss:120b' if provider == 'utopia' else 'gpt-6-luna'),
              'base_url': os.getenv('UTOPIA_BASE_URL', 'https://utopia.hpc4ai.unito.it/api') if provider == 'utopia' else 'https://api.openai.com/v1',
              'max_output_tokens': max_output_tokens, 'timeout_seconds': timeout_seconds}
    if provider == 'openai':
        if pricing_path is None:
            raise ValueError('OpenAI requires an explicit verified pricing file')
        quote = PricingQuote.model_validate(read_json(Path(pricing_path)))
        if date.fromisoformat(quote.verified_at) != date.today():
            raise ValueError('Refresh and verify OpenAI prices today before a paid batch')
        if not 0 <= (date.today() - date.fromisoformat(quote.fx_date)).days <= 4:
            raise ValueError('The documented exchange-rate reference is stale')
        if values['model'] not in quote.models:
            raise ValueError('No verified price is available for the requested model')
        if quote.eur_per_usd < 1 / quote.ecb_usd_per_eur:
            raise ValueError('The budget conversion must cover the documented reference conversion')
        values.update(quote.models[values['model']].to_json_record(), price_date=quote.verified_at,
                      price_source_url=quote.price_source_url, eur_per_usd=quote.eur_per_usd,
                      fx_date=quote.fx_date, fx_source_url=quote.fx_source_url)
    if context_window_tokens is not None:
        known_capacity = values.get('context_window_tokens')
        if known_capacity is not None and context_window_tokens > known_capacity:
            raise ValueError('Requested context window exceeds the documented model capacity')
        values['context_window_tokens'] = context_window_tokens
    return ProviderConfig.model_validate(values)


def index_embedder(manifest, *, pricing_path=None, ledger_path='data/budget/openai.json', local_files_only=False):
    from legal_rag.providers.embeddings import BgeEmbedder, BM25Embedder, OpenAIEmbedder, UtopiaEmbedder

    identity = manifest['embedding']
    if identity['provider'] == 'local':
        embedder = BgeEmbedder(local_files_only=local_files_only, device=identity.get('device', 'cpu'), dtype=identity.get('dtype', 'float32'))
    elif identity['provider'] == 'utopia':
        embedder = UtopiaEmbedder(identity['model'], base_url=identity['endpoint'].removesuffix('/ollama/api/embed'))
    elif identity['provider'] == 'openai':
        config = provider_config('openai', identity['model'], pricing_path=pricing_path)
        embedder = OpenAIEmbedder(config, ledger_path=ledger_path)
    else:
        raise ValueError('Unsupported index embedding provider')
    sparse = manifest.get('sparse_embedding')
    lexical = BM25Embedder(avg_len=sparse['avg_len'], k=sparse['k'], b=sparse['b']) if sparse and sparse.get('model') == 'Qdrant/bm25' else None
    return embedder, lexical


def create_engine(config: ProviderConfig, engine_config: EngineConfig, *, index_dir=None, corpus_dir=None,
                  ledger_path='data/budget/openai.json', pricing_path=None, local_files_only=False):
    from legal_rag.generation import RagEngine
    from legal_rag.indexing import open_index
    from legal_rag.providers import build_chat_client
    from legal_rag.retrieval import Retriever

    client = build_chat_client(config, ledger_path=ledger_path)
    client.preflight()
    from legal_rag.indexing.index import IndexManifest
    manifest = IndexManifest.model_validate(read_json(Path(index_dir) / 'manifest.json')).to_json_record() if index_dir else None
    if manifest is not None and not manifest['ready_for_retrieval']:
        raise ValueError('The index is incomplete')
    counter = BgeTokenCounter(local_files_only=local_files_only)
    retriever = None
    if engine_config.profile != 'no_rag':
        if manifest is None:
            raise ValueError('A retrieval profile requires an index')
        embedder, sparse = index_embedder(manifest, pricing_path=pricing_path, ledger_path=ledger_path, local_files_only=local_files_only)
        qdrant = open_index(index_dir, corpus_dir=corpus_dir)
        try:
            retriever = Retriever(qdrant, embedder, sparse_embedder=sparse, corpus_dir=corpus_dir, manifest=manifest)
        except Exception:
            qdrant.close()
            raise
    return RagEngine(client, retriever, counter, engine_config), manifest


def evaluation_identity(engine, judge, index_manifest, corpus_dir):
    from legal_rag.generation.engine import ANSWER_PROMPT_VERSION

    root = Path(corpus_dir)
    corpus = read_json(root / 'manifest.json')
    if not corpus.get('ready_for_indexing'):
        raise ValueError('Evaluation requires a verified corpus')
    for name, expected in corpus['files'].items():
        if file_hash(root / name) != expected:
            raise ValueError(f'Corpus artifact hash mismatch: {name}')
    if index_manifest and index_manifest['snapshot_id'] != corpus['snapshot_id']:
        raise ValueError('Corpus and index snapshots differ')
    if index_manifest and (not index_manifest.get('ready_for_retrieval') or
                           index_manifest['corpus_manifest_sha256'] != file_hash(root / 'manifest.json') or
                           index_manifest['chunks_sha256'] != file_hash(root / 'chunks.jsonl') or
                           index_manifest['files'] != corpus['files']):
        raise ValueError('Corpus and index identities differ or the index is incomplete')
    return {'generator': engine.client.identity, 'judge': judge.identity,
            'answer_prompt_version': ANSWER_PROMPT_VERSION,
            'filters': {'exclude_ceased': engine.config.exclude_ceased},
            'token_budget': {'tokens': engine.config.context_tokens, 'units': engine.config.context_units},
            'candidate_budget': engine.config.candidate_limit,
            'corpus': {'snapshot_id': corpus['snapshot_id'], 'manifest_sha256': file_hash(root / 'manifest.json')},
            'embedding': {'dense': index_manifest['embedding'], 'sparse': index_manifest.get('sparse_embedding'),
                          'chunks_sha256': index_manifest['chunks_sha256']} if index_manifest else None}


def close_engine(engine):
    if engine.retriever is not None:
        engine.retriever.client.close()
