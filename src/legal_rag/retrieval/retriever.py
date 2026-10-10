"""Dense, lexical and hybrid retrieval over one immutable collection."""

import json
from pathlib import Path

from qdrant_client import models

from legal_rag.artifacts import digest, file_hash, read_jsonl
from legal_rag.contracts import Candidate, Chunk, EngineConfig, Relation, Unit
from legal_rag.providers.embeddings import encode_vectors, validate_dense, validate_sparse


def fuse_rankings(batches: list[list[Candidate]], *, k: int = 60, limit: int = 100) -> list[Candidate]:
    scores, chunks, origins = {}, {}, {}
    for batch in batches:
        seen = set()
        for position, candidate in enumerate(batch):
            key = candidate.chunk.chunk_id
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0) + 1 / (k + position)
            chunks[key] = candidate.chunk
            origins.setdefault(key, set()).add(candidate.origin)
    keys = sorted(scores, key=lambda key: (-scores[key], key))[:limit]
    return [Candidate(chunk=chunks[key], rank=i + 1, score=scores[key], origin="+".join(sorted(origins[key]))) for i, key in enumerate(keys)]


class Retriever:
    def __init__(self, client, embedder, *, sparse_embedder=None, corpus_dir=None, manifest=None):
        self.client, self.embedder, self.sparse_embedder = client, embedder, sparse_embedder
        self.manifest = manifest or {"collection": "chunks", "embedding": embedder.identity, "snapshot_id": None}
        self.collection = self.manifest.get("collection", "chunks")
        self.corpus_dir = Path(corpus_dir or self.manifest["corpus_dir"]) if (corpus_dir or self.manifest.get("corpus_dir")) else None
        self._query_cache = {}
        self._graph = None
        if manifest is not None:
            if not manifest.get("ready_for_retrieval"):
                raise ValueError("Index is not ready for retrieval")
            if manifest.get("embedding") != embedder.identity:
                raise ValueError("Query embedder identity differs from the index")
            sparse_identity = sparse_embedder.identity if sparse_embedder is not None else (embedder.identity if hasattr(embedder, "embed_hybrid") else None)
            if manifest.get("sparse_embedding") != sparse_identity:
                raise ValueError("Sparse query representation differs from the index")

    @property
    def identity(self):
        return {"snapshot_id": self.manifest.get("snapshot_id"), "embedding": self.embedder.identity,
                "corpus": {"snapshot_id": self.manifest.get("snapshot_id"), "manifest_sha256": self.manifest.get("corpus_manifest_sha256")},
                "index_identity": digest(json.dumps(self.manifest, sort_keys=True))}

    def _vectors(self, query, *, need_dense=True, need_sparse=True):
        cached = self._query_cache.setdefault(query, {})
        if self.sparse_embedder is None and hasattr(self.embedder, "embed_hybrid") and not cached:
            dense, sparse = encode_vectors(self.embedder, [query], sparse_embedder=self.sparse_embedder, query=True)
            cached.update(dense=dense[0], sparse=sparse[0] if sparse is not None else None)
        elif self.sparse_embedder is not None or not hasattr(self.embedder, "embed_hybrid"):
            if need_dense and "dense" not in cached:
                cached["dense"] = validate_dense(self.embedder.embed_queries([query]), 1)[0]
            if need_sparse and "sparse" not in cached:
                cached["sparse"] = validate_sparse(self.sparse_embedder.embed_queries([query]), 1)[0] if self.sparse_embedder else None
        return cached.get("dense"), cached.get("sparse")

    @staticmethod
    def policy_filter(config: EngineConfig):
        if config.exclude_ceased:
            return models.Filter(must_not=[models.FieldCondition(key="status", match=models.MatchValue(value="ceased_confirmed"))])
        return None

    def search(self, query: str, *, mode="hybrid", config=None, trace=None):
        config = config or EngineConfig()
        if mode not in {"dense", "sparse", "hybrid"}:
            raise ValueError(f"Unsupported retrieval mode: {mode}")
        if mode != "dense" and self.sparse_embedder is None and not hasattr(self.embedder, "embed_hybrid"):
            raise ValueError("Sparse or hybrid retrieval requires a matching sparse representation")
        dense, sparse = self._vectors(query, need_dense=mode != "sparse", need_sparse=mode != "dense")
        filter_ = self.policy_filter(config)
        if mode != "dense" and sparse is None:
            raise ValueError("Sparse or hybrid retrieval requires a matching sparse representation")
        if mode == "hybrid":
            prefetch = [models.Prefetch(query=dense, using="dense", filter=filter_, limit=config.candidate_limit),
                        models.Prefetch(query=sparse, using="sparse", filter=filter_, limit=config.candidate_limit)]
            response = self.client.query_points(self.collection, prefetch=prefetch, query=models.RrfQuery(rrf=models.Rrf(k=config.rrf_k)),
                                                limit=config.candidate_limit, with_payload=True)
            if trace is not None:
                branches = {}
                for name, vector in (("dense", dense), ("sparse", sparse)):
                    branch = self.client.query_points(self.collection, query=vector, using=name, query_filter=filter_,
                                                      limit=config.candidate_limit, with_payload=["chunk_id"])
                    branches[name] = [{"chunk_id": p.payload["chunk_id"], "rank": i + 1, "score": p.score} for i, p in enumerate(branch.points)]
                trace.append({"query": query, "mode": mode, "branches": branches})
        else:
            response = self.client.query_points(self.collection, query=dense if mode == "dense" else sparse, using=mode,
                                                query_filter=filter_, limit=config.candidate_limit, with_payload=True)
        candidates = []
        for i, point in enumerate(response.points):
            expected_identity = self.manifest.get("embedding_identity_sha256")
            if expected_identity and point.payload.get("embedding_identity_sha256") != expected_identity:
                raise ValueError("Retrieved point has a different embedding identity")
            payload = {key: value for key, value in point.payload.items() if key in Chunk.model_fields}
            chunk = Chunk.model_validate(payload)
            if config.exclude_ceased and chunk.status == "ceased_confirmed":
                raise ValueError("Retrieval returned ceased content despite the active policy")
            if self.manifest.get("snapshot_id") and chunk.snapshot_id != self.manifest["snapshot_id"]:
                raise ValueError("Retrieved chunk belongs to another snapshot")
            candidates.append(Candidate(chunk=chunk, rank=i + 1, score=point.score, origin=mode))
        if trace is not None:
            if mode != "hybrid":
                trace.append({"query": query, "mode": mode})
            trace[-1]["ranking"] = [{"chunk_id": c.chunk.chunk_id, "rank": c.rank, "score": c.score} for c in candidates]
        return candidates

    def expand_graph(self, candidates: list[Candidate], *, config: EngineConfig):
        if not self.corpus_dir:
            raise ValueError("Graph experiments require the indexed corpus snapshot")
        if self._graph is None:
            for name in ("relations.jsonl", "units.jsonl", "chunks.jsonl"):
                expected = self.manifest.get("files", {}).get(name)
                if not expected or file_hash(self.corpus_dir / name) != expected:
                    raise ValueError(f"Graph source hash mismatch: {name}")
            relations = read_jsonl(self.corpus_dir / "relations.jsonl", Relation)
            units = read_jsonl(self.corpus_dir / "units.jsonl", Unit)
            chunks = read_jsonl(self.corpus_dir / "chunks.jsonl", Chunk)
            if any(c.snapshot_id != self.manifest.get("snapshot_id") for c in chunks):
                raise ValueError("Graph chunks belong to another snapshot")
            article_ids = {u.unit_id for u in units if u.kind == "article"}
            adjacency, chunks_by_unit = {}, {}
            for relation in sorted(relations, key=lambda r: r.relation_id):
                if relation.kind in {"references", "amends"} and relation.resolution == "confirmed" and relation.target_id in article_ids:
                    adjacency.setdefault(relation.source_id, []).append(relation)
            for chunk in sorted(chunks, key=lambda c: c.chunk_id):
                for unit_id in set(chunk.unit_ids):
                    chunks_by_unit.setdefault(unit_id, []).append(chunk)
            self._graph = adjacency, chunks_by_unit
        adjacency, chunks_by_unit = self._graph
        seeds = {c.chunk.document_id for c in candidates[:3]} | {uid for c in candidates[:3] for uid in c.chunk.unit_ids}
        present = {c.chunk.chunk_id for c in candidates}
        additions, used = [], []
        related = [relation for seed in seeds for relation in adjacency.get(seed, [])]
        for relation in sorted(related, key=lambda r: r.relation_id):
            for chunk in chunks_by_unit.get(relation.target_id, []):
                if chunk.chunk_id in present or (config.exclude_ceased and chunk.status == "ceased_confirmed"):
                    continue
                if len(additions) >= config.graph_additions:
                    return additions, used
                additions.append(Candidate(chunk=chunk, rank=len(candidates) + len(additions) + 1, score=0, origin="graph"))
                present.add(chunk.chunk_id)
                used.append(relation.relation_id)
        return additions, used

    def verify_oracle(self, chunks: list[Chunk]):
        if not self.corpus_dir or not self.manifest.get("snapshot_id"):
            raise ValueError("Oracle verification requires an indexed corpus snapshot")
        expected = self.manifest.get("files", {}).get("chunks.jsonl")
        if not expected or file_hash(self.corpus_dir / "chunks.jsonl") != expected:
            raise ValueError("Oracle source snapshot differs from the index")
        source = {c.chunk_id: c for c in read_jsonl(self.corpus_dir / "chunks.jsonl", Chunk)}
        for chunk in chunks:
            if source.get(chunk.chunk_id) != chunk:
                raise ValueError("Oracle chunk is not an exact source record")
