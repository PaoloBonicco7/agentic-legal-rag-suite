from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from legal_indexing.io import iter_jsonl

from .qdrant_retrieval import RetrievedChunk


def _safe_article_id(law_id: str | None, article_label_norm: str | None) -> str | None:
    if not law_id or not article_label_norm:
        return None
    lid = str(law_id).strip()
    alabel = str(article_label_norm).strip()
    if not lid or not alabel:
        return None
    return f"{lid}#art:{alabel}"


@dataclass(frozen=True)
class GraphExpansionResult:
    seed_chunk_ids: tuple[str, ...]
    seed_law_ids: tuple[str, ...]
    seed_article_ids: tuple[str, ...]
    seed_passage_ids: tuple[str, ...]
    related_law_ids: tuple[str, ...]
    related_article_ids: tuple[str, ...]
    related_chunk_ids: tuple[str, ...]
    edge_hits: int
    event_hits: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_chunk_ids": list(self.seed_chunk_ids),
            "seed_law_ids": list(self.seed_law_ids),
            "seed_article_ids": list(self.seed_article_ids),
            "seed_passage_ids": list(self.seed_passage_ids),
            "related_law_ids": list(self.related_law_ids),
            "related_article_ids": list(self.related_article_ids),
            "related_chunk_ids": list(self.related_chunk_ids),
            "edge_hits": self.edge_hits,
            "event_hits": self.event_hits,
        }


class LegalGraphAdapter:
    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir
        self._edges_loaded = False
        self._events_loaded = False
        self._law_neighbors_from_edges: dict[str, set[str]] = defaultdict(set)
        self._law_neighbors_from_events: dict[str, set[str]] = defaultdict(set)
        self._related_article_ids: dict[str, set[str]] = defaultdict(set)

    @property
    def edges_path(self) -> Path:
        return self.dataset_dir / "edges.jsonl"

    @property
    def events_path(self) -> Path:
        return self.dataset_dir / "events.jsonl"

    def _load_edges(self) -> None:
        if self._edges_loaded:
            return
        if not self.edges_path.exists():
            self._edges_loaded = True
            return

        for edge in iter_jsonl(self.edges_path):
            src = str(edge.get("src_law_id") or "").strip()
            dst = str(edge.get("dst_law_id") or "").strip()
            if not src or not dst:
                continue
            self._law_neighbors_from_edges[src].add(dst)
            self._law_neighbors_from_edges[dst].add(src)

            dst_article = _safe_article_id(dst, edge.get("dst_article_label_norm"))
            if dst_article:
                self._related_article_ids[src].add(dst_article)
                self._related_article_ids[dst].add(dst_article)
        self._edges_loaded = True

    def _load_events(self) -> None:
        if self._events_loaded:
            return
        if not self.events_path.exists():
            self._events_loaded = True
            return

        for event in iter_jsonl(self.events_path):
            src = str(event.get("source_law_id") or "").strip()
            dst = str(event.get("target_law_id") or "").strip()
            if src and dst:
                self._law_neighbors_from_events[src].add(dst)
                self._law_neighbors_from_events[dst].add(src)

            target_article = _safe_article_id(dst, event.get("target_article_label_norm"))
            if src and target_article:
                self._related_article_ids[src].add(target_article)
            if dst and target_article:
                self._related_article_ids[dst].add(target_article)
        self._events_loaded = True

    def expand_from_retrieved(
        self, retrieved: list[RetrievedChunk], *, max_related_laws: int
    ) -> GraphExpansionResult:
        self._load_edges()
        self._load_events()

        seed_chunk_ids = sorted(
            {x for x in [doc.chunk_id for doc in retrieved] if x is not None and x.strip()}
        )
        seed_law_ids = sorted(
            {x for x in [doc.law_id for doc in retrieved] if x is not None and x.strip()}
        )
        seed_article_ids = sorted(
            {x for x in [doc.article_id for doc in retrieved] if x is not None and x.strip()}
        )
        seed_passage_ids = sorted(
            {
                pid
                for doc in retrieved
                for pid in list(doc.source_passage_ids)
                if pid is not None and str(pid).strip()
            }
        )
        seed_set = set(seed_law_ids)

        related_laws: set[str] = set()
        related_articles: set[str] = set()
        edge_hits = 0
        event_hits = 0

        for law_id in seed_law_ids:
            edge_neighbors = self._law_neighbors_from_edges.get(law_id, set())
            event_neighbors = self._law_neighbors_from_events.get(law_id, set())
            edge_hits += len(edge_neighbors)
            event_hits += len(event_neighbors)
            related_laws.update(edge_neighbors)
            related_laws.update(event_neighbors)
            related_articles.update(self._related_article_ids.get(law_id, set()))

        related_laws = {law_id for law_id in related_laws if law_id not in seed_set}
        related_laws_sorted = sorted(related_laws)[: max(1, int(max_related_laws))]

        related_chunk_ids: set[str] = set()
        for doc in retrieved:
            related_chunk_ids.update({x for x in doc.source_chunk_ids if x})

        return GraphExpansionResult(
            seed_chunk_ids=tuple(seed_chunk_ids),
            seed_law_ids=tuple(seed_law_ids),
            seed_article_ids=tuple(seed_article_ids),
            seed_passage_ids=tuple(seed_passage_ids),
            related_law_ids=tuple(related_laws_sorted),
            related_article_ids=tuple(sorted(related_articles)),
            related_chunk_ids=tuple(sorted(related_chunk_ids)),
            edge_hits=edge_hits,
            event_hits=event_hits,
        )
