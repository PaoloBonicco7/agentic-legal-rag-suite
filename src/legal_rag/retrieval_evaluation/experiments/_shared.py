"""Helpers shared by the retrieval-diagnostics experiment engines."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, TypeVar

from legal_rag.simple_rag.models import RetrievedChunkRecord

from ..evaluator import answer_overlap
from ..models import QuestionTarget

T = TypeVar("T")
U = TypeVar("U")


def compute_answer_overlap_metrics(
    target: QuestionTarget,
    candidates: Sequence[RetrievedChunkRecord],
    *,
    prefix: str,
) -> dict[str, Any]:
    """Diagnostic answer/article overlap for the expected article hits.

    Used by the direct/graph/query-rewriting experiments to score how strongly
    each retrieved chunk shadows the gold answer. The metric is qualitative
    (Jaccard on long tokens) and is *not* used for ranking — it serves the
    failure-analysis section of the notebook.
    """
    expected_articles = set(target.expected_article_ids)
    article_hits = [
        chunk
        for chunk in candidates
        if str(chunk.payload.get("article_id") or "") in expected_articles
    ]
    overlaps = [answer_overlap(target.correct_answer, chunk.text) for chunk in article_hits]
    return {
        f"{prefix}_expected_article_hit_chunk_count": len(article_hits),
        f"{prefix}_first_article_hit_answer_overlap": overlaps[0] if overlaps else None,
        f"{prefix}_best_expected_article_answer_overlap": max(overlaps) if overlaps else None,
    }


def wrap_progress(iterable, *, description: str, enabled: bool):
    """Wrap an iterable with tqdm if available; otherwise return it unchanged."""
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, desc=description)
    except ImportError:
        return iterable


def parallel_map_ordered(
    items: Sequence[T],
    func: Callable[[T], U],
    *,
    max_workers: int,
    description: str,
    show_progress: bool = True,
    request_delay_seconds: float = 0.0,
) -> list[U]:
    """Run independent tasks in parallel and return results in input order.

    Falls back to a serial loop when `max_workers <= 1` or only one item is given.
    `request_delay_seconds` enforces a minimum spacing between task starts across
    all workers — a coarse global rate-limit useful when the downstream service
    (e.g. a shared LLM endpoint) is flaky under bursty load.
    """
    total = len(items)
    delay = max(0.0, float(request_delay_seconds))

    if max_workers <= 1 or total <= 1:
        iterator = wrap_progress(items, description=description, enabled=show_progress)
        out: list[U] = []
        last_start = 0.0
        for item in iterator:
            if delay > 0 and out:
                wait = delay - (time.monotonic() - last_start)
                if wait > 0:
                    time.sleep(wait)
            last_start = time.monotonic()
            out.append(func(item))
        return out

    workers = min(max_workers, total)
    results: list[U | None] = [None] * total
    progress = None
    if show_progress:
        try:
            from tqdm.auto import tqdm

            progress = tqdm(total=total, desc=description)
        except ImportError:
            progress = None

    pacing_lock = threading.Lock()
    next_start = [time.monotonic()]

    def throttled(item: T) -> U:
        if delay > 0:
            with pacing_lock:
                now = time.monotonic()
                wait = next_start[0] - now
                if wait > 0:
                    time.sleep(wait)
                    now = time.monotonic()
                next_start[0] = now + delay
        return func(item)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(throttled, item): idx for idx, item in enumerate(items)}
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            if progress is not None:
                progress.update(1)
    if progress is not None:
        progress.close()
    return [r for r in results]  # type: ignore[misc]
