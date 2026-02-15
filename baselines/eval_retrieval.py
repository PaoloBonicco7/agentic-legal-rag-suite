from __future__ import annotations

from pathlib import Path

from .artifacts import load_docs_from_out_dir
from .benchmark import Question
from .bm25 import BM25Index


def _question_to_query(q: Question, *, query_mode: str) -> str:
    if query_mode == "stem":
        return q.stem
    if query_mode == "stem+options":
        opts = "\n".join([f"{o.label}) {o.text}" for o in q.options])
        return (q.stem + "\n" + opts).strip()
    raise ValueError(f"Unknown query_mode: {query_mode!r}")


def evaluate_bm25_retrieval(
    *,
    questions: tuple[Question, ...],
    out_dir: Path,
    k_values: tuple[int, ...] = (1, 3, 5, 10, 20),
    query_mode: str = "stem+options",
    unit: str = "chunk",
    text_field: str = "text_for_embedding",
    k1: float = 1.2,
    b: float = 0.75,
) -> dict:
    k_values = tuple(sorted({int(k) for k in k_values if int(k) > 0}))
    if not k_values:
        raise ValueError("k_values must contain at least one positive integer")

    docs = load_docs_from_out_dir(out_dir, unit=unit, text_field=text_field)
    index = BM25Index(k1=k1, b=b)
    index.build(((d.doc_id, d.text, d.meta) for d in docs))

    max_k = max(k_values)

    per_k = {k: {"sum_recall": 0.0, "sum_hit": 0, "n": 0} for k in k_values}
    per_level = {}

    for q in questions:
        gold = {(gt.law_id, gt.article_label_norm) for gt in q.gold_targets}
        if not gold:
            continue
        query = _question_to_query(q, query_mode=query_mode)
        top = index.search(query, k=max_k)
        retrieved_keys = [sd.meta.get("article_key") for sd in top if sd.meta.get("article_key") is not None]

        for k in k_values:
            slice_keys = set(retrieved_keys[:k])
            hit = 1 if (slice_keys & gold) else 0
            recall = (len(slice_keys & gold) / len(gold)) if gold else 0.0

            per_k[k]["sum_recall"] += float(recall)
            per_k[k]["sum_hit"] += int(hit)
            per_k[k]["n"] += 1

            lvl = q.level or "UNKNOWN"
            per_level.setdefault(lvl, {}).setdefault(k, {"sum_recall": 0.0, "sum_hit": 0, "n": 0})
            per_level[lvl][k]["sum_recall"] += float(recall)
            per_level[lvl][k]["sum_hit"] += int(hit)
            per_level[lvl][k]["n"] += 1

    def finalize(stats: dict) -> dict:
        n = stats.get("n") or 0
        if not n:
            return {"n": 0, "recall": 0.0, "hit_rate": 0.0}
        return {"n": int(n), "recall": float(stats["sum_recall"]) / n, "hit_rate": float(stats["sum_hit"]) / n}

    return {
        "retrieval": {
            "method": "bm25",
            "k1": k1,
            "b": b,
            "unit": unit,
            "text_field": text_field,
            "query_mode": query_mode,
            "index_docs": index.size,
            "out_dir": str(out_dir),
        },
        "overall": {f"@{k}": finalize(per_k[k]) for k in k_values},
        "by_level": {
            lvl: {f"@{k}": finalize(kstats) for k, kstats in sorted(ks.items())}
            for lvl, ks in sorted(per_level.items())
        },
    }

