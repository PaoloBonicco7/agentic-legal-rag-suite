from __future__ import annotations

import hashlib
import re


_RELATION_PRIORITY: list[tuple[str, re.Pattern[str]]] = [
    ("ABROGATED_BY", re.compile(r"\babrogat[oa]\b.*\bdall['a-z]*\b|\blegge\s+abrogata\b", re.IGNORECASE)),
    ("ABROGATES", re.compile(r"\babroga\b|\babrogano\b|\bsono\s+abrogat", re.IGNORECASE)),
    ("REPLACED_BY", re.compile(r"\bsostituit[oa]\b.*\bdall['a-z]*\b", re.IGNORECASE)),
    ("REPLACES", re.compile(r"\bsostituisc\w*\b|\bsostituit[oa]\b", re.IGNORECASE)),
    ("INSERTED_BY", re.compile(r"\binserit[oa]\b.*\bdall['a-z]*\b", re.IGNORECASE)),
    ("INSERTS", re.compile(r"\binserisc\w*\b|\baggiung\w*\b", re.IGNORECASE)),
    ("MODIFIED_BY", re.compile(r"\bmodificat[oa]\b.*\bdall['a-z]*\b", re.IGNORECASE)),
    ("AMENDS", re.compile(r"\bmodific\w*\b", re.IGNORECASE)),
    ("REFERENCES", re.compile(r"\bart\.?\s*\d+|\blegge\s+regionale\b|\bl\.r\.?\b|\bl:r\.?\b", re.IGNORECASE)),
]


def _norm_relation_type(edge_type: str, evidence_text: str) -> str:
    txt = (evidence_text or "").strip()
    for rel, regex in _RELATION_PRIORITY:
        if regex.search(txt):
            return rel

    base = (edge_type or "").strip().upper()
    if base == "ABROGATED_BY":
        return "ABROGATED_BY"
    if base == "AMENDED_BY":
        return "MODIFIED_BY"
    if base == "REFERS_TO":
        return "REFERENCES"
    return "REFERS_TO"


def _scope(src_article_id: str | None, src_passage_id: str | None) -> str:
    if src_passage_id:
        return "passage"
    if src_article_id:
        return "article"
    return "law"


def _src_article_label_norm(article_id: str | None) -> str | None:
    if not article_id or "#art:" not in article_id:
        return None
    return article_id.split("#art:", 1)[1]


def _edge_norm_id(edge: dict) -> str:
    h = hashlib.sha256()
    parts = [
        str(edge.get("src_law_id") or ""),
        str(edge.get("src_article_id") or ""),
        str(edge.get("src_passage_id") or ""),
        str(edge.get("dst_law_id") or ""),
        str(edge.get("dst_article_label_norm") or ""),
        str(edge.get("relation_type") or ""),
        str(edge.get("relation_scope") or ""),
    ]
    h.update("|".join(parts).encode("utf-8"))
    return h.hexdigest()


def normalize_edges(edges: list[dict]) -> tuple[list[dict], list[dict], dict[str, int]]:
    raw: list[dict] = []
    seen: set[str] = set()
    unresolved = 0

    for edge in sorted(edges, key=lambda x: str(x.get("edge_id") or "")):
        src_law_id = edge.get("src_law_id")
        dst_law_id = edge.get("dst_law_id")
        if not src_law_id or not dst_law_id:
            unresolved += 1
            continue

        relation_type = _norm_relation_type(str(edge.get("edge_type") or ""), str(edge.get("evidence_text") or ""))
        rec = dict(edge)
        rec["relation_type"] = relation_type
        rec["relation_scope"] = _scope(rec.get("src_article_id"), rec.get("src_passage_id"))
        rec["src_article_label_norm"] = _src_article_label_norm(rec.get("src_article_id"))
        rec["is_self_loop"] = bool(src_law_id == dst_law_id)
        rec["confidence"] = float(rec.get("confidence") or 0.0)
        rec["evidence"] = str(rec.get("evidence_text") or "").strip()[:500]
        rec["norm_edge_id"] = _edge_norm_id(rec)

        if rec["norm_edge_id"] in seen:
            continue
        seen.add(rec["norm_edge_id"])
        raw.append(rec)

    clean = [r for r in raw if not r.get("is_self_loop")]
    clean.sort(key=lambda x: str(x.get("norm_edge_id") or ""))

    stats = {
        "edges_raw": len(raw),
        "edges_clean": len(clean),
        "self_loops_raw": sum(1 for r in raw if r.get("is_self_loop")),
        "unresolved_refs_relations": unresolved,
    }
    return raw, clean, stats
