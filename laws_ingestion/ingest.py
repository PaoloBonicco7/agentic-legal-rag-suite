from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import re
from typing import Iterator

from .chunking import chunk_text_words
from .html_blocks import Block, Link, parse_blocks_from_html
from .references import (
    ResolvedLawRef,
    extract_dst_article_label_norm,
    extract_note_anchor_names_from_hrefs,
    resolve_ref_from_href_and_text,
    resolve_refs_from_text,
)
from .registry import CorpusRegistry, LawFile
from .utils import normalize_article_label

_HEADING_RE = re.compile(r"^(PARTE|TITOLO|CAPO|SEZIONE)\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE)
_LAW_HEADER_RE = re.compile(r"\blegge\s+regionale\b.*?,\s*n\.\s*\d+\b", re.IGNORECASE)
_ABROGATION_MARKER_RE = re.compile(r"\b(legge\s+abrogata|\(\s*abrogata\b|abrogata\s+dall')\b", re.IGNORECASE)

_PLAIN_ARTICLE_RE = re.compile(
    r"^(ARTICOLO|Articolo|ART\.)\s+(?P<label>\d+(?:\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?|unico)\b(?P<rest>.*)$",
    re.IGNORECASE,
)

_COMMA_START_RE = re.compile(
    r"^(?P<num>\d+)(?P<suf>bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)?\.\s+",
    re.IGNORECASE,
)
_LETTER_START_RE = re.compile(r"^(?P<lettera>[a-z])\)\s+", re.IGNORECASE)


def _heading_level(text: str) -> int | None:
    m = _HEADING_RE.match((text or "").strip())
    if not m:
        return None
    kind = m.group(1).upper()
    return {"PARTE": 1, "TITOLO": 2, "CAPO": 3, "SEZIONE": 4}.get(kind)


def _is_noise_line(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if t in {"¦"}:
        return True
    if re.fullmatch(r"[_\-]{5,}", t):
        return True
    if re.fullmatch(r"\(\d+\)\.?", t):
        return True
    return False


def _extract_law_title(blocks: list[Block]) -> str:
    for b in blocks:
        if b.kind not in ("h1", "h2"):
            continue
        if _LAW_HEADER_RE.search(b.text):
            idx = b.text.lower().find("legge regionale")
            return b.text[idx:].strip() if idx >= 0 else b.text.strip()
    # fallback: first h1/h2
    for b in blocks:
        if b.kind in ("h1", "h2") and (b.text or "").strip():
            return b.text.strip()
    return ""


def _edge_id(
    src_law_id: str,
    src_article_id: str | None,
    src_passage_id: str | None,
    edge_type: str,
    dst_law_id: str,
    dst_article_label_norm: str | None,
    extraction_method: str,
    evidence_text: str,
) -> str:
    h = hashlib.sha256()
    h.update((src_law_id or "").encode("utf-8"))
    h.update(b"|")
    h.update((src_article_id or "").encode("utf-8"))
    h.update(b"|")
    h.update((src_passage_id or "").encode("utf-8"))
    h.update(b"|")
    h.update(edge_type.encode("utf-8"))
    h.update(b"|")
    h.update(dst_law_id.encode("utf-8"))
    h.update(b"|")
    h.update((dst_article_label_norm or "").encode("utf-8"))
    h.update(b"|")
    h.update(extraction_method.encode("utf-8"))
    h.update(b"|")
    h.update((evidence_text or "").encode("utf-8"))
    return h.hexdigest()


def _classify_relation_type(context: str, evidence_text: str) -> tuple[str, float]:
    t = (evidence_text or "").lower()
    if "abrogat" in t:
        return ("ABROGATED_BY", 0.9) if context in {"preamble", "note"} else ("ABROGATED_BY", 0.7)
    if any(k in t for k in ("modificat", "sostituit", "inserit")):
        return ("AMENDED_BY", 0.7) if context == "note" else ("AMENDED_BY", 0.6)
    return "REFERS_TO", 0.4


def _note_definition_in_block(b: Block) -> tuple[str, str, str] | None:
    """
    Detect start of a note definition block.
    Returns (note_anchor_name, note_number, rest_text).
    """
    for a in b.anchors:
        if not a.name.lower().startswith("nota_"):
            continue
        rest = (b.text or "").strip()
        if a.text and rest.startswith(a.text):
            rest = rest[len(a.text) :].strip()
        rest = rest.lstrip(".").strip()
        if not rest:
            continue
        note_anchor_name = a.name.strip()
        note_number = note_anchor_name.split("nota_", 1)[-1] or ""
        return note_anchor_name, note_number, rest
    return None


def _article_anchor_in_block(b: Block) -> tuple[str, str, str | None] | None:
    """
    Detect an anchored article start.
    Returns (anchor_name, article_label_raw, heading_or_none).
    """
    for a in b.anchors:
        if not a.name.lower().startswith("articolo_"):
            continue
        if not (a.text or "").lower().startswith("art"):
            continue
        label_norm = normalize_article_label(a.text)
        rest = (b.text or "").strip()
        if a.text and rest.startswith(a.text):
            rest = rest[len(a.text) :].strip()
        rest = rest.lstrip(".-:– ").strip()
        heading = rest if rest else None
        return a.name, label_norm, heading
    return None


def _plain_article_in_block(b: Block) -> tuple[str, str | None] | None:
    """
    Fallback for older HTML variants where articles are not anchored.
    Returns (article_label_norm, heading_or_none).
    """
    t = (b.text or "").strip()
    m = _PLAIN_ARTICLE_RE.match(t)
    if not m:
        return None
    label_raw = m.group("label").strip()
    label_norm = normalize_article_label(label_raw)
    rest = (m.group("rest") or "").strip()
    rest = rest.lstrip(".-:– ").strip()
    heading = rest if rest else None
    return label_norm, heading


def _links_out(links: list[Link]) -> list[dict]:
    out = []
    for l in links:
        href = (l.href or "").strip()
        if not href:
            continue
        out.append({"href": href, "text": (l.text or "").strip()})
    return out


@dataclass
class _Line:
    text: str
    links: tuple[Link, ...]


@dataclass
class IngestedLaw:
    law: dict
    articles: list[dict]
    passages: list[dict]
    notes: list[dict]
    edges: list[dict]
    chunks: list[dict]
    parser_backend: str
    unresolved_refs: int
    warnings: list[str]


def ingest_law(
    law_file: LawFile,
    registry: CorpusRegistry,
    *,
    backend: str = "auto",
    strict: bool = False,
    max_words: int = 600,
    overlap_words: int = 80,
) -> IngestedLaw:
    html = law_file.path.read_text(encoding="utf-8", errors="replace")
    blocks, backend_used = parse_blocks_from_html(html, backend=backend)

    warnings: list[str] = []
    unresolved_refs = 0

    law_title = _extract_law_title(blocks) or f"Legge regionale {law_file.law_date.isoformat()}, n. {law_file.law_number}"

    structure: list[str] = []
    preamble_lines: list[_Line] = []

    # Articles collected as drafts (later expanded into passages)
    articles: list[dict] = []
    article_lines_by_id: dict[str, list[_Line]] = {}
    article_order: list[str] = []

    # Note definitions (text) collected now; linked ids filled after passages build
    notes: list[dict] = []
    note_lines_by_anchor: dict[str, list[_Line]] = {}

    # Mapping from note_anchor_name -> linked article/passages (built while creating passages)
    note_links_articles: dict[str, set[str]] = {}
    note_links_passages: dict[str, set[str]] = {}

    edges: list[dict] = []

    def add_edges_from_text(
        *,
        context: str,
        src_article_id: str | None,
        src_passage_id: str | None,
        evidence_text: str,
        links: tuple[Link, ...],
        source_file: str,
        note_anchor_name: str | None = None,
    ) -> None:
        nonlocal unresolved_refs, edges

        # href-driven refs
        href_refs: list[ResolvedLawRef] = []
        for l in links:
            if not (l.href or ""):
                continue
            if "numero_legge=" not in (l.href or ""):
                continue
            ref, u = resolve_ref_from_href_and_text(l.href, l.text, registry)
            unresolved_refs += u
            if ref:
                href_refs.append(ref)

        # text-regex refs
        text_refs, u2 = resolve_refs_from_text(evidence_text, registry)
        unresolved_refs += u2

        # Dedup per dst law_id, keep first (prefer href)
        merged: list[ResolvedLawRef] = []
        seen: set[str] = set()
        for r in href_refs + text_refs:
            if r.law_id in seen:
                continue
            seen.add(r.law_id)
            merged.append(r)

        dst_article_label_norm = extract_dst_article_label_norm(evidence_text)
        edge_type, conf = _classify_relation_type(context, evidence_text)

        for r in merged:
            edges.append(
                {
                    "edge_id": _edge_id(
                        src_law_id=law_file.law_id,
                        src_article_id=src_article_id,
                        src_passage_id=src_passage_id,
                        edge_type=edge_type,
                        dst_law_id=r.law_id,
                        dst_article_label_norm=dst_article_label_norm,
                        extraction_method=r.extraction_method,
                        evidence_text=evidence_text.strip(),
                    ),
                    "edge_type": edge_type,
                    "src_law_id": law_file.law_id,
                    "src_article_id": src_article_id,
                    "src_passage_id": src_passage_id,
                    "dst_law_id": r.law_id,
                    "dst_article_label_norm": dst_article_label_norm,
                    "context": context,
                    "extraction_method": r.extraction_method,
                    "evidence_text": evidence_text.strip(),
                    "confidence": conf,
                    "source_file": source_file,
                    "note_anchor_name": note_anchor_name,
                }
            )

    saw_first_article = False
    note_mode = False
    current_article_id: str | None = None

    def flush_article_if_any() -> None:
        nonlocal current_article_id
        current_article_id = None

    # Collect notes as blocks (definition boundaries)
    current_note_anchor: str | None = None

    def flush_note_if_any() -> None:
        nonlocal current_note_anchor
        current_note_anchor = None

    for b in blocks:
        if _is_noise_line(b.text):
            continue

        lvl = _heading_level(b.text)
        if lvl:
            heading = b.text.strip()
            while len(structure) < lvl:
                structure.append("")
            structure[lvl - 1] = heading
            structure = structure[:lvl]
            continue

        # Skip TOC/index entries early (usually internal href="#articolo_...")
        if not saw_first_article and any((l.href or "").startswith("#articolo_") for l in b.links):
            continue

        note_def = _note_definition_in_block(b)
        if note_def:
            note_mode = True
            flush_article_if_any()
            flush_note_if_any()

            note_anchor_name, note_number, rest = note_def
            current_note_anchor = note_anchor_name
            notes.append(
                {
                    "note_id": f"{law_file.law_id}#note:{note_anchor_name}",
                    "law_id": law_file.law_id,
                    "note_anchor_name": note_anchor_name,
                    "note_number": note_number or None,
                    "note_kind": "",  # filled later
                    "note_text": "",
                    "linked_article_ids": [],
                    "linked_passage_ids": [],
                    "links_out": [],
                }
            )
            note_lines_by_anchor.setdefault(note_anchor_name, []).append(_Line(text=rest, links=b.links))
            add_edges_from_text(
                context="note",
                src_article_id=None,
                src_passage_id=None,
                evidence_text=b.text,
                links=b.links,
                source_file=law_file.source_file,
                note_anchor_name=note_anchor_name,
            )
            continue

        # Article starts: anchored
        art_anchor = _article_anchor_in_block(b)
        if art_anchor:
            note_mode = False
            flush_note_if_any()
            saw_first_article = True
            flush_article_if_any()

            anchor_name, label_norm, heading = art_anchor
            article_id = f"{law_file.law_id}#art:{label_norm}"
            current_article_id = article_id
            articles.append(
                {
                    "article_id": article_id,
                    "law_id": law_file.law_id,
                    "article_label_raw": f"Art. {label_norm}",
                    "article_label_norm": label_norm,
                    "anchor_name": anchor_name,
                    "structure_path": " > ".join([s for s in structure if s]),
                    "article_heading": heading,
                    "article_text": "",
                    "note_anchor_names": [],
                    "is_abrogated": False,
                    "abrogated_by": None,
                    "amended_by_law_ids": [],
                    "links_out": [],
                }
            )
            article_lines_by_id.setdefault(article_id, [])
            article_order.append(article_id)
            continue

        plain_art = _plain_article_in_block(b)
        if plain_art:
            note_mode = False
            flush_note_if_any()
            saw_first_article = True
            flush_article_if_any()

            label_norm, heading = plain_art
            article_id = f"{law_file.law_id}#art:{label_norm}"
            current_article_id = article_id
            articles.append(
                {
                    "article_id": article_id,
                    "law_id": law_file.law_id,
                    "article_label_raw": f"Articolo {label_norm}",
                    "article_label_norm": label_norm,
                    "anchor_name": None,
                    "structure_path": " > ".join([s for s in structure if s]),
                    "article_heading": heading,
                    "article_text": "",
                    "note_anchor_names": [],
                    "is_abrogated": False,
                    "abrogated_by": None,
                    "amended_by_law_ids": [],
                    "links_out": [],
                }
            )
            article_lines_by_id.setdefault(article_id, [])
            article_order.append(article_id)
            continue

        # Continue note mode
        if note_mode and current_note_anchor:
            note_lines_by_anchor.setdefault(current_note_anchor, []).append(_Line(text=b.text, links=b.links))
            add_edges_from_text(
                context="note",
                src_article_id=None,
                src_passage_id=None,
                evidence_text=b.text,
                links=b.links,
                source_file=law_file.source_file,
                note_anchor_name=current_note_anchor,
            )
            continue

        # Preamble vs article body
        if not saw_first_article:
            if (b.text or "").strip().upper() == "INDICE":
                continue
            if any((l.href or "").startswith("#articolo_") for l in b.links):
                continue
            preamble_lines.append(_Line(text=b.text, links=b.links))
            add_edges_from_text(
                context="preamble",
                src_article_id=None,
                src_passage_id=None,
                evidence_text=b.text,
                links=b.links,
                source_file=law_file.source_file,
            )
            continue

        if current_article_id:
            article_lines_by_id[current_article_id].append(_Line(text=b.text, links=b.links))
            # Edges from body are created later at passage granularity.
            continue

        # Unexpected content after first article but without a current article.
        if strict:
            raise ValueError(f"Unattached block after first article in {law_file.source_file}: {b.text[:80]!r}")
        warnings.append(f"Unattached block after first article (skipped): {law_file.source_file}")

    preamble_text = "\n".join([ln.text.strip() for ln in preamble_lines if (ln.text or "").strip()]).strip()

    # No articles found: create pseudo-article unico from preamble
    if not articles and preamble_text:
        article_id = f"{law_file.law_id}#art:unico"
        articles.append(
            {
                "article_id": article_id,
                "law_id": law_file.law_id,
                "article_label_raw": "Articolo unico",
                "article_label_norm": "unico",
                "anchor_name": None,
                "structure_path": "",
                "article_heading": None,
                "article_text": "",
                "note_anchor_names": [],
                "is_abrogated": False,
                "abrogated_by": None,
                "amended_by_law_ids": [],
                "links_out": [],
            }
        )
        article_lines_by_id[article_id] = [_Line(text=preamble_text, links=())]
        article_order.append(article_id)

    # Passages + edges at passage level
    passages: list[dict] = []
    chunks: list[dict] = []

    edges_by_id: dict[str, dict] = {e["edge_id"]: e for e in edges}

    def add_edge_record(e: dict) -> None:
        if e["edge_id"] in edges_by_id:
            return
        edges_by_id[e["edge_id"]] = e

    for art in articles:
        aid = art["article_id"]
        lines = article_lines_by_id.get(aid) or []
        if not lines:
            continue

        # Prepare passages
        cur_label = "intro"
        cur_kind = "intro"
        cur_text_lines: list[str] = []
        cur_links: list[Link] = []
        cur_note_anchors: list[str] = []
        cur_related: list[str] = []
        cur_relation_types: set[str] = set()
        cur_edge_events: list[tuple[str, float, str | None, str, list[ResolvedLawRef]]] = []

        comma_label: str | None = None

        def flush_passage() -> None:
            nonlocal cur_label, cur_kind, cur_text_lines, cur_links, cur_note_anchors, cur_related, cur_relation_types, cur_edge_events
            if not cur_text_lines:
                return
            passage_text = "\n".join([t for t in (s.strip() for s in cur_text_lines) if t]).strip()
            if not passage_text:
                cur_text_lines = []
                cur_links = []
                cur_note_anchors = []
                cur_related = []
                cur_relation_types = set()
                cur_edge_events = []
                return
            pid = f"{aid}#p:{cur_label}"
            passages.append(
                {
                    "passage_id": pid,
                    "article_id": aid,
                    "law_id": art["law_id"],
                    "passage_label": cur_label,
                    "passage_kind": cur_kind,
                    "passage_text": passage_text,
                    "note_anchor_names": sorted(set(cur_note_anchors)),
                    "links_out": _links_out(cur_links),
                    "related_law_ids": sorted(set(cur_related)),
                    "relation_types": sorted(cur_relation_types),
                }
            )

            # passage-level edges
            for edge_type, conf, dst_article_label_norm, evidence_text, refs in cur_edge_events:
                for r in refs:
                    e = {
                        "edge_id": _edge_id(
                            src_law_id=law_file.law_id,
                            src_article_id=aid,
                            src_passage_id=pid,
                            edge_type=edge_type,
                            dst_law_id=r.law_id,
                            dst_article_label_norm=dst_article_label_norm,
                            extraction_method=r.extraction_method,
                            evidence_text=evidence_text,
                        ),
                        "edge_type": edge_type,
                        "src_law_id": law_file.law_id,
                        "src_article_id": aid,
                        "src_passage_id": pid,
                        "dst_law_id": r.law_id,
                        "dst_article_label_norm": dst_article_label_norm,
                        "context": "passage",
                        "extraction_method": r.extraction_method,
                        "evidence_text": evidence_text,
                        "confidence": conf,
                        "source_file": law_file.source_file,
                        "note_anchor_name": None,
                    }
                    add_edge_record(e)

            # link notes
            for na in set(cur_note_anchors):
                note_links_articles.setdefault(na, set()).add(aid)
                note_links_passages.setdefault(na, set()).add(pid)

            # chunks from passage
            for i, chunk in enumerate(chunk_text_words(passage_text, max_words=max_words, overlap_words=overlap_words)):
                prefix = (
                    f"[LR {law_file.law_date.isoformat()} n.{law_file.law_number}] {law_title} | "
                    f"Art. {art.get('article_label_norm')} | {cur_label} | {art.get('structure_path') or ''}"
                ).strip()
                chunks.append(
                    {
                        "chunk_id": f"{pid}#chunk:{i}",
                        "passage_id": pid,
                        "article_id": aid,
                        "law_id": art["law_id"],
                        "chunk_seq": i,
                        "text": chunk,
                        "text_for_embedding": f"{prefix}\n\n{chunk}".strip(),
                        "law_date": law_file.law_date.isoformat(),
                        "law_number": law_file.law_number,
                        "law_title": law_title,
                        "law_status": None,  # filled later
                        "article_label_norm": art.get("article_label_norm"),
                        "article_is_abrogated": False,  # filled later
                        "passage_label": cur_label,
                        "related_law_ids": sorted(set(cur_related)),
                        "relation_types": sorted(cur_relation_types),
                    }
                )

            # reset
            cur_text_lines = []
            cur_links = []
            cur_note_anchors = []
            cur_related = []
            cur_relation_types = set()
            cur_edge_events = []

        for ln in lines:
            txt = (ln.text or "").strip()
            if not txt:
                continue
            # passage boundary detection
            m_comma = _COMMA_START_RE.match(txt)
            m_letter = _LETTER_START_RE.match(txt)
            if m_comma:
                flush_passage()
                num = m_comma.group("num")
                suf = (m_comma.group("suf") or "").lower()
                comma_label = f"c{int(num)}{suf}"
                cur_label = comma_label
                cur_kind = "comma"
            elif m_letter:
                flush_passage()
                lettera = m_letter.group("lettera").lower()
                if comma_label:
                    cur_label = f"{comma_label}.lit_{lettera}"
                else:
                    cur_label = f"lit_{lettera}"
                cur_kind = "lettera"

            cur_text_lines.append(txt)
            cur_links.extend(list(ln.links))

            hrefs = [l.href for l in ln.links if l.href]
            cur_note_anchors.extend(extract_note_anchor_names_from_hrefs(hrefs))

            # references and edges per line (passage context)
            # href refs
            href_refs: list[ResolvedLawRef] = []
            for l in ln.links:
                if not (l.href or "") or "numero_legge=" not in (l.href or ""):
                    continue
                ref, u = resolve_ref_from_href_and_text(l.href, l.text, registry)
                unresolved_refs += u
                if ref:
                    href_refs.append(ref)
            text_refs, u2 = resolve_refs_from_text(txt, registry)
            unresolved_refs += u2

            merged: list[ResolvedLawRef] = []
            seen_dst: set[str] = set()
            for r in href_refs + text_refs:
                if r.law_id in seen_dst:
                    continue
                seen_dst.add(r.law_id)
                merged.append(r)

            if merged:
                edge_type, conf = _classify_relation_type("passage", txt)
                cur_relation_types.add(edge_type)
                cur_related.extend([r.law_id for r in merged])
                dst_article_label_norm = extract_dst_article_label_norm(txt)
                cur_edge_events.append((edge_type, conf, dst_article_label_norm, txt, merged))

        flush_passage()

        # Article fields
        art_text = "\n".join([ln.text.strip() for ln in lines if (ln.text or "").strip()]).strip()
        art["article_text"] = art_text
        art["links_out"] = _links_out([l for ln in lines for l in ln.links])
        # note anchors aggregated from passages later
        # We'll fill note_anchor_names once passages created for this article.

    # Fill article note_anchor_names from passages
    by_article_notes: dict[str, set[str]] = {}
    for p in passages:
        by_article_notes.setdefault(p["article_id"], set()).update(p.get("note_anchor_names") or [])
    for art in articles:
        art["note_anchor_names"] = sorted(by_article_notes.get(art["article_id"], set()))

    # Notes: fill text, classify kind, attach links, attach linked ids, compute abrog/amend refs for status
    def classify_note_kind(text: str) -> str:
        t = (text or "").lower()
        if "comma modific" in t or "modificat" in t or "sostituisc" in t or "sostituit" in t:
            return "modified"
        if "comma inser" in t or "inserit" in t:
            return "inserted"
        if "comma abrog" in t or "articolo abrog" in t or "abrogat" in t:
            return "abrogated"
        return "other"

    note_by_anchor: dict[str, dict] = {n["note_anchor_name"]: n for n in notes}
    note_refs_by_anchor: dict[str, list[str]] = {}
    note_abrogated_by: dict[str, dict | None] = {}

    for anchor, nlines in note_lines_by_anchor.items():
        n = note_by_anchor.get(anchor)
        if not n:
            continue
        text = "\n".join([ln.text.strip() for ln in nlines if (ln.text or "").strip()]).strip()
        n["note_text"] = text
        n["note_kind"] = classify_note_kind(text)
        n["links_out"] = _links_out([l for ln in nlines for l in ln.links])
        n["linked_article_ids"] = sorted(note_links_articles.get(anchor, set()))
        n["linked_passage_ids"] = sorted(note_links_passages.get(anchor, set()))

        # Extract resolved refs from note to populate amended/abrogated fields on articles
        resolved: set[str] = set()
        for ln in nlines:
            href_refs: list[ResolvedLawRef] = []
            for l in ln.links:
                if not (l.href or "") or "numero_legge=" not in (l.href or ""):
                    continue
                ref, u = resolve_ref_from_href_and_text(l.href, l.text, registry)
                unresolved_refs += u
                if ref:
                    href_refs.append(ref)
            text_refs, u2 = resolve_refs_from_text(ln.text, registry)
            unresolved_refs += u2
            for r in href_refs + text_refs:
                resolved.add(r.law_id)
        note_refs_by_anchor[anchor] = sorted(resolved)

        if n["note_kind"] == "abrogated" and resolved:
            # best-effort: take first (stable) resolved as abrogating law
            dst = sorted(resolved)[0]
            note_abrogated_by[anchor] = {
                "law_id": dst,
                "article_label_norm": extract_dst_article_label_norm(text),
                "evidence_text": text[:500],
                "confidence": 0.9,
            }
        else:
            note_abrogated_by[anchor] = None

    # Propagate note edges to linked passages (so edges can be followed from retrieved chunks)
    base_note_edges = [
        e
        for e in list(edges_by_id.values())
        if e.get("context") == "note" and e.get("note_anchor_name") and not e.get("src_passage_id")
    ]
    for base in base_note_edges:
        anchor = base.get("note_anchor_name") or ""
        linked_pids = sorted(note_links_passages.get(anchor, set()))
        if not linked_pids:
            linked_aids = sorted(note_links_articles.get(anchor, set()))
            for aid in linked_aids:
                e2 = dict(base)
                e2["src_article_id"] = aid
                e2["src_passage_id"] = None
                e2["edge_id"] = _edge_id(
                    src_law_id=base.get("src_law_id") or law_file.law_id,
                    src_article_id=aid,
                    src_passage_id=None,
                    edge_type=base.get("edge_type") or "REFERS_TO",
                    dst_law_id=base.get("dst_law_id") or "",
                    dst_article_label_norm=base.get("dst_article_label_norm"),
                    extraction_method=base.get("extraction_method") or "href",
                    evidence_text=base.get("evidence_text") or "",
                )
                add_edge_record(e2)
            continue

        for pid in linked_pids:
            aid = pid.split("#p:", 1)[0]
            e2 = dict(base)
            e2["src_article_id"] = aid
            e2["src_passage_id"] = pid
            e2["edge_id"] = _edge_id(
                src_law_id=base.get("src_law_id") or law_file.law_id,
                src_article_id=aid,
                src_passage_id=pid,
                edge_type=base.get("edge_type") or "REFERS_TO",
                dst_law_id=base.get("dst_law_id") or "",
                dst_article_label_norm=base.get("dst_article_label_norm"),
                extraction_method=base.get("extraction_method") or "href",
                evidence_text=base.get("evidence_text") or "",
            )
            add_edge_record(e2)

    # Apply note-driven status to articles
    notes_by_article: dict[str, list[str]] = {}
    for anchor, aids in note_links_articles.items():
        for aid in aids:
            notes_by_article.setdefault(aid, []).append(anchor)

    for art in articles:
        aid = art["article_id"]
        anchors = notes_by_article.get(aid, [])
        amended_by: set[str] = set()
        abrog: dict | None = None
        for na in anchors:
            n = note_by_anchor.get(na)
            if not n:
                continue
            for law_id in note_refs_by_anchor.get(na, []):
                if n.get("note_kind") in {"modified", "inserted"}:
                    amended_by.add(law_id)
            if not abrog and note_abrogated_by.get(na):
                abrog = note_abrogated_by[na]
        art["amended_by_law_ids"] = sorted(amended_by)
        if abrog:
            art["is_abrogated"] = True
            art["abrogated_by"] = abrog

    # Law status
    is_abrogated = bool(_ABROGATION_MARKER_RE.search(preamble_text or ""))
    law_status = "abrogated" if is_abrogated else "in_force"
    law_abrogated_by = None
    if is_abrogated:
        # pick first preamble ABROGATED_BY edge if any
        candidates = [
            e for e in edges_by_id.values() if e.get("context") == "preamble" and e.get("edge_type") == "ABROGATED_BY"
        ]
        if candidates:
            c = candidates[0]
            law_abrogated_by = {
                "law_id": c.get("dst_law_id"),
                "article_label_norm": c.get("dst_article_label_norm"),
                "evidence_text": (c.get("evidence_text") or "")[:500],
                "confidence": c.get("confidence"),
            }

    law_record = {
        "law_id": law_file.law_id,
        "law_type": "LR",
        "law_date": law_file.law_date.isoformat(),
        "law_number": law_file.law_number,
        "law_title": law_title,
        "source_file": law_file.source_file,
        "preamble_text": preamble_text,
        "status": law_status,
        "abrogated_by": law_abrogated_by,
        "links_out": _links_out([l for ln in preamble_lines for l in ln.links]),
    }

    # Fill denormalized fields in chunks now that law/article status exists
    article_abrogated: dict[str, bool] = {a["article_id"]: bool(a.get("is_abrogated")) for a in articles}
    for c in chunks:
        c["law_status"] = law_status
        c["article_is_abrogated"] = article_abrogated.get(c["article_id"], False)

    # Dedup edges final
    edges_out = list(edges_by_id.values())
    edges_out.sort(key=lambda e: e["edge_id"])

    return IngestedLaw(
        law=law_record,
        articles=articles,
        passages=passages,
        notes=notes,
        edges=edges_out,
        chunks=chunks,
        parser_backend=backend_used,
        unresolved_refs=unresolved_refs,
        warnings=warnings,
    )


def iter_chunks_for_law(law_ingested: IngestedLaw) -> Iterator[dict]:
    for c in law_ingested.chunks:
        yield c
