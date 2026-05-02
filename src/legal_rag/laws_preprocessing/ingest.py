"""Per-law ingestion logic for phase 1 laws preprocessing."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .common import normalize_article_label, normalize_ws
from .html import parse_blocks_from_html
from .models import (
    Block,
    CorpusRegistry,
    IngestedLaw,
    LawFile,
    Line,
    Link,
    article_record,
    chunk_record,
    edge_record,
    law_record,
    note_record,
    passage_record,
)
from .references import (
    classify_relation_type,
    edge_id as make_edge_id,
    extract_dst_article_label_norm,
    resolve_ref_from_href_and_text,
    resolve_refs_from_text,
)

LAW_HEADER_RE = re.compile(r"\blegge\s+regionale\b.*?,\s*n\.\s*\d+\b", re.IGNORECASE)
HEADING_RE = re.compile(r"^(PARTE|TITOLO|CAPO|SEZIONE)\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE)
PLAIN_ARTICLE_RE = re.compile(
    r"^(ARTICOLO|Articolo|ART\.)\s+"
    r"(?P<label>\d+(?:\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?|unico)\b"
    r"(?P<rest>.*)$",
    re.IGNORECASE,
)
TOC_ARTICLE_LINE_RE = re.compile(r"^(ARTICOLO|Articolo|ART\.?)\s+\d+\b", re.IGNORECASE)
COMMA_START_RE = re.compile(
    r"^(?P<num>\d+)(?P<suf>bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)?\.\s+",
    re.IGNORECASE,
)
LETTER_START_RE = re.compile(r"^(?P<lettera>[a-z])\)\s+", re.IGNORECASE)
ABROGATION_FULL_RE = re.compile(
    r"(legge\s+abrogata|\babrogat[oa]\b.*\bdall[ao]\b|\bnon\s+pi[uù]\s+in\s+vigore\b|\bcessat\w*\s+efficac)",
    re.IGNORECASE,
)
PARTIAL_EXCEPTION_RE = re.compile(r"\babrogat[oa]\b.{0,120}\bad\s+eccezione", re.IGNORECASE | re.DOTALL)
INDEX_RE = re.compile(r"\bINDICE\b", re.IGNORECASE)
WORD_RE = re.compile(r"\S+")

def _heading_level(text: str) -> int | None:
    """Map legal section headings to their structure depth."""
    match = HEADING_RE.match((text or "").strip())
    if not match:
        return None
    return {"PARTE": 1, "TITOLO": 2, "CAPO": 3, "SEZIONE": 4}.get(match.group(1).upper())


def _is_noise_line(text: str) -> bool:
    """Identify source artifacts that do not carry legal content."""
    value = (text or "").strip()
    return not value or value == "¦" or bool(re.fullmatch(r"[_\-]{5,}", value)) or bool(re.fullmatch(r"\(\d+\)\.?", value))


def _extract_law_title(blocks: list[Block]) -> str:
    """Find the best law title from heading blocks."""
    for block in blocks:
        if block.kind in {"h1", "h2"} and LAW_HEADER_RE.search(block.text):
            idx = block.text.lower().find("legge regionale")
            return block.text[idx:].strip() if idx >= 0 else block.text.strip()
    for block in blocks:
        if block.kind in {"h1", "h2"} and block.text.strip():
            return block.text.strip()
    return ""


def _article_anchor_in_block(block: Block) -> tuple[str, str, str | None] | None:
    """Recognize article starts when the HTML exposes article anchors."""
    for anchor in block.anchors:
        if not anchor.name.lower().startswith("articolo_"):
            continue
        if not (anchor.text or "").lower().startswith("art"):
            continue
        label_norm = normalize_article_label(anchor.text)
        rest = (block.text or "").strip()
        if anchor.text and rest.startswith(anchor.text):
            rest = rest[len(anchor.text) :].strip()
        rest = rest.lstrip(".-:– ").strip()
        return anchor.name, label_norm, rest or None
    return None


def _plain_article_in_block(block: Block) -> tuple[str, str | None] | None:
    """Recognize article starts from visible text when anchors are absent."""
    match = PLAIN_ARTICLE_RE.match((block.text or "").strip())
    if not match:
        return None
    label_norm = normalize_article_label(match.group("label"))
    rest = (match.group("rest") or "").strip().lstrip(".-:– ").strip()
    return label_norm, rest or None


def _note_definition_in_block(block: Block) -> tuple[str, str, str] | None:
    """Recognize note definitions from note anchors."""
    for anchor in block.anchors:
        if not anchor.name.lower().startswith("nota_"):
            continue
        rest = (block.text or "").strip()
        if anchor.text and rest.startswith(anchor.text):
            rest = rest[len(anchor.text) :].strip()
        rest = rest.lstrip(".").strip()
        if rest:
            return anchor.name.strip(), anchor.name.split("nota_", 1)[-1] or "", rest
    return None


def _looks_like_toc_article_line(text: str) -> bool:
    """Detect article rows that belong to an index rather than article content."""
    return bool(TOC_ARTICLE_LINE_RE.match((text or "").strip()))


def _links_out(links: list[Link] | tuple[Link, ...]) -> list[dict[str, str]]:
    """Serialize and sort outbound links observed in a record."""
    out = [{"href": link.href.strip(), "text": link.text.strip()} for link in links if link.href.strip()]
    return sorted(out, key=lambda item: (item["href"], item["text"]))


def _extract_note_anchor_names_from_hrefs(hrefs: list[str]) -> list[str]:
    """Collect note anchors referenced by a passage line."""
    out: list[str] = []
    seen: set[str] = set()
    for href in hrefs:
        if not href.startswith("#nota_"):
            continue
        anchor = href[1:]
        if anchor not in seen:
            seen.add(anchor)
            out.append(anchor)
    return out



def _chunk_text_words(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into deterministic word-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    words = WORD_RE.findall(text or "")
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - chunk_overlap
    return chunks


def classify_law_status(preamble_text: str, article_count: int, ingest_status: str | None = None) -> tuple[str, float, list[dict[str, str]]]:
    """Classify law status using explicit, deterministic evidence only."""
    text = (preamble_text or "").strip()
    evidence: list[dict[str, str]] = []
    if PARTIAL_EXCEPTION_RE.search(text):
        evidence.append({"kind": "partial_abrogation", "snippet": normalize_ws(text)[:260]})
        return "unknown", 0.65, evidence
    if ABROGATION_FULL_RE.search(text) or ingest_status == "past":
        evidence.append({"kind": "abrogation_phrase", "snippet": normalize_ws(text)[:260]})
        return "past", 0.97, evidence
    if INDEX_RE.search(text) and article_count == 0:
        evidence.append({"kind": "index_without_articles", "snippet": "INDICE con assenza di articoli strutturati"})
        return "index_or_empty", 0.9, evidence
    if article_count > 0:
        return "current", 0.72, evidence
    evidence.append({"kind": "insufficient_evidence", "snippet": "nessuna regola deterministica applicabile"})
    return "unknown", 0.5, evidence


def _article_status(article: dict[str, Any]) -> str:
    """Classify an article as current, past or unknown."""
    if article.get("is_abrogated"):
        return "past"
    return "current" if (article.get("article_text") or "").strip() else "unknown"

def ingest_law(
    law_file: LawFile,
    registry: CorpusRegistry,
    *,
    chunk_size: int,
    chunk_overlap: int,
    strict: bool = False,
) -> IngestedLaw:
    """Parse one HTML law into law, article, passage, note, edge and chunk records."""
    html = law_file.path.read_text(encoding="utf-8", errors="replace")
    blocks = parse_blocks_from_html(html)
    law_title = _extract_law_title(blocks) or f"Legge regionale {law_file.law_date.isoformat()}, n. {law_file.law_number}"

    warnings: list[str] = []
    unresolved_refs = 0
    structure: list[str] = []
    preamble_lines: list[Line] = []
    articles: list[dict[str, Any]] = []
    article_by_id: dict[str, dict[str, Any]] = {}
    article_lines_by_id: dict[str, list[Line]] = {}
    notes: list[dict[str, Any]] = []
    note_lines_by_anchor: dict[str, list[Line]] = {}
    note_links_articles: dict[str, set[str]] = {}
    note_links_passages: dict[str, set[str]] = {}
    note_id_counts: dict[str, int] = defaultdict(int)
    edges_by_id: dict[str, dict[str, Any]] = {}

    def add_edges_from_line(
        *,
        context: str,
        src_article_id: str | None,
        src_passage_id: str | None,
        evidence_text: str,
        links: tuple[Link, ...],
        note_anchor_name: str | None = None,
    ) -> list[str]:
        """Resolve explicit references in one line and register graph edges."""
        nonlocal unresolved_refs
        href_refs: list[ResolvedLawRef] = []
        for link in links:
            if "numero_legge=" not in (link.href or ""):
                continue
            ref, unresolved = resolve_ref_from_href_and_text(link.href, link.text, registry)
            unresolved_refs += unresolved
            if ref:
                href_refs.append(ref)
        text_refs, unresolved = resolve_refs_from_text(evidence_text, registry)
        unresolved_refs += unresolved

        merged: list[ResolvedLawRef] = []
        seen: set[str] = set()
        for ref in href_refs + text_refs:
            if ref.law_id in seen or ref.law_id == law_file.law_id:
                continue
            seen.add(ref.law_id)
            merged.append(ref)

        relation_type, confidence = classify_relation_type(evidence_text)
        dst_article_label_norm = extract_dst_article_label_norm(evidence_text)
        related: list[str] = []
        for ref in merged:
            edge_id = make_edge_id(
                law_file.law_id,
                src_article_id,
                src_passage_id,
                relation_type,
                ref.law_id,
                dst_article_label_norm,
                ref.extraction_method,
                evidence_text.strip(),
            )
            edges_by_id.setdefault(
                edge_id,
                {
                    "edge_id": edge_id,
                    "relation_type": relation_type,
                    "src_law_id": law_file.law_id,
                    "src_article_id": src_article_id,
                    "src_passage_id": src_passage_id,
                    "dst_law_id": ref.law_id,
                    "dst_article_label_norm": dst_article_label_norm,
                    "context": context,
                    "extraction_method": ref.extraction_method,
                    "evidence": evidence_text.strip()[:500],
                    "evidence_text": evidence_text.strip(),
                    "confidence": confidence,
                    "source_file": law_file.source_file,
                    "note_anchor_name": note_anchor_name,
                    "is_self_loop": False,
                },
            )
            related.append(ref.law_id)
        return related

    saw_first_article = False
    in_index_mode = False
    note_mode = False
    current_article_id: str | None = None
    current_note_anchor: str | None = None

    def start_article(label_norm: str, anchor_name: str | None, heading: str | None) -> None:
        """Open or update the current article while preserving stable IDs."""
        nonlocal current_article_id
        article_id = f"{law_file.law_id}#art:{label_norm}"
        if article_id not in article_by_id:
            record: dict[str, Any] = {
                "article_id": article_id,
                "law_id": law_file.law_id,
                "article_label_raw": f"Art. {label_norm}" if label_norm != "unico" else "Articolo unico",
                "article_label_norm": label_norm,
                "anchor_name": anchor_name,
                "structure_path": " > ".join(part for part in structure if part),
                "article_heading": heading,
                "article_text": "",
                "article_status": "unknown",
                "note_anchor_names": [],
                "is_abrogated": False,
                "abrogated_by": None,
                "amended_by_law_ids": [],
                "links_out": [],
            }
            article_by_id[article_id] = record
            article_lines_by_id[article_id] = []
            articles.append(record)
        else:
            record = article_by_id[article_id]
            record["article_heading"] = record.get("article_heading") or heading
            record["anchor_name"] = record.get("anchor_name") or anchor_name
        current_article_id = article_id

    for block in blocks:
        if _is_noise_line(block.text):
            continue
        level = _heading_level(block.text)
        if level:
            while len(structure) < level:
                structure.append("")
            structure[level - 1] = block.text.strip()
            del structure[level:]
            continue

        if not saw_first_article and any(link.href.startswith("#articolo_") for link in block.links):
            continue

        note_def = _note_definition_in_block(block)
        if note_def:
            note_mode = True
            current_article_id = None
            note_anchor_name, note_number, rest = note_def
            current_note_anchor = note_anchor_name
            note_id_counts[note_anchor_name] += 1
            note_suffix = "" if note_id_counts[note_anchor_name] == 1 else f"~{note_id_counts[note_anchor_name]}"
            notes.append(
                {
                    "note_id": f"{law_file.law_id}#note:{note_anchor_name}{note_suffix}",
                    "law_id": law_file.law_id,
                    "note_anchor_name": note_anchor_name,
                    "note_number": note_number or None,
                    "note_kind": "other",
                    "note_text": "",
                    "linked_article_ids": [],
                    "linked_passage_ids": [],
                    "links_out": [],
                }
            )
            note_lines_by_anchor.setdefault(note_anchor_name, []).append(Line(rest, block.links))
            add_edges_from_line(
                context="note",
                src_article_id=None,
                src_passage_id=None,
                evidence_text=block.text,
                links=block.links,
                note_anchor_name=note_anchor_name,
            )
            continue

        anchored_article = _article_anchor_in_block(block)
        if anchored_article:
            note_mode = False
            current_note_anchor = None
            saw_first_article = True
            in_index_mode = False
            _, label_norm, heading = anchored_article
            start_article(label_norm, anchored_article[0], heading)
            continue

        plain_article = _plain_article_in_block(block)
        if plain_article:
            if not saw_first_article and in_index_mode and _looks_like_toc_article_line(block.text):
                continue
            note_mode = False
            current_note_anchor = None
            saw_first_article = True
            in_index_mode = False
            start_article(plain_article[0], None, plain_article[1])
            continue

        if note_mode and current_note_anchor:
            note_lines_by_anchor.setdefault(current_note_anchor, []).append(Line(block.text, block.links))
            add_edges_from_line(
                context="note",
                src_article_id=None,
                src_passage_id=None,
                evidence_text=block.text,
                links=block.links,
                note_anchor_name=current_note_anchor,
            )
            continue

        if not saw_first_article:
            if block.text.strip().upper() == "INDICE":
                in_index_mode = True
                continue
            if any(link.href.startswith("#articolo_") for link in block.links):
                continue
            if in_index_mode and _looks_like_toc_article_line(block.text):
                continue
            preamble_lines.append(Line(block.text, block.links))
            add_edges_from_line(
                context="preamble",
                src_article_id=None,
                src_passage_id=None,
                evidence_text=block.text,
                links=block.links,
            )
            continue

        if current_article_id:
            article_lines_by_id[current_article_id].append(Line(block.text, block.links))
        elif strict:
            raise ValueError(f"Unattached block after first article in {law_file.source_file}: {block.text[:120]!r}")
        else:
            warnings.append(f"Skipped unattached block after first article in {law_file.source_file}")

    preamble_text = "\n".join(line.text.strip() for line in preamble_lines if line.text.strip()).strip()
    if not articles and preamble_text:
        start_article("unico", None, None)
        assert current_article_id is not None
        article_lines_by_id[current_article_id] = [Line(preamble_text, tuple())]

    passages: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []

    for article in articles:
        article_id = article["article_id"]
        lines = article_lines_by_id.get(article_id) or []
        cur_label = "intro"
        cur_kind = "intro"
        cur_lines: list[str] = []
        cur_links: list[Link] = []
        cur_note_anchors: list[str] = []
        cur_related: list[str] = []
        cur_relation_types: set[str] = set()
        passage_label_counts: dict[str, int] = defaultdict(int)
        comma_label: str | None = None

        def flush_passage() -> None:
            """Finalize the current passage and emit its deterministic chunks."""
            nonlocal cur_label, cur_kind, cur_lines, cur_links, cur_note_anchors, cur_related, cur_relation_types
            passage_text = "\n".join(item.strip() for item in cur_lines if item.strip()).strip()
            if not passage_text:
                cur_lines = []
                cur_links = []
                cur_note_anchors = []
                cur_related = []
                cur_relation_types = set()
                return
            passage_label_counts[cur_label] += 1
            passage_id_label = cur_label if passage_label_counts[cur_label] == 1 else f"{cur_label}~{passage_label_counts[cur_label]}"
            passage_id = f"{article_id}#p:{passage_id_label}"
            passage = {
                "passage_id": passage_id,
                "article_id": article_id,
                "law_id": law_file.law_id,
                "passage_label": cur_label,
                "passage_kind": cur_kind,
                "passage_text": passage_text,
                "structure_path": article.get("structure_path") or "",
                "note_anchor_names": sorted(set(cur_note_anchors)),
                "links_out": _links_out(cur_links),
                "related_law_ids": sorted(set(cur_related)),
                "relation_types": sorted(cur_relation_types),
            }
            passages.append(passage)
            for note_anchor in set(cur_note_anchors):
                note_links_articles.setdefault(note_anchor, set()).add(article_id)
                note_links_passages.setdefault(note_anchor, set()).add(passage_id)
            prefix = (
                f"[LR {law_file.law_date.isoformat()} n.{law_file.law_number}] {law_title} | "
                f"Art. {article.get('article_label_norm')} | {cur_label} | {article.get('structure_path') or ''}"
            ).strip()
            for seq, text in enumerate(_chunk_text_words(passage_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)):
                chunks.append(
                    {
                        "chunk_id": f"{passage_id}#chunk:{seq}",
                        "passage_id": passage_id,
                        "article_id": article_id,
                        "law_id": law_file.law_id,
                        "chunk_seq": seq,
                        "text": text,
                        "text_for_embedding": f"{prefix}\n\n{text}".strip(),
                        "law_date": law_file.law_date.isoformat(),
                        "law_number": law_file.law_number,
                        "law_title": law_title,
                        "law_status": "unknown",
                        "article_status": "unknown",
                        "article_label_norm": article.get("article_label_norm"),
                        "passage_label": cur_label,
                        "structure_path": article.get("structure_path") or "",
                        "source_file": law_file.source_file,
                        "index_views": ["historical"],
                        "related_law_ids": sorted(set(cur_related)),
                        "inbound_law_ids": [],
                        "outbound_law_ids": [],
                        "relation_types": sorted(cur_relation_types),
                    }
                )
            cur_lines = []
            cur_links = []
            cur_note_anchors = []
            cur_related = []
            cur_relation_types = set()

        for line in lines:
            text = line.text.strip()
            if not text:
                continue
            comma_match = COMMA_START_RE.match(text)
            letter_match = LETTER_START_RE.match(text)
            if comma_match:
                flush_passage()
                suffix = (comma_match.group("suf") or "").lower()
                comma_label = f"c{int(comma_match.group('num'))}{suffix}"
                cur_label = comma_label
                cur_kind = "comma"
            elif letter_match:
                flush_passage()
                letter = letter_match.group("lettera").lower()
                cur_label = f"{comma_label}.lit_{letter}" if comma_label else f"lit_{letter}"
                cur_kind = "lettera"
            cur_lines.append(text)
            cur_links.extend(line.links)
            cur_note_anchors.extend(_extract_note_anchor_names_from_hrefs([link.href for link in line.links if link.href]))
            line_related = add_edges_from_line(
                context="passage",
                src_article_id=article_id,
                src_passage_id=f"{article_id}#p:{cur_label}",
                evidence_text=text,
                links=line.links,
            )
            if line_related:
                rel_type, _ = classify_relation_type(text)
                cur_related.extend(line_related)
                cur_relation_types.add(rel_type)
        flush_passage()
        article["article_text"] = "\n".join(line.text.strip() for line in lines if line.text.strip()).strip()
        article["links_out"] = _links_out([link for line in lines for link in line.links])

    by_article_notes: dict[str, set[str]] = defaultdict(set)
    for passage in passages:
        by_article_notes[passage["article_id"]].update(passage.get("note_anchor_names") or [])
    for article in articles:
        article["note_anchor_names"] = sorted(by_article_notes.get(article["article_id"], set()))

    note_by_anchor = {note["note_anchor_name"]: note for note in notes}
    note_refs_by_anchor: dict[str, set[str]] = defaultdict(set)
    for note_anchor, note_lines in note_lines_by_anchor.items():
        note = note_by_anchor.get(note_anchor)
        if not note:
            continue
        text = "\n".join(line.text.strip() for line in note_lines if line.text.strip()).strip()
        lowered = text.lower()
        if "abrogat" in lowered:
            note_kind = "abrogated"
        elif "modificat" in lowered or "sostituisc" in lowered or "sostituit" in lowered:
            note_kind = "modified"
        elif "inserit" in lowered:
            note_kind = "inserted"
        else:
            note_kind = "other"
        note["note_text"] = text
        note["note_kind"] = note_kind
        note["links_out"] = _links_out([link for line in note_lines for link in line.links])
        note["linked_article_ids"] = sorted(note_links_articles.get(note_anchor, set()))
        note["linked_passage_ids"] = sorted(note_links_passages.get(note_anchor, set()))
        for line in note_lines:
            refs, unresolved = resolve_refs_from_text(line.text, registry)
            unresolved_refs += unresolved
            for link in line.links:
                if "numero_legge=" not in link.href:
                    continue
                ref, unresolved = resolve_ref_from_href_and_text(link.href, link.text, registry)
                unresolved_refs += unresolved
                if ref:
                    refs.append(ref)
            for ref in refs:
                note_refs_by_anchor[note_anchor].add(ref.law_id)

    for article in articles:
        amended_by: set[str] = set()
        abrogated_by: dict[str, Any] | None = None
        for note_anchor in article.get("note_anchor_names") or []:
            note = note_by_anchor.get(note_anchor)
            if not note:
                continue
            refs = sorted(note_refs_by_anchor.get(note_anchor, set()))
            if note.get("note_kind") in {"modified", "inserted"}:
                amended_by.update(refs)
            if note.get("note_kind") == "abrogated" and refs and abrogated_by is None:
                abrogated_by = {
                    "law_id": refs[0],
                    "article_label_norm": extract_dst_article_label_norm(note.get("note_text") or ""),
                    "evidence_text": (note.get("note_text") or "")[:500],
                    "confidence": 0.9,
                }
        article["amended_by_law_ids"] = sorted(amended_by)
        if abrogated_by:
            article["is_abrogated"] = True
            article["abrogated_by"] = abrogated_by
        article["article_status"] = _article_status(article)

    law_status, confidence, evidence = classify_law_status(preamble_text, len(articles))
    law_record_data = {
        "law_id": law_file.law_id,
        "law_type": "LR",
        "law_date": law_file.law_date.isoformat(),
        "law_number": law_file.law_number,
        "law_title": law_title,
        "law_status": law_status,
        "status_confidence": confidence,
        "status_evidence": evidence,
        "source_file": law_file.source_file,
        "preamble_text": preamble_text,
        "links_out": _links_out([link for line in preamble_lines for link in line.links]),
    }

    article_status_by_id = {article["article_id"]: article["article_status"] for article in articles}
    for chunk in chunks:
        article_status = article_status_by_id.get(chunk["article_id"], "unknown")
        chunk["law_status"] = law_status
        chunk["article_status"] = article_status
        chunk["index_views"] = ["historical"]
        if law_status == "current" and article_status == "current":
            chunk["index_views"].append("current")

    edges = sorted(edges_by_id.values(), key=lambda edge: edge["edge_id"])
    return IngestedLaw(
        law_record(law_record_data),
        [article_record(article) for article in articles],
        [passage_record(passage) for passage in passages],
        [note_record(note) for note in notes],
        [edge_record(edge) for edge in edges],
        [chunk_record(chunk) for chunk in chunks],
        unresolved_refs,
        warnings,
    )
