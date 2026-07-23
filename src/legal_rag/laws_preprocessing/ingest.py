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
    ResolvedLawRef,
    article_record,
    chunk_record,
    edge_record,
    law_record,
    note_record,
    passage_record,
    status_event_record,
)
from .references import (
    classify_relation_type,
    edge_id as make_edge_id,
    extract_dst_article_label_norm,
    resolve_ref_from_href_and_text,
    resolve_refs_from_text,
)
from .status import (
    ALL_EXCEPT_RE,
    BRACKETED_UNKNOWN_RULE,
    DEFAULT_CURRENT_RULE,
    EMPTY_CONTENT_RULE,
    anchorless_note_definition,
    cessation_event_type,
    content_availability,
    exception_article_labels,
    is_fully_bracketed,
    note_kind_from_clause,
    rollup_validity,
    split_later_total_cessation,
    status_event_id,
    target_kind_from_clause,
    visible_note_anchor_names,
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
    r"^\[?\s*(?P<num>\d+)(?P<suf>bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)?\.\s+",
    re.IGNORECASE,
)
LETTER_START_RE = re.compile(r"^\[?\s*(?P<lettera>[a-z])\)\s+", re.IGNORECASE)
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


def _extract_note_anchor_names(
    text: str,
    hrefs: list[str],
    known_anchors: set[str],
) -> list[str]:
    """Collect linked and visible note markers resolved in this document."""
    out: list[str] = []
    seen: set[str] = set()
    for href in hrefs:
        if not href.startswith("#nota_"):
            continue
        anchor = href[1:]
        if anchor in known_anchors and anchor not in seen:
            seen.add(anchor)
            out.append(anchor)
    for anchor in visible_note_anchor_names(text, known_anchors):
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


def classify_law_status(
    preamble_text: str,
    article_count: int,
    ingest_status: str | None = None,
) -> tuple[str, float, list[dict[str, str]]]:
    """Preserve the legacy helper API while returning v2 validity states."""
    text = (preamble_text or "").strip()
    evidence: list[dict[str, str]] = []
    event_type = cessation_event_type(text)
    if event_type and ALL_EXCEPT_RE.search(text):
        evidence.append({"kind": "partial_abrogation", "snippet": normalize_ws(text)[:260]})
        return "partial", 1.0, evidence
    if event_type or ingest_status == "past":
        evidence.append({"kind": f"{event_type or 'repeal'}_phrase", "snippet": normalize_ws(text)[:260]})
        return "past", 1.0, evidence
    if article_count > 0:
        return "current", 1.0, evidence
    evidence.append({"kind": "insufficient_evidence", "snippet": "nessuna regola deterministica applicabile"})
    return "unknown", 1.0, evidence

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
    structured_article_ids: set[str] = set()
    notes: list[dict[str, Any]] = []
    note_lines_by_id: dict[str, list[Line]] = {}
    note_links_articles: dict[str, set[str]] = {}
    note_links_passages: dict[str, set[str]] = {}
    note_id_counts: dict[str, int] = defaultdict(int)
    edges_by_id: dict[str, dict[str, Any]] = {}
    status_diagnostics = {
        "duplicate_note_anchors": 0,
        "anchorless_note_markers": 0,
        "zero_backlinks": 0,
        "multiple_backlinks": 0,
        "scope_kind_mismatches": 0,
        "bracketed_passages": 0,
    }

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
    current_note_id: str | None = None

    def start_article(
        label_norm: str,
        anchor_name: str | None,
        heading: str | None,
        *,
        structured: bool = True,
    ) -> None:
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
                "content_availability": "empty",
                "status_event_ids": [],
                "status_rule_ids": [],
                "note_anchor_names": [],
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
        if structured:
            structured_article_ids.add(article_id)
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
            if note_id_counts[note_anchor_name] > 1:
                status_diagnostics["duplicate_note_anchors"] += 1
                warnings.append(f"Duplicate note anchor {note_anchor_name!r} in {law_file.source_file}")
            note_suffix = "" if note_id_counts[note_anchor_name] == 1 else f"~{note_id_counts[note_anchor_name]}"
            current_note_id = f"{law_file.law_id}#note:{note_anchor_name}{note_suffix}"
            notes.append(
                {
                    "note_id": current_note_id,
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
            note_lines_by_id.setdefault(current_note_id, []).append(Line(rest, block.links))
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
            current_note_id = None
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
            current_note_id = None
            saw_first_article = True
            in_index_mode = False
            start_article(plain_article[0], None, plain_article[1])
            continue

        if note_mode and current_note_anchor and current_note_id:
            anchorless = anchorless_note_definition(block.text)
            if anchorless:
                note_anchor_name, rest = anchorless
                current_note_anchor = note_anchor_name
                note_id_counts[note_anchor_name] += 1
                if note_id_counts[note_anchor_name] > 1:
                    status_diagnostics["duplicate_note_anchors"] += 1
                    warnings.append(f"Duplicate note marker {note_anchor_name!r} in {law_file.source_file}")
                note_suffix = "" if note_id_counts[note_anchor_name] == 1 else f"~{note_id_counts[note_anchor_name]}"
                current_note_id = f"{law_file.law_id}#note:{note_anchor_name}{note_suffix}"
                notes.append(
                    {
                        "note_id": current_note_id,
                        "law_id": law_file.law_id,
                        "note_anchor_name": note_anchor_name,
                        "note_number": note_anchor_name.removeprefix("nota_") or None,
                        "note_kind": "other",
                        "note_text": "",
                        "linked_article_ids": [],
                        "linked_passage_ids": [],
                        "links_out": [],
                    }
                )
                note_lines_by_id[current_note_id] = [Line(rest, block.links)]
                status_diagnostics["anchorless_note_markers"] += 1
                warnings.append(f"Synthesized missing note anchor {note_anchor_name!r} in {law_file.source_file}")
                add_edges_from_line(
                    context="note",
                    src_article_id=None,
                    src_passage_id=None,
                    evidence_text=block.text,
                    links=block.links,
                    note_anchor_name=note_anchor_name,
                )
                continue
            note_lines_by_id.setdefault(current_note_id, []).append(Line(block.text, block.links))
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
        start_article("unico", None, None, structured=False)
        assert current_article_id is not None
        article_lines_by_id[current_article_id] = [Line(preamble_text, tuple())]

    passages: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    known_note_anchors = {str(note["note_anchor_name"]) for note in notes}

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
            bracketed = is_fully_bracketed(passage_text)
            if bracketed:
                status_diagnostics["bracketed_passages"] += 1
            availability = content_availability(
                text=passage_text,
                structured=article_id in structured_article_ids,
            )
            passage = {
                "passage_id": passage_id,
                "article_id": article_id,
                "law_id": law_file.law_id,
                "passage_label": cur_label,
                "passage_kind": cur_kind,
                "passage_text": passage_text,
                "passage_status": "unknown" if bracketed else "current",
                "content_availability": availability,
                "status_event_ids": [],
                "status_rule_ids": [BRACKETED_UNKNOWN_RULE if bracketed else DEFAULT_CURRENT_RULE],
                "is_bracketed": bracketed,
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
                        "passage_status": passage["passage_status"],
                        "content_availability": availability,
                        "status_event_ids": [],
                        "status_rule_ids": list(passage["status_rule_ids"]),
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
            cur_note_anchors.extend(
                _extract_note_anchor_names(
                    text,
                    [link.href for link in line.links if link.href],
                    known_note_anchors,
                )
            )
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
        article["content_availability"] = content_availability(
            text=article["article_text"],
            structured=article_id in structured_article_ids,
            metadata_present=bool(article.get("article_heading")),
        )
        article["links_out"] = _links_out([link for line in lines for link in line.links])

    by_article_notes: dict[str, set[str]] = defaultdict(set)
    for passage in passages:
        by_article_notes[passage["article_id"]].update(passage.get("note_anchor_names") or [])
    for article in articles:
        article["note_anchor_names"] = sorted(by_article_notes.get(article["article_id"], set()))

    notes_by_anchor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for note in notes:
        notes_by_anchor[str(note["note_anchor_name"])].append(note)
    note_refs_by_anchor: dict[str, set[str]] = defaultdict(set)
    decisive_line_by_note_id: dict[str, Line] = {}
    for note in notes:
        note_anchor = str(note["note_anchor_name"])
        note_lines = note_lines_by_id.get(str(note["note_id"])) or []
        text = "\n".join(line.text.strip() for line in note_lines if line.text.strip()).strip()
        decisive_line = next((line for line in note_lines if line.text.strip()), Line("", tuple()))
        decisive_line_by_note_id[str(note["note_id"])] = decisive_line
        note["note_text"] = text
        note["note_kind"] = note_kind_from_clause(decisive_line.text)
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
        for note_anchor in article.get("note_anchor_names") or []:
            refs = sorted(note_refs_by_anchor.get(note_anchor, set()))
            if any(note.get("note_kind") in {"modified", "inserted"} for note in notes_by_anchor.get(note_anchor, [])):
                amended_by.update(refs)
        article["amended_by_law_ids"] = sorted(amended_by)

    passage_by_id = {str(passage["passage_id"]): passage for passage in passages}
    passages_by_article: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for passage in passages:
        passages_by_article[str(passage["article_id"])].append(passage)

    def resolved_law_ids(line: Line) -> list[str]:
        """Resolve modifying laws from the decisive clause only."""
        refs, _ = resolve_refs_from_text(line.text, registry)
        for link in line.links:
            if "numero_legge=" not in link.href:
                continue
            ref, _ = resolve_ref_from_href_and_text(link.href, link.text, registry)
            if ref:
                refs.append(ref)
        return sorted({ref.law_id for ref in refs if ref.law_id != law_file.law_id})

    status_events: list[dict[str, Any]] = []

    def append_status_event(event: dict[str, Any]) -> None:
        event["status_event_id"] = status_event_id(event)
        status_events.append(status_event_record(event))

    for line in preamble_lines:
        split = split_later_total_cessation(line.text)
        clauses = (
            [(split[0], False), (split[1], True)]
            if split
            else [(line.text, False)]
        )
        for clause_text, closes_exceptions in clauses:
            event_type = cessation_event_type(clause_text)
            if not event_type:
                continue
            clause_line = Line(clause_text, tuple()) if split else line
            scope_mode = "all_except" if ALL_EXCEPT_RE.search(clause_text) else "all"
            exception_labels = (
                exception_article_labels(clause_text) if scope_mode == "all_except" else []
            )
            exception_ids = [
                str(article_by_id[f"{law_file.law_id}#art:{label}"]["article_id"])
                for label in exception_labels
                if f"{law_file.law_id}#art:{label}" in article_by_id
            ]
            resolved = scope_mode == "all" or (
                bool(exception_labels) and len(exception_ids) == len(exception_labels)
            )
            append_status_event(
                {
                    "law_id": law_file.law_id,
                    "event_type": event_type,
                    "source_kind": "preamble",
                    "source_anchor": None,
                    "source_clause": clause_text.strip(),
                    "evidence_text": line.text.strip(),
                    "event_sequence": len(status_events),
                    "modifying_law_ids": resolved_law_ids(clause_line),
                    "target_kind": "law",
                    "target_ids": [law_file.law_id],
                    "scope_mode": scope_mode,
                    "exception_labels": exception_labels,
                    "exception_target_ids": sorted(exception_ids),
                    "resolution_status": "resolved" if resolved else "ambiguous",
                    "rule_id": (
                        "status-law-exceptions-later-cessation-v1"
                        if closes_exceptions
                        else (
                            "status-law-all-except-v1"
                            if scope_mode == "all_except"
                            else "status-law-whole-cessation-v1"
                        )
                    ),
                }
            )

    for note in notes:
        decisive_line = decisive_line_by_note_id[str(note["note_id"])]
        event_type = cessation_event_type(decisive_line.text)
        if not event_type:
            continue
        anchor = str(note["note_anchor_name"])
        target_kind = target_kind_from_clause(decisive_line.text, source_kind="note")
        candidate_articles = sorted(note_links_articles.get(anchor, set()))
        candidate_passages = sorted(note_links_passages.get(anchor, set()))
        target_ids: list[str] = []
        resolution = "unapplied"
        rule_id = "status-note-no-backlink-v1"
        if len(notes_by_anchor[anchor]) > 1:
            resolution = "ambiguous"
            rule_id = "status-note-duplicate-anchor-v1"
        elif target_kind == "article":
            target_ids = candidate_articles
            if len(candidate_articles) == 1:
                resolution = "resolved"
                rule_id = "status-article-whole-cessation-v1"
            elif len(candidate_articles) > 1:
                resolution = "ambiguous"
                rule_id = "status-note-multiple-backlinks-v1"
        elif target_kind in {"comma", "letter"}:
            target_ids = candidate_passages
            if len(candidate_passages) == 1:
                passage_kind = str(passage_by_id[candidate_passages[0]]["passage_kind"])
                compatible = (target_kind == "comma" and passage_kind == "comma") or (
                    target_kind == "letter" and passage_kind == "lettera"
                )
                narrower = target_kind == "letter" and passage_kind == "comma"
                if compatible or narrower:
                    resolution = "resolved"
                    rule_id = (
                        "status-passage-narrower-scope-v1"
                        if narrower
                        else "status-passage-whole-cessation-v1"
                    )
                else:
                    resolution = "ambiguous"
                    rule_id = "status-note-scope-mismatch-v1"
                    status_diagnostics["scope_kind_mismatches"] += 1
            elif len(candidate_passages) > 1:
                resolution = "ambiguous"
                rule_id = "status-note-multiple-backlinks-v1"
        elif candidate_articles or candidate_passages:
            target_ids = candidate_passages or candidate_articles
            resolution = "ambiguous"
            rule_id = "status-note-unknown-scope-v1"

        if not candidate_articles and not candidate_passages:
            status_diagnostics["zero_backlinks"] += 1
        elif len(candidate_passages if target_kind in {"comma", "letter"} else candidate_articles) > 1:
            status_diagnostics["multiple_backlinks"] += 1
        append_status_event(
            {
                "law_id": law_file.law_id,
                "event_type": event_type,
                "source_kind": "note",
                "source_anchor": anchor,
                "source_clause": decisive_line.text.strip(),
                "evidence_text": str(note["note_text"]),
                "event_sequence": len(status_events),
                "modifying_law_ids": resolved_law_ids(decisive_line),
                "target_kind": target_kind,
                "target_ids": target_ids,
                "scope_mode": "only",
                "exception_labels": [],
                "exception_target_ids": [],
                "resolution_status": resolution,
                "rule_id": rule_id,
            }
        )

    law_direct_event_ids: list[str] = []
    law_direct_rule_ids: list[str] = []
    law_explicit_status: str | None = None
    explicit_past_articles: set[str] = set()
    ambiguous_articles: set[str] = set()

    def apply_passage_event(passage: dict[str, Any], event: dict[str, Any], status: str) -> None:
        passage["passage_status"] = status
        passage["status_event_ids"] = sorted(set(passage["status_event_ids"]) | {event["status_event_id"]})
        passage["status_rule_ids"] = [event["rule_id"]]

    for event in sorted(status_events, key=lambda item: int(item["event_sequence"])):
        if event["source_kind"] == "preamble":
            law_direct_event_ids.append(event["status_event_id"])
            law_direct_rule_ids.append(event["rule_id"])
            if event["resolution_status"] != "resolved":
                law_explicit_status = "unknown"
                continue
            if event["scope_mode"] == "all":
                law_explicit_status = "past"
                for article in articles:
                    explicit_past_articles.add(str(article["article_id"]))
                    for passage in passages_by_article[str(article["article_id"])]:
                        apply_passage_event(passage, event, "past")
                continue
            if law_explicit_status == "past":
                continue
            law_explicit_status = "partial"
            exceptions = set(event["exception_target_ids"])
            for article in articles:
                article_id = str(article["article_id"])
                if article_id in exceptions:
                    continue
                explicit_past_articles.add(article_id)
                for passage in passages_by_article[article_id]:
                    apply_passage_event(passage, event, "past")
            continue

        if event["resolution_status"] == "resolved":
            if event["target_kind"] == "article":
                article_id = event["target_ids"][0]
                explicit_past_articles.add(article_id)
                for passage in passages_by_article.get(article_id, []):
                    apply_passage_event(passage, event, "past")
            elif event["target_kind"] in {"comma", "letter"}:
                passage = passage_by_id[event["target_ids"][0]]
                status = "partial" if event["rule_id"] == "status-passage-narrower-scope-v1" else "past"
                apply_passage_event(passage, event, status)
        elif event["resolution_status"] == "ambiguous":
            for target_id in event["target_ids"]:
                if target_id in passage_by_id:
                    apply_passage_event(passage_by_id[target_id], event, "unknown")
                elif target_id in article_by_id:
                    ambiguous_articles.add(target_id)
                    article_by_id[target_id]["status_event_ids"].append(event["status_event_id"])
                    article_by_id[target_id]["status_rule_ids"] = [event["rule_id"]]

    for article in articles:
        article_id = str(article["article_id"])
        child_passages = passages_by_article.get(article_id, [])
        child_event_ids = {
            event_id for passage in child_passages for event_id in passage.get("status_event_ids", [])
        }
        child_rule_ids = {
            rule_id for passage in child_passages for rule_id in passage.get("status_rule_ids", [])
        }
        article["status_event_ids"] = sorted(set(article["status_event_ids"]) | child_event_ids)
        if article_id in explicit_past_articles:
            article["article_status"] = "past"
            article["status_rule_ids"] = sorted(
                rule for rule in child_rule_ids if rule != DEFAULT_CURRENT_RULE
            ) or ["status-article-whole-cessation-v1"]
        elif article_id in ambiguous_articles:
            article["article_status"] = "unknown"
            article["status_rule_ids"] = sorted(
                set(article["status_rule_ids"]) | child_rule_ids | {"status-note-ambiguous-target-v1"}
            )
        else:
            default_status = (
                "current"
                if article["content_availability"] in {"substantive", "unstructured"}
                else "unknown"
            )
            article_status, rollup_rule = rollup_validity(
                (str(passage["passage_status"]) for passage in child_passages),
                default=default_status,
            )
            article["article_status"] = article_status
            article["status_rule_ids"] = sorted(
                set(article["status_rule_ids"])
                | (child_rule_ids - {DEFAULT_CURRENT_RULE, BRACKETED_UNKNOWN_RULE})
                | {rollup_rule}
            )

    law_content = "\n".join(str(article.get("article_text") or "") for article in articles)
    law_availability = content_availability(
        text=law_content or preamble_text,
        structured=bool(structured_article_ids),
        metadata_present=bool(law_title),
    )
    if law_explicit_status is None:
        law_status, law_rollup_rule = rollup_validity(
            (str(article["article_status"]) for article in articles),
            default="current" if law_availability in {"substantive", "unstructured"} else "unknown",
        )
        law_rule_ids = sorted(set(law_direct_rule_ids) | {law_rollup_rule})
    else:
        law_status = law_explicit_status
        law_rule_ids = sorted(set(law_direct_rule_ids)) or [
            EMPTY_CONTENT_RULE if law_status == "unknown" else DEFAULT_CURRENT_RULE
        ]
    descendant_event_ids = {
        event_id for article in articles for event_id in article.get("status_event_ids", [])
    }
    law_record_data = {
        "law_id": law_file.law_id,
        "law_type": "LR",
        "law_date": law_file.law_date.isoformat(),
        "law_number": law_file.law_number,
        "law_title": law_title,
        "law_status": law_status,
        "content_availability": law_availability,
        "status_event_ids": sorted(set(law_direct_event_ids) | descendant_event_ids),
        "status_rule_ids": law_rule_ids,
        "source_file": law_file.source_file,
        "preamble_text": preamble_text,
        "links_out": _links_out([link for line in preamble_lines for link in line.links]),
    }

    article_by_id = {str(article["article_id"]): article for article in articles}
    preamble_event_ids = set(law_direct_event_ids)
    preamble_rule_ids = set(law_direct_rule_ids)
    for chunk in chunks:
        article = article_by_id[str(chunk["article_id"])]
        passage = passage_by_id[str(chunk["passage_id"])]
        article_status = str(article["article_status"])
        passage_status = str(passage["passage_status"])
        chunk["law_status"] = law_status
        chunk["article_status"] = article_status
        chunk["passage_status"] = passage_status
        chunk["content_availability"] = passage["content_availability"]
        chunk["status_event_ids"] = sorted(
            preamble_event_ids
            | set(article.get("status_event_ids") or [])
            | set(passage.get("status_event_ids") or [])
        )
        chunk["status_rule_ids"] = sorted(
            preamble_rule_ids
            | set(article.get("status_rule_ids") or [])
            | set(passage.get("status_rule_ids") or [])
            | set(law_rule_ids)
        )
        chunk["index_views"] = ["historical"]
        lineage = {law_status, article_status, passage_status}
        if lineage <= {"current", "partial"} and chunk["content_availability"] in {
            "substantive",
            "unstructured",
        }:
            chunk["index_views"].append("current")
        if "past" not in lineage:
            chunk["index_views"].append("not_explicitly_past")

    edges = sorted(edges_by_id.values(), key=lambda edge: edge["edge_id"])
    return IngestedLaw(
        law_record(law_record_data),
        [article_record(article) for article in articles],
        [passage_record(passage) for passage in passages],
        [note_record(note) for note in notes],
        status_events,
        [edge_record(edge) for edge in edges],
        [chunk_record(chunk) for chunk in chunks],
        unresolved_refs,
        warnings,
        status_diagnostics,
    )
