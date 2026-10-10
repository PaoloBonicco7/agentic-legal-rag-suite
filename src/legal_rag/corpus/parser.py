"""Ordered HTML provenance and deterministic legal structure."""

import re
from datetime import date
from pathlib import Path

from lxml import html

from ..contracts import Block, Document, SourceRecord, SourceSpan, Unit

PARSER_VERSION = "deterministic-parser-3"
MONTHS = dict(zip("gennaio febbraio marzo aprile maggio giugno luglio agosto settembre ottobre novembre dicembre".split(), range(1, 13)))
DATE_RE = re.compile(r"(\d{1,2})[°º]?\s+(" + "|".join(MONTHS) + r")\s+(\d{4})", re.I)
LAW_RE = re.compile(r"(?:legge\s+regionale|l\s*\.\s*r\s*\.)\s*(\d{1,2})[°º]?\s+(" + "|".join(MONTHS) + r")\s+(\d{4})\s*,?\s*n\s*[.°]?\s*(\d+)", re.I)
LABEL = r"\d+\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)?"
PARAGRAPH_RE = re.compile(r"^(" + LABEL + r")\s*\.(?:\s+|$)", re.I)
LETTER_RE = re.compile(r"^([a-z]{1,2})\s*(?:bis|ter)?\s*\)\s*", re.I)
HISTORICAL_RE = re.compile(r"(?:formulazione\s+(?:originaria|precedente)|(?:testo|articolo|comma|lettera).{0,120}\brecitava|testo\s+(?:originario|previgente)|precedentemente\s+recitava)", re.I)
IDENTITY_HEADING_RE = re.compile(r"^legge\s+regionale\s+(?:(?:del\s+)?\d{1,2}[°º]?\s+\w+|n\s*[.°]?\s*\d+\s+del\b)", re.I)
PUBLICATION_RE = re.compile(r"^\(?\s*(?:B\s*\.?\s*U\s*\.?|Bollettino\s+ufficiale)\s", re.I)
CONTAINERS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th", "pre", "blockquote", "div", "section", "table", "tr", "ul", "ol"}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def normalize_label(text: str) -> str:
    return re.sub(r"[\s.-]+", "", text).lower()


def parse_date(text: str) -> str | None:
    match = DATE_RE.search(text)
    if not match:
        numeric = re.search(r"\b(\d{1,2})[ /-](\d{1,2})[ /-](\d{4})\b", text)
        if not numeric:
            return None
        try:
            return date(int(numeric[3]), int(numeric[2]), int(numeric[1])).isoformat()
        except ValueError:
            return None
    try:
        return date(int(match[3]), MONTHS[match[2].lower()], int(match[1])).isoformat()
    except ValueError:
        return None


def law_identity(text: str) -> str | None:
    match = LAW_RE.search(text)
    if not match:
        return None
    enacted = parse_date(match[0])
    return f"vda:lr:{enacted}:{int(match[4])}" if enacted else None


def source_spans(block: Block) -> list[SourceSpan]:
    """Keep exact source slices while removing inline footnote markers."""
    intervals = sorted((int(link["start"]), int(link["end"])) for link in block.links if link.get("href", "").startswith("#nota") and "start" in link)
    spans, cursor = [], 0
    for start, end in intervals + [(len(block.text), len(block.text))]:
        start = max(cursor, start)
        piece = block.text[cursor:start]
        left = cursor + len(piece) - len(piece.lstrip())
        right = start - len(piece) + len(piece.rstrip())
        if left < right:
            spans.append(SourceSpan(block_id=block.block_id, start=left, end=right))
        cursor = max(cursor, end)
    return spans


def span_text(spans: list[SourceSpan], blocks: dict[str, Block]) -> str:
    return "\n".join(blocks[span.block_id].text[span.start:span.end] for span in spans)


def body_text(block: Block) -> str:
    return span_text(source_spans(block), {block.block_id: block})


def substantive(text: str) -> bool:
    text = re.sub(r"\(\s*\d+[a-z]*\s*\)", "", text, flags=re.I).strip()
    text = PARAGRAPH_RE.sub("", text)
    text = LETTER_RE.sub("", text)
    if not text or re.fullmatch(r"[\W\d_]+", text):
        return False
    return not bool(re.fullmatch(r"\(?\s*(?:omissis|abrogat[oaie]|soppresso|soppressa)\s*\)?\s*[.]*", text, re.I))


def article_labels(text: str) -> list[str]:
    text = re.sub(r"\(\d+[a-z]*\)", "", text, flags=re.I).strip()
    match = re.fullmatch(r"(?:art\.?|articolo)\s+(" + LABEL + r"|unico)\s*[.]?", text, re.I)
    if match:
        return [normalize_label(match[1])]
    match = re.fullmatch(r"(?:art\.?|articolo)\s+(" + LABEL + r"|unico)\s*(?:[.:-]\s*\S.{0,500}|\*)", text, re.I)
    if match:
        return [normalize_label(match[1])]
    match = re.fullmatch(r"(?:artt\.?|articoli)\s+([\d\s,–—-]+(?:\s+e\s+\d+)?)\s*[.]?", text, re.I)
    if not match:
        return []
    sequence = match[1]
    numbers = [int(n) for n in re.findall(r"\d+", sequence)]
    if len(numbers) == 2 and re.search(r"[-–—]", sequence) and numbers[0] <= numbers[1] and numbers[1] - numbers[0] <= 200:
        numbers = list(range(numbers[0], numbers[1] + 1))
    return [str(n) for n in numbers]


def extract_blocks(text: str, document_id: str) -> list[Block]:
    tree = html.fromstring(text)
    articles = tree.xpath("self::article | .//article")
    if articles:
        candidates = [node for node in articles if law_identity(normalize(node.text_content()))]
        tree = candidates[0] if candidates else articles[0]
    else:
        candidates = tree.xpath(".//*[@id='tmpl-banche-dati-titolo-secondario']")
        if candidates:
            tree = candidates[0].getparent()
    blocks: list[Block] = []

    def walk(node, tag: str = "text") -> None:
        fragments, links, anchors = [], [], []

        def flush() -> None:
            value = normalize("".join(fragments))
            if value or anchors:
                resolved, cursor = [], 0
                for href, label in links:
                    start = value.find(label, cursor) if label else -1
                    link = {"href": href, "text": label}
                    if start >= 0:
                        link.update(start=str(start), end=str(start + len(label)))
                        cursor = start + len(label)
                    resolved.append(link)
                blocks.append(Block(block_id=f"{document_id}#b:{len(blocks):05d}", document_id=document_id, position=len(blocks), text=value, tag=tag, anchor=next((a for a in anchors if a.startswith("nota")), anchors[0] if anchors else None), links=resolved))
            fragments.clear()
            links.clear()
            anchors.clear()

        def inline(element) -> None:
            name = str(element.tag).lower()
            if name in {"script", "style", "noscript"}:
                return
            if name == "br":
                flush()
                return
            if name in CONTAINERS:
                flush()
                walk(element, name)
                return
            anchor = element.get("name") or element.get("id")
            if anchor:
                anchors.append(anchor)
            if name == "a" and element.get("href"):
                links.append((element.get("href"), normalize(element.text_content())))
            if element.text:
                fragments.append(element.text)
            for child in element:
                inline(child)
                if child.tail:
                    fragments.append(child.tail)

        anchor = node.get("name") or node.get("id")
        if anchor:
            anchors.append(anchor)
        if node.text:
            fragments.append(node.text)
        for child in node:
            inline(child)
            if child.tail:
                fragments.append(child.tail)
        flush()

    walk(tree, str(tree.tag).lower())
    return blocks


def parse_source(source: SourceRecord, snapshot_id: str, html_text: str) -> tuple[Document, list[Block], list[Unit]]:
    tree = html.fromstring(html_text)
    headers = tree.xpath(".//*[@id='tmpl-banche-dati-titolo-secondario']") + tree.xpath("self::h1 | .//h1 | .//h2")
    identity = next((law_identity(normalize(node.text_content())) for node in headers if law_identity(normalize(node.text_content()))), None)
    if identity is None:
        identity = law_identity(Path(source.path).name.replace("-", " ").replace("_LR", " Legge regionale ").replace(" n", ", n"))
    if identity is None:
        raise ValueError(f"Missing legal identity in source {source.source_id}")
    blocks = extract_blocks(html_text, identity)
    by_block = {b.block_id: b for b in blocks}
    heading = next((normalize(node.text_content()) for node in headers if law_identity(normalize(node.text_content())) == identity), identity)
    heading = re.sub(r"\s*[-–—]\s*Testo\s+vigente\s*$", "", heading, flags=re.I)
    document = Document(document_id=identity, identity=heading, source_id=source.source_id, snapshot_id=snapshot_id, source_url=source.url, source_path=source.path, title="", enacted_at=identity.split(":")[2], representation=source.representation, availability="metadata_only")
    units: list[Unit] = []
    used_ids: set[str] = set()
    current_article: Unit | None = None
    current_leaf: Unit | None = None
    current_annex: Unit | None = None
    in_notes, in_historical, in_index, section_heading = False, False, False, False
    seen_body, in_body_quote = False, False
    quote_close = '"'
    quote_scope = "unknown"
    publication_position = next((b.position for b in blocks if PUBLICATION_RE.match(b.text)), -1)

    def add_unit(kind: str, label: str, block: Block, parent: Unit | None = None, *, text: bool = True) -> Unit:
        suffix = {"article": "art", "paragraph": "p", "letter": "l", "annex": "annex", "unstructured": "u"}[kind]
        base = parent.unit_id if parent else identity
        uid = f"{base}#{suffix}:{label}"
        if uid in used_ids:
            document.warnings.append(f"duplicate_unit_label:{uid}")
            uid += f":duplicate:{block.position}"
        used_ids.add(uid)
        spans = source_spans(block) if text else []
        unit = Unit(unit_id=uid, document_id=identity, kind=kind, label=label, article_label=label if kind == "article" else current_article.article_label if current_article and kind not in {"annex", "unstructured"} else None, text=span_text(spans, by_block), spans=spans, parent_id=parent.unit_id if parent else None)
        units.append(unit)
        return unit

    for block in blocks:
        clean = body_text(block)
        if not block.text:
            block.region, block.exclusion_reason = "empty", "empty_html_anchor"
            continue
        note_anchor = bool(block.anchor and block.anchor.lower().startswith("nota"))
        labels = article_labels(clean)
        if note_anchor:
            in_notes, in_historical, in_index = True, False, False
            in_body_quote = False
            current_leaf = None
            block.region = "note"
            if HISTORICAL_RE.search(block.text):
                in_historical = True
            continue
        if in_notes:
            if labels and not in_historical:
                in_notes = False
            else:
                block.region = "historical" if in_historical else "note"
                if HISTORICAL_RE.search(block.text):
                    in_historical = True
                    block.region = "historical"
                continue
        if re.fullmatch(r"note[:.]?", clean, re.I):
            in_notes, in_historical = True, False
            block.region, block.exclusion_reason = "note", "note_heading"
            continue
        if law_identity(block.text) and block.tag in {"h1", "h2"}:
            if law_identity(block.text) != identity:
                document.warnings.append(f"conflicting_identity_heading:{law_identity(block.text)}")
            block.region, block.exclusion_reason = "metadata", "identity_heading"
            continue
        if not seen_body and IDENTITY_HEADING_RE.match(block.text):
            block.region, block.exclusion_reason = "metadata", "identity_heading"
            continue
        if PUBLICATION_RE.match(block.text):
            block.region = "metadata"
            document.published_at = parse_date(block.text)
            continue
        if not seen_body and re.match(r"^SUPPLEMENTO\s+STRAORDINARIO\b", block.text, re.I):
            block.region = "metadata"
            continue
        if re.fullmatch(r"[_\-–—\s]+", block.text):
            block.region, block.exclusion_reason = "separator", "visual_separator"
            continue
        if re.fullmatch(r"(?:indice|sommario)[:.]?", clean, re.I):
            in_index = True
            block.region = "index"
            continue
        heading_links = any(link.get("href", "").startswith("#art") for link in block.links)
        if heading_links and not (block.anchor and block.anchor.startswith("art")):
            block.region = "index"
            continue
        if in_index and not labels:
            block.region = "index"
            continue
        previous = blocks[block.position - 1].text if block.position else ""
        if in_body_quote and quote_scope == "leaf" and labels and block.anchor and block.anchor.lower().startswith("art"):
            document.warnings.append(f"source_quote_boundary_recovered:{block.position}")
            in_body_quote = False
        quote_start = clean.startswith(('"', '“', '«')) and (bool(article_labels(clean[1:].strip())) or previous.rstrip().endswith(":") and bool(re.search(r"sostituit|inserit|aggiunt|seguent", previous, re.I)))
        if current_article and (in_body_quote or quote_start):
            if quote_start and not in_body_quote:
                quote_close = {'"': '"', '“': '”', '«': '»'}[clean[0]]
                quote_scope = "leaf" if PARAGRAPH_RE.match(clean[1:]) or LETTER_RE.match(clean[1:]) else "structure"
            block.region = "body_quote"
            if current_leaf is None:
                current_leaf = add_unit("paragraph", f"intro:{block.position}", block, current_article)
            else:
                current_leaf.spans.extend(source_spans(block))
                current_leaf.text = span_text(current_leaf.spans, by_block)
            current_article.spans.extend(source_spans(block))
            current_article.text = span_text(current_article.spans, by_block)
            content = clean[1:] if quote_start and not in_body_quote else clean
            in_body_quote = not bool(content.count(quote_close) % 2)
            continue
        if labels:
            in_index, section_heading = False, False
            seen_body = True
            current_leaf, current_annex = None, None
            block.region, block.exclusion_reason = "heading", "article_heading"
            for label in labels:
                current_article = add_unit("article", label, block, text=False)
                inline = re.fullmatch(r"(?:art\.?|articolo)\s+(?:" + LABEL + r"|unico)\s*[.:-]\s*(.+)", clean, re.I)
                if inline:
                    current_article.heading = inline[1]
                    if re.search(r"\bomissis\b", inline[1], re.I):
                        current_article.status = "unknown"
                        document.warnings.append(f"omitted_article:{label}")
            continue
        if re.fullmatch(r"\(?\s*omissis\s*\)?\s*[.]*", clean, re.I):
            block.region, block.exclusion_reason = "omitted", "omitted_content"
            if current_article:
                current_article.status = "unknown"
            continue
        if re.fullmatch(r"parte\s+(?:entrata|spesa)", clean, re.I):
            block.region, block.exclusion_reason = "heading", "financial_section_heading"
            continue
        if re.match(r"^(?:capo|titolo|sezione|parte)\s+(?:[ivxlcdm]+|\d+|prima|seconda|terza)\b", clean, re.I):
            block.region, block.exclusion_reason = "heading", "section_heading"
            section_heading = True
            current_article, current_leaf = None, None
            continue
        if section_heading:
            block.region, block.exclusion_reason = "heading", "section_title"
            section_heading = False
            continue
        if current_article and not current_article.spans and current_leaf is None and re.fullmatch(r"\([^)]{1,600}\)[.]?", clean):
            current_article.heading = clean
            block.region, block.exclusion_reason = "heading", "article_title"
            continue
        if re.match(r"^allegat[oi]\b", clean, re.I) and len(clean) < 200:
            current_article, current_leaf = None, None
            omitted = bool(re.search(r"\bomissis\b", clean, re.I))
            current_annex = add_unit("annex", str(sum(u.kind == "annex" for u in units) + 1), block, text=omitted)
            current_annex.heading = clean
            block.region = "annex" if omitted else "heading"
            block.exclusion_reason = "annex_heading"
            if omitted:
                block.exclusion_reason = "omitted_annex"
            continue
        if not seen_body and not current_annex:
            if re.match(r"^\(?\s*(?:legge\s+)?(?:abrogat|non\s+abrogat|cessat|restano\s+salv|ad\s+eccezione)", clean, re.I):
                block.region = "editorial"
                continue
            if not document.title:
                document.title = clean
                block.region = "metadata"
                continue
            if block.position < publication_position:
                document.title += " " + clean
                block.region = "metadata"
                continue
        if current_annex:
            block.region = "annex"
            current_annex.spans.extend(source_spans(block))
            current_annex.text = span_text(current_annex.spans, by_block)
            continue
        block.region = "body"
        if current_article:
            paragraph = PARAGRAPH_RE.match(clean)
            letter = LETTER_RE.match(clean)
            if paragraph:
                current_leaf = add_unit("paragraph", normalize_label(paragraph[1]), block, current_article)
            elif letter:
                parent = current_leaf if current_leaf and current_leaf.kind == "paragraph" else next((u for u in reversed(units) if u.kind == "paragraph" and u.parent_id == current_article.unit_id), current_article)
                current_leaf = add_unit("letter", normalize_label(letter[1]), block, parent)
            elif current_leaf:
                current_leaf.spans.extend(source_spans(block))
                current_leaf.text = span_text(current_leaf.spans, by_block)
            else:
                current_leaf = add_unit("paragraph", f"intro:{block.position}", block, current_article)
            current_article.spans.extend(source_spans(block))
            current_article.text = span_text(current_article.spans, by_block)
        else:
            add_unit("unstructured", str(block.position), block)
    if not document.title:
        document.title = next((b.text.split(" - Testo")[0] for b in blocks if law_identity(b.text) == identity), identity)
        document.warnings.append("missing_thematic_title")
    if any(u.kind == "article" for u in units):
        document.availability = "body"
    elif any(u.kind == "unstructured" and substantive(u.text) for u in units):
        document.availability = "unstructured"
        document.warnings.append("unstructured_body")
    elif units and all("omissis" in u.text.lower() for u in units):
        document.availability = "omitted"
    if any(u.kind == "annex" and "omissis" in u.text.lower() for u in units):
        document.warnings.append("omitted_annex")
    if any("\ufffd" in b.text or re.search(r"[\x80-\x9f]", b.text) for b in blocks):
        document.warnings.append("invalid_text_encoding")
    if in_body_quote:
        document.warnings.append("unterminated_body_quote")
    return document, blocks, units
