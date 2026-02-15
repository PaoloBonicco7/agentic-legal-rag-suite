from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
import re

_WS_RE = re.compile(r"\s+")


def _normalize_ws(text: str) -> str:
    text = (text or "").replace("\xa0", " ")
    return _WS_RE.sub(" ", text).strip()


@dataclass(frozen=True)
class Anchor:
    name: str
    text: str


@dataclass(frozen=True)
class Link:
    href: str
    text: str


@dataclass(frozen=True)
class Block:
    kind: str
    text: str
    anchors: tuple[Anchor, ...]
    links: tuple[Link, ...]


class _BlockHTMLParser(HTMLParser):
    def __init__(self, *, restrict_to_article: bool) -> None:
        super().__init__(convert_charrefs=True)
        self._restrict_to_article = bool(restrict_to_article)
        self._seen_article = False

        self.blocks: list[Block] = []

        self._in_article = False
        self._in_table = False
        self._in_tr = False
        self._in_cell = False

        self._current_kind: str | None = None
        self._text_parts: list[str] = []
        self._anchors: list[Anchor] = []
        self._links: list[Link] = []

        self._a_ctx: dict | None = None
        self._a_text_parts: list[str] = []

        self._row_cells: list[str] = []
        self._cell_text_parts: list[str] = []

    def _allowed(self) -> bool:
        if not self._restrict_to_article:
            return True
        return self._in_article

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_dict = {k.lower(): (v or "") for k, v in attrs}

        if tag == "article":
            self._seen_article = True
            self._in_article = True
            return

        if not self._allowed():
            return

        if tag == "br" and self._current_kind is not None and not self._in_table:
            self._text_parts.append("\n")
            if self._a_ctx is not None:
                self._a_text_parts.append("\n")
            return

        if tag == "table":
            self._in_table = True
            return
        if self._in_table:
            if tag == "tr":
                self._in_tr = True
                self._row_cells = []
                return
            if tag in ("td", "th"):
                self._in_cell = True
                self._cell_text_parts = []
                return

        if tag in ("p", "h1", "h2") and self._current_kind is None and not self._in_table:
            self._current_kind = tag
            self._text_parts = []
            self._anchors = []
            self._links = []
            return

        if tag == "a" and self._current_kind is not None:
            self._a_ctx = {"name": attrs_dict.get("name") or "", "href": attrs_dict.get("href") or ""}
            self._a_text_parts = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "article":
            self._flush_block()
            self._in_article = False
            return

        if not self._allowed():
            return

        if tag == "table":
            self._in_table = False
            return

        if self._in_table:
            if tag in ("td", "th") and self._in_cell:
                self._in_cell = False
                cell = _normalize_ws("".join(self._cell_text_parts))
                self._row_cells.append(cell)
                self._cell_text_parts = []
                return
            if tag == "tr" and self._in_tr:
                self._in_tr = False
                row = " | ".join([c for c in (s.strip() for s in self._row_cells) if c])
                if row:
                    self.blocks.append(Block(kind="table_row", text=row, anchors=(), links=()))
                self._row_cells = []
                return

        if tag == "a" and self._a_ctx is not None and self._current_kind is not None:
            a_text = _normalize_ws("".join(self._a_text_parts))
            name = (self._a_ctx.get("name") or "").strip()
            href = (self._a_ctx.get("href") or "").strip()
            if name:
                self._anchors.append(Anchor(name=name, text=a_text))
            if href:
                self._links.append(Link(href=href, text=a_text))
            self._a_ctx = None
            self._a_text_parts = []
            return

        if self._current_kind == tag and tag in ("p", "h1", "h2"):
            self._flush_block()

    def handle_data(self, data: str) -> None:
        if not self._allowed():
            return

        if self._in_table and self._in_cell:
            self._cell_text_parts.append(data)
            return

        if self._current_kind is None:
            return

        self._text_parts.append(data)
        if self._a_ctx is not None:
            self._a_text_parts.append(data)

    def _flush_block(self) -> None:
        if self._current_kind is None:
            return
        text = _normalize_ws("".join(self._text_parts))
        if text:
            self.blocks.append(
                Block(kind=self._current_kind, text=text, anchors=tuple(self._anchors), links=tuple(self._links))
            )
        self._current_kind = None
        self._text_parts = []
        self._anchors = []
        self._links = []


def _parse_blocks_stdlib(html: str) -> list[Block]:
    restrict = "<article" in (html or "").lower()
    p = _BlockHTMLParser(restrict_to_article=restrict)
    p.feed(html or "")
    p.close()
    if restrict and not p._seen_article:
        p = _BlockHTMLParser(restrict_to_article=False)
        p.feed(html or "")
        p.close()
    return p.blocks


def _parse_blocks_bs4(html: str) -> list[Block]:
    from bs4 import BeautifulSoup  # type: ignore

    soup = BeautifulSoup(html or "", "lxml")
    root = soup.find("article") or soup

    blocks: list[Block] = []

    def collect_anchors_links(tag) -> tuple[tuple[Anchor, ...], tuple[Link, ...]]:
        anchors: list[Anchor] = []
        links: list[Link] = []
        for a in tag.find_all("a"):
            name = (a.get("name") or "").strip()
            href = (a.get("href") or "").strip()
            txt = _normalize_ws(a.get_text(" ", strip=True))
            if name:
                anchors.append(Anchor(name=name, text=txt))
            if href:
                links.append(Link(href=href, text=txt))
        return tuple(anchors), tuple(links)

    for el in root.find_all(["h1", "h2", "p", "table"], recursive=True):
        tag = el.name.lower()
        if tag in ("h1", "h2", "p"):
            txt = _normalize_ws(el.get_text(" ", strip=True))
            if not txt:
                continue
            anchors, links = collect_anchors_links(el)
            blocks.append(Block(kind=tag, text=txt, anchors=anchors, links=links))
        elif tag == "table":
            for tr in el.find_all("tr"):
                cells = []
                for td in tr.find_all(["td", "th"]):
                    cell = _normalize_ws(td.get_text(" ", strip=True))
                    if cell:
                        cells.append(cell)
                row = " | ".join(cells).strip()
                if row:
                    blocks.append(Block(kind="table_row", text=row, anchors=(), links=()))

    return blocks


def parse_blocks_from_html(html: str, *, backend: str = "auto") -> tuple[list[Block], str]:
    """
    Returns (blocks, backend_used).

    backend:
    - "auto": try bs4+lxml, else fallback to stdlib
    - "bs4": require bs4+lxml
    - "stdlib": force stdlib
    """
    backend = (backend or "auto").strip().lower()
    if backend not in {"auto", "bs4", "stdlib"}:
        raise ValueError(f"Unknown backend: {backend!r}")

    if backend in {"auto", "bs4"}:
        try:
            return _parse_blocks_bs4(html), "bs4"
        except Exception:
            if backend == "bs4":
                raise

    return _parse_blocks_stdlib(html), "stdlib"

