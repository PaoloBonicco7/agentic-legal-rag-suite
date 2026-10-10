"""Resumable sequential acquisition of immutable source snapshots."""

import re
import shutil
import time
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

import requests
from lxml import html
from pydantic import Field

from legal_rag.artifacts import digest, file_hash, new_output, read_json, utc_now, write_json, write_jsonl
from legal_rag.contracts import Record, SCHEMA_VERSION, SourceRecord

BASE_URL = "https://www.consiglio.vda.it"
SEARCH_PATH = "/app/leggieregolamenti/risultatiricerca"
ACQUISITION_VERSION = "acquisition-v1"


class AcquisitionConfig(Record):
    base_url: str = BASE_URL
    interval_seconds: float = Field(default=1.0, ge=1.0)
    timeout_seconds: int = Field(default=30, ge=1, le=180)
    retries: int = Field(default=2, ge=0, le=3)


class SnapshotManifest(Record):
    schema_version: str = SCHEMA_VERSION
    acquisition_version: str = ACQUISITION_VERSION
    snapshot_id: str
    acquisition_started_at: str | None = None
    acquisition_finished_at: str | None = None
    imported_at: str | None = None
    complete: bool = False
    legacy: bool = False
    current_completeness_verified: bool = False
    config: AcquisitionConfig | None = None
    errors: list[str] = Field(default_factory=list)
    source_count: int = Field(default=0, ge=0)
    catalog_sha256: str | None = None
    encoding_anomalies: list[str] = Field(default_factory=list)


def _save_manifest(path: Path, state: dict) -> None:
    write_json(path, SnapshotManifest.model_validate(state))


def decode_source(raw: bytes, declared: str | None = None) -> tuple[str, str]:
    encoding = (declared or "utf-8").lower()
    if encoding in {"iso-8859-1", "latin-1", "latin1"} and any(0x80 <= b <= 0x9F for b in raw):
        encoding = "windows-1252"
    text = raw.decode(encoding)
    if "\ufffd" in text or any(0x80 <= ord(char) <= 0x9F for char in text):
        raise ValueError("Source contains replacement or C1 control characters")
    return text, encoding


def parse_catalogue(text: str, base_url: str = BASE_URL) -> tuple[dict[str, dict], int]:
    tree = html.fromstring(text)
    entries: dict[str, dict] = {}
    page_count = re.search(r"\b\d+\s+di\s+(\d+)\s*\(Pagina corrente\)", " ".join(" ".join(tree.itertext()).split()), re.I)
    last_page = int(page_count[1]) if page_count else 1
    for anchor in tree.xpath('//a[@href]'):
        url = urljoin(base_url + SEARCH_PATH, anchor.get("href"))
        parsed = urlparse(url)
        if parsed.netloc != urlparse(base_url).netloc:
            continue
        params = parse_qs(parsed.query)
        if parsed.path.endswith("risultatiricerca") and "pagina" in params:
            last_page = max(last_page, int(params["pagina"][0]))
        identity = " ".join(anchor.text_content().split())
        if parsed.path.endswith("dettaglio") and "pk_lr" in params and identity.lower().startswith("legge regionale"):
            source_id = params["pk_lr"][0]
            if not source_id.isdigit():
                raise ValueError("Unexpected source identifier")
            detail = base_url + "/app/leggieregolamenti/dettaglio?" + urlencode({"pk_lr": source_id, "versione": "V"})
            entry = {"source_id": source_id, "url": detail, "identity": identity}
            if source_id in entries and entries[source_id] != entry:
                raise ValueError(f"Conflicting source identity: {source_id}")
            entries[source_id] = entry
    if not entries:
        raise ValueError("Catalogue page contains no law records")
    return entries, last_page


def validate_detail(text: str, expected_identity: str) -> None:
    tree = html.fromstring(text)
    titles = tree.xpath('//*[@id="tmpl-banche-dati-titolo-secondario"]')
    if not titles:
        titles = tree.xpath('//article//h1|//article//h2')
    normalized = lambda value: re.sub(r"\s+", " ", value).strip().casefold()
    if not any(re.search(re.escape(normalized(expected_identity)) + r"(?!\d)", normalized(title.text_content())) for title in titles):
        raise ValueError(f"Detail identity differs from catalogue: {expected_identity}")


class _Downloader:
    def __init__(self, config: AcquisitionConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "legal-rag-research/0.2"
        self.last_request = 0.0

    def get(self, url: str):
        error = None
        for attempt in range(self.config.retries + 1):
            remaining = self.config.interval_seconds - (time.monotonic() - self.last_request)
            if remaining > 0:
                time.sleep(remaining)
            self.last_request = time.monotonic()
            try:
                response = self.session.get(url, timeout=self.config.timeout_seconds)
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                error = exc
                if attempt < self.config.retries:
                    time.sleep(min(2 ** attempt, 4))
        raise RuntimeError(f"Source request failed: {url}") from error


def _catalogue(downloader: _Downloader, directory: Path, progress=None, *, use_cache: bool = True) -> dict[str, dict]:
    directory.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict] = {}
    page, last_page = 1, 1
    while page <= last_page:
        if page > 2000:
            raise ValueError("Unexpected catalogue pagination")
        path = directory / f"{page:04d}.html"
        metadata = path.with_suffix(".json")
        if use_cache and path.exists() and metadata.exists():
            meta = read_json(metadata)
            if file_hash(path) != meta["sha256"]:
                raise ValueError("Cached catalogue hash mismatch")
            text, _ = decode_source(path.read_bytes(), meta["encoding"])
        else:
            url = downloader.config.base_url + SEARCH_PATH + "?" + urlencode({"tipo": 1, "ricerca_in": 3, "pagina": page})
            response = downloader.get(url)
            text, encoding = decode_source(response.content, response.encoding)
            path.write_bytes(response.content)
            write_json(metadata, {"sha256": file_hash(path), "encoding": encoding, "fetched_at": utc_now()})
        found, linked_last = parse_catalogue(text, downloader.config.base_url)
        for source_id, entry in found.items():
            if source_id in entries and entries[source_id] != entry:
                raise ValueError(f"Catalogue identity conflict: {source_id}")
            entries[source_id] = entry
        last_page = max(last_page, linked_last)
        if progress:
            progress({"stage": directory.name, "page": page, "pages_seen": last_page, "laws": len(entries)})
        page += 1
    return entries


def acquire_snapshot(output_dir: Path, *, config: AcquisitionConfig | None = None, resume: bool = False, progress=None) -> dict:
    config = config or AcquisitionConfig()
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    if output_dir.exists():
        if not resume:
            raise FileExistsError(output_dir)
        state = SnapshotManifest.model_validate(read_json(manifest_path)).to_json_record()
        if state["complete"]:
            raise FileExistsError("Completed snapshots are immutable")
        if state["config"] != config.to_json_record():
            raise ValueError("Acquisition configuration changed on resume")
    else:
        new_output(output_dir)
        state = {"schema_version": SCHEMA_VERSION, "acquisition_version": ACQUISITION_VERSION,
                 "snapshot_id": "pending", "acquisition_started_at": utc_now(),
                 "acquisition_finished_at": None, "complete": False, "legacy": False,
                 "config": config.to_json_record(), "errors": [], "source_count": 0}
        _save_manifest(manifest_path, state)
    downloader = _Downloader(config)
    try:
        initial = _catalogue(downloader, output_dir / "catalogue_initial", progress)
        write_json(output_dir / "discovery.json", initial)
        source_dir = output_dir / "sources"
        source_dir.mkdir(exist_ok=True)
        sources = []
        for position, source_id in enumerate(sorted(initial, key=int), 1):
            entry = initial[source_id]
            path = source_dir / f"{source_id}.html"
            metadata = source_dir / f"{source_id}.json"
            if path.exists() and metadata.exists():
                source = SourceRecord.model_validate(read_json(metadata))
                if file_hash(path) != source.sha256:
                    raise ValueError(f"Source hash mismatch: {source_id}")
                text, _ = decode_source(path.read_bytes(), source.encoding)
            else:
                response = downloader.get(entry["url"])
                text, encoding = decode_source(response.content, response.encoding)
                validate_detail(text, entry["identity"])
                path.write_bytes(response.content)
                source = SourceRecord(source_id=source_id, url=entry["url"], path=str(path.relative_to(output_dir)),
                                      sha256=file_hash(path), fetched_at=utc_now(), encoding=encoding)
                write_json(metadata, source)
            validate_detail(text, entry["identity"])
            sources.append(source)
            if progress:
                progress({"stage": "details", "completed": position, "total": len(initial)})
        final = _catalogue(downloader, output_dir / "catalogue_final", progress, use_cache=False)
        if initial != final:
            raise ValueError("Catalogue changed during acquisition; do not promote this snapshot")
        write_jsonl(output_dir / "catalog.jsonl", sources)
        state.update(snapshot_id="vda-" + digest("\n".join(s.source_id + ":" + s.sha256 for s in sources))[:16],
                     source_count=len(sources), acquisition_finished_at=utc_now(), complete=True,
                     catalog_sha256=file_hash(output_dir / "catalog.jsonl"), errors=[], current_completeness_verified=True)
        _save_manifest(manifest_path, state)
        return state
    except Exception as exc:
        state["errors"] = [f"{type(exc).__name__}: {exc}"]
        _save_manifest(manifest_path, state)
        raise
    finally:
        downloader.session.close()


def import_legacy(input_dir: Path, output_dir: Path) -> dict:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    files = sorted(input_dir.glob("*.html"))
    if not files:
        raise ValueError("No legacy HTML documents found")
    new_output(output_dir)
    (output_dir / "sources").mkdir()
    sources = []
    anomalies = []
    for path in files:
        source_id = path.stem.split("_", 1)[0]
        text = path.read_bytes().decode("utf-8")
        encoding = "utf-8"
        if "\ufffd" in text or any(0x80 <= ord(char) <= 0x9F for char in text):
            anomalies.append(path.name)
        target = output_dir / "sources" / path.name
        shutil.copy2(path, target)
        sources.append(SourceRecord(source_id=source_id, url=BASE_URL + "/app/leggieregolamenti/dettaglio?" + urlencode({"pk_lr": source_id, "versione": "V"}),
                                    path=str(target.relative_to(output_dir)), sha256=file_hash(target), encoding=encoding, representation="legacy"))
    if len({s.source_id for s in sources}) != len(sources):
        raise ValueError("Duplicate legacy source IDs")
    write_jsonl(output_dir / "catalog.jsonl", sources)
    manifest = {"schema_version": SCHEMA_VERSION, "acquisition_version": ACQUISITION_VERSION,
                "snapshot_id": "legacy-" + digest("\n".join(s.source_id + ":" + s.sha256 for s in sources))[:16],
                "acquisition_started_at": None, "acquisition_finished_at": None,
                "imported_at": utc_now(), "legacy": True, "complete": True,
                "current_completeness_verified": False, "source_count": len(sources),
                "catalog_sha256": file_hash(output_dir / "catalog.jsonl"), "encoding_anomalies": anomalies, "errors": []}
    _save_manifest(output_dir / "manifest.json", manifest)
    return manifest
