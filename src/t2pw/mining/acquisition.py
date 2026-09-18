"""Paper acquisition: explicit inputs, uploads, and frontier-driven search.

This is an **adapter**, not a second acquisition stack. Searching, caching,
full-text fetching and eligibility screening are ``t2pw.rag.acquire`` and
``t2pw.rag.eligibility`` doing their existing jobs; what lives here is the
mining-specific part: turning user-supplied identifiers and uploaded files into
the same :class:`~t2pw.mining.models.PaperRecord` shape, and recording which
frontier question found each paper.

Literature services are used freely here. The version-1 restriction is on
biochemical *fact* validation (ChEBI, KEGG, HMDB, UniProt, PathBank), not on
paper *discovery*.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from t2pw.mining.config import MiningConfig, StartingInput
from t2pw.mining.ids import paper_id as make_paper_id
from t2pw.mining.ids import sha256_bytes
from t2pw.mining.models import (
    EVIDENCE_PREPRINT,
    EVIDENCE_PRIMARY,
    EVIDENCE_REVIEW,
    EVIDENCE_UNKNOWN,
    LiteratureQuery,
    PaperRecord,
)

_PMID = re.compile(r"^(?:pmid[:\s]*)?(\d{4,9})$", re.IGNORECASE)
_PMCID = re.compile(r"^(?:pmcid[:\s]*)?(pmc\d{4,9})$", re.IGNORECASE)
_DOI = re.compile(r"^(?:doi[:\s]*)?(10\.\d{4,9}/\S+)$", re.IGNORECASE)
_URL = re.compile(r"^https?://\S+$", re.IGNORECASE)

_REVIEW_MARKERS = ("review", "overview", "perspectives on", "a survey of", "current concepts")
_PREPRINT_MARKERS = ("biorxiv", "medrxiv", "chemrxiv", "preprint", "research square")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PaperReferenceError(ValueError):
    """A supplied paper reference is not one of the accepted forms."""


def parse_paper_reference(raw: Any) -> Tuple[str, str]:
    """Classify a user-supplied paper reference.

    Returns ``(id_type, value)`` where ``id_type`` is ``pmid`` | ``pmcid`` |
    ``doi`` | ``url``. Raises :class:`PaperReferenceError` for anything else --
    guessing at an unparseable reference is how a run ends up citing the wrong
    paper.
    """
    text = str(raw or "").strip()
    if not text:
        raise PaperReferenceError("empty paper reference")

    lowered = text.lower()
    if lowered.startswith("pmcid:"):
        text, lowered = text.split(":", 1)[1].strip(), lowered.split(":", 1)[1].strip()
    elif lowered.startswith("pmid:"):
        value = text.split(":", 1)[1].strip()
        if value.isdigit():
            return "pmid", value
        raise PaperReferenceError(f"PMID must be numeric: {raw!r}")
    elif lowered.startswith("doi:"):
        return "doi", text.split(":", 1)[1].strip()

    match = _PMCID.match(text)
    if match:
        return "pmcid", match.group(1).upper()
    match = _PMID.match(text)
    if match:
        return "pmid", match.group(1)
    match = _DOI.match(text)
    if match:
        return "doi", match.group(1)
    if _URL.match(text):
        # A PMC / PubMed URL carries a better handle than the URL itself.
        pmc = re.search(r"(PMC\d{4,9})", text, re.IGNORECASE)
        if pmc:
            return "pmcid", pmc.group(1).upper()
        pubmed = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d{4,9})", text, re.IGNORECASE)
        if pubmed:
            return "pmid", pubmed.group(1)
        doi = re.search(r"(10\.\d{4,9}/\S+)", text)
        if doi:
            return "doi", doi.group(1).rstrip("/")
        return "url", text
    raise PaperReferenceError(
        f"unrecognised paper reference {raw!r}; expected a PMID, PMCID, DOI or http(s) URL"
    )


def make_paper_starting_input(raw: Any) -> StartingInput:
    id_type, value = parse_paper_reference(raw)
    return StartingInput(kind="paper", raw=str(raw), value=value, id_type=id_type)


def classify_evidence_type(*, title: str = "", abstract: str = "", article_type: str = "",
                           source: str = "", url: str = "") -> str:
    """Best-effort primary / review / preprint label. ``unknown`` when unclear.

    Labelled where it can be determined and honestly ``unknown`` where it cannot
    -- a review is still useful evidence, but a reader must be able to see that
    it is one.
    """
    declared = str(article_type or "").strip().lower()
    if declared:
        if "review" in declared:
            return EVIDENCE_REVIEW
        if "preprint" in declared:
            return EVIDENCE_PREPRINT
        if declared in {"journal article", "research-article", "article", "research article"}:
            return EVIDENCE_PRIMARY
    haystack = f"{source} {url}".lower()
    if any(m in haystack for m in _PREPRINT_MARKERS):
        return EVIDENCE_PREPRINT
    head = f"{title} {abstract[:400]}".lower()
    if any(m in head for m in _REVIEW_MARKERS):
        return EVIDENCE_REVIEW
    return EVIDENCE_UNKNOWN


@dataclass
class LoadedPaper:
    """A paper record plus the text this run will chunk."""

    record: PaperRecord
    text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"record": self.record.to_dict(), "text_chars": len(self.text)}


@dataclass
class AcquisitionResult:
    """What one search produced."""

    query: LiteratureQuery
    papers: List[LoadedPaper] = field(default_factory=list)
    excluded: List[PaperRecord] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query.to_dict(),
            "included": [p.record.paper_id for p in self.papers],
            "excluded": [p.paper_id for p in self.excluded],
            "errors": list(self.errors),
        }


class PaperSource:
    """The seam tests replace.

    The default implementation delegates to ``t2pw.rag.acquire``. A fixture
    source implements the same two methods and the whole mining stack runs
    offline, deterministically, with no API spend.
    """

    name = "rag.acquire"

    def search(self, context: Mapping[str, Any], *, sources: Sequence[str],
               max_papers: int) -> List[Any]:
        from t2pw.rag.acquire import search_candidates

        return search_candidates(dict(context), sources=list(sources), max_papers=int(max_papers))

    def full_text(self, candidate: Any) -> str:
        from t2pw.rag.acquire import fetch_full_text

        return fetch_full_text(candidate)


class OfflinePaperSource(PaperSource):
    """A source that never touches the network. Used by fixtures and replays."""

    name = "offline"

    def __init__(self, papers: Optional[Sequence[Any]] = None) -> None:
        self._papers = list(papers or [])

    def search(self, context: Mapping[str, Any], *, sources: Sequence[str],
               max_papers: int) -> List[Any]:  # noqa: ARG002
        return list(self._papers)[: max(0, int(max_papers))]

    def full_text(self, candidate: Any) -> str:
        return str(getattr(candidate, "full_text", "") or "")


def record_from_candidate(
    candidate: Any,
    *,
    round_index: int,
    query: Optional[LiteratureQuery] = None,
) -> PaperRecord:
    """Build a :class:`PaperRecord` from a ``rag.acquire.CandidatePaper``."""
    ident = str(getattr(candidate, "id", "") or "").strip()
    pmid = pmcid = doi = ""
    if ident.upper().startswith("PMC"):
        pmcid = ident.upper()
    elif ident.isdigit():
        pmid = ident
    elif ident.lower().startswith("10."):
        doi = ident
    url = str(getattr(candidate, "source_uri", "") or "")
    title = str(getattr(candidate, "title", "") or "")
    abstract = str(getattr(candidate, "abstract", "") or "")
    source = str(getattr(candidate, "source", "") or "")

    return PaperRecord(
        paper_id=make_paper_id(pmid=pmid, pmcid=pmcid, doi=doi, url=url, title=title),
        pmid=pmid,
        pmcid=pmcid,
        doi=doi,
        title=title,
        journal=str(getattr(candidate, "journal", "") or ""),
        year=str(getattr(candidate, "year", "") or ""),
        url=url,
        acquisition_source=source,
        retrieved_at=_utcnow(),
        abstract=abstract,
        evidence_type=classify_evidence_type(
            title=title,
            abstract=abstract,
            article_type=str(getattr(candidate, "article_type", "") or ""),
            source=source,
            url=url,
        ),
        discovered_round=int(round_index),
        observed_organisms=[str(o) for o in getattr(candidate, "observed_organisms", []) or []],
        observed_pathways=[str(p) for p in getattr(candidate, "observed_pathways", []) or []],
        discovered_via_query_id=query.query_id if query else "",
        discovered_via_frontier_id=query.frontier_item_id if query else "",
    )


def record_from_upload(
    *,
    data: bytes,
    file_name: str,
    round_index: int = 0,
    title: str = "",
    text: str = "",
) -> LoadedPaper:
    """Turn uploaded bytes into a paper record with a content hash.

    Duplicate uploads are deterministic: the same bytes always produce the same
    ``paper_id`` (``FILE:<sha256[:16]>``), so re-uploading a PDF merges into the
    existing record instead of creating a second citation for one paper.

    PDF text extraction reuses ``t2pw.extraction.pdf_parser`` when the bytes look
    like a PDF; a caller may also pass already-extracted ``text``.
    """
    digest = sha256_bytes(data)
    body = str(text or "")
    notes: List[str] = []
    if not body:
        if data[:5] == b"%PDF-":
            body, note = _extract_pdf_text(data, file_name)
            if note:
                notes.append(note)
        else:
            try:
                body = data.decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                body = ""
                notes.append("uploaded bytes could not be decoded as text")

    record = PaperRecord(
        paper_id=f"FILE:{digest[:16]}",
        title=str(title or Path(file_name).stem),
        file_sha256=digest,
        file_name=str(file_name),
        acquisition_source="upload",
        retrieved_at=_utcnow(),
        evidence_type=classify_evidence_type(title=title),
        discovered_round=int(round_index),
        full_text_chars=len(body),
        notes=notes,
    )
    return LoadedPaper(record=record, text=body)


def _extract_pdf_text(data: bytes, file_name: str) -> Tuple[str, str]:
    """Extract text from PDF bytes via the existing parser. Never raises."""
    import tempfile

    try:
        from t2pw.extraction.pdf_parser import extract_text_from_pdf
    except Exception as exc:  # noqa: BLE001
        return "", f"PDF parsing unavailable: {exc}"
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
            handle.write(data)
            tmp = handle.name
        result = extract_text_from_pdf(tmp)
        if isinstance(result, tuple):
            result = result[0]
        if isinstance(result, Mapping):
            result = result.get("text") or result.get("full_text") or ""
        return str(result or ""), ""
    except Exception as exc:  # noqa: BLE001
        return "", f"PDF text extraction failed for {file_name}: {exc}"
    finally:
        if tmp:
            try:
                Path(tmp).unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass


class Acquirer:
    """Frontier-driven paper acquisition with screening and de-duplication."""

    def __init__(
        self,
        config: MiningConfig,
        *,
        source: Optional[PaperSource] = None,
        screener: Optional[Callable[[Any, MiningConfig], Any]] = None,
    ) -> None:
        self.config = config
        self.source = source or (OfflinePaperSource() if config.offline else PaperSource())
        self._screener = screener
        self.seen_paper_ids: set = set()

    # -- screening ----------------------------------------------------------
    def screen(self, candidate: Any) -> Optional[Any]:
        """Run the existing eligibility screen. ``None`` when screening is off."""
        if not self.config.screen_papers:
            return None
        if self._screener is not None:
            return self._screener(candidate, self.config)
        try:
            from t2pw.rag.eligibility import RequestedScope, screen_candidate

            scope = RequestedScope(
                requested_pathway=self.config.topic,
                requested_organism=self.config.target_species,
            )
            return screen_candidate(candidate, scope)
        except Exception:  # noqa: BLE001 - a screening failure must not lose the paper
            return None

    # -- acquisition --------------------------------------------------------
    def acquire(
        self,
        query: LiteratureQuery,
        context: Mapping[str, Any],
        *,
        round_index: int,
        max_papers: int,
    ) -> AcquisitionResult:
        """Search, screen and load full text for one query."""
        result = AcquisitionResult(query=query)
        try:
            candidates = self.source.search(
                context, sources=self.config.acquisition_sources, max_papers=max_papers
            )
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"{type(exc).__name__}: {exc}")
            query.error = result.errors[-1]
            return result

        query.result_count = len(candidates)
        for candidate in candidates:
            record = record_from_candidate(candidate, round_index=round_index, query=query)
            if record.paper_id in self.seen_paper_ids:
                continue

            decision = self.screen(candidate)
            if decision is not None:
                record.screening_outcome = str(getattr(decision, "outcome", "") or "")
                record.screening_score = float(getattr(decision, "score", 0.0) or 0.0)
                reason = str(getattr(decision, "reason", "") or "")
                if getattr(decision, "ineligible", False):
                    record.exclusion_reasons.append(reason or "screened out by eligibility")
                    self.seen_paper_ids.add(record.paper_id)
                    result.excluded.append(record)
                    continue
                if reason:
                    record.inclusion_reasons.append(reason)
                for attr, target in (
                    ("observed_organisms", record.observed_organisms),
                    ("observed_pathways", record.observed_pathways),
                ):
                    for value in getattr(decision, attr, None) or []:
                        text = str(value or "").strip()
                        if text and text not in target:
                            target.append(text)

            try:
                text = self.source.full_text(candidate)
            except Exception as exc:  # noqa: BLE001
                text = ""
                record.notes.append(f"full-text fetch failed: {type(exc).__name__}: {exc}")
            if not text:
                text = str(getattr(candidate, "full_text", "") or "")
            body = text or record.abstract
            if not body:
                record.exclusion_reasons.append("no retrievable text (no full text and no abstract)")
                self.seen_paper_ids.add(record.paper_id)
                result.excluded.append(record)
                continue

            record.full_text_chars = len(body)
            record.cache_ref = f"{self.source.name}:{getattr(candidate, 'id', '')}"
            self.seen_paper_ids.add(record.paper_id)
            query.new_paper_ids.append(record.paper_id)
            result.papers.append(LoadedPaper(record=record, text=body))
        return result

    def load_explicit(
        self, inputs: Sequence[StartingInput], *, round_index: int = 0
    ) -> Tuple[List[LoadedPaper], List[str]]:
        """Load user-supplied PMIDs / PMCIDs / DOIs / URLs. Returns errors too."""
        loaded: List[LoadedPaper] = []
        errors: List[str] = []
        for item in inputs:
            if item.kind != "paper" or item.id_type == "file":
                continue
            candidate = self._candidate_for(item)
            if candidate is None:
                errors.append(f"could not resolve {item.raw!r} ({item.id_type})")
                continue
            record = record_from_candidate(candidate, round_index=round_index)
            record.acquisition_source = record.acquisition_source or "explicit_input"
            record.inclusion_reasons.append(f"supplied directly by the user as {item.id_type}")
            if item.id_type == "url" and not record.url:
                record.url = item.value
            try:
                text = self.source.full_text(candidate)
            except Exception as exc:  # noqa: BLE001
                text = ""
                record.notes.append(f"full-text fetch failed: {type(exc).__name__}: {exc}")
            body = text or str(getattr(candidate, "abstract", "") or "")
            if not body:
                errors.append(f"no retrievable text for {item.raw!r}")
                record.exclusion_reasons.append("no retrievable text")
                continue
            record.full_text_chars = len(body)
            if record.paper_id in self.seen_paper_ids:
                continue
            self.seen_paper_ids.add(record.paper_id)
            loaded.append(LoadedPaper(record=record, text=body))
        return loaded, errors

    def _candidate_for(self, item: StartingInput) -> Optional[Any]:
        """Build the minimal candidate object the source's fetcher understands."""
        try:
            from t2pw.rag.acquire import CandidatePaper
        except Exception:  # noqa: BLE001
            return None
        ident = item.value
        if item.id_type == "url":
            return CandidatePaper(id=ident, source="explicit_url", source_uri=ident)
        return CandidatePaper(id=ident, source=f"explicit_{item.id_type}")


__all__ = [
    "PaperReferenceError",
    "parse_paper_reference",
    "make_paper_starting_input",
    "classify_evidence_type",
    "LoadedPaper",
    "AcquisitionResult",
    "PaperSource",
    "OfflinePaperSource",
    "Acquirer",
    "record_from_candidate",
    "record_from_upload",
]
