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
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from itertools import zip_longest
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

#: Production-screen classifications the mining rescue may reconsider: the
#: pathway-anchor branch of ``rag.eligibility.screen_paper``, never a veto.
_RESCUABLE = ("context_only", "omics_only", "off_topic")
#: The ones route 1 (score only) may reconsider. ``off_topic`` names no topic term
#: at all, so only route 2, which needs a run name, may bring one back.
_SCORE_RESCUABLE = ("context_only", "omics_only")
#: Without the pathway anchor, a paper must clear the production bar by this much.
#: ponytail: one number, fit on pilot-001's 21 screened papers (wheat 2.5 and a
#: Marchantia primary 2.5 stay out; AtLOX2 enzymology 4.0 and DAD1 3.0 come in).
#: Re-fit when more pilots have been adjudicated.
_UNANCHORED_MARGIN = 1.0
#: A sentence break, but not after an abbreviated genus ("A. thaliana").
_SENTENCE = re.compile(r"(?<=[.!?])(?<!\b[A-Z]\.)\s+")


def _fold(text: str) -> str:
    return " ".join(str(text or "").split()).lower()


def _names(folded: str, name: str) -> bool:
    """``name`` occurs in ``folded`` as a whole word or phrase."""
    return bool(name) and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", folded) is not None


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
        """``search_candidates`` once per source, then the sources taken in turn.

        One call over all sources caps the MERGED list, so EuropePMC's first page
        filled it and NCBI's results were never reached (pilot-001 F2). Here each
        source is searched separately through the same existing API, and the
        lists are interleaved and de-duplicated before the ``max_papers`` cap.
        """
        from t2pw.rag.acquire import search_candidates

        per_source = [
            search_candidates(dict(context), sources=[source], max_papers=int(max_papers))
            for source in sources
        ]
        merged: List[Any] = []
        seen: set = set()
        for candidate in (c for row in zip_longest(*per_source) for c in row if c is not None):
            keys = set(candidate.identity_keys())
            if keys & seen:
                continue
            seen |= keys
            merged.append(candidate)
        return merged[: max(0, int(max_papers))]

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
    def _scope(self) -> Any:
        from t2pw.rag.eligibility import RequestedScope

        return RequestedScope(
            requested_pathway=self.config.topic,
            requested_organism=self.config.target_species,
        )

    def screen(self, candidate: Any) -> Optional[Any]:
        """Run the existing eligibility screen. ``None`` when screening is off."""
        if not self.config.screen_papers:
            return None
        if self._screener is not None:
            return self._screener(candidate, self.config)
        try:
            from t2pw.rag.eligibility import screen_candidate

            return screen_candidate(candidate, self._scope())
        except Exception:  # noqa: BLE001 - a screening failure must not lose the paper
            return None

    def rescue(self, candidate: Any, decision: Any, run_terms: Sequence[str] = ()) -> str:
        """Why mining re-admits a paper the production screen excluded, or ``""``.

        Only an exclusion by the pathway-anchor rule (classification
        ``context_only`` / ``omics_only`` / ``off_topic``) is reconsidered. An
        organism veto or a negative-evidence veto never is. Two routes:

        1. ``context_only`` / ``omics_only`` only (never ``off_topic``): the
           production screen names the target species (``organism_match ==
           "match"``), and ``screen_candidate``'s own thresholds, with the anchor
           rule off and both bars (full and title-only) raised by
           ``_UNANCHORED_MARGIN``, pass the paper.
        2. The title names the target species, and one sentence of the title or
           abstract names it together with an enzyme or metabolite from this
           run's seeds or frontier (``run_terms``). Both guards come from
           pilot-001's wheat transgene paper. Its abstract names *A. thaliana*
           only as a gene source ("ALLENE OXIDE SYNTHASE from Arabidopsis
           thaliana"), and its title names wheat.

        ``rag/eligibility.py`` is not changed. No pathway vocabulary is hard-coded.
        """
        classification = str(getattr(decision, "classification", "") or "")
        reason = str(getattr(decision, "reason", "") or "")
        if classification not in _RESCUABLE or not reason.startswith(f"{classification}:"):
            return ""
        try:
            from t2pw.rag.eligibility import screen_candidate, thresholds_from_config

            if (classification in _SCORE_RESCUABLE
                    and str(getattr(decision, "organism_match", "") or "") == "match"):
                base = thresholds_from_config()
                # Both bars are raised: with no abstract, screen_paper applies the title-only one.
                relaxed = replace(base, require_pathway_anchor=False,
                                  min_score=base.min_score + _UNANCHORED_MARGIN,
                                  title_only_min_score=base.title_only_min_score + _UNANCHORED_MARGIN)
                second = screen_candidate(candidate, self._scope(), thresholds=relaxed, apply=False)
                if not second.ineligible:
                    return (
                        f"{self.config.target_species} named; score {round(second.score, 3)} >= "
                        f"{second.threshold} with the pathway-anchor rule off"
                    )
        except Exception:  # noqa: BLE001 - a failed rescue leaves the exclusion standing
            pass

        from t2pw.mining.species import species_matches_target

        target = self.config.target_species_normalized
        genus, _, epithet = target.partition(" ")
        forms = [target] + ([f"{genus[0]}. {epithet}"] if epithet else [])
        if species_matches_target(genus, target):  # species.py's documented genus convenience
            forms.append(genus)
        title = str(getattr(candidate, "title", "") or "")
        if not any(_names(_fold(title), f) for f in forms):
            return ""
        for sentence in _SENTENCE.split(f"{title}. {getattr(candidate, 'abstract', '') or ''}"):
            folded = _fold(sentence)
            named = next((f for f in forms if _names(folded, f)), "")
            term = next((t for t in run_terms if named and _names(folded, _fold(t))), "")
            if term:
                return f"names {named!r} and {term!r} in one sentence"
        return ""

    # -- acquisition --------------------------------------------------------
    def acquire(
        self,
        query: LiteratureQuery,
        context: Mapping[str, Any],
        *,
        round_index: int,
        max_papers: int,
        run_terms: Sequence[str] = (),
    ) -> AcquisitionResult:
        """Search, screen and load full text for one query.

        ``run_terms`` are this run's seed and frontier names, used only by
        :meth:`rescue`.
        """
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
                # The rescue reconsiders the PRODUCTION screen's verdicts only; an
                # injected test screener's verdict stands as given.
                rescued = (
                    self.rescue(candidate, decision, run_terms)
                    if getattr(decision, "ineligible", False) and self._screener is None
                    else ""
                )
                if getattr(decision, "ineligible", False) and not rescued:
                    record.exclusion_reasons.append(reason or "screened out by eligibility")
                    self.seen_paper_ids.add(record.paper_id)
                    result.excluded.append(record)
                    continue
                if rescued:
                    record.screening_outcome = "mining_rescue"
                    record.inclusion_reasons.append(f"mining_rescue: {rescued}")
                    record.notes.append(f"the production screen excluded it: {reason}")
                elif reason:
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
