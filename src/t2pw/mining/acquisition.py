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

import html
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
#: Route 2 ignores a run name shorter than this: it matches by accident. Pilot
#: follow-up: "ALA" rescued a paper where it is 5-aminolevulinic acid, "Ala" one
#: where it is alanine. An acronym (no lower-case letter) must also match in case.
_MIN_TERM_CHARS = 5
#: A title that makes the target species a host for a gene: the maize ZmLOX2
#: "in Transgenic Arabidopsis" paper. The analogue of the wheat gene-source guard.
#: ponytail: a title word list, not a parse of where the gene came from; it also
#: turns away a native gene overexpressed in the target, which is fail-closed.
_FOREIGN_HOST = re.compile(
    r"\b(?:transgenic|heterologous(?:ly)?|ectopic(?:ally)?|over-?express\w*)\b", re.IGNORECASE)
#: Hosts neither existing organism detector knows: ``rag.eligibility`` has no crop
#: entries and species.py's common-name table has no wheat or grape. Route 1 only.
#: ponytail: four names, each from a live false rescue: the two wheat transgene
#: papers (PMC11478715, PMC10742046) and pilot-002's grape VaPAT1 paper
#: (PMC5104190). A host both detectors and this list miss (barley, soybean,
#: "Nicotiana tabacum" written as a binomial) still passes; add it when a false
#: rescue shows one, or replace the list with a taxonomy lookup.
_EXTRA_HOSTS = ("wheat", "triticum", "grape", "vitis")


def _fold(text: str) -> str:
    return " ".join(str(text or "").split()).lower()


def _names(folded: str, name: str) -> bool:
    """``name`` occurs in ``folded`` as a whole word or phrase."""
    return bool(name) and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", folded) is not None


def _term_in(sentence: str, term: str) -> bool:
    """``term`` occurs in ``sentence``; an acronym (no lower-case letter) only in its own case."""
    if any(c.islower() for c in term):
        return _names(_fold(sentence), _fold(term))
    return _names(" ".join(sentence.split()), term)


#: A fetched full text opens with its own title (EuropePMC fullTextXML front
#: matter). Only this much of it is compared, and this share of the record's
#: title words must be there.
_TITLE_HEAD_CHARS = 3000
_TITLE_MATCH = 0.7


def _title_words(text: str) -> set:
    plain = re.sub(r"<[^>]+>", " ", html.unescape(html.unescape(str(text or ""))))
    return {w for w in re.findall(r"[a-z0-9]+", plain.lower()) if len(w) >= 3}


def full_text_matches_title(title: str, text: str) -> bool:
    """Does ``text`` open with the article called ``title``? Normalised word overlap.

    Pilot-002 fetched another article's full text for two NCBI records (their
    PMCID came from a cited reference) and extracted from it under the included
    paper's title. A title too short to judge (under 3 words of 3+ letters) passes.
    """
    wanted = _title_words(title)
    if len(wanted) < 3:
        return True
    return len(wanted & _title_words(str(text or "")[:_TITLE_HEAD_CHARS])) / len(wanted) >= _TITLE_MATCH


class _CapturingClient:
    """Pass-through HTTP client that keeps the NCBI efetch XML it returned."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.efetch_xml: List[str] = []

    def get(self, url: Any, **kwargs: Any) -> Any:
        resp = self.inner.get(url, **kwargs)
        if "efetch.fcgi" in str(url):
            self.efetch_xml.append(str(getattr(resp, "text", "") or ""))
        return resp


#: A title that calls the paper a review. Word-bounded: "preview" and "reviewed" do not count.
_REVIEW_TITLE = re.compile(
    r"\b(?:mini-?)?(?:review|overview)s?\b|\bperspectives? on\b|\ba survey of\b|\bcurrent concepts\b"
    r"|\brecent (?:advances|progress)\b", re.IGNORECASE)
#: A paper saying it is a review: "this review", "the present mini-review", "in this
#: minireview", "here we review", "we summarize and review", "this article provides a
#: comprehensive review of", "this work reviews", "we provide an overview", "the aim of this
#: paper is to review", "a review of the literature ... is presented". Only self-reference
#: counts, so a primary paper citing someone else's review ("reviewed in [3]", "for a review,
#: see", "(reviewed by ...)", "recently reviewed") never matches; neither do "Writing -
#: review & editing", "Institutional Review Board", "Reviewed by:" (the peer reviewer) or a
#: cited review's title. A bare "review article" is not a rule: in body text it is almost
#: always a citation of someone else's; "this review article" is caught by the first branch.
#: ponytail: a phrase list fit on pilot-001's three reviews (two the old title/abstract
#: markers missed), pilot-003's bryophyte review (no marker until its introduction says
#: "this review makes four specific contributions"), the review phrasings the 6ceef736
#: review listed, all eleven pilot-002/003 primaries and the 23 non-reviews among the 28
#: texts committed under runs/2026-07-27_*. A review that never says so in these words, in
#: its subject heading or in a declared type, stays ``unknown``.
_SELF_REVIEW = re.compile(
    r"\b(?:this|the present|the current) (?:\w+[ -])?(?:mini-?)?(?:review|overview)\b"
    r"|\b(?:this|the present|the current) (?:article|paper|work|chapter|manuscript) (?:\w+ ){0,3}"
    r"(?:reviews?|overview)\b"
    r"|\bwe (?:\w+ )?(?:and )?(?:review|overview)\b"
    r"|\bwe (?:provide|present|give|offer) an? (?:\w+ ){0,2}(?:review|overview)\b"
    r"|\baim of this (?:paper|article|work|review) is to review\b"
    r"|\ban? (?:\w+ ){0,2}review of the literature\b[^.]{0,150}\b(?:is|are) (?:here )?presented\b",
    re.IGNORECASE)
#: A declared type naming a review ("Review", "review-article", "Systematic Review",
#: "Minireview"), but not "reviewed-preprint" or "Peer Review".
_DECLARED_REVIEW = re.compile(r"(?<!peer )(?<!peer-)\b(?:mini-?)?review(?!ed)", re.IGNORECASE)
#: The JATS subject heading EuropePMC flattens just before the record title: "... 779 1 Review
#: Update of the sideroflexin ..." (PMC12180156), "... 1 JBC Reviews The regulation of lipid A
#: biosynthesis" (PMC12444477). Anchored to the end of the words before the title, and
#: case-sensitive, so a CRediT "Writing - review & editing" or a "Reviewed by:" never counts.
_SUBJECT_REVIEW = re.compile(
    r"(?<![\w-])(?:JBC Reviews|Systematic Review|Mini-?[Rr]eview|Minireview|MiniReview"
    r"|Review [Aa]rticle|Reviews?|REVIEW)\s*$")
_PREPRINT_MARKERS = ("biorxiv", "medrxiv", "chemrxiv", "preprint", "research square")
#: Declared types that mark a primary paper. Not "Journal Article": PubMed gives it to
#: reviews too, so a review whose record lists only that would be graded primary.
_PRIMARY_TYPES = frozenset({"research-article", "article", "research article", "research paper"})


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


#: Tag-shaped markup only: "<" then a letter. A "<" before a space or a digit is text
#: ("P < 0.05", "p <0.05"): stripping "<...>" spliced "(P &lt; 0.05) and OPDA rose &gt;2-fold"
#: into "(P 2-fold", a quote the paper never printed.
_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9]*\b[^<>]*>")


def _plain(text: str) -> str:
    """Tags stripped, entities decoded (EuropePMC titles arrive double-escaped), spaces folded."""
    return " ".join(_TAG.sub(" ", html.unescape(html.unescape(str(text or "")))).split())


def _subject_heading(text: str, title: str) -> str:
    """The review subject heading printed just before ``title`` in ``text``'s opening, or ``""``."""
    words = re.findall(r"\w+", _plain(title))[:6]
    if len(words) < 3:
        return ""
    head = " ".join(str(text or "")[:_TITLE_HEAD_CHARS].split())
    at = re.search(r"\W+".join(map(re.escape, words)), head, re.IGNORECASE)
    hit = _SUBJECT_REVIEW.search(head[max(0, at.start() - 40):at.start()]) if at else None
    return hit.group(0).strip() if hit else ""


def grade_evidence_type(*, title: str = "", abstract: str = "", article_type: str = "",
                        source: str = "", url: str = "", full_text: str = "") -> Tuple[str, str]:
    """``(evidence_type, basis)``: the primary / review / preprint label and the rule that decided it.

    A review mislabelled primary is the policy violation (its statements would grade as
    primary evidence); a primary mislabelled review only downgrades its grade. So any
    review signal wins: a declared type naming a review ("Review", "Systematic Review",
    "review-article"); the JATS subject heading printed just before the title in the
    full text's front matter (``_subject_heading``: "Review", "JBC Reviews"; some reviews
    say it nowhere else); a title that calls the paper a review; the abstract or the
    paper's own full text saying so (:data:`_SELF_REVIEW`). Pilot-003's bryophyte review
    (PMC13364333) declares nothing the search returned and says it only in its
    introduction. ``full_text`` is the paper's text as fetched: the self-reference rules
    read it without its front matter and reference list (``retrieval.body_text``), since a
    cited review's title is not this paper. After that: preprint (declared, or by source /
    URL), primary (a declared "research-article", never "Journal Article" alone), else
    honestly ``unknown``. ``article_type`` may list several declared types joined by ``;``.
    """
    from t2pw.mining.retrieval import body_text  # retrieval imports this module

    declared = [t.strip() for t in str(article_type or "").split(";") if t.strip()]
    hit = next((t for t in declared if _DECLARED_REVIEW.search(t)), "")
    if hit:
        return EVIDENCE_REVIEW, f"declared article type {hit!r}"
    hit = _subject_heading(full_text, title)
    if hit:
        return EVIDENCE_REVIEW, f"front-matter subject heading {hit!r} before the title"
    for where, text, rule in (("title", title, _REVIEW_TITLE), ("abstract", abstract, _SELF_REVIEW),
                              ("full text", body_text(full_text), _SELF_REVIEW)):
        text = _plain(text)
        match = rule.search(text)
        if match:
            quote = text[max(0, match.start() - 40):match.end() + 40]
            return EVIDENCE_REVIEW, f"{where} says {match.group(0)!r}: ...{quote}..."
    hit = next((t for t in declared if "preprint" in t.lower()), "")
    if hit:
        return EVIDENCE_PREPRINT, f"declared article type {hit!r}"
    hit = next((m for m in _PREPRINT_MARKERS if m in f"{source} {url}".lower()), "")
    if hit:
        return EVIDENCE_PREPRINT, f"source or URL names {hit!r}"
    hit = next((t for t in declared if t.lower() in _PRIMARY_TYPES), "")
    if hit:
        return EVIDENCE_PRIMARY, f"declared article type {hit!r}"
    return EVIDENCE_UNKNOWN, ""


def classify_evidence_type(*, title: str = "", abstract: str = "", article_type: str = "",
                           source: str = "", url: str = "", full_text: str = "") -> str:
    """Best-effort primary / review / preprint label. ``unknown`` when unclear.

    Labelled where it can be determined and honestly ``unknown`` where it cannot
    -- a review is still useful evidence, but a reader must be able to see that
    it is one. The rules, and why a review signal wins: :func:`grade_evidence_type`.
    """
    return grade_evidence_type(title=title, abstract=abstract, article_type=article_type,
                               source=source, url=url, full_text=full_text)[0]


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
    #: Refuse a fetched full text that does not open with the record's title.
    verify_full_text_title = True

    def search(self, context: Mapping[str, Any], *, sources: Sequence[str],
               max_papers: int) -> List[Any]:
        """``search_candidates`` once per source, then the sources taken in turn.

        One call over all sources caps the MERGED list, so EuropePMC's first page
        filled it and NCBI's results were never reached (pilot-001 F2). Here each
        source is searched separately through the same existing API, and the
        lists are interleaved and de-duplicated before the ``max_papers`` cap.
        NCBI goes through :meth:`_ncbi` (IT2-2).
        """
        from t2pw.rag.acquire import search_candidates

        per_source = [
            self._ncbi(context, int(max_papers)) if source == "ncbi"
            else search_candidates(dict(context), sources=[source], max_papers=int(max_papers))
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

    def _ncbi(self, context: Mapping[str, Any], max_papers: int) -> List[Any]:
        """NCBI candidates whose identity is their PMID alone.

        rag's PubMed parser reads every ``ArticleIdList`` in the record, the
        reference list's included, so the LAST cited reference's PMCID and DOI
        win (``rag/acquire.py`` ``_parse_pubmed_article``). Pilot-002 recorded a
        2021 paper as PMC411572 (Towbin 1979) and fetched two other articles'
        full texts that way. ``rag/`` is frozen, so mining keeps rag's own network
        fetch (``_fetch_ncbi``) and its text fields, but re-reads the returned XML
        for each article's own ``MedlineCitation/PMID``, drops the parsed PMCID,
        DOI and URL, and asks EuropePMC for the PMCID by PMID (rag's
        ``_resolve_europepmc_item``). Without an answer, the PMID is the id.
        ponytail: one EuropePMC lookup per new PMID, cached for the run; this
        path bypasses rag's search cache for NCBI.
        """
        from xml.etree import ElementTree

        from t2pw.rag import acquire as ra

        cap = max(0, int(max_papers))
        capture = _CapturingClient(ra.HttpClient())
        organism = str(context.get("likely_organism") or "")
        ra._fetch_ncbi(capture, dict(context), retmax=max(cap * 2, cap + 5), organism=organism)
        out: List[Any] = []
        for xml in capture.efetch_xml:
            try:
                root = ElementTree.fromstring(xml)
            except ElementTree.ParseError:
                continue
            for node in root.findall(".//PubmedArticle"):
                pmid = (node.findtext("./MedlineCitation/PMID") or "").strip()
                paper = ra._parse_pubmed_article(node)
                if paper is None or not pmid.isdigit() or any(c.pmid == pmid for c in out):
                    continue
                paper.pmid = pmid
                paper.id = pmid
                paper.source_uri = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                paper.requested_organism = organism
                # The record's own declared types ("Journal Article; Review"): F-2.
                paper.article_type = "; ".join(
                    (t.text or "").strip()
                    for t in node.findall("./MedlineCitation/Article/PublicationTypeList/PublicationType")
                    if (t.text or "").strip())
                out.append(paper)
        out = out[:cap]
        for paper in out:
            paper.id = self._pmcid_for(paper.pmid, capture.inner) or paper.pmid
        return out

    def _europepmc_item(self, http: Any, query: str) -> Optional[Dict[str, Any]]:
        """EuropePMC's ``core`` record for ``query``, cached for the run.

        ``{}`` when EuropePMC answered with no record (cached: asking again changes
        nothing); ``None`` when the lookup itself failed -- an exception, a non-200
        or an unreadable body. A failure is NOT cached, so the next call asks again.
        rag's ``_resolve_europepmc_item`` returns the same empty item for both, which
        is how a failed lookup used to stand as "no PMCID" for the whole run (IT2-N3)
        and one article was counted as a ``PMID:`` and a ``PMC:`` paper.
        """
        from t2pw.rag import acquire as ra

        cache: Dict[str, Dict[str, Any]] = self.__dict__.setdefault("_europepmc_items", {})
        if query in cache:
            return cache[query]
        params = {"query": query, "format": "json", "pageSize": 1, "resultType": "core"}
        try:
            resp = http.get(ra._EUROPEPMC_SEARCH, params=params)
            if getattr(resp, "status_code", 0) != 200:
                return None
            results = ra._safe_list(ra._safe_dict(ra._safe_dict(resp.json()).get("resultList")).get("result"))
        except Exception:  # noqa: BLE001 - a failed lookup is reported as None, never raised
            return None
        cache[query] = dict(results[0]) if results and isinstance(results[0], dict) else {}
        return cache[query]

    def _pmcid_for(self, pmid: str, http: Any) -> str:
        """The article's own PMCID from EuropePMC, by PMID. ``""`` when it has none or the lookup failed."""
        pmcid = str((self._europepmc_item(http, f"EXT_ID:{pmid} AND SRC:MED") or {}).get("pmcid") or "")
        return _as_pmcid(pmcid)

    def paper_metadata(self, record: PaperRecord) -> Optional[Dict[str, Any]]:
        """EuropePMC's core record for an INCLUDED paper: publication types, DOI, journal.

        One search by PMID (the query ``_pmcid_for`` already made for an NCBI record,
        so it is a cache hit) or by PMCID. ``{}`` when the paper has neither or
        EuropePMC has no record; ``None`` when the lookup failed.
        """
        from t2pw.rag import acquire as ra

        if record.pmid:
            query = f"EXT_ID:{record.pmid} AND SRC:MED"
        elif record.pmcid:
            query = f"PMCID:{record.pmcid}"
        else:
            return {}
        return self._europepmc_item(ra.HttpClient(), query)

    def full_text(self, candidate: Any) -> str:
        from t2pw.rag.acquire import fetch_full_text

        return fetch_full_text(candidate)


class OfflinePaperSource(PaperSource):
    """A source that never touches the network. Used by fixtures and replays."""

    name = "offline"
    verify_full_text_title = False

    def __init__(self, papers: Optional[Sequence[Any]] = None) -> None:
        self._papers = list(papers or [])

    def search(self, context: Mapping[str, Any], *, sources: Sequence[str],
               max_papers: int) -> List[Any]:  # noqa: ARG002
        return list(self._papers)[: max(0, int(max_papers))]

    def full_text(self, candidate: Any) -> str:
        return str(getattr(candidate, "full_text", "") or "")

    def paper_metadata(self, record: PaperRecord) -> Optional[Dict[str, Any]]:  # noqa: ARG002
        """Offline: no metadata service, and nothing failed."""
        return {}


def _as_pmcid(value: Any) -> str:
    text = str(value or "").strip().upper()
    return (text if text.startswith("PMC") else f"PMC{text}") if text else ""


def _publication_types(item: Mapping[str, Any]) -> List[str]:
    """EuropePMC's ``pubTypeList.pubType``: a list, or a single string."""
    listed = item.get("pubTypeList")
    listed = listed.get("pubType") if isinstance(listed, Mapping) else None
    if isinstance(listed, str):
        listed = [listed]
    return [str(t).strip() for t in listed or [] if str(t).strip()]


def _journal_title(item: Mapping[str, Any]) -> str:
    """``journalTitle`` (the lite shape) or ``journalInfo.journal.title`` (the core shape)."""
    info = item.get("journalInfo")
    journal = info.get("journal") if isinstance(info, Mapping) else None
    nested = journal.get("title") if isinstance(journal, Mapping) else ""
    return str(item.get("journalTitle") or nested or "").strip()


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
    pmid = pmid or str(getattr(candidate, "pmid", "") or "")  # an NCBI record resolved to its PMCID
    url = str(getattr(candidate, "source_uri", "") or "")
    title = str(getattr(candidate, "title", "") or "")
    abstract = str(getattr(candidate, "abstract", "") or "")
    source = str(getattr(candidate, "source", "") or "")
    article_type = str(getattr(candidate, "article_type", "") or "")
    # Discovery time: no full text yet. Acquirer._settle_included grades again once it is known.
    evidence_type, basis = grade_evidence_type(title=title, abstract=abstract,
                                               article_type=article_type, source=source, url=url)

    return PaperRecord(
        paper_id=make_paper_id(pmid=pmid, pmcid=pmcid, doi=doi, url=url, title=title),
        pmid=pmid,
        pmcid=pmcid,
        doi=doi,
        title=title,
        journal=str(getattr(candidate, "journal", "") or ""),
        year=str(getattr(candidate, "year", "") or ""),
        article_type=article_type,
        url=url,
        acquisition_source=source,
        retrieved_at=_utcnow(),
        abstract=abstract,
        evidence_type=evidence_type,
        evidence_type_basis=basis,
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

    evidence_type, basis = grade_evidence_type(title=title, full_text=body)
    record = PaperRecord(
        paper_id=f"FILE:{digest[:16]}",
        title=str(title or Path(file_name).stem),
        file_sha256=digest,
        file_name=str(file_name),
        acquisition_source="upload",
        retrieved_at=_utcnow(),
        evidence_type=evidence_type,
        evidence_type_basis=basis,
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
        #: Every id of every included paper -> that paper's ``paper_id`` (IT2-N3).
        self._included: Dict[str, str] = {}

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

    def _other_organisms_in_title(self, title: str) -> List[str]:
        """Organisms other than the target species that ``title`` names.

        Read-only reuse of both existing detectors: the production screen's
        ``observe_metadata`` on the title alone, counting what its own
        ``compare_organism`` calls a ``mismatch`` (its genus scan reads
        "Arabidopsis jasmonate" as a same-genus binomial), and species.py's
        common-name table (a title word or word pair that ``normalize_species``
        maps to a species). ``_EXTRA_HOSTS`` covers what neither knows.
        """
        from t2pw.mining.species import normalize_species
        from t2pw.rag.eligibility import ObservedContext, compare_organism, observe_metadata

        scope = self._scope()
        found = [o for o in observe_metadata(title, "", scope).observed_organisms
                 if compare_organism(scope, ObservedContext(observed_organisms=[o])) == "mismatch"]
        words = re.findall(r"[A-Za-z]+", title)
        for gram in [*words, *(f"{a} {b}" for a, b in zip(words, words[1:]))]:
            if normalize_species(gram) != gram.lower():  # the table knows this name
                found.append(gram)
        found += [host for host in _EXTRA_HOSTS if _names(_fold(title), host)]
        target = self.config.target_species_normalized
        return [o for o in dict.fromkeys(found) if normalize_species(o) != target]

    def rescue(self, candidate: Any, decision: Any, run_terms: Sequence[str] = ()) -> str:
        """Why mining re-admits a paper the production screen excluded, or ``""``.

        Only an exclusion by the pathway-anchor rule (classification
        ``context_only`` / ``omics_only`` / ``off_topic``) is reconsidered. An
        organism veto or a negative-evidence veto never is. Two routes:

        1. ``context_only`` / ``omics_only`` only (never ``off_topic``): the
           production screen names the target species (``organism_match ==
           "match"``), the title names no other organism
           (:meth:`_other_organisms_in_title`), and ``screen_candidate``'s own
           thresholds, with the anchor rule off and both bars (full and
           title-only) raised by ``_UNANCHORED_MARGIN``, pass the paper. The
           screen says "match" even when the target is only the source of a
           transgene expressed in another plant (PMC10742046, emmer wheat
           "overexpressing ... from Arabidopsis thaliana").
        2. The title names the target species, and one sentence of the title or
           abstract names it together with an enzyme or metabolite from this
           run's seeds or ``target_species_supported`` reactions (``run_terms``,
           built by ``queries.rescue_terms``). A name shorter than
           ``_MIN_TERM_CHARS`` is ignored, and an acronym must match in its own
           case. Both species guards come from pilot-001's wheat transgene
           paper. Its abstract names *A. thaliana* only as a gene source
           ("ALLENE OXIDE SYNTHASE from Arabidopsis thaliana"), and its title
           names wheat.

        Neither route rescues a paper whose title makes the target species a
        transgenic or heterologous host (``_FOREIGN_HOST``): the maize ZmLOX2
        "in Transgenic Arabidopsis" paper.

        ``rag/eligibility.py`` is not changed. No pathway vocabulary is hard-coded.
        """
        classification = str(getattr(decision, "classification", "") or "")
        reason = str(getattr(decision, "reason", "") or "")
        if classification not in _RESCUABLE or not reason.startswith(f"{classification}:"):
            return ""
        title = str(getattr(candidate, "title", "") or "")
        if _FOREIGN_HOST.search(title):
            return ""
        try:
            from t2pw.rag.eligibility import screen_candidate, thresholds_from_config

            if (classification in _SCORE_RESCUABLE
                    and str(getattr(decision, "organism_match", "") or "") == "match"
                    and not self._other_organisms_in_title(title)):
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
        if not any(_names(_fold(title), f) for f in forms):
            return ""
        terms = [t for t in (" ".join(str(r or "").split()) for r in run_terms) if len(t) >= _MIN_TERM_CHARS]
        for sentence in _SENTENCE.split(f"{title}. {getattr(candidate, 'abstract', '') or ''}"):
            folded = _fold(sentence)
            named = next((f for f in forms if _names(folded, f)), "")
            term = next((t for t in terms if named and _term_in(sentence, t)), "")
            if term:
                return f"names {named!r} and {term!r} in one sentence"
        return ""

    # -- an included paper ---------------------------------------------------
    def _settle_included(self, record: PaperRecord, text: str) -> None:
        """Complete an INCLUDED paper's metadata, then grade it on its own text.

        Discovery-time grading sees a title and an abstract and, for EuropePMC, no
        declared type at all (rag's search drops ``pubTypeList``). So each included
        paper -- not every discovered one -- gets one metadata lookup through the
        source seam (:meth:`PaperSource.paper_metadata`; an offline run never asks,
        and offline sources answer ``{}``): its publication types, and an empty DOI,
        journal or abstract filled (A14: the DOI was empty on every pilot paper; the
        abstract because indexing drops the front matter that may hold the only
        copy; stored as plain text). Its PMID and PMCID fill empty ones, so the caller
        can tell when this is an article already included under another id (IT2-N3,
        :meth:`_same_article`). A failed lookup (failures are not cached) is asked once
        more; failing twice, it is a note, never an error, and
        ``evidence_type_verified`` is ``False``, so a later gate can fail closed: the
        grade then rests on the text rules. Then the paper is graded again, on its text
        as fetched (:func:`grade_evidence_type`).
        """
        lookup = None if self.config.offline else getattr(self.source, "paper_metadata", None)
        meta = lookup(record) if lookup is not None else {}
        if meta is None:
            meta = lookup(record)
        if meta is None:
            record.evidence_type_verified = False
            record.notes.append("EuropePMC metadata lookup failed twice: the declared article type "
                                "is unknown, so the evidence type rests on the text rules")
        elif meta:
            record.evidence_type_verified = True
            declared = [t.strip() for t in record.article_type.split(";") if t.strip()]
            record.article_type = "; ".join(dict.fromkeys([*declared, *_publication_types(meta)]))
            record.doi = record.doi or str(meta.get("doi") or "").strip()
            record.journal = record.journal or _journal_title(meta)
            record.abstract = record.abstract or _plain(meta.get("abstractText"))
            record.pmid = record.pmid or str(meta.get("pmid") or "").strip()
            record.pmcid = record.pmcid or _as_pmcid(meta.get("pmcid"))

        record.evidence_type, record.evidence_type_basis = grade_evidence_type(
            title=record.title, abstract=record.abstract, article_type=record.article_type,
            source=record.acquisition_source, url=record.url, full_text=text)
        if record.evidence_type_verified is False:
            record.evidence_type_basis = "; ".join(
                filter(None, [record.evidence_type_basis, "type unverified: EuropePMC lookup failed"]))

    @staticmethod
    def _handles(record: PaperRecord) -> List[str]:
        """Every paper id this record's article could be found under."""
        return [h for h in (record.paper_id,
                            make_paper_id(pmcid=record.pmcid) if record.pmcid else "",
                            make_paper_id(pmid=record.pmid) if record.pmid else "",
                            make_paper_id(doi=record.doi.lower()) if record.doi else "") if h]

    def _same_article(self, record: PaperRecord) -> str:
        """The paper id of an already included paper that is this record's article, or ``""``."""
        return next((self._included[h] for h in self._handles(record)
                     if self._included.get(h, record.paper_id) != record.paper_id), "")

    def remember(self, record: PaperRecord) -> None:
        """Mark an included paper's every id as seen, so the article is one paper (IT2-N3).

        The controller also calls this for each included paper a resumed run restores.
        """
        for handle in self._handles(record):
            self._included.setdefault(handle, record.paper_id)
            self.seen_paper_ids.add(handle)

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

        ``run_terms`` are this run's seed names and the names of its
        ``target_species_supported`` reactions, used only by :meth:`rescue`.
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
            if (text and record.title and getattr(self.source, "verify_full_text_title", False)
                    and not full_text_matches_title(record.title, text)):
                record.notes.append(
                    "full text refused: its opening does not carry this record's title, so it is "
                    "another article; the abstract is used instead"
                )
                text = ""
            if not text:
                text = str(getattr(candidate, "full_text", "") or "")
            body = text or record.abstract
            if not body:
                record.exclusion_reasons.append("no retrievable text (no full text and no abstract)")
                self.seen_paper_ids.add(record.paper_id)
                result.excluded.append(record)
                continue

            self._settle_included(record, body)
            same = self._same_article(record)
            if same:
                # The PMC copy came first and this PMID: record's lookup only now names it.
                record.exclusion_reasons.append(f"same article as {same}, already included")
                self.seen_paper_ids.add(record.paper_id)
                result.excluded.append(record)
                continue
            record.full_text_chars = len(body)
            record.cache_ref = f"{self.source.name}:{getattr(candidate, 'id', '')}"
            self.remember(record)
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
            self._settle_included(record, body)
            same = self._same_article(record)
            if same:
                errors.append(f"{item.raw!r} is the same article as {same}, already loaded; skipped")
                continue
            self.remember(record)
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
    "grade_evidence_type",
    "LoadedPaper",
    "AcquisitionResult",
    "PaperSource",
    "OfflinePaperSource",
    "Acquirer",
    "record_from_candidate",
    "record_from_upload",
    "full_text_matches_title",
]
