"""Input adapters for the public web server (W1).

Turns the three public input modes into plain data a later ``run_pathway`` (R5)
can consume. Nothing here is wired into ``run_pathway`` and nothing here
reimplements acquisition: it calls the existing code.

* name + organism -> :class:`TopicSpec` / ``RequestedScope`` and
  ``t2pw.batch.fetch.fetch_papers`` (Europe PMC / NCBI search + full text).
* PMIDs -> PMCIDs via the NCBI ID converter, then the existing
  ``t2pw.rag.acquire.fetch_full_text`` (Europe PMC ``fullTextXML``).
* uploaded PDFs -> ``t2pw.extraction.pdf_parser.parse_pdf`` (OCR fallback inside).

Every outcome is returned, never dropped: a PMID or PDF that yields no text is a
result with a user-facing ``message``, not an exception and not a missing row.
Network access is injectable (``client`` / ``fetch_text_fn`` / ``fetch_fn``), so
tests make no live calls. NCBI ``tool`` / ``email`` / ``api_key`` come from the
environment through ``map_ids._ncbi_eutils_params``; this module never asks the
user for an email.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from t2pw.batch.fetch import DEFAULT_PER_TOPIC, BatchPaper, TopicSpec, fetch_papers
from t2pw.extraction.pdf_parser import parse_pdf
from t2pw.mapping.map_ids import HttpClient, _ncbi_eutils_params, _ncbi_throttle
from t2pw.rag.acquire import CandidatePaper, fetch_full_text
from t2pw.rag.eligibility import RequestedScope

# NCBI ID converter (PMC). Not present elsewhere in the repo.
IDCONV_URL = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
_IDCONV_CHUNK = 100  # the service accepts up to 200 ids per call

# A PMID is a positive integer, at most 9 digits today. Short ids are real (old
# papers), so unlike batch.fetch._PMID_RE this does not demand 6+ digits.
_PMID_RE = re.compile(r"^[1-9]\d{0,8}$")

MAX_NAME_LEN = 200

# PMID / PDF outcome statuses.
OK = "ok"
INVALID = "invalid"
NO_PMCID = "no_pmcid"
NO_FULL_TEXT = "no_full_text"
LOOKUP_FAILED = "lookup_failed"
EMPTY = "empty"
ERROR = "error"


# -- (1) name + organism ------------------------------------------------------
@dataclass
class NameInputs:
    spec: TopicSpec
    scope: RequestedScope
    papers: List[BatchPaper] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    message: str = ""  # non-empty when RAG found nothing usable (D-W03: say so)


def _clean(value: str, what: str, *, required: bool) -> str:
    text = " ".join(str(value or "").split())
    if required and not text:
        raise ValueError(f"{what} is required")
    if len(text) > MAX_NAME_LEN:
        raise ValueError(f"{what} is longer than {MAX_NAME_LEN} characters")
    return text


def name_to_spec(pathway: str, organism: str = "", count: int = DEFAULT_PER_TOPIC) -> TopicSpec:
    """Validate and build the ``TopicSpec`` for a pathway name + organism."""
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be a positive integer")
    return TopicSpec(
        topic=_clean(pathway, "pathway name", required=True),
        organism=_clean(organism, "organism", required=False),
        count=count,
    )


def resolve_name(
    pathway: str,
    organism: str = "",
    count: int = DEFAULT_PER_TOPIC,
    *,
    fetch_fn: Callable[..., Any] = fetch_papers,
    **fetch_kwargs: Any,
) -> NameInputs:
    """Name mode: build the spec/scope and run the existing paper search.

    ``fetch_kwargs`` pass straight to ``fetch_papers`` (``search_fn``,
    ``fetch_text_fn``, ``client``, ``use_cache`` ...); ``fetch_fn`` replaces it
    wholesale in tests. ``fetch_papers`` never raises; an empty result is
    reported in ``message`` rather than hidden.
    """
    spec = name_to_spec(pathway, organism, count)
    stats: Dict[str, Any] = {}
    papers, skipped = fetch_fn([spec], stats=stats, **fetch_kwargs)
    message = ""
    if not papers:
        message = (
            f"no open-access full text found for '{spec.topic}'"
            + (f" in {spec.organism}" if spec.organism else "")
            + "; try a PMID or upload a PDF"
        )
    return NameInputs(spec, spec.requested_scope(), list(papers), list(skipped), stats, message)


# -- (2) PMIDs ----------------------------------------------------------------
@dataclass
class PmidResult:
    pmid: str  # normalised, or the raw text when invalid
    status: str  # OK | INVALID | NO_PMCID | NO_FULL_TEXT | LOOKUP_FAILED
    pmcid: str = ""
    text: str = ""
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status == OK


def validate_pmid(raw: Any) -> str:
    """Return the normalised PMID or raise ``ValueError`` (trust boundary)."""
    text = str(raw if raw is not None else "").strip()
    if not _PMID_RE.match(text):
        raise ValueError(f"'{text[:40]}' is not a valid PubMed ID (digits only, e.g. 12345678)")
    return text


def _no_oa(pmid: str) -> str:
    return f"no open-access full text found for PMID {pmid}; please upload the PDF"


def _idconv(pmids: Sequence[str], client: Any) -> Dict[str, str]:
    """PMID -> PMCID ('' when none). Raises on transport/parse failure."""
    out: Dict[str, str] = {}
    for i in range(0, len(pmids), _IDCONV_CHUNK):
        chunk = list(pmids[i : i + _IDCONV_CHUNK])
        params = _ncbi_eutils_params({"ids": ",".join(chunk), "idtype": "pmid", "format": "json"})
        resp = client.get(IDCONV_URL, params=params)
        if getattr(resp, "status_code", 0) != 200:
            raise RuntimeError(f"idconv HTTP {getattr(resp, 'status_code', '?')}")
        for rec in resp.json().get("records") or []:
            if isinstance(rec, dict) and rec.get("pmid") and rec.get("status") != "error":
                out[str(rec["pmid"])] = str(rec.get("pmcid") or "")
    return out


def resolve_pmids(
    pmids: Sequence[Any],
    *,
    client: Optional[Any] = None,
    fetch_text_fn: Optional[Callable[[str], str]] = None,
) -> List[PmidResult]:
    """PMIDs -> open-access full text, one :class:`PmidResult` per distinct input.

    Order is preserved and duplicates collapse. Invalid ids are returned as
    ``INVALID`` results, not raised and not dropped. ``client`` needs
    ``.get(url, params=...)`` (default ``HttpClient``, throttled); ``fetch_text_fn``
    maps a PMCID to its full text (default: ``rag.acquire.fetch_full_text``).
    """
    results: Dict[str, PmidResult] = {}
    valid: List[str] = []
    for raw in pmids:
        try:
            pmid = validate_pmid(raw)
        except ValueError as exc:
            results.setdefault(f"bad:{raw}", PmidResult(str(raw).strip(), INVALID, message=str(exc)))
            continue
        if pmid not in results:
            results[pmid] = PmidResult(pmid, LOOKUP_FAILED)
            valid.append(pmid)

    http = client if client is not None else HttpClient()
    mapping: Dict[str, str] = {}
    if valid:
        try:
            if client is None:
                _ncbi_throttle()
            mapping = _idconv(valid, http)
        except Exception as exc:  # noqa: BLE001
            for pmid in valid:
                results[pmid].message = (
                    f"could not look up PMID {pmid} (NCBI unreachable: {exc}); "
                    "try again later or upload the PDF"
                )
            return list(results.values())

    fetch_text = fetch_text_fn or (
        lambda pmcid: fetch_full_text(CandidatePaper(id=pmcid, source="PMC"), client=http)
    )
    for pmid in valid:
        res = results[pmid]
        pmcid = mapping.get(pmid, "")
        if not pmcid:
            res.status, res.message = NO_PMCID, _no_oa(pmid)
            continue
        res.pmcid = pmcid
        try:
            text = (fetch_text(pmcid) or "").strip()
        except Exception as exc:  # noqa: BLE001
            res.message = (
                f"could not fetch full text for PMID {pmid} ({pmcid}): {exc}; "
                "try again later or upload the PDF"
            )
            continue
        if text:
            res.status, res.text = OK, text
        else:
            res.status, res.message = NO_FULL_TEXT, _no_oa(pmid)
    return list(results.values())


# -- (3) uploaded PDFs --------------------------------------------------------
@dataclass
class PdfResult:
    path: Path
    status: str  # OK | EMPTY | ERROR
    text: str = ""
    method: str = ""
    warnings: List[str] = field(default_factory=list)
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status == OK


def read_pdfs(
    paths: Sequence[Any],
    *,
    parse_fn: Callable[..., Dict[str, Any]] = parse_pdf,
) -> List[PdfResult]:
    """Extract text from each uploaded PDF (OCR fallback is inside ``parse_pdf``).

    One result per input path, in order. A file that is not a PDF, is
    unreadable, or yields no text is reported with a message, not dropped.
    """
    out: List[PdfResult] = []
    for raw in paths:
        path = Path(raw)
        name = path.name
        try:
            with path.open("rb") as fh:
                is_pdf = fh.read(5) == b"%PDF-"
        except OSError as exc:
            out.append(PdfResult(path, ERROR, message=f"could not read {name}: {exc}"))
            continue
        if not is_pdf:
            out.append(PdfResult(path, ERROR, message=f"{name} is not a PDF file"))
            continue
        try:
            parsed = parse_fn(str(path))
        except Exception as exc:  # noqa: BLE001
            out.append(PdfResult(path, ERROR, message=f"could not read {name}: {exc}"))
            continue
        text = str(parsed.get("text") or "").strip()
        warnings = [str(w) for w in parsed.get("warnings") or []]
        method = str(parsed.get("method") or "")
        if parsed.get("error"):
            msg = f"could not read {name}: {parsed['error']}"
            out.append(PdfResult(path, ERROR, method=method, warnings=warnings, message=msg))
        elif not text:
            msg = f"no text could be extracted from {name} (scanned or image-only?)"
            out.append(PdfResult(path, EMPTY, method=method, warnings=warnings, message=msg))
        else:
            out.append(PdfResult(path, OK, text, method, warnings))
    return out
