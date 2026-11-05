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
Messages are fixed strings: raw exception or parser text is never placed in a
user-facing field (it can carry URLs with ``api_key`` / ``email``, or server
paths). The exception *class name* only goes in the separate ``detail`` field.
Caps (count, PMID list length, PDF size / pages) are NOT enforced here; they are
a precondition for the caller (R5 / W8).

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
_PMID_RE = re.compile(r"^[1-9][0-9]{0,8}$")  # ASCII digits only

MAX_NAME_LEN = 200
_SEARCH_ERRORS = ("search_failed", "fetch_failed")  # batch.fetch skip reasons that mean an outage

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
    if not papers and any(r.get("reason") in _SEARCH_ERRORS for r in skipped if isinstance(r, dict)):
        message = "could not search for papers right now; please try again later or upload a PDF"
    elif not papers:
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
    message: str = ""  # fixed, user-facing
    detail: str = ""  # internal only: exception class name, never shown

    @property
    def ok(self) -> bool:
        return self.status == OK


def validate_pmid(raw: Any) -> str:
    """Return the normalised PMID or raise ``ValueError`` (trust boundary)."""
    text = str(raw if raw is not None else "").strip()
    if not _PMID_RE.fullmatch(text):
        raise ValueError(f"'{text[:40]}' is not a valid PubMed ID (digits only, e.g. 12345678)")
    return text


def _no_oa(pmid: str) -> str:
    return f"no open-access full text found for PMID {pmid}; please upload the PDF"


def _lookup_failed(pmid: str) -> str:
    return f"could not look up PMID {pmid} right now; please try again later or upload the PDF"


def _idconv(pmids: Sequence[str], client: Any) -> Dict[str, str]:
    """PMID -> PMCID ('' when the record says none). Raises on any malformed reply.

    A requested PMID with no record in a 200 reply is absent from the result (the
    caller treats that as a failed lookup, not as "no PMCID").
    """
    out: Dict[str, str] = {}
    for i in range(0, len(pmids), _IDCONV_CHUNK):
        chunk = list(pmids[i : i + _IDCONV_CHUNK])
        params = _ncbi_eutils_params({"ids": ",".join(chunk), "idtype": "pmid", "format": "json"})
        resp = client.get(IDCONV_URL, params=params)
        if getattr(resp, "status_code", 0) != 200:
            raise RuntimeError(f"idconv HTTP {getattr(resp, 'status_code', '?')}")
        payload = resp.json()
        if not isinstance(payload, dict) or payload.get("status") == "error":
            raise RuntimeError("idconv error reply")
        records = payload.get("records")
        if not isinstance(records, list):
            raise RuntimeError("idconv reply has no records list")
        for rec in records:
            if not isinstance(rec, dict):
                continue
            # an unknown id comes back as {"requested-id": ..., "status": "error"}
            pmid = str(rec.get("pmid") or rec.get("requested-id") or "")
            if pmid in chunk:
                out[pmid] = "" if rec.get("status") == "error" else str(rec.get("pmcid") or "")
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

    Caller notes (R5): ``fetch_full_text`` swallows errors into ``""`` and caches
    that empty result on disk, so a Europe PMC outage reads as ``NO_FULL_TEXT`` and
    sticks. R5 should pass its own ``fetch_text_fn`` (or one built on
    ``use_cache=False``). The number of PMIDs is not capped here.
    """
    results: Dict[str, PmidResult] = {}
    valid: List[str] = []
    for raw in pmids:
        try:
            pmid = validate_pmid(raw)
        except ValueError as exc:
            results.setdefault(f"bad:{raw}", PmidResult(str(raw).strip()[:40], INVALID, message=str(exc)))
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
                results[pmid].message = _lookup_failed(pmid)
                results[pmid].detail = type(exc).__name__
            return list(results.values())

    fetch_text = fetch_text_fn or (
        lambda pmcid: fetch_full_text(CandidatePaper(id=pmcid, source="PMC"), client=http)
    )
    for pmid in valid:
        res = results[pmid]
        if pmid not in mapping:  # a 200 reply that never mentioned this PMID
            res.message, res.detail = _lookup_failed(pmid), "absent_from_reply"
            continue
        pmcid = mapping[pmid]
        if not pmcid:
            res.status, res.message = NO_PMCID, _no_oa(pmid)
            continue
        res.pmcid = pmcid
        try:
            text = (fetch_text(pmcid) or "").strip()
        except Exception as exc:  # noqa: BLE001
            res.message, res.detail = _lookup_failed(pmid), type(exc).__name__
            continue
        if text:
            res.status, res.text = OK, text
        else:
            res.status, res.message = NO_FULL_TEXT, _no_oa(pmid)
    return list(results.values())


# -- (3) uploaded PDFs --------------------------------------------------------
@dataclass
class PdfResult:
    path: str
    status: str  # OK | EMPTY | ERROR
    text: str = ""
    method: str = ""
    ocr_used: bool = False
    message: str = ""  # fixed, user-facing; only the file's base name appears
    detail: str = ""  # internal only: exception class / "parser_error", never shown

    @property
    def ok(self) -> bool:
        return self.status == OK


def read_pdfs(
    paths: Sequence[Any],
    *,
    parse_fn: Callable[..., Dict[str, Any]] = parse_pdf,
) -> List[PdfResult]:
    """Extract text from each uploaded PDF (OCR fallback is inside ``parse_pdf``).

    Contract: ``paths`` are server-owned upload paths (R5 / W8 save the upload
    first); this function does not sandbox arbitrary client-supplied paths. One
    result per input path, in order. A file that is not a PDF, is unreadable, or
    yields no text is reported with a fixed message, not dropped. The parser's
    own error and warning strings (which can contain absolute paths) are not
    copied into the result. Size and page caps are the caller's (W8).
    """
    out: List[PdfResult] = []
    for raw in paths:
        path = Path(raw)
        name = path.name
        key = str(raw)
        try:
            with path.open("rb") as fh:
                is_pdf = fh.read(5) == b"%PDF-"
        except OSError as exc:
            out.append(PdfResult(key, ERROR, message=f"could not read {name}", detail=type(exc).__name__))
            continue
        if not is_pdf:
            out.append(PdfResult(key, ERROR, message=f"{name} is not a PDF file", detail="not_pdf"))
            continue
        try:
            parsed = parse_fn(str(path))
        except Exception as exc:  # noqa: BLE001
            out.append(PdfResult(key, ERROR, message=f"could not read {name}", detail=type(exc).__name__))
            continue
        text = str(parsed.get("text") or "").strip()
        method = str(parsed.get("method") or "")
        ocr = method in ("ocr", "mixed")
        if parsed.get("error"):
            out.append(PdfResult(key, ERROR, method=method, ocr_used=ocr,
                                 message=f"could not read {name}", detail="parser_error"))
        elif not text:
            msg = f"no text could be extracted from {name} (scanned or image-only?)"
            out.append(PdfResult(key, EMPTY, method=method, ocr_used=ocr, message=msg))
        else:
            out.append(PdfResult(key, OK, text, method, ocr))
    return out
