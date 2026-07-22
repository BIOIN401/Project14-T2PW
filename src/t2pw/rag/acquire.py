"""Stage R1 — candidate paper acquisition (WP1).

Given the seed pathway context and its gap terms, fetch candidate papers from
the literature APIs and, on demand, download their full text. The output is a
list of :class:`CandidatePaper` records that feeds WP2 (selection); this stage
does **no** ranking, selection, chunking, or embedding.

Reuse, not reinvention
----------------------
Per the separation invariant (docs/rag/03_separation_invariant.md) and the WP1
brief, the HTTP plumbing is **reused** from the core mapping stage — nothing here
duplicates URL/retry/rate-limit logic:

* ``HttpClient``          — the shared session + retry/backoff client.
* ``_europepmc_full_text``— EuropePMC ``fullTextXML`` fetch + XML→text.
* ``_NCBI_EUTILS_BASE`` / ``_ncbi_eutils_params`` / ``_ncbi_throttle`` — NCBI
  eutils base URL, shared query params, and the 3-req/s throttle.

The dependency arrow points **RAG → core only**: this module imports from
``t2pw.mapping``; nothing in ``t2pw.mapping`` (or any stage module) imports
``t2pw.rag``. WP1 changes no pipeline behavior and uses no core seam — it only
produces data for WP2.

Offline-first
-------------
Every network fetch degrades gracefully: on a missing network or an API error a
source contributes nothing rather than raising, so acquisition returns whatever
is reachable (or an empty list). Results are cached on disk under
``data/rag_index/acquire_cache/`` keyed by a query hash, following the
``id_mapping_cache.json`` precedent, so a re-run is served from cache and never
re-hits the network.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from xml.etree import ElementTree

from t2pw.config import rag_config
from t2pw.mapping.map_ids import (
    _NCBI_EUTILS_BASE,
    HttpClient,
    _canonical_name,
    _europepmc_full_text,
    _ncbi_eutils_params,
    _ncbi_throttle,
    _normalize_name,
    _safe_dict,
    _safe_list,
)
from t2pw.paths import PROJECT_ROOT

# Reused, single-sourced endpoints. The retry/backoff and rate-limit logic lives
# in the reused ``HttpClient`` / ``_ncbi_throttle``; only the search URLs (which
# ``map_ids`` does not expose as a generic helper) are named here, once.
_EUROPEPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_CROSSREF_WORKS = "https://api.crossref.org/works"
_SEMANTIC_SCHOLAR_SEARCH = (
    "https://api.semanticscholar.org/graph/v1/paper/search"
)
_BIORXIV_DETAILS = "https://api.biorxiv.org/details"

# The known acquisition sources. "europepmc" and "ncbi" are the defaults; the
# rest are optional and only queried when named in ``sources``.
_DEFAULT_SOURCES = ("europepmc", "ncbi")
_OPTIONAL_SOURCES = ("crossref", "semanticscholar", "biorxiv")

# Keep OR-clauses bounded so a rich seed context does not build an unusably long
# query. These are query-shaping caps, not the paper cap (``max_papers``).
_MAX_COMPOUND_TERMS = 6
_MAX_PROTEIN_TERMS = 6
_MAX_GAP_TERMS = 6


@dataclass
class CandidatePaper:
    """A single paper fetched by acquisition, before selection (WP2).

    Fields match the WP1 brief exactly. ``id`` carries the best full-text handle
    available for the record (a PMCID when one exists, else a PMID or DOI);
    ``source`` is the acquiring source name. ``full_text`` is populated lazily by
    :func:`fetch_full_text` and is ``""`` straight out of :func:`search_candidates`.
    """

    id: str
    source: str
    title: str = ""
    abstract: str = ""
    organism: str = ""
    full_text: str = ""
    source_uri: str = ""
    year: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidatePaper":
        payload = _safe_dict(data)
        return cls(
            id=str(payload.get("id") or ""),
            source=str(payload.get("source") or ""),
            title=str(payload.get("title") or ""),
            abstract=str(payload.get("abstract") or ""),
            organism=str(payload.get("organism") or ""),
            full_text=str(payload.get("full_text") or ""),
            source_uri=str(payload.get("source_uri") or ""),
            year=str(payload.get("year") or ""),
        )

    def identity_keys(self) -> List[str]:
        """Normalized keys used to dedupe this paper against others / the seed."""
        keys: List[str] = []
        ident = _canonical_name(self.id)
        if ident:
            keys.append(f"id:{ident.lower()}")
        norm_title = _normalize_name(self.title)
        if norm_title:
            keys.append(f"title:{norm_title}")
        return keys


# ---------------------------------------------------------------------------
# On-disk cache (offline-first) — one JSON file per query hash.
# ---------------------------------------------------------------------------
def _default_cache_dir() -> Path:
    return PROJECT_ROOT / "data" / "rag_index" / "acquire_cache"


class AcquireCache:
    """A tiny per-key JSON cache under ``data/rag_index/acquire_cache/``.

    Mirrors the ``MappingCache`` precedent (a rebuildable, git-ignored cache
    under ``data/``) but stores one file per key so a query hash maps cleanly to
    a file. Reads/writes never raise — a corrupt or unwritable cache degrades to
    a miss so acquisition still runs.
    """

    def __init__(self, root: Optional[Path] = None, *, enabled: bool = True) -> None:
        self.root = Path(root) if root is not None else _default_cache_dir()
        self.enabled = enabled

    def _path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a broken cache file is just a miss
            return None
        return raw if isinstance(raw, dict) else None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self._path(key).write_text(
                json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:  # noqa: BLE001 - never let caching break a run
            return


def _hash_key(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


# HTTP statuses that mean "the API refused us", not "nothing matched". 429 is
# NCBI/EuropePMC rate limiting (NCBI allows only 3 req/s without an API key).
_THROTTLE_STATUSES = frozenset({429, 503})


class _RecordingClient:
    """Wrap an HTTP client, recording each call's outcome for diagnostics.

    Every per-source fetcher deliberately swallows transport failures and returns
    ``[]`` so a dead or throttled API can never break a run. The side effect is
    that a rate-limit (HTTP 429), a server error and a genuinely empty search are
    all indistinguishable downstream — they collapse to "0 candidates". This
    proxy records what actually happened so :func:`search_candidates` can report
    the difference without any fetcher needing to change.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls: List[Dict[str, Any]] = []

    def get(self, url: Any, **kwargs: Any) -> Any:
        try:
            resp = self._inner.get(url, **kwargs)
        except Exception as exc:  # noqa: BLE001 - record, then re-raise unchanged
            self.calls.append(
                {"url": str(url), "status": None, "error": type(exc).__name__}
            )
            raise
        self.calls.append(
            {"url": str(url), "status": getattr(resp, "status_code", None), "error": ""}
        )
        return resp

    def __getattr__(self, name: str) -> Any:
        # Pass through anything else the fetchers/stubs may use.
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# Query construction from the seed context.
# ---------------------------------------------------------------------------
def _context_terms(context: Dict[str, Any]) -> Dict[str, Any]:
    ctx = _safe_dict(context)

    def _str(value: Any) -> str:
        return _canonical_name(str(value or ""))

    def _list(value: Any, limit: int) -> List[str]:
        items = [_str(item) for item in _safe_list(value)]
        out: List[str] = []
        seen: set = set()
        for item in items:
            key = item.lower()
            if item and key not in seen:
                seen.add(key)
                out.append(item)
            if len(out) >= limit:
                break
        return out

    organism = _str(ctx.get("likely_organism")) or _str(ctx.get("organism"))
    return {
        "pathway_name": _str(ctx.get("pathway_name")),
        "organism": organism,
        "compounds": _list(ctx.get("key_compounds"), _MAX_COMPOUND_TERMS),
        "proteins": _list(ctx.get("key_proteins"), _MAX_PROTEIN_TERMS),
        "gaps": _list(ctx.get("gap_terms"), _MAX_GAP_TERMS),
    }


def _core_terms(terms: Dict[str, Any]) -> List[str]:
    ordered: List[str] = []
    seen: set = set()
    candidates: List[str] = []
    if terms["pathway_name"]:
        candidates.append(terms["pathway_name"])
    candidates.extend(terms["gaps"])
    candidates.extend(terms["compounds"])
    candidates.extend(terms["proteins"])
    for term in candidates:
        key = term.lower()
        if term and key not in seen:
            seen.add(key)
            ordered.append(term)
    return ordered


def build_query(context: Dict[str, Any]) -> str:
    """Build an organism-scoped EuropePMC/Crossref query from the seed context.

    Mirrors how ``map_ids`` scopes literature lookups: the topical terms are
    OR'd, then ``AND "<organism>"`` narrows to the target organism when known.
    """
    terms = _context_terms(context)
    core = _core_terms(terms)
    if not core:
        return f'"{terms["organism"]}"' if terms["organism"] else ""
    or_clause = " OR ".join(f'"{term}"' for term in core)
    query = f"({or_clause})" if len(core) > 1 else or_clause
    if terms["organism"]:
        query = f'{query} AND "{terms["organism"]}"'
    return query


def _ncbi_term(context: Dict[str, Any]) -> str:
    # NCBI free-text term: same topical OR-clause, organism-restricted.
    terms = _context_terms(context)
    core = _core_terms(terms)
    if not core:
        return f'{terms["organism"]}[Organism]' if terms["organism"] else ""
    or_clause = " OR ".join(f'"{term}"' for term in core)
    query = f"({or_clause})"
    if terms["organism"]:
        query = f'{query} AND {terms["organism"]}[Organism]'
    return query


# ---------------------------------------------------------------------------
# Per-source fetchers. Each is fail-safe: any error yields an empty list.
# ---------------------------------------------------------------------------
def _year(value: Any) -> str:
    text = _canonical_name(str(value or ""))
    return text[:4] if text[:4].isdigit() else ""


def _europepmc_primary_id(item: Dict[str, Any]) -> str:
    pmcid = _canonical_name(str(item.get("pmcid") or ""))
    if pmcid:
        return pmcid if pmcid.upper().startswith("PMC") else f"PMC{pmcid}"
    source = _canonical_name(str(item.get("source") or ""))
    raw_id = _canonical_name(str(item.get("id") or ""))
    if source.upper() == "PMC" and raw_id:
        return raw_id if raw_id.upper().startswith("PMC") else f"PMC{raw_id}"
    pmid = _canonical_name(str(item.get("pmid") or ""))
    if pmid:
        return pmid
    doi = _canonical_name(str(item.get("doi") or ""))
    return doi or raw_id


def _europepmc_uri(item: Dict[str, Any], primary_id: str) -> str:
    doi = _canonical_name(str(item.get("doi") or ""))
    if doi:
        return f"https://doi.org/{doi}"
    if primary_id.upper().startswith("PMC"):
        return f"https://europepmc.org/article/PMC/{primary_id}"
    pmid = _canonical_name(str(item.get("pmid") or "")) or primary_id
    if pmid.isdigit():
        return f"https://europepmc.org/article/MED/{pmid}"
    return ""


def _fetch_europepmc(
    client: Any, context: Dict[str, Any], *, page_size: int, organism: str
) -> List[CandidatePaper]:
    query = build_query(context)
    if not query:
        return []
    params = {
        "query": query,
        "format": "json",
        "pageSize": max(1, min(page_size, 100)),
        "resultType": "core",
    }
    try:
        resp = client.get(_EUROPEPMC_SEARCH, params=params)
    except Exception:  # noqa: BLE001 - offline / retries exhausted -> no results
        return []
    if getattr(resp, "status_code", 0) != 200:
        return []
    try:
        payload = resp.json()
    except Exception:  # noqa: BLE001
        return []
    results = _safe_list(_safe_dict(payload.get("resultList")).get("result"))
    papers: List[CandidatePaper] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        primary_id = _europepmc_primary_id(item)
        if not primary_id:
            continue
        papers.append(
            CandidatePaper(
                id=primary_id,
                source="europepmc",
                title=str(item.get("title") or "").strip(),
                abstract=str(item.get("abstractText") or "").strip(),
                organism=organism,
                source_uri=_europepmc_uri(item, primary_id),
                year=_year(item.get("pubYear")),
            )
        )
    return papers


def _ncbi_esearch_pmids(client: Any, term: str, retmax: int) -> List[str]:
    _ncbi_throttle()
    params = _ncbi_eutils_params(
        {"db": "pubmed", "term": term, "retmode": "json", "retmax": retmax}
    )
    try:
        resp = client.get(f"{_NCBI_EUTILS_BASE}/esearch.fcgi", params=params)
    except Exception:  # noqa: BLE001
        return []
    if getattr(resp, "status_code", 0) != 200:
        return []
    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        return []
    pmids: List[str] = []
    for value in _safe_list(_safe_dict(data.get("esearchresult")).get("idlist")):
        pmid = _canonical_name(str(value))
        if pmid.isdigit():
            pmids.append(pmid)
    return pmids


def _parse_pubmed_article(node: ElementTree.Element) -> Optional[CandidatePaper]:
    pmid = (node.findtext(".//MedlineCitation/PMID") or "").strip()
    title = " ".join(
        "".join(el.itertext()).strip()
        for el in node.findall(".//Article/ArticleTitle")
    ).strip()
    abstract = " ".join(
        "".join(el.itertext()).strip()
        for el in node.findall(".//Article/Abstract/AbstractText")
    ).strip()
    year = (node.findtext(".//Article/Journal/JournalIssue/PubDate/Year") or "").strip()
    doi = ""
    pmcid = ""
    for aid in node.findall(".//ArticleIdList/ArticleId"):
        id_type = (aid.get("IdType") or "").lower()
        text = (aid.text or "").strip()
        if id_type == "doi":
            doi = text
        elif id_type == "pmc":
            pmcid = text if text.upper().startswith("PMC") else f"PMC{text}"
    primary_id = pmcid or pmid or doi
    if not primary_id:
        return None
    if doi:
        uri = f"https://doi.org/{doi}"
    elif pmid:
        uri = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    else:
        uri = ""
    return CandidatePaper(
        id=primary_id,
        source="ncbi",
        title=title,
        abstract=abstract,
        source_uri=uri,
        year=_year(year),
    )


def _fetch_ncbi(
    client: Any, context: Dict[str, Any], *, retmax: int, organism: str
) -> List[CandidatePaper]:
    term = _ncbi_term(context)
    if not term:
        return []
    pmids = _ncbi_esearch_pmids(client, term, retmax)
    if not pmids:
        return []
    _ncbi_throttle()
    params = _ncbi_eutils_params(
        {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    )
    try:
        resp = client.get(f"{_NCBI_EUTILS_BASE}/efetch.fcgi", params=params)
    except Exception:  # noqa: BLE001
        return []
    if getattr(resp, "status_code", 0) != 200:
        return []
    text = getattr(resp, "text", "") or ""
    if not text.strip():
        return []
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError:
        return []
    papers: List[CandidatePaper] = []
    for node in root.findall(".//PubmedArticle"):
        paper = _parse_pubmed_article(node)
        if paper is None:
            continue
        if organism and not paper.organism:
            paper.organism = organism
        papers.append(paper)
    return papers


def _fetch_crossref(
    client: Any, context: Dict[str, Any], *, rows: int, organism: str
) -> List[CandidatePaper]:
    query = build_query(context)
    if not query:
        return []
    params = {"query": query, "rows": max(1, min(rows, 100))}
    try:
        resp = client.get(_CROSSREF_WORKS, params=params)
    except Exception:  # noqa: BLE001
        return []
    if getattr(resp, "status_code", 0) != 200:
        return []
    try:
        payload = resp.json()
    except Exception:  # noqa: BLE001
        return []
    items = _safe_list(_safe_dict(payload.get("message")).get("items"))
    papers: List[CandidatePaper] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        doi = _canonical_name(str(item.get("DOI") or ""))
        if not doi:
            continue
        titles = _safe_list(item.get("title"))
        title = str(titles[0]).strip() if titles else ""
        year = ""
        parts = _safe_list(
            _safe_dict(item.get("published-print") or item.get("published-online")).get(
                "date-parts"
            )
        )
        if parts and _safe_list(parts[0]):
            year = _year(_safe_list(parts[0])[0])
        papers.append(
            CandidatePaper(
                id=doi,
                source="crossref",
                title=title,
                abstract=str(item.get("abstract") or "").strip(),
                organism=organism,
                source_uri=f"https://doi.org/{doi}",
                year=year,
            )
        )
    return papers


def _fetch_semanticscholar(
    client: Any, context: Dict[str, Any], *, limit: int, organism: str
) -> List[CandidatePaper]:
    query = build_query(context)
    if not query:
        return []
    params = {
        "query": query,
        "limit": max(1, min(limit, 100)),
        "fields": "title,abstract,year,externalIds",
    }
    try:
        resp = client.get(_SEMANTIC_SCHOLAR_SEARCH, params=params)
    except Exception:  # noqa: BLE001
        return []
    if getattr(resp, "status_code", 0) != 200:
        return []
    try:
        payload = resp.json()
    except Exception:  # noqa: BLE001
        return []
    papers: List[CandidatePaper] = []
    for item in _safe_list(payload.get("data")):
        if not isinstance(item, dict):
            continue
        external = _safe_dict(item.get("externalIds"))
        doi = _canonical_name(str(external.get("DOI") or ""))
        pmid = _canonical_name(str(external.get("PubMed") or ""))
        primary_id = doi or pmid or _canonical_name(str(item.get("paperId") or ""))
        if not primary_id:
            continue
        uri = f"https://doi.org/{doi}" if doi else ""
        papers.append(
            CandidatePaper(
                id=primary_id,
                source="semanticscholar",
                title=str(item.get("title") or "").strip(),
                abstract=str(item.get("abstract") or "").strip(),
                organism=organism,
                source_uri=uri,
                year=_year(item.get("year")),
            )
        )
    return papers


def _fetch_biorxiv(
    client: Any, context: Dict[str, Any], *, organism: str, **_: Any
) -> List[CandidatePaper]:
    # bioRxiv has no keyword search API; fetch a recent details window and filter
    # locally by the seed terms. Kept intentionally minimal (optional source).
    terms = _context_terms(context)
    needles = [t.lower() for t in _core_terms(terms)]
    if not needles:
        return []
    url = f"{_BIORXIV_DETAILS}/biorxiv/2020-01-01/2030-01-01/0"
    try:
        resp = client.get(url)
    except Exception:  # noqa: BLE001
        return []
    if getattr(resp, "status_code", 0) != 200:
        return []
    try:
        payload = resp.json()
    except Exception:  # noqa: BLE001
        return []
    papers: List[CandidatePaper] = []
    for item in _safe_list(payload.get("collection")):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        abstract = str(item.get("abstract") or "").strip()
        haystack = f"{title} {abstract}".lower()
        if not any(needle in haystack for needle in needles):
            continue
        doi = _canonical_name(str(item.get("doi") or ""))
        if not doi:
            continue
        papers.append(
            CandidatePaper(
                id=doi,
                source="biorxiv",
                title=title,
                abstract=abstract,
                organism=organism,
                source_uri=f"https://doi.org/{doi}",
                year=_year(item.get("date")),
            )
        )
    return papers


_SOURCE_FETCHERS = {
    "europepmc": _fetch_europepmc,
    "ncbi": _fetch_ncbi,
    "crossref": _fetch_crossref,
    "semanticscholar": _fetch_semanticscholar,
    "biorxiv": _fetch_biorxiv,
}


# ---------------------------------------------------------------------------
# Dedup against the seed + among candidates.
# ---------------------------------------------------------------------------
def _seed_identity_keys(context: Dict[str, Any]) -> set:
    ctx = _safe_dict(context)
    seed = _safe_dict(ctx.get("seed_paper"))
    keys: set = set()

    def _add_id(value: Any) -> None:
        text = _canonical_name(str(value or ""))
        if text:
            keys.add(f"id:{text.lower()}")

    def _add_title(value: Any) -> None:
        norm = _normalize_name(str(value or ""))
        if norm:
            keys.add(f"title:{norm}")

    for source in (seed, ctx):
        _add_id(source.get("doi"))
        _add_id(source.get("pmid"))
        _add_id(source.get("pmcid"))
        _add_id(source.get("id") if source is seed else source.get("seed_id"))
        _add_title(source.get("title") if source is seed else source.get("seed_title"))
    _add_id(ctx.get("seed_doi"))
    _add_id(ctx.get("seed_pmid"))
    _add_id(ctx.get("seed_pmcid"))
    return keys


def _dedupe(
    papers: Sequence[CandidatePaper], seed_keys: set
) -> List[CandidatePaper]:
    out: List[CandidatePaper] = []
    seen: set = set()
    for paper in papers:
        keys = paper.identity_keys()
        if any(key in seed_keys for key in keys):
            continue  # matches the seed paper -> not a new candidate
        if any(key in seen for key in keys):
            continue  # duplicate of an already-kept candidate
        seen.update(keys)
        out.append(paper)
    return out


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------
def search_candidates(
    context: Dict[str, Any],
    *,
    sources: Sequence[str] = _DEFAULT_SOURCES,
    max_papers: Optional[int] = None,
    client: Optional[Any] = None,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
    refresh: bool = False,
    status: Optional[Dict[str, Any]] = None,
) -> List[CandidatePaper]:
    """Fetch candidate papers for a seed pathway context.

    Builds organism-scoped queries from ``context`` (``pathway_name``,
    ``likely_organism``, ``key_compounds``, ``key_proteins``, ``gap_terms``),
    queries each named source over the shared ``HttpClient``, dedupes against the
    seed and among candidates, and caps at ``max_papers``.

    ``max_papers`` defaults to ``rag_config()["acquire_max_papers"]`` (resolved
    at call time). ``sources`` may include the defaults ``("europepmc", "ncbi")``
    plus the optional flags ``"crossref"``, ``"semanticscholar"``, ``"biorxiv"``.

    Offline-first: results are cached per query hash under
    ``data/rag_index/acquire_cache/``. A cache hit is returned without any
    network call; a missing network yields whatever is cached or an empty list,
    never a raised exception.

    Two rules keep the cache from memoizing a silent zero: an empty query (no
    EuropePMC query *and* no NCBI term — an upstream/Stage-0 failure) returns
    ``[]`` immediately without reading, writing, or querying anything; and a
    zero-result search is never persisted, so it is retried on the next run.
    """
    cfg = rag_config()
    cap = max_papers if isinstance(max_papers, int) and max_papers > 0 else int(
        cfg["acquire_max_papers"]
    )
    requested = [str(s).strip().lower() for s in sources if str(s).strip()]
    active = [s for s in requested if s in _SOURCE_FETCHERS]

    cache = AcquireCache(cache_dir, enabled=use_cache)
    terms = _context_terms(context)
    cache_payload = {
        "europepmc_query": build_query(context),
        "ncbi_term": _ncbi_term(context),
        "sources": sorted(active),
        "max_papers": cap,
    }
    key = _hash_key(cache_payload)

    # An empty query is an upstream error (a failed/empty Stage-0 context), not a
    # search result: never consult or write the cache for it, and do not pretend a
    # search happened. Caching it would memoize a permanent, silent zero for every
    # later run whose context also came back empty (the hash of an empty query is
    # stable), so short-circuit before touching the cache or the network.
    if not cache_payload["europepmc_query"] and not cache_payload["ncbi_term"]:
        if isinstance(status, dict):
            status.update(
                {"query": "", "empty_query": True, "from_cache": False, "requests": []}
            )
        return []

    if use_cache and not refresh:
        cached = cache.get(key)
        if cached is not None:
            if isinstance(status, dict):
                status.update(
                    {
                        "query": cache_payload["europepmc_query"],
                        "empty_query": False,
                        "from_cache": True,
                        "requests": [],
                    }
                )
            return [
                CandidatePaper.from_dict(item)
                for item in _safe_list(cached.get("candidates"))
            ]

    http = client if client is not None else HttpClient()
    recorder = _RecordingClient(http)
    per_source = max(cap * 2, cap + 5)  # over-fetch a little before dedupe/cap
    collected: List[CandidatePaper] = []
    for source in active:
        fetcher = _SOURCE_FETCHERS[source]
        try:
            collected.extend(
                _call_fetcher(fetcher, recorder, context, per_source, terms["organism"])
            )
        except Exception:  # noqa: BLE001 - a bad source must not sink the rest
            continue

    if isinstance(status, dict):
        seen_statuses = [c["status"] for c in recorder.calls if c["status"] is not None]
        status.update(
            {
                "query": cache_payload["europepmc_query"],
                "empty_query": False,
                "from_cache": False,
                "requests": len(recorder.calls),
                "http_statuses": sorted(set(seen_statuses)),
                "errors": sorted({c["error"] for c in recorder.calls if c["error"]}),
                "rate_limited": any(s in _THROTTLE_STATUSES for s in seen_statuses),
                "transport_failed": any(c["status"] is None for c in recorder.calls),
            }
        )

    seed_keys = _seed_identity_keys(context)
    deduped = _dedupe(collected, seed_keys)[:cap]

    # Only NON-empty results are memoized. A zero-result search is usually a
    # transient failure (offline, API error, rate limit) and must be retried on the
    # next run rather than frozen into the cache as a permanent zero.
    if deduped:
        cache.set(
            key,
            {
                "query": cache_payload,
                "candidates": [paper.to_dict() for paper in deduped],
            },
        )
    return deduped


def _call_fetcher(
    fetcher: Any,
    client: Any,
    context: Dict[str, Any],
    per_source: int,
    organism: str,
) -> List[CandidatePaper]:
    # Adapt the shared per-source count to each fetcher's keyword name.
    if fetcher is _fetch_europepmc:
        return _fetch_europepmc(
            client, context, page_size=per_source, organism=organism
        )
    if fetcher is _fetch_ncbi:
        return _fetch_ncbi(client, context, retmax=per_source, organism=organism)
    if fetcher is _fetch_crossref:
        return _fetch_crossref(client, context, rows=per_source, organism=organism)
    if fetcher is _fetch_semanticscholar:
        return _fetch_semanticscholar(
            client, context, limit=per_source, organism=organism
        )
    return _fetch_biorxiv(client, context, organism=organism)


def _resolve_europepmc_item(client: Any, candidate: CandidatePaper) -> Dict[str, Any]:
    """Resolve a candidate to an EuropePMC item dict for full-text fetch.

    A PMCID resolves directly. A PMID or DOI is resolved to its PMCID via an
    EuropePMC search (``PMC/DOI resolution as needed``). Fail-safe: returns a
    best-effort item, possibly without a PMCID (then full text is unavailable).
    """
    ident = _canonical_name(candidate.id)
    if ident.upper().startswith("PMC"):
        return {"source": "PMC", "id": ident, "pmcid": ident}

    if ident.isdigit():
        search = f"EXT_ID:{ident} AND SRC:MED"
    elif ident.lower().startswith("10."):
        search = f'DOI:"{ident}"'
    else:
        return {"source": "", "id": ident, "pmcid": ""}

    params = {"query": search, "format": "json", "pageSize": 1, "resultType": "core"}
    try:
        resp = client.get(_EUROPEPMC_SEARCH, params=params)
    except Exception:  # noqa: BLE001
        return {"source": "", "id": ident, "pmcid": ""}
    if getattr(resp, "status_code", 0) != 200:
        return {"source": "", "id": ident, "pmcid": ""}
    try:
        payload = resp.json()
    except Exception:  # noqa: BLE001
        return {"source": "", "id": ident, "pmcid": ""}
    results = _safe_list(_safe_dict(payload.get("resultList")).get("result"))
    if results and isinstance(results[0], dict):
        item = results[0]
        return {
            "source": str(item.get("source") or ""),
            "id": str(item.get("id") or ident),
            "pmcid": str(item.get("pmcid") or ""),
        }
    return {"source": "", "id": ident, "pmcid": ""}


def fetch_full_text(
    candidate: CandidatePaper,
    *,
    client: Optional[Any] = None,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
    refresh: bool = False,
) -> str:
    """Download the full text for a candidate, reusing ``_europepmc_full_text``.

    Resolves a PMCID from the candidate's id (a PMID/DOI is resolved via an
    EuropePMC search first), then fetches ``fullTextXML`` and converts it to
    plain text through the reused core helper. Offline-first: the result is
    cached under ``acquire_cache/fulltext/`` and returned from cache without a
    network call; any error degrades to ``""`` (never raises).
    """
    ident = _canonical_name(candidate.id)
    if not ident:
        return ""

    fulltext_dir = (cache_dir or _default_cache_dir()) / "fulltext"
    cache = AcquireCache(fulltext_dir, enabled=use_cache)
    key = _hash_key({"id": ident.lower(), "source": candidate.source})

    if use_cache and not refresh:
        cached = cache.get(key)
        if cached is not None:
            return str(cached.get("full_text") or "")

    http = client if client is not None else HttpClient()
    item = _resolve_europepmc_item(http, candidate)
    try:
        text = _europepmc_full_text(http, item)
    except Exception:  # noqa: BLE001
        text = ""

    cache.set(key, {"id": ident, "source": candidate.source, "full_text": text})
    return text


__all__ = [
    "CandidatePaper",
    "AcquireCache",
    "search_candidates",
    "fetch_full_text",
    "build_query",
]
