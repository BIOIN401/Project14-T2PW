"""Smart search (W10): one input box, classified and routed.

``classify(text, organisms)`` is deterministic: paper identifiers by regex (PMID, PMCID,
DOI, publisher/PubMed URLs), pasted text by length, else a pathway phrase with an
optional organism found by exact organism-list match ("glycolysis in yeast",
"enterobactin biosynthesis E. coli").

``llm_extract(text, organisms, chat_fn)`` is the only LLM use: it is called by the web
layer only when the deterministic reading found no PathBank pathway and the text reads
like a sentence. It returns the same ``Route`` shape; every identifier it returns is
re-checked with the regexes and an organism is kept only if it is in the organism list.
The router prompt lives here on purpose: ``t2pw/llm/prompts/`` is frozen for this track.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional, Sequence, Tuple

log = logging.getLogger("t2pw.web.smart_search")

MAX_QUERY = 20_000  # a pasted passage; the form caps the box at this too
TEXT_MIN_CHARS = 400  # shorter than this is never treated as pasted paper text
LLM_TIMEOUT_S = 25.0

PATHWAY, PAPER, TEXT, URL_ONLY, EMPTY = "pathway", "paper", "text", "url", "empty"

_PMCID = re.compile(r"\bPMC\d{3,9}\b", re.I)
_DOI = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>]+", re.I)
_PMID_TAGGED = re.compile(r"\bPMID\s*:?\s*(\d{1,9})\b", re.I)
_PUBMED_URL = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d{1,9})|europepmc\.org/(?:abstract|article)/MED/(\d{1,9})", re.I)
_URL = re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.I)
_ID_LIST = re.compile(r"[\d\s,;]+")


@dataclass(frozen=True)
class Route:
    kind: str
    query: str
    pathway_name: str = ""
    organism: Any = None  # a t2pw.service.organisms.Organism, or None
    organism_text: str = ""  # what the user typed for the organism, when it matched nothing
    pmids: Tuple[str, ...] = ()
    pmcids: Tuple[str, ...] = ()
    dois: Tuple[str, ...] = ()
    used_llm: bool = False
    notes: Tuple[str, ...] = field(default=())


def _dedupe(items) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def _ids(text: str) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    pmcids = _dedupe(m.upper() for m in _PMCID.findall(text))
    dois = _dedupe(d.rstrip(".,;)]") for d in _DOI.findall(text))
    pmids = [m for m in _PMID_TAGGED.findall(text)]
    pmids += [a or b for a, b in _PUBMED_URL.findall(text)]
    if not (pmcids or dois or pmids) and _ID_LIST.fullmatch(text) and re.search(r"\d", text):
        pmids = re.findall(r"\d+", text)  # a bare list of numbers is a PMID list
    return _dedupe(p.lstrip("0") for p in pmids if p.strip("0")), pmcids, dois


def find_organism(phrase: str, organisms: Sequence[Any]):
    """An organism-list entry whose label, common name or genus abbreviation ("E. coli")
    equals ``phrase`` (case and dots ignored). Exact only: a guess here would scope a run."""
    want = normalize_org(phrase)
    if not want:
        return None
    hits = []
    for o in organisms:
        label = normalize_org(o.label)
        head, _, rest = label.partition(" ")
        keys = {normalize_org(o.common_name or "")} | ({head[0] + " " + rest} if rest else set())
        if want == label or want in keys:
            hits.append((not o.has_pathways, o.id, o))
    # ponytail: "yeast" and "human" name several PathBank species (some are junk entries labelled
    # "human"); the lowest species id is the curated main one (Homo sapiens 1, S. cerevisiae 18).
    # Rank by pathway count if that ever fails.
    return min(hits, key=lambda h: h[:2])[2] if hits else None


def normalize_org(text: str) -> str:
    return " ".join(str(text).casefold().replace(".", " ").split())


def _split_organism(text: str, organisms: Sequence[Any]):
    """``("glycolysis", <S. cerevisiae>)`` from "glycolysis in Saccharomyces cerevisiae"; the
    organism is the text after the last " in "/" of "/"," or the last 1-3 words."""
    words = text.split()
    m = re.match(r"(.+?)\s*(?:\bin\b|\bfor\b|,|\(|\bof\b)\s*([^,()]+?)\)?\s*$", text, re.I)
    if m:
        org = find_organism(m.group(2), organisms)
        if org is not None:
            return m.group(1).strip(), org
    for k in (3, 2, 1):
        if len(words) > k:
            org = find_organism(" ".join(words[-k:]), organisms)
            if org is not None:
                return " ".join(words[:-k]), org
    return text, None


def classify(text: str, organisms: Sequence[Any]) -> Route:
    q = " ".join(str(text or "").split()) if len(str(text or "")) < TEXT_MIN_CHARS else str(text).strip()
    if not q:
        return Route(EMPTY, "")
    long = len(q) >= TEXT_MIN_CHARS
    if long and len(re.findall(r"[.!?](\s|$)", q)) >= 2:
        return Route(TEXT, q)
    pmids, pmcids, dois = _ids(q)
    if pmids or pmcids or dois:
        return Route(PAPER, q, pmids=pmids, pmcids=pmcids, dois=dois)
    if _URL.search(q):
        return Route(URL_ONLY, q)
    if long:
        return Route(TEXT, q)
    name, org = _split_organism(q, organisms)
    name = re.sub(r"^(show me|find|the)\s+", "", name, flags=re.I).strip(" .?") or q
    return Route(PATHWAY, q, pathway_name=name[:200], organism=org)


def wants_llm(route: Route) -> bool:
    """Only a pathway phrase that reads like a sentence is worth an LLM call."""
    return route.kind == PATHWAY and (len(route.pathway_name.split()) >= 4 or "?" in route.query)


ROUTER_PROMPT = """You route requests for a biological pathway web server.
Read the user's request and answer with ONE JSON object, no prose:
{"kind": "pathway" | "paper" | "other",
 "pathway_name": "<the pathway's usual name, e.g. 'Glycolysis', 'TCA Cycle', 'Ergosterol Biosynthesis'; empty if none>",
 "organism": "<the organism's scientific name, e.g. 'Saccharomyces cerevisiae'; empty if none is stated or clearly implied>",
 "identifiers": ["<PMIDs, PMCIDs or DOIs written in the request, copied exactly>"]}
Use "paper" only when the request gives an identifier. Never invent identifiers."""


def llm_extract(route: Route, organisms: Sequence[Any], chat_fn: Optional[Callable[..., str]] = None,
                timeout_s: float = LLM_TIMEOUT_S) -> Optional[Route]:
    """A better reading of ``route.query`` from the LLM, or ``None`` (no answer, a bad
    answer, a timeout, no provider configured). Never raises."""
    if chat_fn is None:
        chat_fn = _default_chat
    messages = [{"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": route.query[:2000]}]
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        raw = pool.submit(chat_fn, messages).result(timeout=timeout_s)
        data = json.loads(raw) if isinstance(raw, str) else None
    except FutureTimeout:
        log.warning("search router: LLM timed out after %.0f s", timeout_s)
        return None
    except Exception as exc:  # noqa: BLE001 - any provider/parse failure means "no better reading"
        log.warning("search router: LLM unavailable (%s)", type(exc).__name__)
        return None
    finally:
        pool.shutdown(wait=False)  # ponytail: a hung call keeps its thread until the client's own timeout
    if not isinstance(data, dict):
        return None
    pmids, pmcids, dois = _ids(" ".join(str(i) for i in data.get("identifiers") or [] if isinstance(i, (str, int))))
    if (pmids or pmcids or dois) and data.get("kind") == PAPER:
        return replace(route, kind=PAPER, pmids=pmids, pmcids=pmcids, dois=dois, used_llm=True)
    name = data.get("pathway_name")
    if not isinstance(name, str) or not name.strip():
        return None
    org_text = data.get("organism") if isinstance(data.get("organism"), str) else ""
    org = find_organism(org_text, organisms) if org_text.strip() else None
    return replace(route, pathway_name=" ".join(name.split())[:200], organism=org or route.organism,
                   organism_text="" if org else org_text.strip()[:200], used_llm=True)


IDCONV_URL = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"  # same service as service/inputs.py


def to_pmids(pmcids: Sequence[str], dois: Sequence[str], get: Optional[Callable[..., Any]] = None) -> dict:
    """PMCID / DOI -> PMID through the NCBI ID converter: ``{id: pmid or ""}``; an id the
    lookup could not answer is absent. The pipeline takes PMIDs, so the web layer converts
    before queueing. Never raises."""
    if get is None:
        import httpx

        get = lambda url, params: httpx.get(url, params=params, timeout=10.0)  # noqa: E731
    from t2pw.mapping.map_ids import _ncbi_eutils_params

    out = {}
    for idtype, ids in (("pmcid", pmcids), ("doi", dois)):
        if not ids:
            continue
        try:
            resp = get(IDCONV_URL, params=_ncbi_eutils_params({"ids": ",".join(ids), "idtype": idtype, "format": "json"}))
            records = resp.json().get("records") if getattr(resp, "status_code", 0) == 200 else None
        except Exception as exc:  # noqa: BLE001 - network or a bad reply: leave these ids unanswered
            log.warning("search: idconv failed (%s)", type(exc).__name__)
            continue
        wanted = {i.casefold(): i for i in ids}
        for rec in records or []:
            key = str(rec.get("requested-id") or rec.get(idtype) or "").casefold() if isinstance(rec, dict) else ""
            if key in wanted:
                pmid = str(rec.get("pmid") or "")
                out[wanted[key]] = pmid if pmid.isdigit() else ""
    return out


def _default_chat(messages) -> str:
    from t2pw.llm.client import chat_detailed  # lazy: importing the client reads provider config

    return chat_detailed(messages, temperature=0.0, response_json=True, stage_name="web_search_router",
                         model_env_var="OPENROUTER_SEARCH_ROUTER_MODEL", timeout=LLM_TIMEOUT_S,
                         max_retries=0).text
