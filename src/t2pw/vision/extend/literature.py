"""Literature extension (VISION section 2 step 3): steps next to the paper core from OTHER
peer-reviewed papers, each with ``other_paper`` provenance carrying the citation and a verbatim quote.

Flow, per run (``extend_literature``):

1. Frontier = the paper's terminal substrates (consumed, never produced) and terminal products
   (produced, never consumed), currency excluded, plus the enzymes of the steps that touch them.
2. A few Europe PMC queries, one frontier compound each, scoped to the paper's organism
   (``rag.acquire.build_query``). Kept only: a journal record, not a preprint, not retracted,
   not the input paper (PMID / PMCID / DOI / title), and an abstract that names no other organism
   only. Every candidate's keep / drop reason is in the report.
3. Text: the abstract always; full text (``rag.acquire.fetch_full_text``) when open access, cut to
   the few passages that name a frontier term.
4. ``rag.extract`` turns each passage into reactions (one LLM call per passage, counted and
   bounded). A reaction is kept only when its quote is a normalised exact substring of the
   passage (>= 20 characters) and ``rag.admission.validate_evidence_span`` finds that quote states
   it (participants, direction, catalysts). Currency the quote does not name is dropped; a catalyst
   the quote does not attach is dropped; no compartment is ever written (the extractor reads none).
   A step must touch a grounded entity already in the payload (otherwise it is not "next to" it).
   The stored quote is the passage's own text at the match, not the model's string. Organism
   (VD-19): a quote or paper naming another organism is refused; one that names none, or only a
   same-genus one, is admitted with ``related_organism: true`` on its records (FULL view only).
   A step counts as accepted only once ``to_rows`` keeps it.
5. Kept steps go through ``convert.to_rows`` (the DB extension's identity rules VD-11 / VD-13) and
   ``payload.merge_extension``; records are relabelled ``other_paper`` / ``extension_lit``.

Fail-open per paper: any error, or the deadline passing mid-paper, means no rows from that paper,
with the reason recorded. Off unless T2PW_EXTEND is on and "literature" is requested; nothing in
the pipeline calls this.
"""

from __future__ import annotations

import html
import json
import re
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from itertools import zip_longest
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlencode

from t2pw.vision.payload import (_blank_source, _Index, is_currency, keyed_rows, merge_extension, norm)

from . import LITERATURE, extend_enabled, paper_context
from .convert import to_rows
from .http import get_text

STAGE = "extension_lit"
MIN_QUOTE = 20
PASSAGE_CHARS = 3000
_PREPRINT_SERVERS = ("biorxiv", "medrxiv", "research square", "researchsquare", "ssrn", "arxiv",
                     "preprints.org", "chemrxiv", "authorea", "qeios")
_BAD_TYPES = ("preprint", "retracted publication", "retraction of publication")


# ---------------------------------------------------------------- frontier and queries

def frontier(big: Dict[str, Any]) -> Dict[str, List[str]]:
    """Terminal substrates / products of the paper's reactions (no currency) and their enzymes."""
    consumed, produced, steps = {}, {}, []
    for section, _, row, is_base in keyed_rows(big):
        if not is_base or section != "processes.reactions" or not isinstance(row, dict):
            continue
        ins = [n for n in row.get("inputs") or [] if isinstance(n, str) and not is_currency(n)]
        outs = [n for n in row.get("outputs") or [] if isinstance(n, str) and not is_currency(n)]
        for n in ins:
            consumed.setdefault(norm(n), n)
        for n in outs:
            produced.setdefault(norm(n), n)
        steps.append((ins, outs, row.get("enzymes") or []))
    subs = [n for k, n in consumed.items() if k not in produced]
    prods = [n for k, n in produced.items() if k not in consumed]
    edge = {norm(n) for n in subs + prods}
    enzymes: Dict[str, str] = {}
    for ins, outs, actors in steps:
        if edge & {norm(n) for n in ins + outs}:
            for a in actors:
                name = a.get("entity") or a.get("protein") or a.get("name") if isinstance(a, dict) else a
                if isinstance(name, str) and name.strip():
                    enzymes.setdefault(norm(name), name)
    return {"substrates": subs, "products": prods, "enzymes": list(enzymes.values())}


def queries(front: Dict[str, List[str]], organism: Optional[str], max_queries: int) -> List[str]:
    """One query per frontier compound, substrates and products alternating."""
    from t2pw.rag.acquire import build_query

    terms = [t for pair in zip_longest(front["substrates"], front["products"]) for t in pair if t]
    out = []
    for t in terms:
        q = build_query({"key_compounds": [t], "organism": organism or ""})
        if q and q not in out:
            out.append(q)
        if len(out) >= max_queries:
            break
    return out


# ---------------------------------------------------------------- candidates

def _ids(item: Dict[str, Any]) -> Dict[str, Optional[str]]:
    pmcid = str(item.get("pmcid") or "").strip().upper() or None
    if pmcid and not pmcid.startswith("PMC"):
        pmcid = "PMC" + pmcid
    pmid = str(item.get("pmid") or "").strip() or None
    doi = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", str(item.get("doi") or "").strip().lower()) or None
    return {"pmid": pmid, "pmcid": pmcid, "doi": doi}


def source_id(ids: Dict[str, Optional[str]]) -> str:
    return ids["pmcid"] or (f"PMID:{ids['pmid']}" if ids["pmid"] else f"DOI:{ids['doi']}")


def journal_of(item: Dict[str, Any]) -> Optional[str]:
    j = ((item.get("journalInfo") or {}).get("journal") or {})
    title = j.get("title") or j.get("isoabbreviation") or item.get("journalTitle")
    return str(title).strip() or None if title else None


def screen(item: Dict[str, Any], paper: Dict[str, Any]) -> Optional[str]:
    """None when the record may be used, else the reason it is dropped."""
    ids = _ids(item)
    if not any(ids.values()):
        return "no_identifier"
    mine = _ids(paper)
    title = normalise(clean_text(paper.get("title")))
    if any(ids[k] and ids[k] == mine[k] for k in mine) or (title and title == normalise(clean_text(item.get("title")))):
        return "input_paper"
    types = [str(t).lower() for t in ((item.get("pubTypeList") or {}).get("pubType") or [])]
    publisher = str((item.get("bookOrReportDetails") or {}).get("publisher") or "").lower()
    journal = journal_of(item)
    if str(item.get("source") or "").upper() == "PPR" or "preprint" in types or any(
            s in publisher or s in (journal or "").lower() for s in _PREPRINT_SERVERS):
        return "preprint"
    if any(t in _BAD_TYPES for t in types):
        return "retracted"
    if not journal:
        return "no_journal_record"
    return None


def clean_text(value: Any) -> str:
    # unescape first: Europe PMC titles carry '&lt;i&gt;' as well as '<i>'
    text = re.sub(r"<[^>]+>", " ", html.unescape(str(value or "")))
    return re.sub(r"\s+", " ", text).strip()


def organism_verdict(requested: Optional[str], text: str) -> Tuple[str, List[str]]:
    from t2pw.rag.admission import compare_organism, organisms_in_span

    seen = organisms_in_span(text)
    return compare_organism(requested or "", seen), seen


# ---------------------------------------------------------------- quotes and passages

def _normalise_map(text: Any) -> Tuple[str, List[int]]:
    """(normalised text, source index of each normalised character). Per character: NFKC, casefold;
    whitespace runs -> one space; no space before closing / after opening punctuation (full text from
    XML puts a space wherever a tag ended: 'in <i>B. subtilis</i>.' -> 'subtilis .'); trimmed."""
    chars: List[Tuple[str, int]] = []
    for i, c in enumerate(str(text or "")):
        for d in unicodedata.normalize("NFKC", c).casefold():
            d = " " if d.isspace() else d
            if d == " " and (not chars or chars[-1][0] == " " or chars[-1][0] in "(["):
                continue
            if d in ".,;:!?)]" and chars and chars[-1][0] == " ":
                chars.pop()
            chars.append((d, i))
    if chars and chars[-1][0] == " ":
        chars.pop()
    return "".join(c for c, _ in chars), [i for _, i in chars]


def normalise(text: Any) -> str:
    return _normalise_map(text)[0]


def source_span(quote: Any, text: str) -> Optional[str]:
    """The passage's own text that the quote matches (normalised exact substring, >= MIN_QUOTE), or None."""
    q = normalise(quote)
    norm_text, where = _normalise_map(text)
    at = norm_text.find(q) if len(q) >= MIN_QUOTE else -1
    return None if at < 0 else text[where[at]:where[at + len(q) - 1] + 1]


def quote_in(quote: Any, text: str) -> bool:
    return source_span(quote, text) is not None


def passages(full_text: str, terms: List[str], k: int) -> List[str]:
    """Up to ``k`` windows (~PASSAGE_CHARS, whole statements) naming the most frontier terms."""
    from t2pw.rag.admission import split_statements, states_name

    windows, cur = [], ""
    for s in split_statements(full_text):
        if cur and len(cur) + len(s) + 1 > PASSAGE_CHARS:
            windows.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        windows.append(cur)
    scored = [(sum(states_name(w, t) for t in terms), i, w) for i, w in enumerate(windows)]
    best = sorted((x for x in scored if x[0]), key=lambda x: (-x[0], x[1]))[:k]
    return [w for _, _, w in sorted(best, key=lambda x: x[1])]


# ---------------------------------------------------------------- one reaction

def check_reaction(rx: Dict[str, Any], passage: str, requested: Optional[str], index: _Index,
                   paper_match: str = "unknown"):
    """(neutral step or None, reason, dropped-part counts). The stored quote is the passage's own text.
    Organism (VD-19): the quote's own organisms decide when it names any, else the paper's verdict
    (title + abstract); 'mismatch' is refused, anything but an exact 'match' is admitted as
    ``related_organism`` (shown in the FULL view only)."""
    from t2pw.rag.admission import states_name, validate_evidence_span

    drops: Counter = Counter()
    model_quote = str(rx.get("quote") or "").strip()
    found = source_span(model_quote, passage)
    if found is None:
        return None, "quote_not_verbatim" if len(normalise(model_quote)) >= MIN_QUOTE else "quote_too_short", drops
    quote = re.sub(r"\s+", " ", found).strip()   # what the checks read; the record keeps ``found``
    names = {}
    for side in ("inputs", "outputs"):
        kept = []
        for p in rx.get(side) or []:
            n = p["name"] if isinstance(p, dict) else str(p)
            if is_currency(n) and not states_name(quote, n):
                drops["currency_not_in_quote"] += 1
                continue
            kept.append(n)
        names[side] = kept
    if not names["inputs"] or not names["outputs"]:
        return None, "one_sided", drops
    enzymes = list(rx.get("enzymes") or [])
    verdict = validate_evidence_span(quote, inputs=names["inputs"], outputs=names["outputs"],
                                     enzymes=enzymes, reversible=bool(rx.get("reversible")))
    if not verdict.ok and enzymes and all("unsupported_catalyst" in r for r in verdict.reasons):
        drops["enzyme_not_in_quote"] += len(enzymes)   # the quote states the chemistry, not that catalyst
        enzymes = []
        verdict = validate_evidence_span(quote, inputs=names["inputs"], outputs=names["outputs"],
                                         enzymes=[], reversible=bool(rx.get("reversible")))
    if not verdict.ok:
        return None, "admission: " + (verdict.reasons[0].split(":")[0] if verdict.reasons else "refused"), drops
    quote_match, seen = organism_verdict(requested, quote)
    if quote_match == "mismatch":
        return None, "quote_organism_mismatch", drops
    match = quote_match if seen else paper_match
    if match == "mismatch":   # the paper screen drops these already; kept for direct callers
        return None, "paper_organism_mismatch", drops
    if not any(index.edge(n) for n in names["inputs"] + names["outputs"]):
        return None, "not_connected", drops
    reversible = verdict.normalized_reversible if verdict.normalized_reversible is not None else bool(
        rx.get("reversible"))
    related = match != "match"
    step = {"kind": "reaction", "name": str(rx.get("name") or "").strip() or None,
            "inputs": [{"type": "compound", "name": n, "synonyms": [], "ids": {}} for n in names["inputs"]],
            "outputs": [{"type": "compound", "name": n, "synonyms": [], "ids": {}} for n in names["outputs"]],
            "enzymes": enzymes, "location": None, "direction": "reversible" if reversible else None,
            "quote": found, "related_organism": related,
            "organism_note": f"organism {match} (quote names {', '.join(seen) or 'none'}; paper {paper_match})"
                             + ("; related_organism: FULL view only" if related else "")}
    return step, None, drops


def _rows_for(big: Dict[str, Any], step: Dict[str, Any], sid: str, n: int, title: Optional[str], note: str):
    """to_rows on a one-step pathway, relabelled to other_paper / extension_lit."""
    index = _Index(big)
    complexes, proteins = [], []
    for e in step["enzymes"]:   # a name the payload already uses for a complex: that complex
        key = index.ent_key.get(norm(e))
        if key and key.startswith("entities.protein_complexes:"):
            complexes.append({"entity": index.key_name.get(key) or e, "entity_type": "protein_complex",
                              "role": "catalyst"})
        else:
            proteins.append({"type": "protein", "name": e, "synonyms": [], "ids": {}})
    pw = {"source_id": sid, "db": "literature", "accession": sid, "name": title, "species": None,
          "steps": [dict(step, accession=f"s{n}", enzymes=proteins)]}
    rows, provs, _, counts = to_rows(big, pw)
    if complexes and rows.get("processes.reactions"):
        rows["processes.reactions"][-1]["enzymes"] += complexes
    for rs in rows.values():
        for r in rs:
            if r.get("provenance") == "extension_db":
                r["provenance"] = STAGE
    # VD-19: ``related_organism`` is a boolean field on every record (view.py shows a row whose records
    # are all related_organism in FULL only); the note repeats it in words
    provs = {s: [[{"kind": "other_paper", "source": sid, "quote": step["quote"], "stage": STAGE,
                   "note": f"{note}; {step['organism_note']}; {rec.get('note')}",
                   "related_organism": step["related_organism"]} for rec in recs] for recs in v]
             for s, v in provs.items()}
    return rows, provs, counts


# ---------------------------------------------------------------- the stage

def _default_chat():
    from t2pw.rag.extract import _default_chat_fn

    return _default_chat_fn


def extend_literature(big: Dict[str, Any], **kw) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(extended copy of ``big``, report); arguments as ``_extend_literature``. Fail-open as a whole:
    an error outside the per-paper guard (frontier, queries, screening, placement) returns ``big``
    unchanged with status 'failed'."""
    try:
        return _extend_literature(big, **kw)
    except Exception as exc:  # noqa: BLE001 - fail open by design
        return big, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:300], "papers": [],
                     "candidates": [], "queries": [], "llm": {"calls": None}, "placement": {}}


def _extend_literature(big: Dict[str, Any], *, paper: Optional[Dict[str, Any]] = None, max_papers: int = 5,
                      budget_s: float = 120, max_llm_calls: int = 8, max_queries: int = 3, page_size: int = 10,
                      passages_per_paper: int = 2, cache_dir=None,
                      get: Optional[Callable[..., Optional[str]]] = None,
                      chat_fn: Optional[Callable[[str, str], str]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(extended copy of ``big``, report). ``paper`` = the input paper's {pmid, pmcid, doi, title}
    (default: ``big["sources"]["paper"]``). ``get(url, *, deadline, cache_dir) -> body | None`` and
    ``chat_fn(system, user) -> str`` are injectable (tests); defaults: ``http.get_text`` and the
    shared LLM client through ``rag.extract``."""
    from t2pw.rag.acquire import _EUROPEPMC_SEARCH, CandidatePaper, fetch_full_text
    from t2pw.rag.extract import extract_reactions_from_text

    deadline = time.monotonic() + budget_s
    get = get or get_text
    chat = chat_fn or _default_chat()
    paper = paper or (big.get("sources") or {}).get("paper") or {}
    cache = f"{cache_dir}/literature" if cache_dir else None
    ctx = paper_context(big)
    organism = next((s["name"] for s in ctx["species"] if s.get("name")), None)
    front = frontier(big)
    terms = front["substrates"] + front["products"] + front["enzymes"]
    report: Dict[str, Any] = {"budget_s": budget_s, "max_papers": max_papers, "max_llm_calls": max_llm_calls,
                              "organism": organism, "frontier": front, "input_paper": {k: paper.get(k) for k in (
                                  "pmid", "pmcid", "doi", "title")}, "queries": [], "candidates": [],
                              "papers": [], "llm": {"calls": 0, "errors": {}}, "status": "ok"}
    llm_errors: Counter = Counter()

    def counted(system: str, user: str) -> str:
        report["llm"]["calls"] += 1
        try:
            return chat(system, user)
        except Exception as exc:  # noqa: BLE001 - extract fails closed; we only count
            llm_errors[type(exc).__name__] += 1
            raise

    class _Resp:   # what fetch_full_text's client contract needs
        def __init__(self, body):
            self.status_code, self.text = (404, "") if body is None else (200, body)

        def json(self):
            return json.loads(self.text)

    class _Client:   # records what fetch_full_text swallows (live: 7 of 9 open-access texts timed out silently)
        def __init__(self):
            self.errors: List[str] = []

        def get(self, url, params=None, **_):
            try:
                return _Resp(get(url + ("?" + urlencode(params) if params else ""), deadline=deadline,
                                 cache_dir=cache))
            except Exception as exc:  # noqa: BLE001
                self.errors.append(type(exc).__name__)
                raise

    # 1-2: search and screen
    hits: List[List[Dict[str, Any]]] = []
    for q in queries(front, organism, max_queries):
        entry = {"query": q, "status": "ok", "hits": 0}
        report["queries"].append(entry)
        url = _EUROPEPMC_SEARCH + "?" + urlencode({"query": q, "format": "json", "pageSize": page_size,
                                                   "resultType": "core"})
        try:
            body = get(url, deadline=deadline, cache_dir=cache)
            items = [i for i in ((json.loads(body or "{}").get("resultList") or {}).get("result") or [])
                     if isinstance(i, dict)]
        except Exception as exc:  # noqa: BLE001 - fail open per query
            entry.update(status="failed", error=type(exc).__name__)
            continue
        entry["hits"] = len(items)
        hits.append(items)
    chosen, seen_ids = [], set()
    for item in (i for row in zip_longest(*hits) for i in row if i):
        ids = _ids(item)
        if any(v and v in seen_ids for v in ids.values()):
            continue
        seen_ids |= {v for v in ids.values() if v}
        reason = screen(item, paper)
        abstract = clean_text(item.get("abstractText"))
        match, orgs = organism_verdict(organism, f"{item.get('title') or ''} {abstract}") if not reason else (None, [])
        if not reason and match == "mismatch":
            reason = "abstract_names_other_organism"
        if not reason and len(chosen) >= max_papers:
            reason = "over_max_papers"
        report["candidates"].append({"id": source_id(ids) if any(ids.values()) else None,
                                     "title": item.get("title"), "journal": journal_of(item),
                                     "year": item.get("pubYear"), "source": item.get("source"),
                                     "organisms": orgs, "organism_match": match,
                                     "decision": "dropped" if reason else "kept", "reason": reason})
        if not reason:
            chosen.append((item, ids, abstract, match))
    if not report["queries"]:
        report["status"] = "no_frontier"
    elif not chosen:
        report["status"] = "no_candidates"

    # 3-5: per paper, atomic
    retrieved = datetime.now(timezone.utc).date().isoformat()
    for item, ids, abstract, match in chosen:
        sid = source_id(ids)
        entry: Dict[str, Any] = {"source_id": sid, "status": "ok", "passages": 0, "extracted": 0, "accepted": 0,
                                 "rejected": Counter(), "dropped_parts": Counter(), "rejected_samples": [],
                                 "organism_match": match}
        report["papers"].append(entry)
        if time.monotonic() >= deadline:
            entry["status"] = report["status"] = "budget_exhausted"
            break
        left = max_llm_calls - report["llm"]["calls"]
        if left <= 0:
            entry["status"] = report["status"] = "llm_budget_exhausted"
            break
        title = clean_text(item.get("title")) or None
        try:
            texts = [abstract] if abstract else []
            if ids["pmcid"] and (item.get("isOpenAccess") == "Y" or item.get("inEPMC") == "Y"):
                cand = CandidatePaper(id=ids["pmcid"], source="europepmc", title=title or "")
                client = _Client()
                full = fetch_full_text(cand, client=client, use_cache=False)
                entry["full_text_chars"] = len(full)
                if client.errors:
                    entry["full_text_errors"] = client.errors
                texts += passages(full, terms, passages_per_paper)
            texts = texts[:left]
            entry["passages"] = len(texts)
            trial, tried = big, 0
            for p_i, text in enumerate(texts):
                if time.monotonic() >= deadline:
                    raise TimeoutError("deadline passed mid-paper")
                where = "abstract" if p_i == 0 and abstract else f"full text passage {p_i}"
                for rx in extract_reactions_from_text(text, chat_fn=counted):
                    entry["extracted"] += 1
                    step, why, drops = check_reaction(rx, text, organism, _Index(trial), match or "unknown")
                    entry["dropped_parts"].update(drops)
                    rows = None
                    if step:
                        tried += 1
                        rows, provs, counts = _rows_for(trial, step, sid, tried, title, where)
                        entry.setdefault("counts", Counter()).update(counts)
                        if not rows.get("processes.reactions"):   # to_rows skipped it (self loop, one side)
                            why = "to_rows: " + next((k for k in counts if k.startswith("skipped")), "skipped")
                    if not rows or not rows.get("processes.reactions"):
                        entry["rejected"][why] += 1
                        if len(entry["rejected_samples"]) < 10:
                            entry["rejected_samples"].append({"reason": why, "quote": str(rx.get("quote"))[:300],
                                                              "inputs": rx.get("inputs"), "outputs": rx.get("outputs")})
                        continue
                    entry["accepted"] += 1
                    if step["related_organism"]:
                        entry["related_organism"] = entry.get("related_organism", 0) + 1
                    src = {sid: {**_blank_source("other_paper"), **ids, "title": title, "journal": journal_of(item),
                                 "year": item.get("pubYear"), "url": f"https://europepmc.org/article/"
                                 f"{item.get('source') or 'MED'}/{item.get('id')}", "retrieved": retrieved}}
                    trial = merge_extension(trial, rows, provs, src)
            big = trial   # the whole paper, or nothing from it
        except Exception as exc:  # noqa: BLE001 - fail open per paper
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:200], accepted_discarded=entry["accepted"])
        for k in ("rejected", "dropped_parts", "counts"):
            entry[k] = dict(entry.get(k) or {})
    report["llm"]["errors"] = dict(llm_errors)
    report["placement"] = placement(big)
    return big, report


def placement(big: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """Per paper: steps merged into base rows, and new steps by hop."""
    out: Dict[str, Counter] = {}
    hops = big.get("hops") or {}
    for section, key, _, is_base in keyed_rows(big):
        if not section.startswith("processes."):
            continue
        for sid in {r.get("source") for r in big["provenance"].get(key, []) if r.get("stage") == STAGE}:
            h = hops.get(key)
            bucket = "merged_into_base" if is_base else (
                "hop_none" if h is None else "hop_1" if h <= 1 else "hop_2_plus")
            out.setdefault(sid, Counter())[bucket] += 1
    return {k: dict(v) for k, v in out.items()}


def run_if_enabled(big: Dict[str, Any], *, sources=(), **kw) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """``(big, None)`` untouched unless T2PW_EXTEND is on AND "literature" is in ``sources``."""
    if extend_enabled() and LITERATURE in sources:
        return extend_literature(big, **kw)
    return big, None
