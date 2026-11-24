"""Final payload, phase 1 (PO ruling D-W12): a deterministic database id fill before the freeze.

Flag ``T2PW_FINAL_PAYLOAD`` (default ON since D-W19a; unset or empty is on, only 0/false/no/off turns it off). Reuses the clean model's online fill layer
(``clean_model.builder`` + ``clean_model.fill``) to find ids, then writes only the accepted ones into the payload rows, under
the keys the PWML writer already reads: compounds ``chebi_id kegg_id hmdb_id pubchem_cid cas drugbank_id`` and species
``taxonomy_id``. Proteins get nothing written: the fill layer only proposes UniProt candidates for them (P-20, D-106/D-108/
D-117), so those stay report suggestions. No LLM, no gene names, no reaction EC/Rhea inference.

Rules: an existing id is never touched; one exact label/synonym match only; an id already used by another row of the same
list is refused; complexes, families/classes and the Unknown sentinel are skipped. Any failure leaves ``payload`` exactly as it
was. Provenance goes under one new top-level key, ``final_payload_fill``.

Order: the fill runs BEFORE the pre-freeze resolver (``prefreeze_with_final_fill``), so a filled id reaches the D-015 resolver
(which matches by chebi/kegg id and renames, e.g. H+ -> Hydrogen Ion) and the freeze/hash sees the canonical name. If a filled id
makes the resolver raise ``PrefreezeResolutionError``, the payload is restored in place and the resolver re-run, so a fill can
never cause a refusal the flag-off run would not have (the title/reference key is dropped with it).

Known limits: (1) the title and reference follow the payload key, not the flag, at export time (``effective_pathway_name`` and
the IR read ``final_payload_pathway`` whenever it is present). (2) The id fill and the nucleic lookup share one 300 s budget, the
title lookup has its own 60 s, and the throttle is per fetcher: no NEW request starts after about 360 s, but a request already in
flight (4 tries, up to ~127 s on a blackholed host) can run past that. Fix round 3 bounds this further: (a) both budgets are capped
to the time left before the job deadline (``state["pathway_run_deadline"]``, epoch seconds) minus ``DEADLINE_MARGIN_S``, and under a
deadline the live fetcher uses 10 s timeouts and 2 retries (worst request ~33 s), so a web job that finished with the flag off
cannot run out of time here; (b) ``_Budget`` opens a circuit per HOST after ``CIRCUIT_BREAKER_N`` consecutive ``FetchError`` s from it (a success
resets that host's count; D-W23 c), so a dead network costs ``CIRCUIT_BREAKER_N`` failed requests per host (worst case: the global time budget plus one request in flight); each step report carries ``circuit_open``; (c) ``T2PW_FINAL_PAYLOAD_BUDGET_S``
caps every budget including the title's (non-finite or negative values are ignored). ``T2PW_FINAL_PAYLOAD_BUDGET_S=0`` is the
OFFLINE SWITCH: the layer makes no network call at all and fills nothing that needs one (cached answers are still used). (3) The
``final_payload_fill`` key carries dates, so the canonical hash changes from day to day for the same run. (4) Role/class terms:
a name that is a ChEBI role or class (``electron acceptor``, ``antioxidant``) carries no structure in ChEBI, so a chebi_id fill
is accepted only when the same lookup also returned an InChIKey (``no_structure_in_chebi`` otherwise). That is the only signal
available without extra requests; it also refuses the rare real molecule ChEBI holds without a structure (and any entity whose
details lookup failed), which stays a suggestion-free miss, never a wrong id. (5) A bare PMID/PMCID/DOI in the paper id is read
as an id, but a reference is written only when the NCBI lookup confirmed that paper.
"""

from __future__ import annotations

import copy
import datetime
import math
import os
import re
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

FINAL_PAYLOAD_FILL_FIELD = "final_payload_fill"
BUDGET_SECONDS = 300.0  # wall clock for live requests; cached answers are free, so a re-run finishes the rest
TITLE_BUDGET_SECONDS = 60.0  # the title / reference lookup (one or two requests) has its own, so a spent id fill cannot starve it (review 2, R2-3)
HOST_INTERVAL = 1.0  # at most 1 request per second per host (the fill layer's own default is 3/s)
CIRCUIT_BREAKER_N = 5  # consecutive FetchErrors after which every further request is refused (dead network: seconds, not minutes)
DEADLINE_MARGIN_S = 60.0  # kept free for the freeze, audit and export between the end of this layer and the job deadline
DEADLINE_STATE_KEY = "pathway_run_deadline"  # = t2pw.service.extraction.DEADLINE_STATE_KEY (absolute time.time() seconds; a test pins them equal)

_COMPOUND_ALIASES = {"chebi_id": ("chebi",), "kegg_id": ("kegg",), "hmdb_id": ("hmdb",),
                     "pubchem_cid": ("pubchem", "pubchem_id"), "cas": ("cas",), "drugbank_id": ("drugbank",)}
_TARGETS = {"compound": ("compounds", _COMPOUND_ALIASES), "species": ("species", {"taxonomy_id": ()})}
_PLACEHOLDERS = {"", "n/a", "na", "none", "null", "nan", "unknown", "unmapped", "-"}
_NOT_A_MOLECULE = re.compile(r"\bcomplex of\b|\b(family|superfamily|class|classes)\b|^unknown$", re.I)


def final_payload_enabled() -> bool:
    return os.environ.get("T2PW_FINAL_PAYLOAD", "").strip().casefold() not in ("0", "false", "no", "off")


def _empty(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip().casefold() in _PLACEHOLDERS)


def _short_abbreviation(name: Any) -> bool:
    """Two to five capital letters once spaces are removed: no digits, no charge signs, no lowercase (MK, PMEA, CDG; not H2O, H+, O2, NAD+)."""
    return re.fullmatch(r"[A-Z]{2,5}", re.sub(r"\s+", "", str(name))) is not None


def _norm(v: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v).casefold()).removeprefix("chebi").removeprefix("hmdb").lstrip("0")


class _Budget:
    """Wraps the live fetcher: once the wall-clock budget is spent, every further request is a FetchError; so is every request to a host
    that gave ``breaker`` FetchErrors in a row (counted in ``refused``, not fatal). The circuit is per host (D-W23 c): a success resets
    only its own host's count, and a dead host never stops requests to another. The time budget stays global."""

    def __init__(self, inner: Any, seconds: float, breaker: int = CIRCUIT_BREAKER_N) -> None:
        self.inner, self.deadline, self.breaker = inner, time.monotonic() + seconds, breaker
        self.refused, self.errors_in_a_row, self.open_hosts = 0, {}, set()

    @property
    def circuit_open(self) -> bool:  # any host's circuit is open (what the step reports carry)
        return bool(self.open_hosts)

    @property
    def exhausted(self) -> bool:  # a request was refused (budget spent or circuit open)
        return self.refused > 0

    def limit(self, seconds: float, *, tighten: bool = False) -> None:
        """Never run past ``seconds`` from now (only ever shortens). ``tighten``: a hard job deadline, so also bound one request."""
        self.deadline = min(self.deadline, time.monotonic() + max(0.0, seconds))
        if tighten and hasattr(self.inner, "timeout") and hasattr(self.inner, "retries"):
            self.inner.timeout, self.inner.retries = min(self.inner.timeout, 10.0), min(self.inner.retries, 2)

    def _call(self, fn: Any, url: str, params: Optional[Dict[str, Any]]) -> Any:
        from t2pw.clean_model.fill.http import FetchError
        host = urlsplit(str(url)).netloc.casefold()
        if host in self.open_hosts or time.monotonic() >= self.deadline:
            self.refused += 1
            raise FetchError("final payload fill circuit open" if host in self.open_hosts else "final payload fill time budget exhausted")
        try:
            out = fn(url, params)
        except FetchError:
            n = self.errors_in_a_row[host] = self.errors_in_a_row.get(host, 0) + 1
            if n >= self.breaker:
                self.open_hosts.add(host)
            raise
        self.errors_in_a_row[host] = 0
        return out

    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._call(self.inner.get_json, url, params)

    def get_text(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[str]:
        return self._call(self.inner.get_text, url, params)


def _env_budget() -> Optional[float]:
    """T2PW_FINAL_PAYLOAD_BUDGET_S as a finite number >= 0, else None (unset, empty, nan, inf, negative or not a number: use the default)."""
    try:
        v = float(os.environ.get("T2PW_FINAL_PAYLOAD_BUDGET_S", ""))
    except ValueError:
        return None
    return v if math.isfinite(v) and v >= 0 else None


def _live_fetcher(seconds: Optional[float] = None) -> Any:
    from t2pw.clean_model.fill.http import Throttle, UrllibFetcher
    if seconds is None:
        env = _env_budget()
        seconds = BUDGET_SECONDS if env is None else env
    return _Budget(UrllibFetcher(throttle=Throttle(interval=HOST_INTERVAL)), seconds)


def _title_budget() -> float:
    env = _env_budget()
    return TITLE_BUDGET_SECONDS if env is None else min(TITLE_BUDGET_SECONDS, env)


def _cap_to_job_deadline(fetcher: Any, state: Any) -> Any:
    """Shorten a live budgeted fetcher to the time left before ``state["pathway_run_deadline"]`` minus the margin. No deadline in
    state (Streamlit, tests), or a fetcher that is not ours (a caller-given fake): untouched."""
    st = state if hasattr(state, "get") else {}
    dl = st.get(DEADLINE_STATE_KEY)
    if isinstance(dl, (int, float)) and not isinstance(dl, bool) and math.isfinite(dl) and hasattr(fetcher, "limit"):
        fetcher.limit(dl - time.time() - DEADLINE_MARGIN_S, tighten=True)
    return fetcher


def _refused(fetcher: Any) -> int:
    """Requests the budgeted fetcher has refused so far (a fake has no budget: 0)."""
    return int(getattr(fetcher, "refused", 0) or 0)


def _circuit_open(fetcher: Any) -> bool:
    return bool(getattr(fetcher, "circuit_open", False))


def _plan(payload: Dict[str, Any], items: List[Dict[str, Any]], report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Accepted writes: [{row, field, value, entry}] with every hard rule checked against the payload as it stands."""
    ents = payload.get("entities") if isinstance(payload.get("entities"), dict) else {}
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for it in items:
        if it.get("kind") in _TARGETS and it.get("field") in _TARGETS[it["kind"]][1] and isinstance(it.get("entity"), str):
            groups.setdefault((it["kind"], it["entity"], it["field"]), []).append(it)
    writes: List[Dict[str, Any]] = []
    taken: Dict[tuple, set] = {}  # (kind, field) -> normalised ids written in this pass
    refused = report["refused"]

    def refuse(reason: str, key: tuple) -> None:
        refused[reason] = refused.get(reason, 0) + 1
        report["refused_detail"].append({"kind": key[0], "entity": key[1], "field": key[2], "reason": reason})

    # A short all-capitals abbreviation matched only as a ChEBI synonym is often a different compound's synonym (MK -> Met-Lys,
    # PMEA -> adefovir, CDG -> Cys-Asp-Gly; hand check 2026-11-11): refuse the entity, with every cross-reference derived from it.
    # Exactly "exact_synonym": exact_label and the charged_form fills (NAD+) were all correct.
    abbreviation = {(i["kind"], i["entity"]) for i in items if i.get("kind") == "compound" and i.get("field") == "chebi_id"
                    and (i.get("prov") or {}).get("match_type") == "exact_synonym" and _short_abbreviation(i.get("entity"))}
    # F4: a chebi_id found by name whose details lookup returned no InChIKey is a ChEBI role or class (no structure): refuse it
    # and every cross-reference derived from it.
    structured = {(i["kind"], i["entity"]) for i in items if i.get("field") == "inchikey" and i.get("value")}
    by_name = {(i["kind"], i["entity"]) for i in items if i.get("kind") == "compound" and i.get("field") == "chebi_id"
               and str((i.get("prov") or {}).get("match_type") or "").startswith("exact_")}
    for key, its in groups.items():
        kind, name, fld = key
        if (kind, name) in abbreviation:
            refuse("short_abbreviation_synonym", key)
            continue
        if len({_norm(i["value"]) for i in its}) != 1:
            refuse("ambiguous", key)
            continue
        it = its[0]
        rows = [r for r in ents.get(_TARGETS[kind][0]) or [] if isinstance(r, dict)]
        hit = [r for r in rows if r.get("name") == name]
        if len(hit) != 1:
            refuse("row_not_unique", key)
            continue
        row = hit[0]
        mapped = row.get("mapped_ids") if isinstance(row.get("mapped_ids"), dict) else {}
        if kind == "compound" and _NOT_A_MOLECULE.search(name):
            refuse("not_a_single_molecule", key)
        elif (kind, name) in by_name and (kind, name) not in structured:
            refuse("no_structure_in_chebi", key)
        elif not _empty(row.get(fld)) or (isinstance(row.get("db_row"), dict) and not _empty(row["db_row"].get(fld))):
            refuse("slot_filled", key)
        elif any(not _empty(mapped.get(a)) for a in _TARGETS[kind][1][fld]):
            refuse("mapped_ids_filled", key)
        elif any(r is not row and not _empty(r.get(fld)) and _norm(r[fld]) == _norm(it["value"]) for r in rows) \
                or _norm(it["value"]) in taken.get((kind, fld), set()):
            refuse("id_used_by_other_row", key)
        else:
            taken.setdefault((kind, fld), set()).add(_norm(it["value"]))
            p = it.get("prov") or {}
            writes.append({"row": row, "field": fld, "value": str(it["value"]),
                           "entry": {"kind": kind, "entity": name, "field": fld, "accession": str(it["value"]),
                                     "source": p.get("source"), "match_type": p.get("match_type"),
                                     "retrieved": p.get("retrieved")}})
    return writes


def apply_final_id_fill(payload: Dict[str, Any], meta: Optional[Dict[str, Any]] = None, *, fetcher: Any = None,
                        cache: Any = None, today: Optional[str] = None, paper_text: Optional[str] = None,
                        paper_source: str = "paper text") -> Dict[str, Any]:
    """Fill empty id slots of ``payload`` in place. Fails open: on any error ``payload`` is restored and the report says why."""
    report: Dict[str, Any] = {"enabled": True, "written": 0, "by_field": {}, "refused": {}, "refused_detail": [],
                              "suggestions": [], "error": None, "budget_exhausted": False, "circuit_open": False}
    snapshot = copy.deepcopy(payload)
    fetcher = fetcher or _live_fetcher()
    refused0 = _refused(fetcher)
    try:
        from t2pw.clean_model import build_from_payload
        from t2pw.clean_model.fill import fill_model

        today = today or datetime.date.today().isoformat()
        # D-W20a: ontology ids for location / cell-type / tissue names, first (few requests) so a spent budget costs compound fills
        from t2pw.service.final_payload_ontology import fill_ontology_ids
        onto = fill_ontology_ids(payload, fetcher, cache, today, paper_text=paper_text, paper_source=paper_source)  # P5: text for numbered cell types
        report["ontology"] = {k: onto[k] for k in ("counts", "refused", "requests", "cache_hits", "extra")}
        for e in onto["entries"]:
            report["by_field"]["%s.%s" % (e["kind"], e["field"])] = report["by_field"].get("%s.%s" % (e["kind"], e["field"]), 0) + 1
        model = build_from_payload(payload, dict(meta or {}))
        rep = fill_model(model, fetcher=fetcher, cache=cache, today=today, sources=("chebi", "uniprot", "ncbi"))
        report["fill"] = rep.to_dict()
        report["suggestions"] = copy.deepcopy(model.db_layer.get("suggestions") or [])
        writes = _plan(payload, model.db_layer.get("items") or [], report)
        for w in writes:
            w["row"][w["field"]] = w["value"]
            report["by_field"][w["entry"]["kind"] + "." + w["field"]] = report["by_field"].get(w["entry"]["kind"] + "." + w["field"], 0) + 1
        entries = onto["entries"] + [w["entry"] for w in writes]
        if entries:
            payload[FINAL_PAYLOAD_FILL_FIELD] = {"schema": 1, "date": today, "written": len(entries),
                                                 "by_field": dict(report["by_field"]), "entries": entries}
        report["written"] = len(entries)
    except Exception as exc:  # fail open: the payload is exactly what it was
        payload.clear()
        payload.update(snapshot)
        report.update(written=0, by_field={}, error=f"{type(exc).__name__}: {exc}")
    report["budget_exhausted"], report["circuit_open"] = _refused(fetcher) > refused0, _circuit_open(fetcher)  # refused DURING this step
    return report


def fill_meta(pathway_context: Any = None, state: Any = None) -> Dict[str, Any]:
    """Run metadata for the fill: organism from the Stage-0 context; pmid/pmcid only if the caller's state carries them."""
    ctx = pathway_context if isinstance(pathway_context, dict) else {}
    st = state if hasattr(state, "get") else {}
    meta: Dict[str, Any] = {}
    org = ctx.get("likely_organism") or ctx.get("organism")
    if org:
        meta["organism"] = str(org)
    for k in ("pmid", "pmcid"):
        v = ctx.get(k) or st.get(k)
        if v:
            meta[k] = str(v)
    return meta


def _paper_kw(state: Any) -> Dict[str, Any]:
    """P5: the paper text of the run (``state["final_payload_paper_text"]``, set by ``run_pathway`` with the flag on), for the ontology fill."""
    st = state if hasattr(state, "get") else {}
    txt = st.get("final_payload_paper_text")
    return {"paper_text": txt, "paper_source": str(st.get("final_payload_paper_text_source") or "paper text")} if isinstance(txt, str) and txt else {}


def maybe_apply_final_id_fill(payload: Dict[str, Any], pathway_context: Any = None, state: Any = None, *,
                              chat_fn: Any = None, **kw: Any) -> Dict[str, Any]:
    """The postpipeline seam: {} (payload untouched) when the flag is off; otherwise the fill report, whose ``merge`` key is the
    D-W24 (5) state / duplicate-reaction merge (``final_payload_merge``) and whose ``compartments`` key is the D-W24 (2, 7a)
    compartment estimate (``final_payload_compartments``, run last; ``chat_fn`` replaces its LLM call). Never raises."""
    if not final_payload_enabled():
        return {}
    shared: Dict[str, Any] = {}
    try:
        # ONE fetcher (one time budget, T2PW_FINAL_PAYLOAD_BUDGET_S from its construction) for the id fill and the nucleic lookup: a
        # fill that spends it leaves the nucleic rows unknown with lookup_error (each report says ``budget_exhausted`` and ``circuit_open``). Both budgets are capped to the job deadline. The title /
        # reference lookup is NOT in it: it builds its own small budget (TITLE_BUDGET_SECONDS) in ``apply_final_pathway_meta``, so a
        # slow fill cannot starve it (review 2, R2-3; the title reads only the paper metadata, nothing the fill writes). A caller-given
        # ``fetcher`` is used as it is by all three steps.
        shared = {**kw, "fetcher": kw.get("fetcher") or _cap_to_job_deadline(_live_fetcher(), state)}
        report = apply_final_id_fill(payload, fill_meta(pathway_context, state), **{**_paper_kw(state), **shared})
        report["meta_keys"] = sorted(fill_meta(pathway_context, state))
        report["nucleic_types"] = maybe_apply_paper_nucleic_types(payload, state, **{k: shared[k] for k in ("fetcher", "cache", "today") if k in shared})
        report["pathway_meta"] = maybe_apply_final_pathway_meta(payload, pathway_context, state, **kw)
    except Exception as exc:
        report = {"enabled": True, "written": 0, "error": f"{type(exc).__name__}: {exc}"}
    # D-W24 (5), L6: last, offline, whatever the fill did; never raises (fails open, payload restored).
    try:
        from t2pw.service.final_payload_merge import merge_states_and_duplicates
        report["merge"] = merge_states_and_duplicates(payload)
    except Exception as exc:
        report["merge"] = {"error": f"{type(exc).__name__}: {exc}"}
    # D-W24 (2, 7a), L7: after the merge, so its neighbours and candidates are the merged states; never raises (fails open).
    try:
        from t2pw.service.final_payload_compartments import estimate_compartments
        report["compartments"] = estimate_compartments(payload, state, cache=kw.get("cache"), chat_fn=chat_fn,
                                                       fetcher=shared.get("fetcher"), today=kw.get("today"),
                                                       organism_hint=fill_meta(pathway_context, state).get("organism") or "")
    except Exception as exc:
        report["compartments"] = {"error": f"{type(exc).__name__}: {exc}"}
    # Lane loc/assign: flagged inferred locations (organism default, LLM quote), after L7, before L8; never raises.
    try:
        from t2pw.service.final_payload_locations import assign_locations
        report["locations"] = assign_locations(payload, state, cache=kw.get("cache"), chat_fn=chat_fn,
                                               fetcher=shared.get("fetcher"), today=kw.get("today"),
                                               organism_hint=fill_meta(pathway_context, state).get("organism") or "")
    except Exception as exc:
        report["locations"] = {"error": f"{type(exc).__name__}: {exc}"}
    # D-W24 (6, 7c), L8: evidence checks, last: after L7's compartment estimate, whose rows it checks, offline; never raises.
    try:
        from t2pw.service.final_payload_evidence import check_evidence
        ctx = pathway_context if isinstance(pathway_context, dict) else {}
        report["evidence"] = check_evidence(payload, _paper_kw(state).get("paper_text"), {
            "pathway_name": str(ctx.get("pathway_name") or ""), "organism": fill_meta(pathway_context, state).get("organism") or ""})
    except Exception as exc:
        report["evidence"] = {"error": f"{type(exc).__name__}: {exc}"}
    return report


def maybe_apply_paper_nucleic_types(payload: Dict[str, Any], state: Any = None, *, fetcher: Any = None, cache: Any = None,
                                    today: Optional[str] = None) -> Dict[str, Any]:
    """D-W20c / D-W21d: DNA/RNA type of unsettled nucleic acids read from the paper text in ``state["final_payload_paper_text"]``
    (set by ``run_pathway`` with the flag on; absent in the Streamlit path), then NCBI Gene / RNAcentral by exact name in the
    pathway species, then ``nucleotide_type_status: "unknown"``. ``fetcher`` defaults to the live, budgeted one (a NEW budget when
    called on its own; ``maybe_apply_final_id_fill`` passes the one fetcher it shares with the id fill)."""
    from t2pw.service.final_payload_nucleic import apply_nucleic_types
    st = state if hasattr(state, "get") else {}
    fetcher = fetcher or _cap_to_job_deadline(_live_fetcher(), st)
    refused0 = _refused(fetcher)
    rep = apply_nucleic_types(payload, st.get("final_payload_paper_text"), fetcher=fetcher, cache=cache, today=today,
                              source=str(st.get("final_payload_paper_text_source") or "paper text"))
    rep["budget_exhausted"], rep["circuit_open"] = _refused(fetcher) > refused0, _circuit_open(fetcher)
    return rep


def prefreeze_with_final_fill(payload: Dict[str, Any], resolve: Any, pathway_context: Any = None, state: Any = None,
                              **kw: Any) -> tuple:
    """The postpipeline seam: (``resolve(payload)``, fill report). Flag off: exactly ``resolve(payload)`` and {}.
    Flag on: fill first, then the resolver; a ``PrefreezeResolutionError`` after a fill restores ``payload`` in place and
    re-runs the resolver on it, as the flag-off run would."""
    if not final_payload_enabled():
        return resolve(payload), {}
    snapshot = copy.deepcopy(payload)
    report = maybe_apply_final_id_fill(payload, pathway_context, state, **kw)
    if payload == snapshot:
        return resolve(payload), report
    from t2pw.pwml.prefreeze_resolution import PrefreezeResolutionError
    try:
        return resolve(payload), report
    except PrefreezeResolutionError as exc:
        payload.clear()
        payload.update(snapshot)
        report.update(written=0, restored_after_resolver_error=f"{type(exc).__name__}: {exc}")
        return resolve(payload), report


# ---- pathway title and reference (D-W12: the PWML carries the paper title and one reference) -------------------------

FINAL_PAYLOAD_PATHWAY_FIELD = "final_payload_pathway"
_PW_FIELDS = ("title", "pmid", "pmcid", "doi", "year", "authors")


def _fld(f: Any) -> Optional[Dict[str, Any]]:
    """{value, source, match_type, retrieved} of a clean-model Field, or None when absent/empty."""
    if getattr(f, "absent", True) or f.value in (None, "", []):
        return None
    p = f.prov
    return {"value": f.value, "source": p.source or p.label, "match_type": p.match_type or p.label, "retrieved": p.retrieved}


def _citation(b: Dict[str, Any]) -> str:
    au = (b.get("authors") or {}).get("value")
    au = ", ".join(str(a) for a in au) if isinstance(au, list) else (str(au) if au else "")
    parts = [(au + ": " if au else "") + str(b["title"]["value"]).rstrip(".") + "."]
    if b.get("year"):
        parts.append(f"{b['year']['value']}.")
    if b.get("doi"):
        parts.append(f"doi: {b['doi']['value']}.")
    if b.get("pmid"):
        parts.append(f"PMID: {b['pmid']['value']}.")
    return " ".join(parts)


def apply_final_pathway_meta(payload: Dict[str, Any], meta: Optional[Dict[str, Any]] = None, *, fetcher: Any = None,
                             cache: Any = None, today: Optional[str] = None,
                             pathway_name: Optional[str] = None) -> Dict[str, Any]:
    """Set ``payload["final_payload_pathway"]`` (title, pmid, pmcid, doi, year, authors, each with provenance, plus the one
    ``reference``). Paper mode (a pmid/pmcid/doi is known): the title comes from NCBI esummary through the fill layer, else
    ``meta["title"]``, else the pathway name; the reference only when NCBI confirmed the paper. Name mode: the pathway name. Nothing is invented; with no text the key is not
    written. Fails open: on any error the payload is exactly what it was."""
    report: Dict[str, Any] = {"enabled": True, "written": False, "title_source": None, "error": None, "budget_exhausted": False,
                              "circuit_open": False}
    snapshot = copy.deepcopy(payload)
    fetcher = fetcher or _live_fetcher(_title_budget())  # its own small budget, never the id fill's
    refused0 = _refused(fetcher)
    try:
        from t2pw.clean_model import build_from_payload
        from t2pw.clean_model.fill import fill_model

        today = today or datetime.date.today().isoformat()
        meta = dict(meta or {})
        if meta.pop("pmid_source", None):  # a guessed pmid (token position in the PMC header) is not an identifier
            meta.pop("pmid", None)
        meta.setdefault("source", "run input")
        meta_title = meta.pop("title", None)  # a fallback only: the NCBI title wins when the lookup answers
        model = build_from_payload(payload, meta)
        if any(not getattr(model.pathway, k).absent for k in ("pmid", "pmcid")):
            fill_model(model, fetcher=fetcher, cache=cache, today=today, sources=("ncbi",))
        b: Dict[str, Any] = {"schema": 1, "date": today}
        for k in _PW_FIELDS:
            v = _fld(getattr(model.pathway, k))
            if v:
                b[k] = v
        if "title" not in b:
            for txt, src in ((meta_title, "paper metadata"), (pathway_name, "pathway name")):
                if isinstance(txt, str) and txt.strip():
                    b["title"] = {"value": " ".join(txt.split()), "source": src, "match_type": "given", "retrieved": None}
                    break
        if "title" in b:
            if b["title"]["source"] == "NCBI" and any(k in b for k in ("pmid", "pmcid", "doi")):  # only a confirmed paper
                b["reference"] = {"pubmed_id": b["pmid"]["value"] if "pmid" in b else None, "text": _citation(b)}
            payload[FINAL_PAYLOAD_PATHWAY_FIELD] = b
            report.update(written=True, title_source=b["title"]["source"], has_reference="reference" in b)
    except Exception as exc:  # fail open
        payload.clear()
        payload.update(snapshot)
        report.update(written=False, error=f"{type(exc).__name__}: {exc}")
    report["budget_exhausted"], report["circuit_open"] = _refused(fetcher) > refused0, _circuit_open(fetcher)
    return report


def paper_meta(pathway_context: Any = None, state: Any = None) -> Dict[str, Any]:
    """pmid/pmcid/doi/title/year of the paper when the caller put them in ``state["final_payload_paper"]`` (or pathway_context)."""
    ctx = pathway_context if isinstance(pathway_context, dict) else {}
    st = state if hasattr(state, "get") else {}
    src = st.get("final_payload_paper") if isinstance(st.get("final_payload_paper"), dict) else {}
    return {k: str(v) for k in ("pmid", "pmcid", "doi", "title", "year") if (v := src.get(k) or ctx.get(k))}


def maybe_apply_final_pathway_meta(payload: Dict[str, Any], pathway_context: Any = None, state: Any = None,
                                   **kw: Any) -> Dict[str, Any]:
    """The postpipeline seam for the title/reference: {} when the flag is off; otherwise the report. Never raises."""
    if not final_payload_enabled():
        return {}
    try:
        ctx = pathway_context if isinstance(pathway_context, dict) else {}
        meta = {**fill_meta(pathway_context, state), **paper_meta(pathway_context, state)}
        if not kw.get("fetcher"):
            kw = {**kw, "fetcher": _cap_to_job_deadline(_live_fetcher(_title_budget()), state)}
        return apply_final_pathway_meta(payload, meta, pathway_name=ctx.get("pathway_name"), **kw)
    except Exception as exc:
        return {"enabled": True, "written": False, "error": f"{type(exc).__name__}: {exc}"}


def effective_pathway_name(payload: Any, name: str) -> str:
    """The title from ``final_payload_pathway`` in place of the default name; ``name`` unchanged when the key is absent."""
    b = payload.get(FINAL_PAYLOAD_PATHWAY_FIELD) if isinstance(payload, dict) else None
    t = (b.get("title") or {}).get("value") if isinstance(b, dict) and isinstance(b.get("title"), dict) else None
    return " ".join(t.split()) if name == "Generated Pathway" and isinstance(t, str) and t.strip() else name


def paper_meta_from_paper(paper: Any) -> Dict[str, str]:
    """pmid/pmcid/doi/title/year from a batch/service paper object or dict. ``paper_id`` is read as an id only when it looks
    like one; a web id ``web:PMID123+text1+x.pdf`` is split on ``+`` and each kind is taken only when exactly one was submitted."""
    def get(n: str) -> str:
        v = paper.get(n) if isinstance(paper, dict) else getattr(paper, n, None)
        return str(v).strip() if v not in (None, "") else ""
    out = {k: get(k) for k in ("pmid", "pmcid", "doi", "title", "year")}
    pid = get("paper_id")
    web = pid.startswith("web:")
    toks = [t for t in pid[4:].split("+") if t] if web else [pid]
    for k, rx in (("pmcid", r"PMC\d+"), ("pmid", r"PMID\d+" if web else r"(?:PMID)?\d+"), ("doi", r"10\.\S+")):
        hits = {re.sub(r"(?i)^pmid", "", t) if k == "pmid" else t for t in toks if re.fullmatch(rx, t, re.I)}
        if len(hits) == 1 and not out[k]:
            out[k] = hits.pop()
    return {k: v for k, v in out.items() if v}
