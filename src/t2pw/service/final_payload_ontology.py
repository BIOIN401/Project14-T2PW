"""Final payload phase 4 (PO ruling D-W20a): ontology ids for location / cell-type / tissue names PathWhiz lacks.

Used by ``final_payload.apply_final_id_fill`` under T2PW_FINAL_PAYLOAD. A vocabulary name the reviewed map (``pwml.vocabulary_map``)
does not resolve to a PathWhiz row, and that carries no ontology id, is looked up in the EBI Ontology Lookup Service (OLS4):
GO cellular component for locations, CL for cell types, BTO for tissues (PathWhiz validates ``GO:``/``CL:``/``BTO:`` + 7 digits, so
UBERON ids would be rejected on import and are not used).

Rules: an EXACT label or EXACT synonym match only (verified here, not trusted from the index), match type recorded; the rank of
a fuzzy hit is never used. The OLS4 search ``synonym`` field mixes exact, related, broad and narrow synonyms (``nerve fiber`` is a
RELATED synonym of GO:0043005 "neuron projection"), so a synonym hit counts only after the term itself is fetched (OLS4 v2
``classes`` endpoint) and the name is in its ``oboInOwl#hasExactSynonym`` list; a failed fetch refuses the name. After the first
fetch error of a run the remaining lookups are skipped and counted (``skipped_after_fetch_error``).
More than one distinct term matching exactly -> refused as ``ambiguous``; an existing id is never
overwritten. A name never seen in ``entities`` but present in ``biological_states`` (the IR invents a bare row for it later) gets
a row ``{"name", "ontology_id"}`` appended, so the id reaches the IR. Failures are counted, never raised.

Phase 5 (PO ruling D-W21 b/e), all still exact-only:
* singular form: a name with no exact hit is tried once more with the plural -s cut from its last word (``synaptic vesicles``); the
  form that matched is recorded as ``matched_form``.
* numbered cell types (``Type II cell``): CL has no term with that exact name. The paper text (``paper_text``) and the payload's own
  cell-type names are scanned for full phrases ``type <numeral> <1-3 words> cell``; the name resolves only when those phrases, each an
  exact CL label / exact synonym (or a reviewed vocabulary synonym), name exactly ONE CL term. The phrase and a quote are recorded.
* cell lines (MG-63, Saos-2) are not CL terms. After CL fails, Cellosaurus is asked for an exact identifier or exact synonym, in a
  species the payload also carries, with exactly one such cell line. The accession goes on the row as ``cellosaurus_id`` (NOT
  ``ontology_id``: PathWhiz ``cell_types.ontology_id`` must match CL:\\d{7}, so the PWML writer cannot use it) with ``db`` provenance.
* locations: whether a name is already resolved by the vocabulary map now depends on the organism of the states that use it (inner /
  outer membrane are bacterial PathWhiz rows, see ``pwml.vocabulary_map``).
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Any, Dict, Iterable, List, Optional, Tuple

from t2pw.clean_model.fill.cache import Cache, CachedFetcher
from t2pw.clean_model.fill.http import FetchError
from t2pw.clean_model.fill.sources import OLS4, _cf
from t2pw.pwml.vocabulary_map import PATHWHIZ_ROWS, SYNONYMS, map_vocabulary_name, membrane_row, organism_kinds, row_id, singular_form

# payload key, PWML kind (vocabulary_map), biological_states key(s), OLS ontology, id prefix, restrict-to-branch IRI
_SPECS = (
    ("subcellular_locations", "subcellular-location", ("subcellular_location", "subcellular-location", "compartment",
                                                         "compartment_canonical"), "go", "GO:",
     "http://purl.obolibrary.org/obo/GO_0005575"),  # cellular_component: GO also holds processes and functions
    ("cell_types", "cell-type", ("cell_type", "cell-type"), "cl", "CL:", None),
    ("tissues", "tissue", ("tissue",), "bto", "BTO:", None),
)
OLS4_TERM = "https://www.ebi.ac.uk/ols4/api/v2/ontologies/{onto}/classes/{iri}"  # iri: double URL-encoded
EXACT_SYNONYM_KEY = "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym"
_KIND_LABEL = {"subcellular-location": "location", "cell-type": "cell_type", "tissue": "tissue"}


def exact_synonyms(f: Any, onto: str, obo_id: str) -> List[str]:
    """The exact synonyms of one term, from the OLS4 v2 entity. A FetchError propagates; an unknown term gives []."""
    iri = urllib.parse.quote(urllib.parse.quote("http://purl.obolibrary.org/obo/" + obo_id.replace(":", "_"), safe=""), safe="")
    d = f.get_json(OLS4_TERM.format(onto=onto, iri=iri))
    vals = (d or {}).get(EXACT_SYNONYM_KEY) or []
    return [str(v.get("value") if isinstance(v, dict) else v) for v in (vals if isinstance(vals, list) else [vals])]


def ols_exact(f: Any, name: str, ontology: str, prefix: str, branch: Optional[str]) -> List[Dict[str, str]]:
    """Distinct terms whose label or synonym equals ``name`` (case/space-folded): [{id, label, match_type}]."""
    q = _cf(name)
    params = {"q": name, "ontology": ontology, "exact": "true", "queryFields": "label,synonym",
              "fieldList": "obo_id,label,synonym", "rows": 50}
    if branch:
        params["allChildrenOf"] = branch
    d = f.get_json(OLS4, params)
    out: Dict[str, Dict[str, str]] = {}
    for r in ((d or {}).get("response") or {}).get("docs") or []:
        oid, label = r.get("obo_id") or "", r.get("label") or ""
        if not oid.startswith(prefix):
            continue
        if _cf(label) == q:
            mt = "exact_label"
        elif q in [_cf(s) for s in r.get("synonym") or []] and q in [_cf(s) for s in exact_synonyms(f, ontology, oid)]:
            mt = "exact_synonym"
        else:
            continue
        if oid not in out or mt == "exact_label":
            out[oid] = {"id": oid, "label": label, "match_type": mt}
    return list(out.values())


# ---- P5 (B): cell lines through Cellosaurus -------------------------------------------------------------------------------
CELLOSAURUS = "https://api.cellosaurus.org/search/cell-line"


class CellosaurusPageFull(Exception):
    """The search returned a full page, so an exact match cannot be excluded: the name is refused, the service is not failing."""


def cellosaurus_exact(f: Any, name: str) -> List[Dict[str, Any]]:
    """Distinct cell lines whose identifier or synonym equals ``name`` (case/space-folded): [{id, label, match_type, taxa}] with ``taxa``
    the NCBI taxon ids and ``species`` the labels of the line. The search is Cellosaurus ``idsy`` (it also returns derivatives such as
    ``MG-63/CISR8``; those are dropped here by the equality test). A full page (100) cannot prove there is no further match: ``CellosaurusPageFull``."""
    q = _cf(name)
    if not q or '"' in name:
        return []
    d = f.get_json(CELLOSAURUS, {"q": f'idsy:"{name}"', "format": "json", "rows": 100, "fields": "id,ac,sy,ox"})
    lines = ((d or {}).get("Cellosaurus") or {}).get("cell-line-list") or []
    if len(lines) >= 100:
        raise CellosaurusPageFull(name)
    out: Dict[str, Dict[str, Any]] = {}
    for c in lines:
        names = [n for n in c.get("name-list") or [] if isinstance(n, dict)]
        hit = [n for n in names if _cf(n.get("value")) == q]
        accs = [a.get("value") for a in c.get("accession-list") or [] if isinstance(a, dict) and a.get("value")]
        if not hit or not accs:
            continue
        label = next((n["value"] for n in names if n.get("type") == "identifier"), hit[0]["value"])
        sp = [s for s in c.get("species-list") or [] if isinstance(s, dict)]
        out[accs[0]] = {"id": accs[0], "label": label,
                        "match_type": "exact_label" if any(n.get("type") == "identifier" for n in hit) else "exact_synonym",
                        "taxa": [str(s.get("accession")) for s in sp if s.get("accession")],
                        "species": [str(s.get("label")) for s in sp if s.get("label")]}
    return list(out.values())


def _payload_species(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    ents = payload.get("entities") if isinstance(payload.get("entities"), dict) else {}
    out = []
    for r in ents.get("species") or []:
        if isinstance(r, dict) and str(r.get("name") or "").strip():
            out.append({"name": str(r["name"]).strip(), "classification": str(r.get("classification") or ""),
                        "taxid": str(r.get("taxonomy_id") or r.get("taxonomy-id") or "").strip()})
    return out


def _species_match(line: Dict[str, Any], species: List[Dict[str, str]]) -> bool:
    """The cell line's species is one the payload carries: same NCBI taxon id, else same scientific name (``Homo sapiens (Human)``)."""
    taxa, sci = set(line["taxa"]), {_cf(s.split(" (")[0]) for s in line["species"]}
    return any((sp["taxid"] and sp["taxid"] in taxa) or _cf(sp["name"]) in sci for sp in species)


def cellosaurus_for(f: Any, name: str, species: List[Dict[str, str]]) -> Tuple[Optional[Dict[str, Any]], str]:
    """(the one cell line, "found") or (None, reason): no_exact_match / no_species_context / species_mismatch / ambiguous."""
    try:
        lines = cellosaurus_exact(f, name)
    except CellosaurusPageFull:
        return None, "result_page_full"
    if not lines:
        return None, "no_exact_match"
    if not species:
        return None, "no_species_context"
    ok = [x for x in lines if _species_match(x, species)]
    return (ok[0], "found") if len(ok) == 1 else (None, "species_mismatch" if not ok else "ambiguous")


# ---- P5 (B): numbered cell types resolved by the paper's own full phrase ----------------------------------------------------
_NUMERAL = r"(?:i{1,3}|iv|v|vi{1,3}|ix|x|\d{1,2})"
_NUMBERED = re.compile(rf"^type ({_NUMERAL}) cell$", re.I)
_FILLER = {"and", "or", "the", "of", "in", "are", "is", "to", "with", "from", "by", "that", "which", "as", "for", "an", "these", "those",
           "both", "all", "other", "between", "but", "not", "only", "type", "cell", "cells", "also", "than", "were", "was", "have", "has"}
MAX_PHRASES = 6  # more distinct phrases than this: refused (too_many_phrases), never cut to the most frequent ones


def _numbered_name(name: Any) -> bool:
    """True for a ``Type <numeral> cell`` name (singular or plural)."""
    text = str(name or "")
    return _NUMBERED.match(re.sub(r"\s+", " ", singular_form(text) or text).strip()) is not None


def numbered_phrases(name: str, texts: Iterable[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """Full phrases ``type <numeral> <1-3 words> cell(s)`` for a numbered cell type name, from (source, text) pairs, most frequent first:
    [{phrase, source, count, quote}]. Empty when ``name`` (singular) is not ``Type <numeral> cell``."""
    m = _NUMBERED.match(re.sub(r"\s+", " ", singular_form(name) or name).strip())
    if not m:
        return []
    rx = re.compile(rf"\btype\s+{re.escape(m.group(1))}\s+((?:[A-Za-z][A-Za-z-]*\s+){{1,3}})cells?\b", re.I)
    found: Dict[str, Dict[str, Any]] = {}
    for source, text in texts:
        for mm in rx.finditer(text or ""):
            words = mm.group(1).split()
            if any(w.casefold() in _FILLER for w in words):
                continue
            phrase = f"type {m.group(1)} {' '.join(words)} cell".casefold()
            e = found.setdefault(phrase, {"phrase": phrase, "source": source, "count": 0,
                                          "quote": " ".join((text[max(0, mm.start() - 60):mm.end() + 60]).split())})
            e["count"] += 1
    return sorted(found.values(), key=lambda e: (-e["count"], e["phrase"]))


def numbered_context(f: Any, name: str, texts: Iterable[Tuple[str, str]]) -> Tuple[Optional[Dict[str, Any]], str]:
    """(hit, "found") when the phrases found for a numbered name give exactly one CL term (exact label / exact synonym through OLS4, or a
    reviewed vocabulary synonym); (None, reason) otherwise: no_context_phrase / no_exact_phrase / ambiguous_context."""
    phrases = numbered_phrases(name, texts)
    if not phrases:
        return None, "no_context_phrase"
    if len(phrases) > MAX_PHRASES:  # a cut list could hide a second CL term, so "exactly one" would pass falsely
        return None, "too_many_phrases"
    ids: Dict[str, Dict[str, Any]] = {}
    for ph in phrases:
        hits = ols_exact(f, ph["phrase"], "cl", "CL:", None)
        if not hits and (tgt := SYNONYMS["cell-type"].get(ph["phrase"])):
            hits = [{"id": PATHWHIZ_ROWS["cell-type"][tgt], "label": tgt, "match_type": "reviewed_synonym"}]
        for h in hits:
            ids.setdefault(h["id"], dict(h, context=ph))
    if len(ids) > 1:
        return None, "ambiguous_context"
    return (next(iter(ids.values())), "found") if ids else (None, "no_exact_phrase")


# ---- P5 (E): organisms of the states that use a location ---------------------------------------------------------------------
def location_organisms(payload: Dict[str, Any], name: str, state_keys: Iterable[str]) -> frozenset:
    """Organism kinds of the biological states whose location is ``name`` (a state without a species counts every species row of the
    payload; no state at all likewise)."""
    rows = _payload_species(payload)
    by_name = {_cf(r["name"]): r for r in rows}
    pairs: List[Tuple[Any, Any]] = []
    for s in payload.get("biological_states") or []:
        if not isinstance(s, dict):
            continue
        n = next((str(s[k]).strip() for k in state_keys if s.get(k) not in (None, "")), "")
        if _cf(n) != _cf(name):
            continue
        sp = str(s.get("species") or "").strip()
        pairs += [(sp, (by_name.get(_cf(sp)) or {}).get("classification"))] if sp else [(r["name"], r["classification"]) for r in rows]
    return organism_kinds(pairs or [(r["name"], r["classification"]) for r in rows])


def fill_ontology_ids(payload: Dict[str, Any], fetcher: Any, cache: Optional[Cache], today: str, *,
                      paper_text: Optional[str] = None, paper_source: str = "paper text") -> Dict[str, Any]:
    """Write ``ontology_id`` onto vocabulary rows in place (cell lines: ``cellosaurus_id``). Returns {"entries": [...provenance...],
    "counts": {...}, "refused": [...], "extra": {...P5 counters...}}. ``paper_text`` is the paper (numbered cell types only)."""
    cf = CachedFetcher(fetcher, cache if cache is not None else Cache(), today)
    ents = payload.get("entities") if isinstance(payload.get("entities"), dict) else None
    rep: Dict[str, Any] = {"entries": [], "refused": [], "counts": {"looked_up": 0, "resolved": 0, "ambiguous": 0,
                                                                      "no_exact_match": 0, "fetch_error": 0,
                                                                      "skipped_after_fetch_error": 0},
                           "extra": {"singular_form_resolved": 0, "numbered_context_resolved": 0, "numbered_context_refused": {}, "id_used_by_other_row": 0,
                                     "cellosaurus_found": 0, "cellosaurus_refused": {}, "membrane_rows_by_organism": []}}
    extra = rep["extra"]
    if ents is None:
        return rep
    states = [s for s in payload.get("biological_states") or [] if isinstance(s, dict)]
    species = _payload_species(payload)
    ols_failed = False
    for key, kind, state_keys, onto, prefix, branch in _SPECS:
        rows = ents.get(key)
        if not isinstance(rows, list):
            rows = []
        have = {_cf(r.get("name")) for r in rows if isinstance(r, dict)}
        for s in states:  # names only the biological states carry; the IR would invent a bare row for them
            n = next((str(s[k]).strip() for k in state_keys if s.get(k) not in (None, "")), "")
            if n and _cf(n) not in have:
                have.add(_cf(n))
                rows.append({"name": n, "_from_state": True})
        # ids already held by a row of this list, or reached by a row's reviewed name mapping: a second row must not resolve to the same
        # ontology id / PathWhiz row (PathWhiz ontology ids are unique, and two rows for one id are two names for one entity)
        claimed: Dict[str, str] = {}
        for r in rows:
            if isinstance(r, dict) and str(r.get("name") or "").strip():
                n0 = str(r["name"]).strip()
                m0 = map_vocabulary_name(kind, n0, None, location_organisms(payload, n0, state_keys) if kind == "subcellular-location" else None)
                held = r.get("ontology_id") or r.get("ontology-id") or (row_id(kind, m0[0]) if m0[2] in ("exact", "mapped") else None)
                if held:
                    claimed.setdefault(str(held), n0)
        # plural-looking names later, so the exact (singular) name of the same entity claims the id first; numbered cell type names
        # ("Type II cell", resolved from the paper's context) last of all, so a row with its own exact CL label ("Type II taste cell")
        # claims that id before a context guess does
        for row in sorted((r for r in rows if isinstance(r, dict)),
                          key=lambda r: (kind == "cell-type" and _numbered_name(r.get("name")),
                                         singular_form(r.get("name")) is not None)):
            if not str(row.get("name") or "").strip():
                continue
            name, cur = str(row["name"]).strip(), row.get("ontology_id") or row.get("ontology-id")
            orgs = location_organisms(payload, name, state_keys) if kind == "subcellular-location" else None
            mapped = map_vocabulary_name(kind, name, None, orgs)
            if orgs is not None and membrane_row(name):
                extra["membrane_rows_by_organism"].append({"name": name, "organisms": sorted(orgs), "outcome": mapped[2], "row": mapped[0]})
            if cur not in (None, "") or mapped[2] in ("exact", "mapped"):
                continue
            if ols_failed:  # OLS is failing: stop asking, count what was not asked (a Cellosaurus failure does not stop OLS lookups)
                rep["counts"]["skipped_after_fetch_error"] += 1
                rep["refused"].append({"kind": _KIND_LABEL[kind], "name": name, "reason": "skipped_after_fetch_error"})
                continue
            rep["counts"]["looked_up"] += 1
            form, via, cello_note = name, None, None
            try:
                hits = ols_exact(cf, name, onto, prefix, branch)
                if not hits and (sing := singular_form(name)) is not None:  # B: the singular, as a second exact lookup
                    form, hits = sing, ols_exact(cf, sing, onto, prefix, branch)
                if kind == "cell-type" and len(hits) != 1:  # B: "Type II cell": the paper's own full phrase, exact, one CL term
                    own = [str(r.get("name")) for r in ents.get("cell_types") or [] if isinstance(r, dict) and r.get("name") and r is not row]
                    own += [str(st[k]) for st in states for k in ("cell_type", "cell-type") if st.get(k)]
                    texts = ([(paper_source, paper_text)] if paper_text else []) + [("payload cell types", " . ".join(own))]
                    ctx, why = numbered_context(cf, name, texts)
                    if ctx is not None and (not hits or ctx["id"] in {h["id"] for h in hits}):
                        hits, via = [ctx], ctx["context"]
                    elif why != "no_context_phrase":
                        extra["numbered_context_refused"][why] = extra["numbered_context_refused"].get(why, 0) + 1
                if kind == "cell-type" and not hits and not row.get("cellosaurus_id"):  # B: a cell line is not a CL term
                    try:
                        line, cello_note = cellosaurus_for(cf, name, species)
                    except FetchError:  # Cellosaurus failing (or refusing an odd name) must not stop the OLS lookups
                        line, cello_note = None, "fetch_error"
                        rep["counts"]["fetch_error"] += 1  # still a fetch error in the totals; ``cellosaurus_refused`` counts it apart
                    if line is None:
                        extra["cellosaurus_refused"][cello_note] = extra["cellosaurus_refused"].get(cello_note, 0) + 1
                    else:
                        row["cellosaurus_id"] = line["id"]
                        row.pop("_from_state", None)
                        extra["cellosaurus_found"] += 1
                        rep["entries"].append({"kind": _KIND_LABEL[kind], "entity": name, "field": "cellosaurus_id", "accession": line["id"],
                                               "source": "Cellosaurus", "label": line["label"], "match_type": line["match_type"],
                                               "species": line["species"], "pathwhiz_can_use": False,
                                               "retrieved": cf.dates[-1] if cf.dates else today})
            except FetchError as exc:
                ols_failed = True
                rep["counts"]["fetch_error"] += 1
                rep["refused"].append({"kind": _KIND_LABEL[kind], "name": name, "reason": f"fetch_error: {exc}"})
                continue
            if len(hits) > 1:
                rep["counts"]["ambiguous"] += 1
                rep["refused"].append({"kind": _KIND_LABEL[kind], "name": name, "reason": "ambiguous",
                                       "candidates": [h["id"] for h in hits]})
            elif hits and hits[0]["id"] in claimed and _cf(claimed[hits[0]["id"]]) != _cf(name):
                extra["id_used_by_other_row"] += 1
                rep["refused"].append({"kind": _KIND_LABEL[kind], "name": name, "reason": "id_used_by_other_row",
                                       "candidates": [hits[0]["id"]], "other_row": claimed[hits[0]["id"]]})
            elif not hits:
                rep["counts"]["no_exact_match"] += 1
                rep["refused"].append({"kind": _KIND_LABEL[kind], "name": name, "reason": "no_exact_match",
                                       **({"cellosaurus": row["cellosaurus_id"]} if row.get("cellosaurus_id") and cello_note == "found"
                                          else {"cellosaurus": cello_note} if cello_note else {})})
            else:
                h = hits[0]
                row["ontology_id"] = h["id"]
                row.pop("_from_state", None)
                claimed[h["id"]] = name
                rep["counts"]["resolved"] += 1
                e = {"kind": _KIND_LABEL[kind], "entity": name, "field": "ontology_id", "accession": h["id"],
                     "source": "OLS4", "label": h["label"], "match_type": h["match_type"],
                     "retrieved": cf.dates[-1] if cf.dates else today}
                if via is not None:
                    e.update(via="paper_context", context_phrase=via["phrase"], context_source=via["source"], context_count=via["count"],
                             context_quote=via["quote"])
                    extra["numbered_context_resolved"] += 1
                elif form != name:
                    e["matched_form"] = form
                    extra["singular_form_resolved"] += 1
                rep["entries"].append(e)
        rows[:] = [r for r in rows if not (isinstance(r, dict) and r.get("_from_state") and not r.get("ontology_id")
                                           and not r.get("cellosaurus_id"))]
        if rows and key not in ents:
            ents[key] = rows
    rep["requests"], rep["cache_hits"] = cf.requests, cf.hits
    return rep
