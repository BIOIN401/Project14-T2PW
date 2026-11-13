"""Final payload phase 4 (PO ruling D-W20a): ontology ids for location / cell-type / tissue names PathWhiz lacks.

Used by ``final_payload.apply_final_id_fill`` under T2PW_FINAL_PAYLOAD. A vocabulary name the reviewed map (``pwml.vocabulary_map``)
does not resolve to a PathWhiz row, and that carries no ontology id, is looked up in the EBI Ontology Lookup Service (OLS4):
GO cellular component for locations, CL for cell types, BTO for tissues (PathWhiz validates ``GO:``/``CL:``/``BTO:`` + 7 digits, so
UBERON ids would be rejected on import and are not used).

Rules: an EXACT label or EXACT synonym match only (verified here, not trusted from the index), match type recorded; the rank of
a fuzzy hit is never used; more than one distinct term matching exactly -> refused as ``ambiguous``; an existing id is never
overwritten. A name never seen in ``entities`` but present in ``biological_states`` (the IR invents a bare row for it later) gets
a row ``{"name", "ontology_id"}`` appended, so the id reaches the IR. Failures are counted, never raised.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from t2pw.clean_model.fill.cache import Cache, CachedFetcher
from t2pw.clean_model.fill.http import FetchError
from t2pw.clean_model.fill.sources import OLS4, _cf
from t2pw.pwml.vocabulary_map import map_vocabulary_name

# payload key, PWML kind (vocabulary_map), biological_states key(s), OLS ontology, id prefix, restrict-to-branch IRI
_SPECS = (
    ("subcellular_locations", "subcellular-location", ("subcellular_location", "subcellular-location", "compartment",
                                                         "compartment_canonical"), "go", "GO:",
     "http://purl.obolibrary.org/obo/GO_0005575"),  # cellular_component: GO also holds processes and functions
    ("cell_types", "cell-type", ("cell_type", "cell-type"), "cl", "CL:", None),
    ("tissues", "tissue", ("tissue",), "bto", "BTO:", None),
)
_KIND_LABEL = {"subcellular-location": "location", "cell-type": "cell_type", "tissue": "tissue"}


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
        elif q in [_cf(s) for s in r.get("synonym") or []]:
            mt = "exact_synonym"
        else:
            continue
        if oid not in out or mt == "exact_label":
            out[oid] = {"id": oid, "label": label, "match_type": mt}
    return list(out.values())


def fill_ontology_ids(payload: Dict[str, Any], fetcher: Any, cache: Optional[Cache], today: str) -> Dict[str, Any]:
    """Write ``ontology_id`` onto vocabulary rows in place. Returns {"entries": [...provenance...], "counts": {...}, "refused": [...]}."""
    cf = CachedFetcher(fetcher, cache if cache is not None else Cache(), today)
    ents = payload.get("entities") if isinstance(payload.get("entities"), dict) else None
    rep: Dict[str, Any] = {"entries": [], "refused": [], "counts": {"looked_up": 0, "resolved": 0, "ambiguous": 0,
                                                                      "no_exact_match": 0, "fetch_error": 0}}
    if ents is None:
        return rep
    states = [s for s in payload.get("biological_states") or [] if isinstance(s, dict)]
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
        for row in rows:
            if not isinstance(row, dict) or not str(row.get("name") or "").strip():
                continue
            name, cur = str(row["name"]).strip(), row.get("ontology_id") or row.get("ontology-id")
            if cur not in (None, "") or map_vocabulary_name(kind, name, None)[2] in ("exact", "mapped"):
                continue
            rep["counts"]["looked_up"] += 1
            try:
                hits = ols_exact(cf, name, onto, prefix, branch)
            except FetchError as exc:
                rep["counts"]["fetch_error"] += 1
                rep["refused"].append({"kind": _KIND_LABEL[kind], "name": name, "reason": f"fetch_error: {exc}"})
                continue
            if len(hits) > 1:
                rep["counts"]["ambiguous"] += 1
                rep["refused"].append({"kind": _KIND_LABEL[kind], "name": name, "reason": "ambiguous",
                                       "candidates": [h["id"] for h in hits]})
            elif not hits:
                rep["counts"]["no_exact_match"] += 1
                rep["refused"].append({"kind": _KIND_LABEL[kind], "name": name, "reason": "no_exact_match"})
            else:
                h = hits[0]
                row["ontology_id"] = h["id"]
                row.pop("_from_state", None)
                rep["counts"]["resolved"] += 1
                rep["entries"].append({"kind": _KIND_LABEL[kind], "entity": name, "field": "ontology_id", "accession": h["id"],
                                       "source": "OLS4", "label": h["label"], "match_type": h["match_type"],
                                       "retrieved": cf.dates[-1] if cf.dates else today})
        rows[:] = [r for r in rows if not (isinstance(r, dict) and r.get("_from_state") and not r.get("ontology_id"))]
        if rows and key not in ents:
            ents[key] = rows
    rep["requests"], rep["cache_hits"] = cf.requests, cf.hits
    return rep
