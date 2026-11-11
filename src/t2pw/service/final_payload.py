"""Final payload, phase 1 (PO ruling D-W12): a deterministic database id fill before the freeze.

Flag ``T2PW_FINAL_PAYLOAD`` (1/true/yes, default off). Reuses the clean model's online fill layer
(``clean_model.builder`` + ``clean_model.fill``) to find ids, then writes only the accepted ones into the payload rows, under
the keys the PWML writer already reads: compounds ``chebi_id kegg_id hmdb_id pubchem_cid cas drugbank_id`` and species
``taxonomy_id``. Proteins get nothing written: the fill layer only proposes UniProt candidates for them (P-20, D-106/D-108/
D-117), so those stay report suggestions. No LLM, no gene names, no reaction EC/Rhea inference.

Rules: an existing id is never touched; one exact label/synonym match only; an id already used by another row of the same
list is refused; complexes, families/classes and the Unknown sentinel are skipped. Any failure leaves ``payload`` exactly as it
was. Provenance goes under one new top-level key, ``final_payload_fill``.
"""

from __future__ import annotations

import copy
import datetime
import os
import re
import time
from typing import Any, Dict, List, Optional

FINAL_PAYLOAD_FILL_FIELD = "final_payload_fill"
BUDGET_SECONDS = 300.0  # wall clock for live requests; cached answers are free, so a re-run finishes the rest
HOST_INTERVAL = 1.0  # at most 1 request per second per host (the fill layer's own default is 3/s)

_COMPOUND_ALIASES = {"chebi_id": ("chebi",), "kegg_id": ("kegg",), "hmdb_id": ("hmdb",),
                     "pubchem_cid": ("pubchem", "pubchem_id"), "cas": ("cas",), "drugbank_id": ("drugbank",)}
_TARGETS = {"compound": ("compounds", _COMPOUND_ALIASES), "species": ("species", {"taxonomy_id": ()})}
_PLACEHOLDERS = {"", "n/a", "na", "none", "null", "nan", "unknown", "unmapped", "-"}
_NOT_A_MOLECULE = re.compile(r"\bcomplex of\b|\b(family|superfamily|class|classes)\b|^unknown$", re.I)


def final_payload_enabled() -> bool:
    return os.environ.get("T2PW_FINAL_PAYLOAD", "").strip().casefold() in ("1", "true", "yes")


def _empty(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip().casefold() in _PLACEHOLDERS)


def _norm(v: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v).casefold()).removeprefix("chebi").removeprefix("hmdb").lstrip("0")


class _Budget:
    """Wraps the live fetcher: once the wall-clock budget is spent every further request is a FetchError (counted, not fatal)."""

    def __init__(self, inner: Any, seconds: float) -> None:
        self.inner, self.deadline = inner, time.monotonic() + seconds

    def _check(self) -> None:
        if time.monotonic() > self.deadline:
            from t2pw.clean_model.fill.http import FetchError
            raise FetchError("final payload fill time budget exhausted")

    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        self._check()
        return self.inner.get_json(url, params)

    def get_text(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[str]:
        self._check()
        return self.inner.get_text(url, params)


def _live_fetcher() -> Any:
    from t2pw.clean_model.fill.http import Throttle, UrllibFetcher
    try:
        seconds = float(os.environ.get("T2PW_FINAL_PAYLOAD_BUDGET_S", BUDGET_SECONDS))
    except ValueError:
        seconds = BUDGET_SECONDS
    return _Budget(UrllibFetcher(throttle=Throttle(interval=HOST_INTERVAL)), seconds)


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

    for key, its in groups.items():
        kind, name, fld = key
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
                        cache: Any = None, today: Optional[str] = None) -> Dict[str, Any]:
    """Fill empty id slots of ``payload`` in place. Fails open: on any error ``payload`` is restored and the report says why."""
    report: Dict[str, Any] = {"enabled": True, "written": 0, "by_field": {}, "refused": {}, "refused_detail": [],
                              "suggestions": [], "error": None}
    snapshot = copy.deepcopy(payload)
    try:
        from t2pw.clean_model import build_from_payload
        from t2pw.clean_model.fill import fill_model

        today = today or datetime.date.today().isoformat()
        model = build_from_payload(payload, dict(meta or {}))
        rep = fill_model(model, fetcher=fetcher or _live_fetcher(), cache=cache, today=today, sources=("chebi", "uniprot", "ncbi"))
        report["fill"] = rep.to_dict()
        report["suggestions"] = copy.deepcopy(model.db_layer.get("suggestions") or [])
        writes = _plan(payload, model.db_layer.get("items") or [], report)
        for w in writes:
            w["row"][w["field"]] = w["value"]
            report["by_field"][w["entry"]["kind"] + "." + w["field"]] = report["by_field"].get(w["entry"]["kind"] + "." + w["field"], 0) + 1
        if writes:
            payload[FINAL_PAYLOAD_FILL_FIELD] = {"schema": 1, "date": today, "written": len(writes),
                                                 "by_field": dict(report["by_field"]), "entries": [w["entry"] for w in writes]}
        report["written"] = len(writes)
    except Exception as exc:  # fail open: the payload is exactly what it was
        payload.clear()
        payload.update(snapshot)
        report.update(written=0, by_field={}, error=f"{type(exc).__name__}: {exc}")
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


def maybe_apply_final_id_fill(payload: Dict[str, Any], pathway_context: Any = None, state: Any = None,
                              **kw: Any) -> Dict[str, Any]:
    """The postpipeline seam: {} (payload untouched) when the flag is off; otherwise the fill report. Never raises."""
    if not final_payload_enabled():
        return {}
    try:
        report = apply_final_id_fill(payload, fill_meta(pathway_context, state), **kw)
        report["meta_keys"] = sorted(fill_meta(pathway_context, state))
        return report
    except Exception as exc:
        return {"enabled": True, "written": 0, "error": f"{type(exc).__name__}: {exc}"}
