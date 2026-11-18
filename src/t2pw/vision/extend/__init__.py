"""Database extension stage (VISION section 2 step 3): curated pathways around the paper core.

``extend_big_payload(big)`` finds curated pathways (PathBank, Reactome, WikiPathways) for the
payload's species whose compounds / proteins overlap the paper's grounded ids, converts their steps
to overlay rows with ``database`` provenance and merges them with ``payload.merge_extension``
(a step already in the paper becomes an extra provenance record on the base row). Every source
fails open: an error means no extension from it, recorded in the report. Off by default
(``T2PW_EXTEND``); nothing in the pipeline calls this yet.
"""

from __future__ import annotations

import os
import time
from collections import Counter
from typing import Any, Callable, Dict, Optional, Tuple

from t2pw.vision.payload import (_Index, currency_family, grounded_ids, is_currency, keyed_rows,
                                 merge_extension)

from .convert import to_rows

FLAG = "T2PW_EXTEND"
SOURCES = ("pathbank", "reactome", "wikipathways")


def extend_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    value = (os.environ if env is None else env).get(FLAG, "")
    return str(value).strip().lower() not in ("", "0", "false", "no", "off")


def _default_fetcher(name: str) -> Callable:
    if name == "pathbank":
        from .pathbank import fetch
    elif name == "reactome":
        from .reactome import fetch
    elif name == "wikipathways":
        from .wikipathways import fetch
    else:
        raise KeyError(name)
    return fetch


def paper_context(big: Dict[str, Any]) -> Dict[str, Any]:
    """Species (name, taxonomy id) and the grounded ids of the paper's non-currency entities."""
    species, ids = [], {}
    for section, _, row, is_base in keyed_rows(big):
        if not is_base or not isinstance(row, dict):
            continue
        if section == "entities.species":
            species.append({"name": row.get("name"), "taxonomy_id": str(row.get("taxonomy_id") or "") or None})
        elif section in ("entities.compounds", "entities.proteins", "entities.protein_complexes"):
            if is_currency(row.get("name")):
                continue
            found = set(grounded_ids(row))
            for c in row.get("components") or []:
                if isinstance(c, dict):
                    found |= grounded_ids(c)
            for ns, v in found:
                ids.setdefault(ns, set()).add(v)
    return {"species": species, "ids": ids}


def _safe_error(exc: BaseException) -> str:
    # only our own and the public REST clients' messages are kept: a DB driver error can carry a
    # host or user name
    from .http import SourceError
    own = isinstance(exc, SourceError) or type(exc).__module__.split(".")[0] in ("requests", "urllib3")
    return f"{type(exc).__name__}: {exc}"[:300] if own else type(exc).__name__


def extend_big_payload(big: Dict[str, Any], *, sources=SOURCES, budget_s: float = 120,
                       max_pathways: int = 3, cache_dir=None,
                       fetchers: Optional[Dict[str, Callable]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(extended copy of ``big``, report). ``fetchers`` maps a source name to
    ``fetch(ctx, *, max_pathways, deadline, cache_dir) -> (pathways, notes)`` (tests inject fakes)."""
    deadline = time.monotonic() + budget_s
    ctx = paper_context(big)
    report: Dict[str, Any] = {"budget_s": budget_s, "max_pathways": max_pathways, "sources": {},
                              "species": ctx["species"], "failed": []}
    for name in sources:
        entry: Dict[str, Any] = {"status": "ok", "pathways": [], "notes": [], "counts": Counter()}
        report["sources"][name] = entry
        if time.monotonic() >= deadline:
            entry["status"] = "skipped_budget"
            continue
        try:
            fetch = (fetchers or {}).get(name) or _default_fetcher(name)
            pathways, notes = fetch(ctx, max_pathways=max_pathways, deadline=deadline,
                                    cache_dir=(os.path.join(str(cache_dir), name) if cache_dir else None))
        except Exception as exc:  # noqa: BLE001 - fail open by design
            entry.update(status="failed", error=_safe_error(exc))
            report["failed"].append(name)
            continue
        entry["notes"] = list(notes or [])
        for pw in pathways:
            try:   # fail open per pathway: one bad record never stops the source or the run
                rows, provs, src, counts = to_rows(big, pw)
                big = merge_extension(big, rows, provs, src)
            except Exception as exc:  # noqa: BLE001
                entry["notes"].append(f"{pw.get('source_id')}: conversion failed, {type(exc).__name__}")
                entry["counts"]["pathways_failed"] += 1
                continue
            entry["counts"].update(counts)
            entry["pathways"].append({"source_id": pw["source_id"], "accession": pw.get("accession"),
                                      "name": pw.get("name"), "overlap": pw.get("overlap"),
                                      "steps": len(pw.get("steps") or [])})
        if not pathways:
            entry["status"] = "no_candidates"
        elif not entry["pathways"]:
            entry["status"] = "failed"
            report["failed"].append(name)
    for entry in report["sources"].values():
        entry["counts"] = dict(entry["counts"])
    report["placement"] = placement(big)
    hops = big.get("hops") or {}
    report["overlay_entities_hop_none"] = sum(   # in FULL only: nothing grounded ties them to the core
        1 for s, k, _, is_base in keyed_rows(big) if not is_base and s.startswith("entities.") and hops.get(k) is None)
    report["currency_only_twins"] = currency_only_twins(big)
    return big, report


def run_if_enabled(big: Dict[str, Any], **kw) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """The entry point for the pipeline wiring lane: ``(big, None)`` untouched when T2PW_EXTEND is off."""
    return extend_big_payload(big, **kw) if extend_enabled() else (big, None)


def placement(big: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """Per source id: process steps merged into base rows, and extension steps by hop."""
    out: Dict[str, Counter] = {}
    hops = big.get("hops") or {}
    for section, key, _, is_base in keyed_rows(big):
        if not section.startswith("processes."):
            continue
        for sid in {r.get("source") for r in big["provenance"].get(key, [])
                    if isinstance(r, dict) and r.get("kind") == "database" and r.get("stage") == "extension_db"}:
            h = hops.get(key)
            bucket = "merged_into_base" if is_base else (
                "hop_none" if h is None else "hop_1" if h <= 1 else "hop_2_plus")
            out.setdefault(sid, Counter())[bucket] += 1
    return {k: dict(v) for k, v in out.items()}


def currency_only_twins(big: Dict[str, Any]) -> Dict[str, Any]:
    """Extension reactions that stayed separate from another reaction with the same core sets
    (either orientation) because their currency differs ONLY by H+ / H2O (VD-11 note a)."""
    index = _Index(big)
    small = {currency_family("H+"), currency_family("H2O")}
    rx = [(k, r, is_base) for s, k, r, is_base in keyed_rows(big) if s == "processes.reactions" and isinstance(r, dict)]
    sets = {k: index._rx_sets(r) for k, r, _ in rx}
    pairs = []
    for k, r, is_base in rx:
        if is_base:
            continue
        ins, outs, ci, co = sets[k]
        for k2, _, _ in rx:
            if k2 == k:
                continue
            i2, o2, ci2, co2 = sets[k2]
            if (ins, outs) == (i2, o2):
                diff = (ci ^ ci2) | (co ^ co2)
            elif (ins, outs) == (o2, i2):
                diff = (ci ^ co2) | (co ^ ci2)
            else:
                continue
            if diff and diff <= small:
                pairs.append([k, k2, sorted(diff)])
    return {"count": len(pairs), "pairs": pairs[:50]}
