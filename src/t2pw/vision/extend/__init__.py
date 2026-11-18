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

from t2pw.vision.payload import (STATES, _Index, currency_family, grounded_ids, is_currency, keyed_rows,
                                 merge_extension, norm, refs)

from .convert import to_rows

FLAG = "T2PW_EXTEND"
SOURCES = ("pathbank", "reactome", "wikipathways")
LITERATURE = "literature"   # other papers: literature.extend_literature, its own entry (other_paper provenance)
# Budget share per source. Each source gets its share of what is left, so time a source does not
# use passes on to the ones after it (one PathBank extract takes ~35-40 s and used to starve the rest).
SHARES = {"pathbank": 2.0, "reactome": 1.0, "wikipathways": 1.0}


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


def used_species(big: Dict[str, Any]) -> Dict[str, str]:
    """{normalized name: name as written} of the species the states of the paper's processes use,
    else (no state names one) the species of its proteins / complexes. A species row nothing uses
    (an extraction artefact, e.g. a stray 'Arabidopsis thaliana') never chooses the organism."""
    base = big.get("base") or {}
    state_species = {norm(r.get("name")): r.get("species") for r in base.get(STATES) or [] if isinstance(r, dict)}
    by_states: Dict[str, str] = {}
    by_proteins: Dict[str, str] = {}
    for section, _, row, is_base in keyed_rows(big):
        if not is_base or not isinstance(row, dict):
            continue
        if section in ("entities.proteins", "entities.protein_complexes"):
            names, into = [row.get(f) for f in ("species", "organism")], by_proteins
        elif section.startswith("processes."):
            names, into = [state_species.get(norm(st)) for st in refs(section, row)[1]], by_states
        else:
            continue
        for n in names:
            if isinstance(n, str) and norm(n):
                into.setdefault(norm(n), n.strip())
    # the states the processes run in name the organism; protein species only when no state does
    # (live: menaquinone's MenA/MenI wrappers carry a borrowed 'Arabidopsis thaliana' identity)
    return by_states or by_proteins


def paper_context(big: Dict[str, Any]) -> Dict[str, Any]:
    """Species in use (name, taxonomy id) and the grounded ids of the paper's non-currency entities."""
    species, ids = [], {}
    used = used_species(big)
    for section, _, row, is_base in keyed_rows(big):
        if not is_base or not isinstance(row, dict):
            continue
        if section == "entities.species":
            if norm(row.get("name")) in used:
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
    listed = {norm(s["name"]) for s in species}
    species += [{"name": used[n], "taxonomy_id": None} for n in sorted(set(used) - listed)]   # used, no row
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
    sources = list(sources)
    for i, name in enumerate(sources):
        entry: Dict[str, Any] = {"status": "ok", "pathways": [], "notes": [], "counts": Counter()}
        report["sources"][name] = entry
        now = time.monotonic()
        if now >= deadline:
            entry["status"] = "skipped_budget"
            continue
        later = sum(SHARES.get(s, 1.0) for s in sources[i:])
        own_deadline = now + (deadline - now) * SHARES.get(name, 1.0) / later
        entry["budget_s"] = round(own_deadline - now, 1)
        try:
            fetch = (fetchers or {}).get(name) or _default_fetcher(name)
            pathways, notes = fetch(ctx, max_pathways=max_pathways, deadline=own_deadline,
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
            if diff and diff <= small and not any({k, k2} == {a, b} for a, b, _ in pairs):  # each pair once
                pairs.append([k, k2, sorted(diff)])
    return {"count": len(pairs), "pairs": pairs[:50]}
