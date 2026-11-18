"""Reactome source (ContentService REST): pathways that hold the paper's UniProt / ChEBI ids, for
the paper's species only, and their reactions with participants and catalysts.

Not converted (counted in notes): black-box events, polymerisation, regulation (regulatedBy), and
any reaction with a set, polymer, drug or other non-molecule participant.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .http import get_text

BASE = "https://reactome.org/ContentService"
_SUFFIX = re.compile(r"\s*\[([^\[\]]+)\]\s*$")


def _get(path: str, deadline: float, cache_dir) -> Any:
    body = get_text(BASE + path, deadline=deadline, cache_dir=cache_dir)
    return json.loads(body) if body else None


def split_name(display: str) -> Tuple[str, Optional[str]]:
    """'ATP [cytosol]' -> ('ATP', 'cytosol')."""
    m = _SUFFIX.search(display or "")
    return ((display[:m.start()].strip(), m.group(1)) if m else ((display or "").strip(), None))


def _protein(ref: Dict[str, Any]) -> Dict[str, Any]:
    disp = str(ref.get("displayName") or "")       # 'UniProt:P17858 PFKL'
    name = disp.split(" ", 1)[1] if " " in disp else ref.get("identifier")
    return {"type": "protein", "name": name, "synonyms": [], "ids": {"uniprot": ref.get("identifier")}}


SETS = ("DefinedSet", "CandidateSet", "OpenSet")


def part_of(pe: Dict[str, Any], subunits=None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """(neutral part or None when not convertible, compartment) for one /participants entry.
    A complex gets its UniProt members as components only when ``subunits(dbId)`` (the
    /data/complex/{id}/subunits list) shows no nested set: participants' refEntities flatten sets,
    and "any of" members must not become co-subunits. Otherwise it is written by name only."""
    name, comp = split_name(pe.get("displayName") or "")
    cls = pe.get("schemaClass")
    refs = pe.get("refEntities") or []
    prot = [r for r in refs if r.get("schemaClass") in ("ReferenceGeneProduct", "ReferenceIsoform")]
    if cls == "SimpleEntity" and len(refs) == 1:
        ref_name = str(refs[0].get("displayName") or "").split(" [ChEBI:")[0].strip()
        return {"type": "compound", "name": name, "synonyms": [ref_name] if ref_name != name else [],
                "ids": {"chebi": refs[0].get("identifier")}}, comp
    if cls == "EntityWithAccessionedSequence" and len(prot) == 1:
        p = _protein(prot[0])
        return {**p, "name": name, "synonyms": [p["name"]] if p["name"] != name else []}, comp
    if cls == "Complex":
        subs = subunits(pe.get("peDbId")) if subunits else None
        flat = bool(prot) and subs is not None and not any(
            isinstance(s, dict) and s.get("schemaClass") in SETS for s in subs)
        return {"type": "complex", "name": name, "synonyms": [],
                "components": [_protein(r) for r in prot] if flat else []}, comp
    return None, comp


def _ids_of(v) -> List[int]:
    return [x if isinstance(x, int) else x.get("dbId") for x in v or [] if isinstance(x, (int, dict))]


def reaction_step(st_id: str, deadline: float, cache_dir) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    q = _get(f"/data/query/{st_id}", deadline, cache_dir)
    pes = {p["peDbId"]: p for p in _get(f"/data/participants/{st_id}", deadline, cache_dir) or []}
    subunits = lambda db_id: _get(f"/data/complex/{db_id}/subunits", deadline, cache_dir)
    side = {}
    for f in ("input", "output"):
        side[f] = []
        for i in dict.fromkeys(_ids_of(q.get(f))):   # repeats are stoichiometry; not carried
            part, comp = part_of(pes.get(i, {}), subunits)
            if part is None:
                return None, f"{st_id}: {f} {pes.get(i, {}).get('schemaClass')} not converted"
            side[f].append((part, comp))
    enzymes, notes = [], []
    for ca in _ids_of(q.get("catalystActivity")):
        act = _get(f"/data/query/{ca}", deadline, cache_dir) or {}
        pe = pes.get((act.get("physicalEntity") or {}).get("dbId"), {})
        part, _ = part_of(pe, subunits)
        if part is not None:
            enzymes.append(part)
        elif pe.get("schemaClass") in SETS:
            # "any of": each member alone, but only when every member is a single protein; refEntities
            # flatten complex members into subunits, which the source never lists as alternatives
            name = split_name(pe.get("displayName", ""))[0]
            members = (_get(f"/data/query/{pe.get('peDbId')}", deadline, cache_dir) or {}).get("hasMember") or []
            refs = [r for r in pe.get("refEntities") or [] if r.get("schemaClass") == "ReferenceGeneProduct"]
            if members and all(isinstance(m, dict) and m.get("schemaClass") == "EntityWithAccessionedSequence"
                               for m in members) and len(refs) == len(members):
                enzymes += [_protein(r) for r in refs]
                notes.append(f"catalyst set {name}: members listed as alternatives")
            else:
                enzymes.append({"type": "complex", "name": name, "synonyms": [], "components": []})
                notes.append(f"catalyst set {name}: members not all single proteins, written by name only")
    comp_rows = [c for c in q.get("compartment") or [] if isinstance(c, dict)]
    comps = [c.get("displayName") for c in comp_rows]
    go = (f"GO:{comp_rows[0]['accession']}" if len(comp_rows) == 1 and comp_rows[0].get("accession")
          and str(comp_rows[0].get("databaseName", "GO")).upper() == "GO" else None)
    key = lambda p: json.dumps([p["type"], p["name"], p.get("ids"), [c.get("ids") for c in p.get("components") or []]],
                               sort_keys=True)
    ins, outs = side["input"], side["output"]
    if len(ins) == 1 and len(outs) == 1 and key(ins[0][0]) == key(outs[0][0]) and ins[0][1] != outs[0][1]:
        return {"kind": "transport", "accession": st_id, "name": q.get("displayName"), "cargo": ins[0][0],
                "enzymes": enzymes, "from_location": ins[0][1], "to_location": outs[0][1],
                "note": "; ".join(notes)}, None
    return {"kind": "reaction", "accession": st_id, "name": q.get("displayName"),
            "inputs": [p for p, _ in ins], "outputs": [p for p, _ in outs], "enzymes": enzymes,
            "location": comps[0] if len(comps) == 1 else None, "location_go": go,
            "note": "; ".join(notes + ([f"compartments {comps}: none chosen"] if len(comps) > 1 else []))}, None


def species_of(ctx: Dict[str, Any], deadline: float, cache_dir) -> Optional[Dict[str, Any]]:
    want_tax = {str(s.get("taxonomy_id")) for s in ctx.get("species") or [] if s.get("taxonomy_id")}
    want_name = {str(s.get("name") or "").strip().lower() for s in ctx.get("species") or []}
    for sp in _get("/data/species/all", deadline, cache_dir) or []:
        if str(sp.get("taxId")) in want_tax or str(sp.get("displayName", "")).lower() in want_name:
            return sp
    return None


def fetch(ctx: Dict[str, Any], *, max_pathways: int = 3, deadline: float, cache_dir=None,
          min_overlap: int = 2, max_ids: int = 40):
    notes: List[str] = []
    sp = species_of(ctx, deadline, cache_dir)
    if not sp:
        return [], ["species not in Reactome: " + ", ".join(str(s.get("name")) for s in ctx.get("species") or [])]
    ids = ctx.get("ids") or {}
    probes = [("UniProt", v.upper()) for v in sorted(ids.get("uniprot", ()))[:max_ids]] + \
             [("ChEBI", v) for v in sorted(ids.get("chebi", ()))[:max_ids]]
    hits: Dict[str, Dict[str, Any]] = {}
    for res, ident in probes:
        if time.monotonic() >= deadline:
            notes.append("budget exhausted during candidate search")
            break
        for pw in _get(f"/data/mapping/{res}/{ident}/pathways?species={sp['taxId']}", deadline, cache_dir) or []:
            h = hits.setdefault(pw["stId"], {"pw": pw, "ids": set()})
            h["ids"].add(f"{res}:{ident}")
    ranked = sorted((h for h in hits.values() if len(h["ids"]) >= min_overlap),
                    key=lambda h: (-len(h["ids"]), h["pw"]["stId"]))
    notes.append(f"{len(probes)} ids probed; candidates: " + "; ".join(
        f"{h['pw']['stId']} {h['pw'].get('displayName')} {len(h['ids'])}" for h in ranked[:10]))
    pathways = []
    for h in ranked[:max_pathways]:
        pw, steps, pnotes = h["pw"], [], []
        try:
            events = _get(f"/data/pathway/{pw['stId']}/containedEvents", deadline, cache_dir) or []
            for ev in events:
                if not isinstance(ev, dict) or ev.get("schemaClass") != "Reaction":
                    if isinstance(ev, dict) and ev.get("schemaClass") != "Pathway":
                        pnotes.append(f"{ev.get('stId')} {ev.get('schemaClass')} not converted")
                    continue
                step, why = reaction_step(ev["stId"], deadline, cache_dir)
                if step:
                    steps.append(step)
                else:
                    pnotes.append(why)
        except Exception as exc:  # noqa: BLE001 - keep the steps already read (budget or network)
            pnotes.append(f"stopped early: {type(exc).__name__}: {exc}"[:200])
        pathways.append({
            "source_id": f"reactome:{pw['stId']}", "db": "reactome", "accession": pw["stId"],
            "name": pw.get("displayName"), "url": f"https://reactome.org/content/detail/{pw['stId']}",
            "updated_at": pw.get("lastUpdatedDate") or pw.get("releaseDate"),
            "content_sha256": hashlib.sha256(json.dumps(steps, sort_keys=True).encode()).hexdigest(),
            "retrieved": time.strftime("%Y-%m-%d"), "species": sp.get("displayName"),
            "taxonomy_id": sp.get("taxId"), "steps": steps, "notes": pnotes,
            "overlap": sorted(h["ids"])})
        notes += pnotes[:20]
        if time.monotonic() >= deadline:
            notes.append("budget exhausted")
            break
    return pathways, notes
