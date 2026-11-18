"""WikiPathways source: the public xref index (findPathwaysByXref.json) for candidates, the GPML
file for content. Read as stated, nothing more:

- metabolite -> metabolite line with an arrow (``mim-conversion`` or ``Arrow``): a reaction in the
  arrow's direction; metabolites joined to that line's anchor are co-substrates (no arrowhead /
  ``mim-conversion`` into the anchor) or co-products (an arrow out of the anchor);
- gene product / protein / complex group -> anchor with ``mim-catalysis`` or ``Arrow``: its enzyme;
- ``mim-inhibition`` / ``TBar`` and ``mim-stimulation`` / ``mim-necessary-stimulation`` between two
  nodes: inhibition / activation interactions.
Everything else is counted in notes. GPML gives no compartments, so no step gets a state.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from .http import get_text

INDEX = "https://www.wikipathways.org/json/findPathwaysByXref.json"
GPML = "https://www.wikipathways.org/wikipathways-assets/pathways/{id}/{id}.gpml"
XREF = {"chebi": "chebi", "kegg compound": "kegg", "hmdb": "hmdb", "pubchem-compound": "pubchem",
        "uniprot-trembl": "uniprot", "uniprot": "uniprot", "uniprot-swissprot": "uniprot"}
NODE = {"Metabolite": "compound", "GeneProduct": "protein", "Protein": "protein"}
CONVERSION, CATALYSIS = ("mim-conversion", "Arrow"), ("mim-catalysis", "Arrow")
RELATION = {"mim-inhibition": "inhibition", "TBar": "inhibition",
            "mim-stimulation": "activation", "mim-necessary-stimulation": "activation"}


def _tag(e) -> str:
    return e.tag.rsplit("}", 1)[-1]


def _ids(field: str) -> set:
    return {x.split(":", 1)[-1].strip().upper() for x in re.split(r"[,;]", field or "") if x.strip()}


def parse_gpml(text: str) -> Dict[str, Any]:
    root = ET.fromstring(text)
    nodes: Dict[str, Dict[str, Any]] = {}
    groups: Dict[str, List[str]] = {}       # group GraphId -> member node GraphIds (complex groups only)
    group_ref: Dict[str, str] = {}          # GroupId -> GraphId
    for g in root:
        if _tag(g) == "Group" and g.get("Style") == "Complex" and g.get("GraphId"):
            group_ref[g.get("GroupId")] = g.get("GraphId")
            groups[g.get("GraphId")] = []
    for n in root:
        if _tag(n) != "DataNode" or n.get("Type") not in NODE:
            continue
        xref = next((x for x in n if _tag(x) == "Xref"), None)
        ids = {}
        if xref is not None and xref.get("ID") and XREF.get(str(xref.get("Database")).lower()):
            ids[XREF[str(xref.get("Database")).lower()]] = xref.get("ID").strip()
        label = " ".join(str(n.get("TextLabel") or "").split())
        if not label:
            continue
        nodes[n.get("GraphId")] = {"type": NODE[n.get("Type")], "name": label, "synonyms": [], "ids": ids}
        if n.get("GroupRef") in group_ref:
            groups[group_ref[n.get("GroupRef")]].append(n.get("GraphId"))
    lines, anchor_of = [], {}
    for it in root:
        if _tag(it) != "Interaction":
            continue
        gfx = next((x for x in it if _tag(x) == "Graphics"), None)
        if gfx is None:
            continue
        pts = [p for p in gfx if _tag(p) == "Point"]
        for a in gfx:
            if _tag(a) == "Anchor" and a.get("GraphId"):
                anchor_of[a.get("GraphId")] = it.get("GraphId")
        if len(pts) >= 2:
            heads = [p.get("ArrowHead") for p in pts if p.get("ArrowHead")]
            lines.append({"id": it.get("GraphId"), "src": pts[0].get("GraphRef"), "tgt": pts[-1].get("GraphRef"),
                          "head": heads[-1] if heads else None})
    return {"nodes": nodes, "groups": groups, "lines": lines, "anchor_of": anchor_of}


def to_steps(g: Dict[str, Any]):
    nodes, groups, anchor_of = g["nodes"], g["groups"], g["anchor_of"]

    def part(ref):
        if ref in nodes:
            return dict(nodes[ref])
        members = [nodes[m] for m in groups.get(ref, []) if nodes.get(m, {}).get("type") == "protein"]
        if members:
            return {"type": "complex", "name": ":".join(m["name"] for m in members), "synonyms": [],
                    "components": [dict(m) for m in members]}
        return None

    rx: Dict[str, Dict[str, Any]] = {}
    for ln in g["lines"]:
        s, t = nodes.get(ln["src"]), nodes.get(ln["tgt"])
        if ln["head"] in CONVERSION and s and t and s["type"] == t["type"] == "compound":
            rx[ln["id"]] = {"kind": "reaction", "accession": ln["id"], "inputs": [dict(s)], "outputs": [dict(t)],
                            "enzymes": [], "location": None, "direction": "GPML arrow"}
    steps, notes = [], []
    for ln in g["lines"]:
        if ln["id"] in rx:
            continue
        src_rx, tgt_rx = rx.get(anchor_of.get(ln["src"])), rx.get(anchor_of.get(ln["tgt"]))
        s, t = part(ln["src"]), part(ln["tgt"])
        if tgt_rx and s and s["type"] in ("protein", "complex") and ln["head"] in CATALYSIS:
            tgt_rx["enzymes"].append(s)
        elif tgt_rx and s and s["type"] == "compound" and ln["head"] in (None, "mim-conversion"):
            tgt_rx["inputs"].append(s)
        elif src_rx and t and t["type"] == "compound" and ln["head"] in CONVERSION:
            src_rx["outputs"].append(t)
        elif ln["head"] in RELATION and s and t:
            steps.append({"kind": "interaction", "accession": ln["id"], "relationship": RELATION[ln["head"]],
                          "entity_1": s, "entity_2": t, "location": None})
        else:
            notes.append(f"line {ln['id']} ({ln['head']}) not converted")
    return list(rx.values()) + steps, notes


def fetch(ctx: Dict[str, Any], *, max_pathways: int = 3, deadline: float, cache_dir=None, min_overlap: int = 2):
    want = set()
    for s in ctx.get("species") or []:
        w = str(s.get("name") or "").split()
        if len(w) >= 2:
            want.add(f"{w[0]} {w[1]}".lower())
    ids = ctx.get("ids") or {}
    chebi = {v.upper() for v in ids.get("chebi", ())}
    uniprot = {v.upper() for v in ids.get("uniprot", ())}
    index = json.loads(get_text(INDEX, deadline=deadline, cache_dir=cache_dir, timeout=90) or "{}")
    ranked = []
    for p in index.get("pathwayInfo") or []:
        if str(p.get("species") or "").lower() not in want:
            continue
        shared = (_ids(p.get("chebi")) & chebi) | (_ids(p.get("uniprot")) & uniprot)
        if len(shared) >= min_overlap:
            ranked.append((len(shared), p, sorted(shared)))
    ranked.sort(key=lambda x: (-x[0], x[1]["id"]))
    notes = [f"species {sorted(want)}; candidates: " + "; ".join(f"{p['id']} {p['name']} {n}" for n, p, _ in ranked[:10])]
    if not ranked:
        return [], notes
    pathways = []
    for n, p, shared in ranked[:max_pathways]:
        if time.monotonic() >= deadline:
            notes.append(f"budget exhausted before {p['id']}")
            break
        text = get_text(GPML.format(id=p["id"]), deadline=deadline, cache_dir=cache_dir)
        if not text:
            notes.append(f"{p['id']}: no GPML")
            continue
        steps, pnotes = to_steps(parse_gpml(text))
        pathways.append({"source_id": f"wikipathways:{p['id']}", "db": "wikipathways", "accession": p["id"],
                         "name": p.get("name"), "url": p.get("url"), "updated_at": p.get("revision"),
                         "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                         "retrieved": time.strftime("%Y-%m-%d"), "species": p.get("species"),
                         "taxonomy_id": None, "steps": steps, "notes": pnotes, "overlap": shared})
        notes.append(f"{p['id']}: {len(steps)} steps, {len(pnotes)} lines not converted")
    return pathways, notes
