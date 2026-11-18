"""PathBank source: candidate pathways by overlap with the paper's grounded ids, read-only.

Connection and pathway content come from ``bench/eval_v2/pathbank_gold.py`` (``connect`` reads the
``PATHBANK_DB_*`` settings; ``extract`` resolves a public PW accession through
``super_pathway_visualizations.pw_id``). Only the candidate search is new here: every statement is
a SELECT.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from t2pw.bench.eval_v2.pathbank_gold import _Reader, connect, extract

GENERATED = re.compile(r"\s*Generated Pathway\b", re.I)


def _clean(text) -> str:
    """PathBank names carry tabs and padding ('\\t Chlorophyllide ')."""
    return " ".join(str(text or "").split())


TYPES = {"Compound": "compound", "Protein": "protein", "ProteinComplex": "complex", "NucleicAcid": "nucleic_acid"}
ID_KEYS = {"hmdb_id": "hmdb", "kegg_id": "kegg", "chebi_id": "chebi", "pubchem_cid": "pubchem",
           "uniprot_id": "uniprot"}


def _in(values) -> Tuple[str, list]:
    values = list(dict.fromkeys(values))
    return ",".join(["%s"] * len(values)), values


def species_ids(r: _Reader, species: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """PathBank species rows for the paper's species: same taxonomy id or name, else the same
    binomial (``Genus species%``, strains included)."""
    taxids = [s["taxonomy_id"] for s in species if s.get("taxonomy_id")]
    names = [s["name"] for s in species if s.get("name")]
    found: Dict[int, Dict[str, Any]] = {}
    if taxids or names:
        conds, params = [], []
        if taxids:
            ph, v = _in(taxids)
            conds.append(f"taxonomy_id IN ({ph})")
            params += v
        if names:
            ph, v = _in(names)
            conds.append(f"name IN ({ph})")
            params += v
        for row in r.rows(f"SELECT id, name, taxonomy_id FROM species WHERE {' OR '.join(conds)}", params):
            found[row["id"]] = {**row, "match": "exact"}
    for n in names:
        words = n.replace("[", "").replace("]", "").split()
        if len(words) >= 2 and words[0][:1].isupper() and words[1].islower():
            for row in r.rows("SELECT id, name, taxonomy_id FROM species WHERE name LIKE %s",
                              (f"{words[0]} {words[1]}%",)):
                found.setdefault(row["id"], {**row, "match": "binomial"})
    return found


def compound_ids(r: _Reader, ids: Dict[str, set]) -> set:
    out = {int(v) for v in ids.get("pathbank_compound", ()) if str(v).isdigit()}
    probes = []
    for ns, col, forms in (("kegg", "kegg_id", lambda v: [v.upper()]),
                           ("chebi", "chebi_id", lambda v: [v, f"CHEBI:{v}"]),
                           ("hmdb", "hmdb_id", lambda v: [v.upper(), "HMDB" + v.upper()[4:].lstrip("0").rjust(5, "0"),
                                                          "HMDB" + v.upper()[4:].lstrip("0").rjust(7, "0")])):
        vals = [f for v in ids.get(ns, ()) for f in forms(str(v))]
        if vals:
            ph, v = _in(vals)
            probes.append((f"{col} IN ({ph})", v))
    if probes:
        sql = "SELECT id FROM compounds WHERE " + " OR ".join(p for p, _ in probes)
        out |= {row["id"] for row in r.rows(sql, [x for _, v in probes for x in v])}
    return out


def candidates(r: _Reader, ctx: Dict[str, Any], *, min_overlap: int = 2) -> Tuple[List[Dict[str, Any]], List[str]]:
    notes: List[str] = []
    sp = species_ids(r, ctx.get("species") or [])
    if not sp:
        return [], ["no PathBank species for " + ", ".join(str(s.get("name")) for s in ctx.get("species") or [])]
    notes.append("species: " + ", ".join(f"{v['name']} ({v['match']})" for v in sp.values()))
    sp_ph, sp_v = _in(sp)
    score: Dict[int, Dict[str, int]] = {}
    cids = compound_ids(r, ctx.get("ids") or {})
    if cids:
        ph, v = _in(cids)
        for row in r.rows(
                "SELECT pe.pathway_id pid, COUNT(DISTINCT re.element_id) n FROM reaction_elements re "
                "JOIN pathway_elements pe ON pe.process_type='Reaction' AND pe.process_id=re.reaction_id "
                f"JOIN pathways p ON p.id=pe.pathway_id WHERE re.element_type='Compound' AND re.element_id IN ({ph}) "
                f"AND p.species_id IN ({sp_ph}) GROUP BY pe.pathway_id", v + sp_v):
            score.setdefault(row["pid"], {})["compounds"] = int(row["n"])
    uniprots = sorted(ctx.get("ids", {}).get("uniprot", ()))
    if uniprots:
        ph, v = _in(u.upper() for u in uniprots)
        for row in r.rows(
                "SELECT pe.pathway_id pid, COUNT(DISTINCT pr.uniprot_id) n FROM proteins pr "
                "JOIN protein_complex_proteins pcp ON pcp.protein_id=pr.id "
                "JOIN reaction_enzymes rz ON rz.protein_complex_id=pcp.protein_complex_id "
                "JOIN pathway_elements pe ON pe.process_type='Reaction' AND pe.process_id=rz.reaction_id "
                f"JOIN pathways p ON p.id=pe.pathway_id WHERE pr.uniprot_id IN ({ph}) AND p.species_id IN ({sp_ph}) "
                "GROUP BY pe.pathway_id", v + sp_v):
            score.setdefault(row["pid"], {})["proteins"] = int(row["n"])
    notes.append(f"query ids: {len(cids)} PathBank compounds, {len(uniprots)} UniProt")
    keep = {pid: s for pid, s in score.items() if sum(s.values()) >= min_overlap}
    if not keep:
        return [], notes + [f"no pathway shares >= {min_overlap} entities"]
    ph, v = _in(keep)
    rows = r.rows("SELECT p.id pid, p.name, p.species_id, spv.pw_id FROM pathways p "
                  "JOIN super_pathway_visualizations spv ON spv.named_for_type='Pathway' AND spv.named_for_id=p.id "
                  f"WHERE p.id IN ({ph}) AND spv.pw_id IS NOT NULL AND COALESCE(spv.private_pathway,0)=0", v)
    out, seen = [], set()
    for row in sorted(rows, key=lambda x: (-sum(keep[x["pid"]].values()),
                                           sp[x["species_id"]]["match"] != "exact", x["pid"])):
        if row["pid"] in seen:
            continue
        if GENERATED.match(str(row.get("name") or "")):   # machine-made record, not a curated pathway
            notes.append(f"{row['pw_id']} '{_clean(row.get('name'))}' skipped: not curated")
            seen.add(row["pid"])
            continue
        seen.add(row["pid"])
        out.append({"pw_id": row["pw_id"], "name": _clean(row["name"]), "pathway_id": row["pid"],
                    "species": sp[row["species_id"]]["name"], "species_match": sp[row["species_id"]]["match"],
                    "overlap": keep[row["pid"]]})
    # propagated copies of one pathway (one per strain) share a name: keep the best of each
    best: Dict[str, Dict[str, Any]] = {}
    for c in out:
        best.setdefault(str(c["name"]).strip().lower(), c)
    return list(best.values()), notes


def _part(p: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    t = TYPES.get(str(p.get("element_type")))
    if not t or not _clean(p.get("name")):
        return None
    ids = {ID_KEYS[k]: v for k, v in (p.get("identifiers") or {}).items() if k in ID_KEYS}
    if t == "compound":
        ids["pathbank_compound_id"] = p.get("element_id")
    if t == "protein":
        ids["pathbank_protein_id"] = p.get("element_id")
    syn = lambda x: [s for s in map(_clean, x.get("synonyms") or []) if s]
    part = {"type": t, "name": _clean(p["name"]), "synonyms": syn(p), "ids": ids}
    if t == "complex":
        part["components"] = [{"type": "protein", "name": _clean(c.get("name")), "synonyms": syn(c),
                               "ids": {"uniprot": c.get("uniprot"), "pathbank_protein_id": c.get("protein_id")},
                               "stoichiometry": c.get("stoichiometry")}
                              for c in p.get("components") or [] if _clean(c.get("name"))]
    return part


def _loc(state: Optional[Dict[str, Any]]) -> Optional[str]:
    return (state or {}).get("location") or None


RELATION = (("inhib", "inhibition"), ("repress", "inhibition"), ("activ", "activation"), ("stimul", "activation"))


def to_neutral(ex: Dict[str, Any]) -> Dict[str, Any]:
    acc = ex["accession"]
    steps, notes = [], []
    for proc in ex.get("processes") or []:
        ptype, step_acc = proc["process_type"], proc.get("accession") or proc.get("process_id")
        parts = [(p.get("side"), p, _part(p)) for p in proc.get("participants") or []]
        if any(x is None for _, _, x in parts):
            notes.append(f"{ptype} {step_acc}: participant type not converted, step skipped")
            continue
        cats = [c for c in (_part({**z, "element_type": "ProteinComplex"}) for z in proc.get("catalysts") or []) if c]
        if ptype == "Reaction":
            steps.append({"kind": "reaction", "accession": step_acc,
                          "inputs": [x for side, _, x in parts if side == "left"],
                          "outputs": [x for side, _, x in parts if side == "right"],
                          "enzymes": cats, "location": _loc(proc.get("biological_state")),
                          "direction": proc.get("direction"), "spontaneous": proc.get("spontaneous")})
        elif ptype == "Transport":
            for _, p, x in parts:
                steps.append({"kind": "transport", "accession": step_acc, "cargo": x, "enzymes": cats,
                              "from_location": _loc(p.get("from_state")), "to_location": _loc(p.get("to_state")),
                              "direction": p.get("transport_direction")})
        elif ptype == "Interaction":
            itype = str(proc.get("interaction_type") or "").lower()
            rel = next((r for k, r in RELATION if k in itype), None)
            left = [x for side, _, x in parts if side == "left"]
            right = [x for side, _, x in parts if side == "right"]
            if not rel or len(left) != 1 or len(right) != 1:
                notes.append(f"Interaction {step_acc}: type {itype or 'none'}, {len(left)}:{len(right)} "
                             "elements, not converted")
                continue
            steps.append({"kind": "interaction", "accession": step_acc, "relationship": rel,
                          "entity_1": left[0], "entity_2": right[0], "location": _loc(proc.get("biological_state"))})
        else:
            notes.append(f"{ptype} {step_acc}: not converted")
    return {"source_id": f"pathbank:{acc}", "db": "pathbank", "accession": acc, "name": _clean(ex.get("name")),
            "url": f"https://pathbank.org/view/{acc}", "updated_at": (ex.get("curator") or {}).get("updated_at"),
            "content_sha256": ex.get("content_sha256"), "retrieved": ex.get("extracted_at"),
            "species": (ex.get("species") or {}).get("name"),
            "taxonomy_id": (ex.get("species") or {}).get("taxonomy_id"), "steps": steps, "notes": notes,
            "references": [x.get("pubmed_id") for x in ex.get("references") or [] if x.get("pubmed_id")]}


def fetch(ctx: Dict[str, Any], *, max_pathways: int = 3, deadline: float, cache_dir=None, conn=None,
          min_overlap: int = 2):
    """(pathways in the neutral shape, notes). ``cache_dir`` is unused: PathBank is read live.
    The socket read timeout is the budget left at connect time; it bounds each query, not their sum
    (extract runs many), so the budget is checked between pathways and can be overrun by one extract."""
    own = conn is None
    left = max(5, int(deadline - time.monotonic()))
    conn = conn or connect({"read_timeout": left, "connect_timeout": min(10, left)})
    try:
        r = _Reader(conn)
        cands, notes = candidates(r, ctx, min_overlap=min_overlap)
        notes.append(f"read timeout per query {left} s; the budget is not hard inside one extract")
        pathways = []
        for c in cands[:max_pathways]:
            if time.monotonic() >= deadline:
                notes.append(f"budget exhausted before {c['pw_id']}")
                break
            pw = to_neutral(extract(conn, c["pw_id"]))
            pw["overlap"] = c["overlap"]
            pw["species_match"] = c["species_match"]
            pathways.append(pw)
        notes.append("candidates: " + "; ".join(f"{c['pw_id']} {c['name']} {c['overlap']}" for c in cands[:10]))
        return pathways, notes
    finally:
        if own:
            conn.close()
