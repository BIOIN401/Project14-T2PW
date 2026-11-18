"""Source pathways (one neutral shape for every source) -> overlay rows for ``merge_extension``.

Neutral shape, filled by pathbank.py / reactome.py / wikipathways.py::

    pathway = {"source_id", "db", "accession", "name", "url", "updated_at", "content_sha256",
               "species", "taxonomy_id", "steps": [step], "notes": [str]}
    step    = {"kind": "reaction", "accession", "name", "inputs": [part], "outputs": [part],
               "enzymes": [part], "location", "direction", "spontaneous"}
            | {"kind": "transport", "accession", "name", "cargo": part, "from_location",
               "to_location", "enzymes": [part], "direction"}
            | {"kind": "interaction", "accession", "name", "relationship": "activation" | "inhibition",
               "entity_1": part, "entity_2": part, "location"}
    part    = {"type": "compound" | "protein" | "complex" | "nucleic_acid", "name", "synonyms",
               "ids": {chebi, kegg, hmdb, pubchem, pathbank_compound_id, uniprot, pathbank_protein_id},
               "components": [part], "stoichiometry"}

Only what the source states is written: no direction, stoichiometry, compartment or enzyme is
added. A step without a compartment gets no biological_state (counted in ``counts``).
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from t2pw.vision.payload import _Index, grounded_ids, keyed_rows, norm

STAGE = "extension_db"
SECTION = {"compound": "entities.compounds", "protein": "entities.proteins",
           "complex": "entities.protein_complexes", "nucleic_acid": "entities.nucleic_acids"}
COMPOUND_ID_FIELDS = {"chebi": "chebi_id", "kegg": "kegg_id", "hmdb": "hmdb_id", "pubchem": "pubchem_cid",
                      "pathbank_compound_id": "pathbank_compound_id"}

# Currency compounds by database id -> the spelling payload.CURRENCY knows. Sources spell them
# their own way ("Hydrogen Ion", "Water", "Adenosine triphosphate"), which payload.is_currency does
# not fold, so without this they would join identity sets and carry hop edges.
_CURRENCY_IDS = {
    "H2O": ("kegg:c00001", "chebi:15377"), "H+": ("kegg:c00080", "chebi:15378", "chebi:24636"),
    "ATP": ("kegg:c00002", "chebi:15422", "chebi:30616"), "ADP": ("kegg:c00008", "chebi:16761", "chebi:456216"),
    "AMP": ("kegg:c00020", "chebi:16027", "chebi:456215"), "GTP": ("kegg:c00044", "chebi:15996", "chebi:37565"),
    "GDP": ("kegg:c00035", "chebi:17552", "chebi:58189"), "UTP": ("kegg:c00075", "chebi:15713", "chebi:46398"),
    "UDP": ("kegg:c00015", "chebi:17659", "chebi:58223"), "CTP": ("kegg:c00063", "chebi:17677", "chebi:37563"),
    "CDP": ("kegg:c00112", "chebi:17239", "chebi:58069"),
    "NAD+": ("kegg:c00003", "chebi:15846", "chebi:57540"), "NADH": ("kegg:c00004", "chebi:16908", "chebi:57945"),
    "NADP+": ("kegg:c00006", "chebi:18009", "chebi:58349"), "NADPH": ("kegg:c00005", "chebi:16474", "chebi:57783"),
    "FAD": ("kegg:c00016", "chebi:16238", "chebi:57692"), "FADH2": ("kegg:c01352", "chebi:17877", "chebi:58307"),
    "O2": ("kegg:c00007", "chebi:15379"), "CO2": ("kegg:c00011", "chebi:16526"),
    "H2O2": ("kegg:c00027", "chebi:16240"), "NH3": ("kegg:c00014", "kegg:c01342", "chebi:16134", "chebi:28938"),
    "phosphate": ("kegg:c00009", "chebi:18367", "chebi:43474", "chebi:26078"),
    "PPi": ("kegg:c00013", "chebi:29888", "chebi:33019", "chebi:18361"),
}
CURRENCY_BY_ID = {i: fam for fam, ids in _CURRENCY_IDS.items() for i in ids}


def _clean_ids(part: Dict[str, Any]) -> Dict[str, str]:
    out = {}
    for k, v in (part.get("ids") or {}).items():
        v = str(v).strip() if v not in (None, "") else ""
        if not v:
            continue
        if k == "chebi":
            v = v.split(":")[-1]
        out[k] = v
    return out


def currency_name(part: Dict[str, Any]) -> Optional[str]:
    ids = _clean_ids(part)
    for ns in ("kegg", "chebi"):
        if ns in ids and f"{ns}:{ids[ns].lower()}" in CURRENCY_BY_ID:
            return CURRENCY_BY_ID[f"{ns}:{ids[ns].lower()}"]
    return None


def _entity_row(part: Dict[str, Any], species: Optional[str], db: str) -> Dict[str, Any]:
    ids = _clean_ids(part)
    row: Dict[str, Any] = {"name": part["name"], "provenance": STAGE}
    syn = [s for s in part.get("synonyms") or [] if s and s != part["name"]]
    if syn:
        row["synonyms"] = list(dict.fromkeys(syn))[:20]
    t = part["type"]
    if t == "compound":
        row["class"] = "compound"
        mapped = {k: v for k, v in ids.items() if k in COMPOUND_ID_FIELDS}
        if mapped:
            row["mapped_ids"] = mapped
        for k, v in mapped.items():
            row[COMPOUND_ID_FIELDS[k]] = int(v) if k == "pathbank_compound_id" and v.isdigit() else v
    elif t == "protein":
        row["class"] = "protein"
        if ids.get("uniprot"):
            row["mapped_ids"] = {"uniprot": ids["uniprot"]}
            row["uniprot_id"] = ids["uniprot"]
        if ids.get("pathbank_protein_id"):
            row["pathbank_protein_id"] = ids["pathbank_protein_id"]
        if species:
            row["species"] = species
    elif t == "complex":
        comps = []
        for c in part.get("components") or []:
            comp = {"name": c["name"]}
            if c.get("stoichiometry") not in (None, ""):
                comp["stoichiometry"] = c["stoichiometry"]
            if _clean_ids(c).get("uniprot"):
                comp["mapped_ids"] = {"uniprot": _clean_ids(c)["uniprot"]}
            comps.append(comp)
        row["components"] = comps
        if species:
            row["species"] = species
    elif t == "nucleic_acid":
        row["class"] = "nucleic_acid"
        for k, v in ids.items():
            row.setdefault("mapped_ids", {})[k] = v
    row["note"] = f"from {db}"
    return row


def _uniprots_of_complex(row: Dict[str, Any], protein_uniprot: Dict[str, set]) -> frozenset:
    """UniProt set of a complex row: component mapped_ids, else the named protein row's ids."""
    out = set()
    for c in row.get("components") or []:
        if not isinstance(c, dict):
            continue
        ids = {v for ns, v in grounded_ids(c) if ns == "uniprot"}
        out |= ids or protein_uniprot.get(norm(c.get("name") or c.get("entity") or c.get("protein")), set())
    return frozenset(out)


def to_rows(big: Dict[str, Any], pw: Dict[str, Any]):
    """(rows_by_section, provenance_by_row, sources, counts) for one source pathway."""
    db, sid = pw["db"], pw["source_id"]
    species = pw.get("species") or None
    counts: Counter = Counter()
    index = _Index(big)
    existing = list(keyed_rows(big))
    by_key = {k: (s, r) for s, k, r, _ in existing}
    protein_uniprot: Dict[str, set] = {}
    for s, k, r, _ in existing:
        if s == "entities.proteins" and isinstance(r, dict):
            protein_uniprot[norm(r.get("name"))] = {v for ns, v in grounded_ids(r) if ns == "uniprot"}
    complex_by_set: Dict[frozenset, str] = {}
    for s, k, r, _ in existing:
        if s == "entities.protein_complexes" and isinstance(r, dict):
            u = _uniprots_of_complex(r, protein_uniprot)
            if u:
                complex_by_set.setdefault(u, r.get("name"))

    rows: Dict[str, List[Dict[str, Any]]] = {}
    provs: Dict[str, List[List[Dict[str, Any]]]] = {}
    seen: Dict[Tuple[str, str], str] = {}   # (section, source name) -> name written

    def rec(note: str) -> List[Dict[str, Any]]:
        return [{"kind": "database", "source": sid, "quote": None, "stage": STAGE, "note": note}]

    def emit(section: str, row: Dict[str, Any], note: str) -> None:
        rows.setdefault(section, []).append(row)
        provs.setdefault(section, []).append(rec(note))

    def entity(part: Dict[str, Any]) -> str:
        """Write the part's entity row once; return the name process rows must use."""
        part = dict(part)
        cur = currency_name(part) if part["type"] == "compound" else None
        if cur and norm(cur) != norm(part["name"]):
            part["synonyms"] = [part["name"]] + list(part.get("synonyms") or [])
            part["name"] = cur
            counts["currency_renamed"] += 1
        section = SECTION[part["type"]]
        ck = (section, norm(part["name"]))
        if ck in seen:
            return seen[ck]
        if part["type"] == "complex":
            part["components"] = [dict(c, name=entity(c)) for c in part.get("components") or []]
        row = _entity_row(part, species, db)
        name = row["name"]
        if part["type"] == "complex":
            u = _uniprots_of_complex(row, protein_uniprot)
            base_name = complex_by_set.get(u) if u else None
            if base_name:   # same member set as an existing complex: one enzyme, not two
                name = base_name
                counts["complex_joined_by_members"] += 1
            elif norm(name) in index.ent_key:  # same name, other (or unknown) members: keep apart
                name = f"{name} [{db}]"
                counts["renamed_name_conflict"] += 1
        else:
            hit, _ = index.match(section, row)
            if hit:
                name = index.key_name.get(hit) or name
            elif norm(name) in index.ent_key:  # F4: same name, conflicting ids -> separate row
                name = f"{name} [{db}]"
                counts["renamed_name_conflict"] += 1
        if name != row["name"]:
            if norm(name) != norm(row["name"]):
                row.setdefault("synonyms", [])
                if not name.endswith(f"[{db}]"):
                    row["synonyms"] = [row["name"]] + row["synonyms"]
            row["name"] = name
        if part["type"] == "protein":
            protein_uniprot.setdefault(norm(name), {v for ns, v in grounded_ids(row) if ns == "uniprot"})
        ids = ", ".join(f"{k}:{v}" for k, v in sorted(_clean_ids(part).items()))
        emit(section, row, f"{db} {pw['accession']} entity" + (f" ({ids})" if ids else ""))
        seen[ck] = name
        return name

    if species:   # proteins, complexes and states name it
        emit("entities.species", {"name": species, "taxonomy_id": pw.get("taxonomy_id"), "provenance": STAGE},
             f"{db} {pw['accession']} species")
    states_written: set = set()

    def state(location: Optional[str]) -> Optional[str]:
        if not location:
            return None
        name = f"{location} ({species or 'species not given'})"
        if name not in states_written:
            row = {"name": name, "subcellular_location": location}
            if species:
                row["species"] = species
            emit("entities.subcellular_locations", {"name": location, "provenance": STAGE},
                 f"{db} {pw['accession']} location")
            emit("biological_states", row, f"{db} {pw['accession']} state")
            states_written.add(name)
        return name

    def actor(part: Dict[str, Any]) -> Dict[str, Any]:
        name = entity(part)
        if part["type"] == "protein":   # a lone protein that an existing complex wraps: use that complex
            u = frozenset(v for k, v in _clean_ids(part).items() if k == "uniprot")
            if u in complex_by_set:
                return {"entity": complex_by_set[u], "entity_type": "protein_complex", "role": "catalyst"}
        etype = {"complex": "protein_complex"}.get(part["type"], part["type"])
        return {"entity": name, "entity_type": etype, "role": "catalyst"}

    def where(location: Optional[str], hit_state: Optional[str]) -> str:
        return f"source compartment: {location or 'not given'}" + (
            f"; matched row state: {hit_state}" if hit_state is not None else "")

    for step in pw.get("steps") or []:
        kind, acc = step.get("kind"), step.get("accession") or ""
        head = f"{db} {pw['accession']} {kind} {acc}".rstrip()
        if step.get("direction"):
            head += f"; source direction {step['direction']}"
        if step.get("note"):
            head += f"; {step['note']}"
        if kind == "reaction":
            ins = [entity(p) for p in step.get("inputs") or []]
            outs = [entity(p) for p in step.get("outputs") or []]
            if not ins or not outs:
                counts["skipped_one_sided"] += 1
                continue
            row: Dict[str, Any] = {"name": step.get("name") or f"{' + '.join(ins)} -> {' + '.join(outs)}",
                                   "inputs": ins, "outputs": outs,
                                   "enzymes": [actor(e) for e in step.get("enzymes") or []]}
            if step.get("spontaneous") is not None:
                row["spontaneous"] = bool(step["spontaneous"])
            hit, _ = index.match("processes.reactions", {"inputs": ins, "outputs": outs})
            hit_state = None
            if hit:   # an existing step: drop our state so the identity merge is state-blind
                hit_state = (by_key.get(hit, (None, {}))[1] or {}).get("biological_state")
                if step.get("location") and hit_state and not _same_place(big, hit_state, step["location"]):
                    counts["merged_compartment_differs"] += 1
            else:
                st = state(step.get("location"))
                if st:
                    row["biological_state"] = st
                else:
                    counts["no_compartment"] += 1
            emit("processes.reactions", row, head + "; " + where(step.get("location"), hit_state))
        elif kind == "transport":
            cargo = entity(step["cargo"])
            row = {"name": step.get("name") or f"{cargo} transport", "cargo": cargo,
                   "transporters": [dict(actor(e), role="transporter") for e in step.get("enzymes") or []]}
            for f, loc in (("from_biological_state", step.get("from_location")),
                           ("to_biological_state", step.get("to_location"))):
                st = state(loc)
                if st:
                    row[f] = st
                else:
                    counts["no_compartment"] += 1
            emit("processes.transports", row, head)
        elif kind == "interaction" and step.get("relationship") in ("activation", "inhibition"):
            e1, e2 = entity(step["entity_1"]), entity(step["entity_2"])
            row = {"name": step.get("name") or f"{e1} {step['relationship']} of {e2}",
                   "entity_1": e1, "entity_2": e2, "relationship": step["relationship"]}
            pair = sorted([norm(e1), norm(e2)])
            twin = next((r for s, k, r, _ in existing if s == "processes.interactions" and isinstance(r, dict)
                         and sorted([norm(r.get("entity_1")), norm(r.get("entity_2"))]) == pair
                         and norm(r.get("relationship")) == norm(step["relationship"])), None)
            if twin is not None:   # same pair and relationship: take its state so the merge joins
                if twin.get("biological_state"):
                    row["biological_state"] = twin["biological_state"]
            else:
                st = state(step.get("location"))
                if st:
                    row["biological_state"] = st
                else:
                    counts["no_compartment"] += 1
            emit("processes.interactions", row, head)
        else:
            counts[f"skipped_{kind}"] += 1
            continue
        counts[f"steps_{kind}"] += 1

    source = {"kind": "database", "pmid": None, "pmcid": None, "doi": None, "title": pw.get("name"),
              "db": db, "accession": pw.get("accession"), "url": pw.get("url"),
              "retrieved": pw.get("retrieved"), "updated_at": pw.get("updated_at"),
              "content_sha256": pw.get("content_sha256"), "species": species}
    return rows, provs, {sid: source}, counts


def _same_place(big: Dict[str, Any], state_name: str, location: str) -> bool:
    for s, _, r, _ in keyed_rows(big):
        if s == "biological_states" and isinstance(r, dict) and norm(r.get("name")) == norm(state_name):
            return norm(r.get("subcellular_location")) == norm(location)
    return False
