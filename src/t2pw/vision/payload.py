"""The big payload: the frozen paper payload (``base``, never edited) plus an overlay.

Shape (docs/vision/PLAN.md section 2.1)::

    {"schema": "t2pw.big_payload/1", "base": <final_mapped.json>, "extension": <rows in base section shapes>,
     "provenance": {<element key>: [record, ...]}, "hops": {<element key>: int | None},
     "sources": {<source id>: {kind, pmid, pmcid, doi, title, db, accession, url, retrieved}}}

Element key = ``<section path>:<signature>``; the signature is the normalized name for entities and
states, and a content signature for processes and element locations. Repeats get ``#2``, ``#3`` ...
in row order, base first, then extension, so appending extension rows never renames an existing key.

Provenance record = ``{kind, source, quote, stage, note}`` with kind one of paper | other_paper |
database | inferred | curation | unclassified. How today's fields map is in :func:`row_provenance`.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from t2pw.pipeline.cofactor_policy import _CURRENCY, _normalize as _currency_fold  # private, read-only
from t2pw.pipeline.name_keys import reference_key
from t2pw.rag.provenance import SEED_SOURCE_ID as SEED_TAG

SCHEMA = "t2pw.big_payload/1"
CONTAINERS = ("entities", "processes", "element_locations")
STATES = "biological_states"
LOCATION_ENTITY_KEYS = ("compound", "protein", "nucleic_acid", "element_collection")

ORIGIN_KIND = {
    "paper_stated": "paper",
    "rag_literature": "other_paper",
    "database_grounded": "database",
    "inferred": "inferred",
    "audit_modified": "curation",
}
# Lineage source_type values that name a paper (anything else is a database record).
PAPER_SOURCE_TYPES = ("", "paper")
# RAG tags chunks of the supplied paper SEED_TAG ("seed_paper"). rag/conform.py (~line 283) warns the
# tag means "a seed reaction registered this first", not "the base has it", so it never makes a row
# kind "paper": a rag_literature entry, or a seed-tagged row without lineage, stays other_paper.

# Ubiquitous currency metabolites (pipeline/cofactor_policy.py _CURRENCY, folded the same way). They
# are left out of reaction identity and never carry a hop edge.
# Spellings only, not family keys: the module drops bare "pi" (phosphatidylinositol) on purpose.
CURRENCY = frozenset(_currency_fold(s) for spellings in _CURRENCY.values() for s in spellings)


def is_currency(name: Any) -> bool:
    return isinstance(name, str) and _currency_fold(name) in CURRENCY

ID_FIELDS = {"chebi_id": "chebi", "kegg_id": "kegg", "hmdb_id": "hmdb", "pubchem_cid": "pubchem",
             "pathbank_compound_id": "pathbank_compound", "drugbank_id": "drugbank", "uniprot": "uniprot",
             "uniprot_id": "uniprot", "pathbank_protein_id": "pathbank_protein"}
NO_ID = ("", "unknown", "none", "null", "n/a", "na", "0")


def norm(value: Any) -> str:
    """The shared reference key (pipeline/name_keys.py), '' for non-strings."""
    return reference_key(value) if isinstance(value, str) else ""


# ---------------------------------------------------------------- sections and references

def iter_sections(payload: Dict[str, Any]) -> Iterator[Tuple[str, List[Any]]]:
    """(section path, rows) for every list-valued row section, in payload order, states last."""
    for container in CONTAINERS:
        sub = payload.get(container)
        if isinstance(sub, dict):
            for name, rows in sub.items():
                if isinstance(rows, list):
                    yield f"{container}.{name}", rows
    if isinstance(payload.get(STATES), list):
        yield STATES, payload[STATES]


def section_rows(payload: Dict[str, Any], section: str, create: bool = False) -> Optional[List[Any]]:
    if section == STATES:
        if create:
            payload.setdefault(STATES, [])
        return payload.get(STATES)
    container, name = section.split(".", 1)
    if create:
        return payload.setdefault(container, {}).setdefault(name, [])
    sub = payload.get(container)
    return sub.get(name) if isinstance(sub, dict) else None


def refs(section: str, row: Any, rename: Optional[Callable[[str], str]] = None):
    """(entity names, state names, row) a row references. With ``rename``, the returned row is a
    copy with every entity reference passed through it (states are never renamed)."""
    ents: List[str] = []
    states: List[str] = []
    if not isinstance(row, dict):
        return ents, states, row
    out = dict(row)

    def ent(v):
        if isinstance(v, str) and v.strip():
            ents.append(v)
            return rename(v) if rename else v
        return v

    def actor(x):  # enzyme / modifier / transporter / complex component: a name or a dict naming one
        if isinstance(x, dict):
            x = dict(x)
            for k in [k for k in ("entity", "protein") if k in x] or ["name"]:
                if k in x:
                    x[k] = ent(x[k])
            return x
        return ent(x)

    def st(v):
        if isinstance(v, str) and v.strip():
            states.append(v)

    if section.startswith("processes."):
        for f in ("inputs", "outputs"):
            if isinstance(row.get(f), list):
                out[f] = [ent(x) for x in row[f]]
        for f in ("enzymes", "modifiers", "transporters"):
            if isinstance(row.get(f), list):
                out[f] = [actor(x) for x in row[f]]
        for f in ("cargo", "entity_1", "entity_2"):
            if f in row:
                out[f] = ent(row[f])
        for f in ("biological_state", "from_biological_state", "to_biological_state"):
            st(row.get(f))
        if isinstance(row.get("elements_with_states"), list):
            ews = []
            for x in row["elements_with_states"]:
                if isinstance(x, dict):
                    x = dict(x)
                    if "element" in x:
                        x["element"] = ent(x["element"])
                    st(x.get("biological_state"))
                ews.append(x)
            out["elements_with_states"] = ews
    elif section.startswith("element_locations."):
        for f in LOCATION_ENTITY_KEYS:
            if f in row:
                out[f] = ent(row[f])
        st(row.get("biological_state"))
    elif section in ("entities.proteins", "entities.protein_complexes"):
        if isinstance(row.get("components"), list):
            out["components"] = [actor(x) for x in row["components"]]
        if isinstance(row.get("species"), str):
            out["species"] = ent(row["species"])
    elif section == STATES:
        for f in ("species", "subcellular_location", "cell_type", "tissue"):
            if f in row:
                out[f] = ent(row[f])
    return ents, states, (out if rename else row)


# ---------------------------------------------------------------- element keys

def signature(section: str, row: Any) -> str:
    if not isinstance(row, dict):
        return ""
    if section == "processes.reactions":
        ins = "+".join(sorted(norm(x) for x in row.get("inputs") or []))
        outs = "+".join(sorted(norm(x) for x in row.get("outputs") or []))
        return f"{ins}>{outs}@{norm(row.get('biological_state'))}"
    if section == "processes.transports":
        return f"{norm(row.get('cargo'))}:{norm(row.get('from_biological_state'))}>{norm(row.get('to_biological_state'))}"
    if section == "processes.interactions":
        pair = "+".join(sorted([norm(row.get("entity_1")), norm(row.get("entity_2"))]))
        return f"{pair}:{norm(row.get('relationship'))}@{norm(row.get('biological_state'))}"
    if section.startswith("element_locations."):
        ents, states, _ = refs(section, row)
        return f"{norm(ents[0]) if ents else ''}@{norm(states[0]) if states else ''}"
    return norm(row.get("name"))


def next_key(section: str, row: Any, taken: Counter) -> str:
    base = f"{section}:{signature(section, row)}"
    taken[base] += 1
    return base if taken[base] == 1 else f"{base}#{taken[base]}"


def keyed_rows(big: Dict[str, Any], taken: Optional[Counter] = None) -> Iterator[Tuple[str, str, Any, bool]]:
    """(section, key, row, is_base) for every base row, then every extension row."""
    taken = Counter() if taken is None else taken
    for part, is_base in ((big.get("base") or {}, True), (big.get("extension") or {}, False)):
        for section, rows in iter_sections(part):
            for row in rows:
                yield section, next_key(section, row, taken), row, is_base


def element_keys(big: Dict[str, Any]) -> List[str]:
    return [key for _, key, _, _ in keyed_rows(big)]


# ---------------------------------------------------------------- provenance

def _blank_source(kind: str) -> Dict[str, Any]:
    return {"kind": kind, "pmid": None, "pmcid": None, "doi": None, "title": None, "db": None,
            "accession": None, "url": None, "retrieved": None}


def register_source(sources: Dict[str, Any], sid: str, stype: str, uri: str = "", title: Any = None) -> str:
    """Add a lineage / rag source to the table under a stable id and return the id."""
    sid, stype = str(sid or uri or "").strip(), str(stype or "").strip()
    paperish = stype.lower() in PAPER_SOURCE_TYPES
    if paperish or sid.lower().startswith(stype.lower() + ":"):
        table_id = sid
    else:
        table_id = f"{stype}:{sid}"
    if table_id not in sources:
        entry = _blank_source("other_paper" if paperish else "database")
        entry["title"] = title or None
        entry["url"] = uri or None
        if paperish:
            up = sid.upper()
            if up.startswith("PMC"):
                entry["pmcid"] = sid
            elif up.startswith("PMID:") or sid.isdigit():
                entry["pmid"] = sid.split(":")[-1]
            elif sid.startswith("10."):
                entry["doi"] = sid
            if "doi.org/" in (uri or ""):
                entry["doi"] = uri.split("doi.org/", 1)[1]
            if sid == SEED_TAG:
                entry["note"] = ("RAG tag for a chunk indexed from the supplied paper; not a reliable "
                                 "seed test (rag/conform.py), so rows citing it stay other_paper")
        else:
            entry["db"], entry["accession"] = stype, sid
        sources[table_id] = entry
    return table_id


def evidence_quote(row: Dict[str, Any]) -> Optional[str]:
    ev = row.get("evidence")
    if isinstance(ev, list) and ev:
        ev = ev[0].get("text") if isinstance(ev[0], dict) else ev[0]
    return ev.strip() if isinstance(ev, str) and ev.strip() else None


def row_provenance(row: Any, sources: Dict[str, Any], paper_source: Optional[str] = None) -> List[Dict[str, Any]]:
    """Provenance records for one payload row; registers every cited source in ``sources``.

    - each ``provenance_lineage`` entry -> one record (origin mapped by ORIGIN_KIND; unresolved,
      excluded and unknown origins -> kind ``unclassified`` with the origin kept in ``note``);
    - ``rag_provenance`` naming a source other than the seed tag -> other_paper / database record,
      unless lineage already gave that (kind, source); with lineage, the seed tag adds nothing;
    - no lineage and a seed-tagged ``rag_provenance`` -> one ``other_paper`` record (the row came in
      through RAG, so its evidence is a retrieved passage, not proof the paper states it);
    - otherwise no lineage but an ``evidence`` quote -> one ``paper`` record; neither -> ``unclassified``.
    """
    if not isinstance(row, dict):
        return [{"kind": "unclassified", "source": None, "quote": None, "stage": None, "note": "not a row"}]
    quote = evidence_quote(row)
    recs: List[Dict[str, Any]] = []
    lineage = [e for e in row.get("provenance_lineage") or [] if isinstance(e, dict)]
    for e in lineage:
        origin = str(e.get("origin") or "")
        kind = ORIGIN_KIND.get(origin, "unclassified")
        note = str(e.get("reason") or "")
        if kind == "unclassified":
            note = f"{origin}: {note}"
        ids = [register_source(sources, s.get("source_id"), s.get("source_type"), s.get("uri") or "")
               for s in e.get("sources") or [] if isinstance(s, dict) and (s.get("source_id") or s.get("uri"))]
        if len(ids) > 1:
            note += "; sources: " + ", ".join(ids)
        source = ids[0] if ids else (paper_source if kind == "paper" else None)
        recs.append({"kind": kind, "source": source, "quote": quote if kind in ("paper", "other_paper") else None,
                     "stage": e.get("stage"), "note": note})
    rp = row.get("rag_provenance")
    if isinstance(rp, dict) and str(rp.get("source_id") or "").strip() not in ("", SEED_TAG):
        sid = register_source(sources, rp["source_id"], rp.get("source_type"), rp.get("source_uri") or "",
                              rp.get("source_title"))
        kind = sources[sid]["kind"]
        if not any(r["kind"] == kind and r["source"] == sid for r in recs):
            recs.append({"kind": kind, "source": sid, "quote": quote, "stage": "rag_retrieval",
                         "note": "from rag_provenance"})
    if not lineage and not recs:
        if isinstance(rp, dict) and str(rp.get("source_id") or "").strip() == SEED_TAG:
            sid = register_source(sources, SEED_TAG, rp.get("source_type"), rp.get("source_uri") or "",
                                  rp.get("source_title"))
            recs.append({"kind": "other_paper", "source": sid, "quote": quote, "stage": "rag_retrieval",
                         "note": "no lineage; seed-tagged rag_provenance"})
        elif quote:
            recs.append({"kind": "paper", "source": paper_source, "quote": quote, "stage": None,
                         "note": "no lineage; evidence quote"})
        else:
            recs.append({"kind": "unclassified", "source": None, "quote": None, "stage": None,
                         "note": "no lineage, no evidence"})
    return recs


# ---------------------------------------------------------------- build / save / load

def build_big_payload(base: Dict[str, Any], *, paper_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base = copy.deepcopy(base)
    sources: Dict[str, Any] = {}
    paper_source = None
    if paper_meta:
        sources["paper"] = {**_blank_source("paper"), **paper_meta}
        paper_source = "paper"
    provenance, hops = {}, {}
    big = {"schema": SCHEMA, "base": base, "extension": {}, "provenance": provenance, "hops": hops,
           "sources": sources}
    for _, key, row, _ in keyed_rows(big):
        provenance[key] = row_provenance(row, sources, paper_source)
        hops[key] = 0
    return big


def save_big_payload(big: Dict[str, Any], path) -> None:
    Path(path).write_text(json.dumps(big, ensure_ascii=False, indent=1), encoding="utf-8")


def load_big_payload(path) -> Dict[str, Any]:
    big = json.loads(Path(path).read_text(encoding="utf-8"))
    if big.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA} file")
    return big


def big_from_run(run_dir, *, paper_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base = json.loads((Path(run_dir) / "final_mapped.json").read_text(encoding="utf-8"))
    return build_big_payload(base, paper_meta=paper_meta)


# ---------------------------------------------------------------- merge and hops

def grounded_ids(row: Dict[str, Any]) -> set:
    pairs = [(f, row.get(f)) for f in ID_FIELDS]
    if isinstance(row.get("mapped_ids"), dict):
        pairs += list(row["mapped_ids"].items())
    out = set()
    for field, value in pairs:
        ns = ID_FIELDS.get(field, field).lower()
        ns = ns[:-3] if ns.endswith("_id") else ns
        for v in value if isinstance(value, list) else [value]:
            if v is None or isinstance(v, bool):
                continue
            v = str(v).strip().lower()
            v = v[len(ns) + 1:] if v.startswith(ns + ":") else v
            if v not in NO_ID:
                out.add((ns, v))
    return out


def _ids_conflict(a: set, b: set) -> bool:
    """True when two rows have ids in a shared namespace and none of those ids agree."""
    shared = {ns for ns, _ in a} & {ns for ns, _ in b}
    return any(not ({v for n, v in a if n == ns} & {v for n, v in b if n == ns}) for ns in shared)


def _names(row: Dict[str, Any]) -> List[str]:
    names = [row.get("name")] + list(row.get("synonyms") or [])
    return [n for n in (norm(x) for x in names) if n]


class _Index:
    """Identity lookups over base + extension rows (rebuilt per merge; payloads are small)."""

    def __init__(self, big: Dict[str, Any]):
        self.taken: Counter = Counter()
        self.by_id: Dict[tuple, str] = {}
        self.by_name: Dict[tuple, str] = {}
        self.ent_key: Dict[str, str] = {}     # normalized name / synonym -> entity element key
        self.canon: Dict[str, str] = {}       # normalized name / synonym -> the row's own name
        self.proc: Dict[tuple, str] = {}
        self.rx: Dict[tuple, List[Tuple[str, str]]] = {}  # (inputs, outputs) -> [(state, key)]
        self.key_name: Dict[str, Any] = {}    # entity element key -> the row's own name
        self.key_ids: Dict[str, set] = {}     # entity element key -> its grounded ids
        rows = list(keyed_rows(big, self.taken))  # self.taken: new rows continue the ordinals
        for primary in (True, False):         # primary names win over synonyms
            for section, key, row, _ in rows:
                if section.startswith("entities.") and isinstance(row, dict):
                    self.key_name[key] = row.get("name")
                    self.key_ids[key] = grounded_ids(row)
                    names = _names(row)
                    for n in names[:1] if primary else names[1:]:
                        self.ent_key.setdefault(n, key)
                        self.canon.setdefault(n, row.get("name"))
        for section, key, row, _ in rows:
            self._add(section, row, key)

    def ident(self, name: Any) -> str:
        n = norm(name)
        return self.ent_key.get(n, "name:" + n)

    def rename(self, name: str) -> str:
        return self.canon.get(norm(name)) or name

    def edge(self, name: Any) -> Optional[str]:
        """The entity key a name gives a hop edge through: grounded and not a currency metabolite."""
        if is_currency(name):
            return None
        key = self.ident(name)
        return key if self.key_ids.get(key) else None

    def _rx_sets(self, row: Dict[str, Any]) -> tuple:
        """Identity sets of inputs and outputs without currency metabolites (all of them if that
        leaves both sides empty)."""
        ins, outs = row.get("inputs") or [], row.get("outputs") or []
        core_in = [n for n in ins if not is_currency(n)]
        core_out = [n for n in outs if not is_currency(n)]
        if core_in or core_out:
            ins, outs = core_in, core_out
        return frozenset(map(self.ident, ins)), frozenset(map(self.ident, outs))

    def _add(self, section: str, row: Any, key: str) -> None:
        if not isinstance(row, dict):
            return
        if section.startswith("entities."):
            for i in grounded_ids(row):
                self.by_id.setdefault((section,) + i, key)
            for n in _names(row):
                self.by_name.setdefault((section, n), key)
        elif section == "processes.reactions":
            self.rx.setdefault(self._rx_sets(row), []).append((norm(row.get("biological_state")), key))
        else:
            self.proc.setdefault((section, signature(section, row)), key)

    def _rx_match(self, sets: tuple, state: str) -> Optional[str]:
        # a state on both sides must agree; a row without a state matches any state
        return next((k for s, k in self.rx.get(sets, []) if not (s and state) or s == state), None)

    def match(self, section: str, row: Any) -> Tuple[Optional[str], bool]:
        if not isinstance(row, dict):
            return None, False
        if section.startswith("entities."):
            ids = grounded_ids(row)
            for i in sorted(ids):
                if (section,) + i in self.by_id:
                    return self.by_id[(section,) + i], False
            for n in _names(row):
                key = self.by_name.get((section, n))
                if key and not _ids_conflict(ids, self.key_ids.get(key, set())):
                    return key, False
            return None, False
        if section == "processes.reactions":
            ins, outs = self._rx_sets(row)
            state = norm(row.get("biological_state"))
            hit = self._rx_match((ins, outs), state)
            if hit or ins == outs:
                return hit, False
            hit = self._rx_match((outs, ins), state)
            return hit, hit is not None
        return self.proc.get((section, signature(section, row))), False

    def alias(self, row: Dict[str, Any], key: str, canonical_name: Any) -> None:
        for n in _names(row):
            self.ent_key.setdefault(n, key)
            self.canon.setdefault(n, canonical_name)

    def add_new(self, section: str, row: Dict[str, Any]) -> str:
        key = next_key(section, row, self.taken)
        if section.startswith("entities.") and isinstance(row, dict):
            self.key_name[key] = row.get("name")
            self.key_ids[key] = grounded_ids(row)
            self.alias(row, key, row.get("name"))
        self._add(section, row, key)
        return key


def _rank(section: str) -> int:
    return {"entities": 0, STATES: 1, "processes": 2, "element_locations": 3}[section.split(".")[0]]


def merge_extension(big: Dict[str, Any], rows_by_section: Dict[str, List[Dict[str, Any]]],
                    provenance_by_row: Optional[Dict[str, List[List[Dict[str, Any]]]]] = None,
                    sources: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a copy of ``big`` with incoming rows merged into the extension.

    ``rows_by_section`` maps a section path (``entities.compounds``, ``processes.reactions``,
    ``biological_states`` ...) to rows in the base shape; ``provenance_by_row`` maps the same path to
    one record list per row. A row matching an existing element (base or extension) by identity is
    not added: its records are appended to that element. Identity: entities by a shared grounded id,
    else a normalized name or synonym (same section) unless both rows carry different ids in one
    namespace; reactions by the identity sets of inputs and outputs without currency metabolites,
    either orientation (note "orientation reversed"), and the same normalized state when both rows
    have one; other rows by their signature. Entity names an incoming row references (participants,
    actors, complex components, species, state parts) are rewritten to the matched row's name, so a
    view stays reference-closed. Hops are recomputed.
    """
    big = copy.deepcopy(big)
    for sid, entry in (sources or {}).items():
        big["sources"].setdefault(sid, entry)
    index = _Index(big)
    ext = big.setdefault("extension", {})
    for section in sorted(rows_by_section, key=_rank):
        recs_list = (provenance_by_row or {}).get(section) or []
        for i, row in enumerate(rows_by_section[section]):
            recs = copy.deepcopy(recs_list[i]) if i < len(recs_list) else []
            _, _, row = refs(section, copy.deepcopy(row), rename=index.rename)
            hit, reversed_ = index.match(section, row)
            if hit:
                if reversed_:
                    for r in recs:
                        r["note"] = (r["note"] + "; " if r.get("note") else "") + "orientation reversed"
                big["provenance"].setdefault(hit, []).extend(recs)
                if section.startswith("entities."):
                    index.alias(row, hit, index.key_name.get(hit))
                continue
            section_rows(ext, section, create=True).append(row)
            big["provenance"][index.add_new(section, row)] = recs
    big["hops"] = compute_hops(big, index)
    return big


def compute_hops(big: Dict[str, Any], index: Optional[_Index] = None) -> Dict[str, Optional[int]]:
    """Base rows 0. Edges run only through shared GROUNDED, non-currency entities (PLAN 2.1; ATP or
    water would otherwise make everything hop 1). An extension process is 1 + the smallest hop among
    its edge entities; an extension entity takes the smallest hop of the processes naming it; a
    location row or state takes its entity's / users' hop. Unreachable -> None (full view only)."""
    index = index or _Index(big)
    rows = list(keyed_rows(big))
    hops: Dict[str, Optional[int]] = {}
    ent_hop: Dict[str, int] = {}   # edge entity key -> hop
    ext_procs = []
    for section, key, row, is_base in rows:
        if is_base:
            hops[key] = 0
            if section.startswith("entities."):
                ent_hop[key] = 0
        elif section.startswith("processes."):
            names = refs(section, row)[0]
            ext_procs.append((key, {e for e in map(index.edge, names) if e}, {index.ident(n) for n in names}))
    changed = True
    while changed:  # ponytail: relax-until-stable, O(rounds * processes); BFS if overlays get large
        changed = False
        for key, edges, _ in ext_procs:
            near = [ent_hop[e] for e in edges if e in ent_hop]
            if near and (hops.get(key) is None or min(near) + 1 < hops[key]):
                hops[key] = min(near) + 1
                changed = True
            if hops.get(key) is not None:
                for e in edges:
                    if e not in ent_hop or hops[key] < ent_hop[e]:
                        ent_hop[e] = hops[key]
                        changed = True
    member_hop = dict(ent_hop)  # any entity a reached process names, edge or not
    for key, _, members in ext_procs:
        if hops.get(key) is not None:
            for e in members:
                member_hop[e] = min(member_hop.get(e, hops[key]), hops[key])

    def ext(h):  # an extension row is never hop 0, even when it touches the base directly
        return None if h is None else max(h, 1)

    state_hop: Dict[str, int] = {}
    for section, key, row, is_base in rows:
        if is_base:
            continue
        if section.startswith("entities."):
            hops[key] = ext(member_hop.get(key))
        elif section.startswith("element_locations."):
            ents = refs(section, row)[0]
            hops[key] = ext(member_hop.get(index.ident(ents[0]))) if ents else None
        elif section.startswith("processes."):
            hops.setdefault(key, None)
        if section != STATES and hops.get(key) is not None:
            for s in refs(section, row)[1]:
                state_hop[norm(s)] = min(state_hop.get(norm(s), hops[key]), hops[key])
    for section, key, row, is_base in rows:
        if not is_base and section == STATES:
            hops[key] = state_hop.get(norm(row.get("name") if isinstance(row, dict) else None))
    return hops
