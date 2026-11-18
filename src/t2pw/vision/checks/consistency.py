"""Cross-format consistency (development only): parse PWML, SBML, SBGN and KGML back and compare what they say.

    cross_format({"pwml": bytes, "sbml": bytes, "sbgn": bytes, "kgml": bytes}) -> report

Every format is read into one small model: entities (names, database ids, compartments used), processes
(reactions and transports: participants on each side, catalysts), interactions (entity to entity), compartments.
Entities of different formats are the same entity when they share a database id (chebi, kegg, hmdb, pubchem, cas,
drugbank, uniprot) or the same normalized name (union-find across all formats). A process is its multiset of
(input entities, output entities); compartments are compared as the grouping of entities (names are not trusted).

Known, legitimate differences get their own entry under ``known`` and are not failures:
  a  PWML writes canonical (mapped) names, SBML / SBGN / KGML write the stored payload names; also compartment names
  b  SBGN has no entity-to-entity interactions and no non-regulatory ones (the V2 omission report): interactions are
     not compared for SBGN
  c  KGML has no compartments (and no catalysts): neither is compared for KGML
Everything else that differs is a failure, listed by class with examples. lxml is imported inside the functions.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from t2pw.vision.payload import NO_ID, norm

ORDER = ("pwml", "sbml", "sbgn", "kgml")
STRONG = {"chebi", "kegg", "hmdb", "pubchem", "cas", "drugbank", "uniprot"}
_ALIAS = {"kegg.compound": "kegg", "cpd": "kegg", "pubchem.compound": "pubchem", "pubchem_cid": "pubchem",
          "chebi_id": "chebi", "kegg_id": "kegg", "hmdb_id": "hmdb", "drugbank_id": "drugbank", "uniprot_id": "uniprot"}
PWML_TYPES = {"Compound": "compounds", "Protein": "proteins", "ProteinComplex": "protein-complexes",
              "NucleicAcid": "nucleic-acids", "ElementCollection": "element-collections"}
_PWML_ID_TAGS = {"chebi-id": "chebi", "kegg-id": "kegg", "hmdb-id": "hmdb", "pubchem-cid": "pubchem", "cas": "cas",
                 "drugbank-id": "drugbank", "uniprot-id": "uniprot"}
_LOC_TAGS = {"compound-locations": ("compound-id", "Compound"), "protein-locations": ("protein-id", "Protein"),
             "protein-complex-visualizations": ("protein-complex-id", "ProteinComplex"),
             "nucleic-acid-locations": ("nucleic-acid-id", "NucleicAcid"),
             "element-collection-locations": ("element-collection-id", "ElementCollection")}
_REF_TAGS = {"compound-id": "Compound", "protein-id": "Protein", "protein-complex-id": "ProteinComplex",
             "nucleic-acid-id": "NucleicAcid", "element-collection-id": "ElementCollection"}


# ---------------------------------------------------------------- model

class Model:
    def __init__(self, fmt: str):
        self.fmt = fmt
        self.ents: List[Dict[str, Any]] = []      # {name, ids:set, comps:set}
        self.procs: List[Dict[str, Any]] = []     # {name, ins:[ent], outs:[ent], cats:[ent]}
        self.ints: List[Tuple[int, int, str]] = []
        self.derived: List[Tuple[int, int]] = []   # SBML: (non-catalyst modifier, catalyst of the same reaction)
        self.wrappers: List[Tuple[int, int]] = []  # PWML: (single-member complex, its protein)
        self.regs: List[Tuple[int, int, str]] = []  # SBGN arcs / SBML non-catalyst modifiers: (entity, process index, class)
        self.comp_go: Dict[str, Set[str]] = {}      # compartment name -> GO ids it carries (PWML ontology-id, SBML annotation)
        self.has_comps = True
        self.problems: List[str] = []

    def ent(self, name: Any, ids=(), comp: Optional[str] = None, kind: str = "") -> int:
        self.ents.append({"name": str(name or "").strip(), "ids": set(ids), "comps": {comp} if comp else set(), "kind": kind})
        return len(self.ents) - 1


def nid(ns: str, value: Any) -> Optional[Tuple[str, str]]:
    """Normalized (namespace, id), None for an empty / placeholder / non-strong id."""
    ns = _ALIAS.get(str(ns).strip().lower(), str(ns).strip().lower())
    v = str(value or "").strip()
    if ns not in STRONG or v.lower() in NO_ID:
        return None
    v = v.lower()
    if ns in ("chebi", "hmdb", "pubchem"):
        v = re.sub(r"^(chebi:|hmdb)?0*", "", v)
    return ns, v.upper() if ns in ("kegg", "uniprot", "drugbank") else v


def _ids(pairs) -> Set[Tuple[str, str]]:
    return {i for i in (nid(a, b) for a, b in pairs) if i}


def _local(el) -> str:
    t = el.tag
    return t.split("}", 1)[1] if isinstance(t, str) and "}" in t else t


def _kids(el, name: str):
    return [c for c in el if _local(c) == name]


def _text(el, name: str) -> str:
    for c in el:
        if _local(c) == name:
            return (c.text or "").strip()
    return ""


# ---------------------------------------------------------------- PWML

def parse_pwml(data: bytes) -> Model:
    from lxml import etree
    m = Model("pwml")
    root = etree.fromstring(data)
    pv = next(iter(root.iter("pathway-visualization")), None)
    if pv is None:
        m.problems.append("no pathway-visualization")
        return m
    sect = lambda tag: [c for s in pv if _local(s) == tag for c in s]  # noqa: E731
    loc = {_text(e, "id"): _text(e, "name") for e in sect("subcellular-locations")}
    for e in sect("subcellular-locations"):
        if _text(e, "ontology-id").upper().startswith("GO:"):
            m.comp_go.setdefault(_text(e, "name"), set()).add(_text(e, "ontology-id").upper())
    state = {_text(e, "id"): loc.get(_text(e, "subcellular-location-id"), "") for e in sect("biological-states")}
    idx: Dict[Tuple[str, str], int] = {}
    for typ, tag in PWML_TYPES.items():
        for e in sect(tag):
            ids = [(_PWML_ID_TAGS[_local(c)], c.text) for c in e if _local(c) in _PWML_ID_TAGS and (c.text or "").strip()]
            idx[(typ, _text(e, "id"))] = m.ent(_text(e, "name"), _ids(ids), kind=typ)
    for e in sect("protein-complexes"):
        members = [idx.get(("Protein", (c.text or "").strip())) for x in e if _local(x) == "protein_complex-proteins"
                   for y in x for c in y if _local(c) == "protein-id"]
        c_i, own = idx[("ProteinComplex", _text(e, "id"))], m.ents[idx[("ProteinComplex", _text(e, "id"))]]["name"]
        if len(members) == 1 and members[0] is not None and norm(own) == norm(m.ents[members[0]]["name"] + " complex"):
            m.wrappers.append((c_i, members[0]))
    for tag, (ref, typ) in _LOC_TAGS.items():
        for e in sect(tag):
            i = idx.get((typ, _text(e, ref)))
            if i is None:
                m.problems.append("%s names unknown %s %s" % (tag, typ, _text(e, ref)))
                continue
            if state.get(_text(e, "biological-state-id")):
                m.ents[i]["comps"].add(state[_text(e, "biological-state-id")])

    def side(parent, tag):
        out = []
        for e in [c for p in parent if _local(p) == tag for c in p]:
            i = idx.get((_text(e, "element-type"), _text(e, "element-id")))
            if i is None:
                m.problems.append("unresolved %s %s" % (_text(e, "element-type"), _text(e, "element-id")))
            else:
                out.append(i)
        return out

    def actors(parent, tag):
        out = []
        for e in [c for p in parent if _local(p) == tag for c in p]:
            for c in e:
                if _local(c) in _REF_TAGS and (c.text or "").strip():
                    i = idx.get((_REF_TAGS[_local(c)], c.text.strip()))
                    if i is None:
                        m.problems.append("unresolved actor %s" % c.text)
                    else:
                        out.append(i)
        return out

    for r in sect("reactions"):
        m.procs.append({"name": _text(r, "id"), "ins": side(r, "reaction-left-elements"),
                        "outs": side(r, "reaction-right-elements"), "cats": actors(r, "reaction-enzymes")})
    for t in sect("transports"):
        cargo = side(t, "transport-elements")
        m.procs.append({"name": _text(t, "id"), "ins": cargo, "outs": cargo, "cats": actors(t, "transport-transporters")})
    for x in sect("interactions"):
        for a in side(x, "interaction-left-elements"):
            for b in side(x, "interaction-right-elements"):
                m.ints.append((a, b, _text(x, "interaction-type")))
    return m


# ---------------------------------------------------------------- SBML

def parse_sbml(data: bytes) -> Model:
    from lxml import etree
    m = Model("sbml")
    root = etree.fromstring(data)
    model = next((c for c in root if _local(c) == "model"), None)
    if model is None:
        m.problems.append("no model")
        return m
    comp_name = {c.get("id"): (c.get("name") or c.get("id")) for s in _kids(model, "listOfCompartments") for c in s}
    for s in _kids(model, "listOfCompartments"):
        for c in s:
            for li in c.iter():
                go = re.search(r"/(GO:\d+)$", li.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource") or "", re.I) if _local(li) == "li" else None
                if go:
                    m.comp_go.setdefault(comp_name[c.get("id")], set()).add(go.group(1).upper())
    sp: Dict[str, int] = {}
    by_name: Dict[str, int] = {}
    for s in [c for l in _kids(model, "listOfSpecies") for c in l]:
        ids = [(m_.group(1), m_.group(2)) for li in s.iter() if _local(li) == "li"
               for m_ in [re.search(r"identifiers\.org/([^/]+)/(?:CHEBI:)?([^/]+)$", li.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource") or "")] if m_]
        i = m.ent(s.get("name") or s.get("id"), _ids(ids), comp_name.get(s.get("compartment"), s.get("compartment")))
        sp[s.get("id")] = i
        by_name.setdefault(norm(s.get("name") or ""), i)
    for rx in [c for l in _kids(model, "listOfReactions") for c in l]:
        refs = lambda tag: [sp[c.get("species")] for l in _kids(rx, tag) for c in l if c.get("species") in sp]  # noqa: E731
        mods = [c for l in _kids(rx, "listOfModifiers") for c in l if c.get("species") in sp]
        cats = [sp[c.get("species")] for c in mods if c.get("sboTerm") == "SBO:0000013"]
        m.derived += [(sp[c.get("species")], k) for c in mods if c.get("sboTerm") != "SBO:0000013" for k in cats]
        m.regs += [(sp[c.get("species")], len(m.procs), c.get("sboTerm") or "") for c in mods if c.get("sboTerm") != "SBO:0000013"]
        m.procs.append({"name": rx.get("name") or rx.get("id"), "ins": refs("listOfReactants"), "outs": refs("listOfProducts"), "cats": cats})
    for it in root.iter():
        if _local(it) == "interaction" and it.get("sourceName") is not None:
            a, b = by_name.get(norm(it.get("sourceName"))), by_name.get(norm(it.get("targetName")))
            if a is None or b is None:
                m.problems.append("interaction endpoint unresolved: %s -> %s" % (it.get("sourceName"), it.get("targetName")))
            else:
                m.ints.append((a, b, it.get("kind") or ""))
    return m


# ---------------------------------------------------------------- SBGN

def parse_sbgn(data: bytes) -> Model:
    from lxml import etree
    m = Model("sbgn")
    root = etree.fromstring(data)
    mp = next((c for c in root if _local(c) == "map"), None)
    if mp is None:
        m.problems.append("no map")
        return m
    comp = {g.get("id"): (next((_c.get("text") for _c in g if _local(_c) == "label"), "")) for g in _kids(mp, "glyph")
            if g.get("class") == "compartment"}
    node: Dict[str, int] = {}   # glyph id -> entity
    proc: Dict[str, int] = {}   # process glyph id or its port id -> process index

    def entity(g, parent_comp):
        label = next((c.get("text") for c in g if _local(c) == "label"), "")
        ids = [(k, v) for e in g if _local(e) == "extension" for x in e if _local(x) == "ids" for k, v in x.attrib.items()]
        c = comp.get(g.get("compartmentRef")) or parent_comp
        node[g.get("id")] = i = m.ent(label, _ids(ids), c)
        for sub in _kids(g, "glyph"):  # complex members sit in the complex's compartment
            entity(sub, c)

    for g in _kids(mp, "glyph"):
        cls = g.get("class")
        if cls == "compartment":
            continue
        if cls == "process":
            proc[g.get("id")] = len(m.procs)
            for p in _kids(g, "port"):
                proc[p.get("id")] = len(m.procs)
            nm = next((x.get("name") for e in g if _local(e) == "extension" for x in e if _local(x) == "process"), "")
            m.procs.append({"name": nm, "ins": [], "outs": [], "cats": []})
        else:
            entity(g, None)
    for a in _kids(mp, "arc"):
        cls, s, t = a.get("class"), a.get("source"), a.get("target")
        if cls == "consumption" and s in node and t in proc:
            m.procs[proc[t]]["ins"].append(node[s])
        elif cls == "production" and s in proc and t in node:
            m.procs[proc[s]]["outs"].append(node[t])
        elif cls == "catalysis" and s in node and t in proc:
            m.procs[proc[t]]["cats"].append(node[s])
        elif cls in ("stimulation", "inhibition", "modulation") and s in node and t in proc:
            m.regs.append((node[s], proc[t], cls))
    return m


# ---------------------------------------------------------------- KGML

def parse_kgml(data: bytes) -> Model:
    from lxml import etree
    m = Model("kgml")
    m.has_comps = False
    root = etree.fromstring(data)
    ent: Dict[str, int] = {}
    for e in _kids(root, "entry"):
        if e.get("type") == "reaction":
            continue
        gr = next((c for c in e if _local(c) == "graphics"), None)
        ids = [tuple(tok.split(":", 1)) for tok in (e.get("name") or "").split() if ":" in tok]
        ent[e.get("id")] = m.ent(gr.get("name") if gr is not None else "", _ids(ids))
    for r in _kids(root, "reaction"):
        refs = lambda tag: [ent[c.get("id")] for c in _kids(r, tag) if c.get("id") in ent]  # noqa: E731
        m.procs.append({"name": r.get("id"), "ins": refs("substrate"), "outs": refs("product"), "cats": []})
    for rel in _kids(root, "relation"):
        if rel.get("entry1") in ent and rel.get("entry2") in ent:
            sub = next((c.get("name") for c in rel if _local(c) == "subtype"), "")
            m.ints.append((ent[rel.get("entry1")], ent[rel.get("entry2")], sub))
    return m


PARSERS = {"pwml": parse_pwml, "sbml": parse_sbml, "sbgn": parse_sbgn, "kgml": parse_kgml}


# ---------------------------------------------------------------- comparison

class _UF:
    def __init__(self):
        self.p: Dict[Any, Any] = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


UNKNOWN = "unknown"                                   # the PathBank "Unknown" sentinel protein (clean_model/builder.py)
UNSPECIFIED = frozenset(norm(x) for x in ("unspecified location", "unspecified"))  # where SBML / SBGN put an entity with no stated location


def _is_sentinel(fmt: str, e: Dict[str, Any]) -> bool:
    """The PathBank Unknown protein: in PWML only a Protein-type entity of that name (a compound called Unknown is a
    compound); in the other formats any entity of that name (SBML and KGML must not have it, SBGN draws it)."""
    return e["name"].strip().casefold() == UNKNOWN and (fmt != "pwml" or e.get("kind") == "Protein")


def _clusters(models: Dict[str, Model]) -> Dict[Tuple[str, int], Any]:
    """(fmt, entity index) -> cluster id: shared strong id or shared normalized name. A PWML single-member complex
    ("<protein> complex") is the same entity as its protein. The Unknown sentinel has no cluster."""
    uf, owner = _UF(), {}
    for fmt, m in models.items():
        for i, e in enumerate(m.ents):
            if _is_sentinel(fmt, e):
                continue
            node = (fmt, i)
            uf.find(node)
            for tok in [("id",) + t for t in e["ids"]] + ([("name", norm(e["name"]))] if norm(e["name"]) else []):
                if tok in owner:
                    uf.union(node, owner[tok])
                else:
                    owner[tok] = node
        for c, p in m.wrappers:
            uf.union((fmt, c), (fmt, p))
    return {(f, i): uf.find((f, i)) for f, m in models.items() for i, e in enumerate(m.ents) if not _is_sentinel(f, e)}


def _add(fails: Dict[str, Dict[str, Any]], cls: str, example: str, n: int = 1) -> None:
    f = fails.setdefault(cls, {"class": cls, "n": 0, "examples": []})
    f["n"] += n
    if len(f["examples"]) < 5:
        f["examples"].append(example)


def cross_format(outputs: Dict[str, bytes], *, pwml_ir: Optional[dict] = None, sbgn_report: Optional[dict] = None) -> Dict[str, Any]:
    """Compare the formats in ``outputs`` (keys among pwml, sbml, sbgn, kgml; others are ignored).

    ``pwml_ir`` (the IR the PWML export returned) lets the check tell an interaction PWML withheld on purpose (its own
    ``withheld`` record) from one it lost; ``sbgn_report`` is the SBGN exporter's own report, echoed under ``known``."""
    models: Dict[str, Model] = {}
    fails: Dict[str, Dict[str, Any]] = {}
    for fmt in ORDER:
        if fmt in outputs:
            try:
                models[fmt] = PARSERS[fmt](outputs[fmt])
            except Exception as e:  # a file that does not parse is a failure of that format, not a crash
                _add(fails, "parse_error:" + fmt, "%s: %s" % (type(e).__name__, str(e)[:200]))
    for fmt, m in models.items():
        for p in m.problems:
            _add(fails, "unresolved_reference:" + fmt, p)
    cl = _clusters(models)
    label: Dict[Any, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))  # cluster -> fmt -> names
    for (f, i), c in cl.items():
        label[c][f].add(models[f].ents[i]["name"])
    name_of = lambda c: next((n for f in ORDER for n in sorted(label[c].get(f, ())) if n), "?")  # noqa: E731
    present = {f: {cl[(f, i)] for i in range(len(m.ents)) if (f, i) in cl} for f, m in models.items()}
    everything = set().union(*present.values()) if present else set()
    cs = lambda f, idxs: frozenset(cl[(f, i)] for i in idxs if (f, i) in cl)  # noqa: E731  (the sentinel has no cluster)
    in_proc = {c for f, m in models.items() for p in m.procs for c in cs(f, p["ins"] + p["outs"] + p["cats"])}
    in_int = {c for f, m in models.items() for a, b, _ in m.ints for c in cs(f, (a, b))}
    known: Dict[str, Any] = {
        "d_pathwhiz_complex_wrapper": {"n": sum(len(m.wrappers) for m in models.values()),
                                       "note": "PWML single-member complex '<protein> complex' compared as its protein (Stage 6 wrapper: PathWhiz has no bare-protein enzyme or interaction member)"},
        "f_unknown_sentinel": {"n": sum(_is_sentinel(f, e) for f, m in models.items() for e in m.ents),
                               "per_format": {f: sum(_is_sentinel(f, e) for e in m.ents) for f, m in models.items()},
                               "note": "the PathBank Unknown protein is a placeholder: PWML (Protein type) and SBGN carry it, the clean model (SBML, KGML) must not"}}
    for f in ("sbml", "kgml"):
        if f in models:
            for e in models[f].ents:
                if _is_sentinel(f, e):
                    _add(fails, "unknown_sentinel_in_" + f, e["name"])

    # entities: the same set in every format
    only_int = []
    for f, have in present.items():
        for c in sorted(everything - have, key=name_of):
            if f == "sbgn" and c not in in_proc and c in in_int:  # (b) SBGN draws no entity that only appears in an interaction
                only_int.append(name_of(c))
            else:
                _add(fails, "entity_missing_in_" + f, name_of(c))
    known["b_sbgn_entities_only_in_interactions"] = {"n": len(only_int), "examples": only_int[:5]}

    # (a) names: PWML canonical vs payload-format names; the payload formats must agree with each other
    pay = [f for f in ("sbml", "sbgn", "kgml") if f in models]
    canon: List[str] = []
    for c, by in label.items():
        keys = {f: {norm(n) for n in ns if n} for f, ns in by.items()}
        have = [f for f in pay if keys.get(f)]
        for f in have[1:]:
            if not keys[have[0]] & keys[f]:
                _add(fails, "name_differs_between_payload_formats", "%s=%s vs %s=%s" % (have[0], sorted(by[have[0]]), f, sorted(by[f])))
        if keys.get("pwml") and have and not any(keys["pwml"] & keys[f] for f in have):
            canon.append("%s -> %s" % (sorted(by["pwml"])[0], sorted(by[have[0]])[0]))
    known["a_canonical_names"] = {"n": len(canon), "examples": canon[:5],
                                  "note": "PWML canonical (mapped) name differs from the stored payload name; same entity by id"}

    # processes: the same multiset of (inputs, outputs) in every format
    def sigs(f):
        return Counter((cs(f, p["ins"]), cs(f, p["outs"])) for p in models[f].procs)
    show = lambda s: "{%s} -> {%s}" % (", ".join(sorted(name_of(c) for c in s[0])), ", ".join(sorted(name_of(c) for c in s[1])))  # noqa: E731
    ref = next(iter(models), None)
    if ref:
        rs = sigs(ref)
        for f in models:
            if f == ref:
                continue
            fs = sigs(f)
            for s, n in (rs - fs).items():
                _add(fails, "process_missing_in_" + f, show(s), n)
            for s, n in (fs - rs).items():
                _add(fails, "process_extra_in_" + f, show(s), n)

    # catalysts (not in KGML: no slot): for processes both formats have once
    def catalysts(f):
        d = defaultdict(list)
        for p in models[f].procs:
            d[(cs(f, p["ins"]), cs(f, p["outs"]))].append(cs(f, p["cats"]))
        return d
    cat_fmts = [f for f in ("pwml", "sbml", "sbgn") if f in models]
    if len(cat_fmts) > 1:
        base = catalysts(cat_fmts[0])
        for f in cat_fmts[1:]:
            other = catalysts(f)
            for s in base.keys() & other.keys():
                if sorted(map(sorted, base[s])) != sorted(map(sorted, other[s])):
                    _add(fails, "catalyst_mismatch:%s_vs_%s" % (cat_fmts[0], f),
                         "%s: %s=%s %s=%s" % (show(s), cat_fmts[0], [sorted(name_of(c) for c in x) for x in base[s]], f, [sorted(name_of(c) for c in x) for x in other[s]]))

    # interactions (not in SBGN: class b)
    def pairs(f, ints):
        return {(cl[(f, a)], cl[(f, b)]) for a, b, *_ in ints if (f, a) in cl and (f, b) in cl}
    P = pairs("pwml", models["pwml"].ints) if "pwml" in models else set()
    Es = pairs("sbml", models["sbml"].ints) if "sbml" in models else set()
    Ds = pairs("sbml", models["sbml"].derived) if "sbml" in models else set()
    K = pairs("kgml", models["kgml"].ints) if "kgml" in models else set()
    shown = lambda a, b: "%s -> %s" % (name_of(a), name_of(b))  # noqa: E731
    by_norm = {norm(n): c for c, by in label.items() for ns in by.values() for n in ns if norm(n)}
    withheld = {}
    for it in ((pwml_ir or {}).get("processes") or {}).get("interactions") or []:
        w = it.get("withheld") if isinstance(it, dict) and it.get("export_status") == "withheld" else None
        if w and norm(w.get("left")) in by_norm and norm(w.get("right")) in by_norm:
            withheld[(by_norm[norm(w["left"])], by_norm[norm(w["right"])])] = w.get("reason")
    kept_back, modifier = [], []
    if "pwml" in models and "sbml" in models:
        for a, b in sorted(Es - P, key=str):
            if (a, b) in withheld:  # PWML's own record of an interaction it cannot store
                kept_back.append("%s (%s)" % (shown(a, b), withheld[(a, b)]))
            else:
                _add(fails, "interaction_missing_in_pwml", shown(a, b))
        for a, b in sorted(P - Es, key=str):
            if (a, b) in Ds:  # SBML lists a reaction's regulator as a modifier; PWML draws it as an interaction with the enzyme
                modifier.append(shown(a, b))
            else:
                _add(fails, "interaction_extra_in_pwml", shown(a, b))
    if "sbml" in models and "kgml" in models:
        for a, b in sorted(Es - K, key=str):
            _add(fails, "interaction_missing_in_kgml", shown(a, b))
        for a, b in sorted(K - Es, key=str):
            _add(fails, "interaction_extra_in_kgml", shown(a, b))
    known["e_pwml_withheld_interactions"] = {"n": len(kept_back), "examples": kept_back[:5],
                                             "note": "PWML stores only some interaction types and no bare-protein endpoint; ir.processes.interactions[].withheld says which it held back"}
    known["g_pwml_modifier_interactions"] = {"n": len(modifier), "examples": modifier[:5],
                                             "note": "a reaction's regulator is a modifier in SBML and an interaction with the enzyme complex in PWML"}
    known["b_sbgn_interactions_not_compared"] = {
        "n": len(P | Es | K) if "sbgn" in models else 0, "sbgn_report_omitted": (sbgn_report or {}).get("omitted"),
        "note": "SBGN PD has no entity-to-entity interaction and the exporter omits non-regulatory ones (V2 / VD-8)"}

    # regulatory arcs: SBGN stimulation / inhibition / modulation against what SBML says. SBML's non-catalyst modifier of a
    # reaction must be an SBGN arc on that process; an SBGN arc must be a modifier, or come from an interaction whose target
    # catalyses the process (V2: interactions resolve to the processes the target catalyses). Class b covers only what the
    # SBGN exporter's own report lists as omitted reaction arcs.
    if "sbml" in models and "sbgn" in models:
        def regset(f):
            sig = lambda p: (cs(f, models[f].procs[p]["ins"]), cs(f, models[f].procs[p]["outs"]))  # noqa: E731
            return {(cl[(f, e)], sig(p)) for e, p, _ in models[f].regs if (f, e) in cl}
        Rs, Rg = regset("sbml"), regset("sbgn")
        cats_of = defaultdict(set)
        for s_, lists in catalysts("sbgn").items():
            cats_of[s_] = set().union(*lists)
        from t2pw.vision.export_sbgn import _rel_class  # the SBGN exporter's closed word list (VD-8): binding, component_of, ... are not regulation
        reg_pairs = lambda f: {(cl[(f, a)], cl[(f, b)]) for a, b, t in models[f].ints  # noqa: E731
                               if (f, a) in cl and (f, b) in cl and _rel_class(t)} if f in models else set()
        pair_set = reg_pairs("pwml") | reg_pairs("sbml") | Ds  # only a regulatory interaction explains an SBGN arc
        missing = sorted(Rs - Rg, key=str)
        allowed = sum(((sbgn_report or {}).get("omitted") or {}).get("reaction_arcs", {}).values())
        for x, s_ in missing[allowed:]:
            _add(fails, "sbgn_regulatory_arc_missing", "%s -> %s" % (name_of(x), show(s_)))
        for x, s_ in sorted(Rg - Rs, key=str):
            if not any((x, k) in pair_set for k in cats_of.get(s_, ())):
                _add(fails, "sbgn_regulatory_arc_extra", "%s -> %s" % (name_of(x), show(s_)))
        known["b_sbgn_omitted_regulatory_arcs"] = {"n": min(len(missing), allowed), "report_allows": allowed,
                                                   "note": "SBML modifiers the SBGN exporter reports as omitted reaction arcs (role_not_regulatory, unresolved_entity)"}

    # compartments: the same grouping of entities (names are not trusted); (c) KGML has none
    def groups(f):  # compartment -> entities placed there; "unspecified" placements are not placements
        g = defaultdict(set)
        for i, e in enumerate(models[f].ents):
            for c in e["comps"]:
                if (f, i) in cl and norm(c) not in UNSPECIFIED:
                    g[c].add(cl[(f, i)])
        return g
    cfmts = [f for f in ("pwml", "sbml", "sbgn") if f in models]
    grp = {f: groups(f) for f in cfmts}
    placed = {f: set().union(*grp[f].values()) if grp[f] else set() for f in cfmts}
    known["f_unplaced_entities"] = {"n": sum(1 for f in cfmts for m in [models[f]] for i, e in enumerate(m.ents)
                                             if (f, i) in cl and e["comps"] and all(norm(c) in UNSPECIFIED for c in e["comps"])),
                                    "note": "entities the payload gives no location are in 'unspecified location' (SBML) and are not compared; PWML and SBGN place them"}
    names_a: Counter = Counter()
    extra_pw = 0
    # an entity placed in a payload format and present in PWML must be placed in PWML too (PWML places every entity it draws)
    for f in ("sbml", "sbgn"):
        if "pwml" in models and f in models:
            for c in sorted((placed[f] & present["pwml"]) - placed["pwml"], key=name_of):
                _add(fails, "compartment_placement_missing_in_pwml", "%s (placed in %s)" % (name_of(c), f))
    # between the payload formats: the same entities placed; SBGN assumes a place for an entity SBML leaves unplaced (class f)
    if "sbml" in models and "sbgn" in models:
        both = present["sbml"] & present["sbgn"]
        unplaced_sbml = {cl[("sbml", i)] for i, e in enumerate(models["sbml"].ents)
                         if ("sbml", i) in cl and e["comps"] and all(norm(c) in UNSPECIFIED for c in e["comps"])}
        for c in sorted((placed["sbml"] & both) - placed["sbgn"], key=name_of):
            _add(fails, "compartment_placement_missing_in_sbgn", name_of(c))
        for c in sorted(((placed["sbgn"] & both) - placed["sbml"]) - unplaced_sbml, key=name_of):
            _add(fails, "compartment_placement_missing_in_sbml", name_of(c))
    # PWML vs a payload format: every placement the payload format has must lie inside one PWML compartment (PWML merges
    # compartments under canonical names, and places the ends of the interactions and transports it draws, which the
    # payload formats leave where the payload put them)
    # Only entities with exactly one placement in both formats are grouped: SBML lists an entity in every compartment the
    # payload's states give it, PWML and SBGN draw it only where a process uses it (V7a proof, 3 runs diagnosed).
    pl = {f: defaultdict(set) for f in cfmts}  # cluster -> compartment names as written
    for f in cfmts:
        for c, v in grp[f].items():
            for cluster in v:
                pl[f][cluster].add(c)
    k_pwml = 0  # placements SBML lists and PWML does not draw
    no_overlap: List[str] = []  # diagnostic: entities with several placements where no PWML placement maps to any SBML one
    for f in ("sbml", "sbgn"):
        if "pwml" in models and f in models:
            both = placed["pwml"] & placed[f]
            anchored = {c for c in both if len(pl["pwml"][c]) == 1 and len(pl[f][c]) == 1}
            gp, gf = defaultdict(set), defaultdict(set)
            for c in anchored:
                gp[next(iter(pl["pwml"][c]))].add(c)
                gf[next(iter(pl[f][c]))].add(c)
            if f == "sbml":  # PWML compartment -> the SBML names the anchored entities show it stands for
                stands_for = {cp: {norm(next(iter(pl[f][c]))) for c in block} for cp, block in gp.items()}
                for c in sorted(both - anchored, key=name_of):
                    s = {norm(x) for x in pl[f][c]}
                    if not any(norm(p) in s or stands_for.get(p, set()) & s for p in pl["pwml"][c]):
                        no_overlap.append("%s: pwml=%s sbml=%s" % (name_of(c), sorted(pl["pwml"][c]), sorted(pl[f][c])))
            for c in both:
                d = len(pl["pwml"][c]) - len(pl[f][c])
                extra_pw += max(d, 0)
                k_pwml += max(-d, 0) if f == "sbml" else 0
            for cf, block in gf.items():
                hit = [cp for cp, pset in gp.items() if block <= pset]
                if not hit:
                    _add(fails, "compartment_grouping_differs:pwml_vs_" + f, "%s '%s' (%d entities) is not inside one PWML compartment: %s" % (
                        f, cf, len(block), sorted(name_of(c) for c in block)[:4]))
                else:
                    named = [h for h in hit if norm(h) == norm(cf)]
                    cp = named[0] if named else hit[0]
                    if not named:
                        names_a["pwml=%s %s=%s" % (cp, f, cf)] += 1
                    go_p, go_f = models["pwml"].comp_go.get(cp), models[f].comp_go.get(cf)
                    if go_p and go_f and not go_p & go_f:  # both carry an ontology term and they disagree
                        _add(fails, "compartment_ontology_differs:pwml_vs_" + f, "pwml %s %s vs %s %s %s" % (cp, sorted(go_p), f, cf, sorted(go_f)))
    # the two payload formats (same names): an entity with one placement in each must have the same one; SBML-only placements
    # are known (SBGN draws one glyph per (entity, compartment) a process uses, export_sbgn docstring); placements only SBGN
    # has are bounded by the exporter's own compartment_assumed notes (transporter, interaction source, reaction state from IR)
    k_sbgn = 0
    if "sbml" in models and "sbgn" in models:
        allowed = sum(v for k, v in ((sbgn_report or {}).get("notes") or {}).items() if str(k).startswith("compartment_assumed"))
        dev = []
        for c in sorted(placed["sbml"] & placed["sbgn"], key=name_of):
            s, g = {norm(x): x for x in pl["sbml"][c]}, {norm(x): x for x in pl["sbgn"][c]}
            if len(s) == 1 and len(g) == 1:
                if set(s) != set(g):
                    dev.append("%s: sbml=%s sbgn=%s" % (name_of(c), next(iter(s.values())), next(iter(g.values()))))
            else:
                k_sbgn += len(set(s) - set(g))
                dev += ["%s: sbgn only %s" % (name_of(c), g[n]) for n in sorted(set(g) - set(s))]
        for ex in dev[allowed:]:
            _add(fails, "compartment_placement_differs:sbml_vs_sbgn", ex)
        known["j_sbgn_assumed_placements"] = {"n": min(len(dev), allowed), "report_allows": allowed, "examples": dev[:5],
                                              "note": "placements SBGN has that SBML does not, up to the exporter's compartment_assumed notes"}
    known["m_nonanchored_no_placement_overlap"] = {"n": len(no_overlap), "examples": no_overlap[:5],
                                                    "note": "diagnostic, not a failure: an entity with several placements in PWML or SBML where no PWML placement maps to any SBML one"}
    known["k_sbml_only_placements"] = {"n": k_pwml + k_sbgn, "vs_pwml": k_pwml, "vs_sbgn": k_sbgn,
                                       "note": "compartments SBML lists for an entity that PWML / SBGN do not draw it in (no process uses it there)"}
    known["a_compartment_names"] = {"n": sum(names_a.values()), "pairs": dict(names_a), "examples": list(names_a)[:5],
                                    "note": "PWML canonical compartment name differs from the stored name; every pair listed with its count"}
    known["h_pwml_extra_placements"] = {"n": extra_pw, "note": "placements PWML adds (an interaction or transport end drawn in a state the payload does not give the entity)"}
    known["c_kgml_has_no_compartments_or_catalysts"] = "kgml" in models

    counts = {f: {"entities": len(present[f]), "processes": len(m.procs), "interactions": len(m.ints),
                  "compartments_used": len({c for e in m.ents for c in e["comps"]})} for f, m in models.items()}
    return {"formats": list(models), "counts": counts, "failures": sorted(fails.values(), key=lambda x: x["class"]),
            "known": known, "ok": not fails}
