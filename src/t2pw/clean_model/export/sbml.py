"""Clean model -> SBML Level 3 Version 2 text. Stdlib only, deterministic (same model, same bytes).

Nothing is invented: no initial amounts, no "Unknown" species, stoichiometry only when the model says the paper stated
it, `reversible` is a labelled placeholder unless the paper stated the direction. `bqbiol:is` carries only ids
labelled paper/db; inferred values, suggestions and conflicts stay in the t2pw annotation. KEGG ids appear only if
the model already holds them. Interactions have no SBML core object and are written in the model annotation.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from ..builder import _is_sentinel
from ..model import CleanModel, Field
from . import sbo as S

SBML_NS = "http://www.sbml.org/sbml/level3/version2/core"
T2PW_NS = "urn:t2pw:clean-model:1"
XHTML_NS = "http://www.w3.org/1999/xhtml"
UNSPEC_ID = "c_unspecified_location"
_BAD_XML = re.compile("[^\t\n\r -퟿-�\U00010000-\U0010ffff]")


# ---- tiny XML tree ------------------------------------------------------------------------------------------------

def _clean(s: Any) -> str:
    return _BAD_XML.sub("", str(s))


def _esc(s: Any, attr: bool = False) -> str:
    s = _clean(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return s.replace('"', "&quot;").replace("\n", "&#10;").replace("\t", "&#9;").replace("\r", "&#13;") if attr else s


class El:
    def __init__(self, tag: str, attrs: Optional[Dict[str, Any]] = None, kids: Optional[List["El"]] = None,
                 text: Optional[str] = None) -> None:
        self.tag, self.text = tag, text
        self.attrs = {k: ("true" if v is True else "false" if v is False else v)
                      for k, v in (attrs or {}).items() if v is not None}
        self.kids = [k for k in (kids or []) if k is not None]

    def add(self, *kids: Optional["El"]) -> "El":
        self.kids += [k for k in kids if k is not None]
        return self

    def render(self, depth: int = 0) -> str:
        pad = "  " * depth
        a = "".join(' %s="%s"' % (k, _esc(v, True)) for k, v in self.attrs.items())
        if not self.kids and self.text is None:
            return "%s<%s%s/>\n" % (pad, self.tag, a)
        if not self.kids:
            return "%s<%s%s>%s</%s>\n" % (pad, self.tag, a, _esc(self.text), self.tag)
        return "%s<%s%s>\n%s%s</%s>\n" % (pad, self.tag, a, "".join(k.render(depth + 1) for k in self.kids), pad, self.tag)


def notes(paras: List[str]) -> Optional[El]:
    paras = [p for p in paras if p]
    if not paras:
        return None
    return El("notes", kids=[El("body", {"xmlns": XHTML_NS}, [El("p", text=p) for p in paras])])


# ---- helpers ------------------------------------------------------------------------------------------------------

def slug(s: Any) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.casefold()).strip("_")[:40] or "x"


def _lkey(name: Any) -> str:
    return re.sub(r"\s+", " ", str(name)).strip().casefold()


def _val(f: Any) -> Any:
    return None if f is None or f.absent or f.value in (None, "") else f.value


def _str(f: Any) -> Optional[str]:
    v = _val(f)
    return v if isinstance(v, str) else None


def _prov_attrs(p: Any) -> Dict[str, Any]:
    g = (lambda k: p.get(k)) if isinstance(p, dict) else (lambda k: getattr(p, k, None))
    return {"label": g("label"), "source": g("source"), "matchType": g("match_type"), "retrieved": g("retrieved")}


# id type -> (identifiers.org prefix, normaliser returning the id or None, rdf qualifier)
def _digits(v: str) -> Optional[str]:
    m = re.fullmatch(r"(?:[A-Za-z]+[:_\s]?)?(\d+)", v.strip())
    return m.group(1) if m else None


def _rx(pattern: str):
    return lambda v: v.strip() if re.fullmatch(pattern, v.strip()) else None


_UNIPROT = r"([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})"
IDTYPES = {
    "chebi": ("chebi", lambda v: ("CHEBI:" + _digits(v)) if _digits(v) and re.match(r"(?i)^(chebi[:_]?)?\d+$", v.strip()) else None, "bqbiol:is"),
    "uniprot": ("uniprot", _rx(_UNIPROT), "bqbiol:is"),
    "ec-code": ("ec-code", _rx(r"\d+\.(\d+|-)\.(\d+|-)\.(\d+|-|n\d+)"), "bqbiol:isVersionOf"),
    "go": ("go", lambda v: v.strip().upper() if re.fullmatch(r"(?i)GO:\d{7}", v.strip()) else None, "bqbiol:is"),
    "rhea": ("rhea", lambda v: _digits(v) if re.fullmatch(r"(?i)(rhea:)?\d+", v.strip()) else None, "bqbiol:is"),
    "taxonomy": ("taxonomy", lambda v: _digits(v) if re.fullmatch(r"\d+", v.strip()) else None, "bqbiol:hasTaxon"),
    "pubchem.compound": ("pubchem.compound", lambda v: _digits(v) if re.fullmatch(r"\d+", v.strip()) else None, "bqbiol:is"),
    "hmdb": ("hmdb", _rx(r"HMDB\d{5,7}"), "bqbiol:is"),
    "cas": ("cas", _rx(r"\d{2,7}-\d\d-\d"), "bqbiol:is"),
    "drugbank": ("drugbank", _rx(r"DB\d{5}"), "bqbiol:is"),
    "kegg.compound": ("kegg.compound", _rx(r"C\d{5}"), "bqbiol:is"),
    "kegg.reaction": ("kegg.reaction", _rx(r"R\d{5}"), "bqbiol:is"),
    # KEGG gene ids and Ensembl transcript/gene ids name the gene a polypeptide is encoded by, not the polypeptide itself
    "kegg.genes": ("kegg.genes", _rx(r"[a-z]{2,5}:\S+"), "bqbiol:isEncodedBy"),
    "ensembl": ("ensembl", lambda v: v.strip().split(".")[0] if re.fullmatch(r"ENS[A-Z]*P\d+(\.\d+)?", v.strip()) else None, "bqbiol:is"),
    "ensembl.encoded": ("ensembl", lambda v: v.strip().split(".")[0] if re.fullmatch(r"ENS[A-Z]*[GT]\d+(\.\d+)?", v.strip()) else None,
                        "bqbiol:isEncodedBy"),
}
# (entity kind, field name) -> id type. Fields are payload slots on the model object and/or db_layer item fields.
FIELD_IDTYPE = {
    ("compound", "chebi_id"): "chebi", ("compound", "kegg_id"): "kegg.compound", ("compound", "hmdb_id"): "hmdb",
    ("compound", "pubchem_cid"): "pubchem.compound", ("compound", "cas"): "cas", ("compound", "drugbank_id"): "drugbank",
    ("compound", "drugbank_mapped"): "drugbank",
    ("protein", "uniprot"): "uniprot", ("protein", "ec_number"): "ec-code", ("protein", "ec_numbers"): "ec-code",
    ("protein", "taxonomy_id"): "taxonomy", ("protein", "taxon_id"): "taxonomy", ("protein", "kegg_gene"): "kegg.genes",
    ("protein", "ensembl"): "ensembl", ("protein", "ensembl_gene"): "ensembl.encoded",
    ("location", "go_id"): "go",
    ("reaction", "rhea_id"): "rhea", ("reaction", "ec_number"): "ec-code", ("reaction", "kegg_reaction_id"): "kegg.reaction",
    ("reaction", "go_mf"): "go",
}
OBJ_FIELDS = {"compound": ("chebi_id", "kegg_id", "hmdb_id", "pubchem_cid", "cas", "drugbank_id", "drugbank_mapped"),
              "protein": ("uniprot", "ec_number", "ec_numbers", "taxonomy_id"),
              "location": ("go_id",), "reaction": ("rhea_id", "ec_number", "kegg_reaction_id", "go_mf")}


class _Ids:
    """Collected ids of one object: valid ones for rdf, everything else for the t2pw provenance element."""

    def __init__(self) -> None:
        self.valid: List[Tuple[str, str, Dict[str, Any]]] = []  # (idtype, id, prov attrs)
        self.invalid: List[Tuple[str, str, Dict[str, Any]]] = []
        self.inferred: List[Tuple[str, Any, Dict[str, Any]]] = []
        self._seen: set = set()

    def add(self, idtype: str, raw: Any, prov: Any, field: str) -> None:
        label = _prov_attrs(prov)["label"]
        vals = raw if isinstance(raw, list) else [raw]
        for v in vals:
            if v in (None, ""):
                continue
            if label not in ("paper", "db"):
                self.inferred.append((field, v, _prov_attrs(prov)))
                continue
            norm = IDTYPES[idtype][1](str(v))
            key = (idtype, norm or str(v))
            if key in self._seen:
                continue
            self._seen.add(key)
            (self.valid if norm else self.invalid).append((idtype, norm or str(v), _prov_attrs(prov)))


# ---- exporter -----------------------------------------------------------------------------------------------------

class _Plain:
    stoichiometry = Field()


class _Ent:
    def __init__(self, kind: str, obj: Any, name: str, idx: int) -> None:
        self.kind, self.obj, self.name, self.idx = kind, obj, name, idx
        self.eid = ""
        self.n = 0


class _Exporter:
    def __init__(self, model: CleanModel) -> None:
        self.m = model
        self.omissions: List[Dict[str, Any]] = []
        self.sentinel_names: set = set()  # names of flagged Unknown objects: never an entity, however referenced
        self.sentinel_refs: Dict[str, int] = {}
        self.sloc = {_str(s.name): _str(s.location) for s in (model.states or []) if _str(s.name) and _str(s.location)}
        self.state_ctx = {_str(s.name): s for s in (model.states or []) if _str(s.name)}
        self.layer = {k: list((model.db_layer or {}).get(k) or []) for k in ("items", "suggestions", "conflicts", "refused")}
        self._entities()
        self._compartments()
        self.species: Dict[Tuple[int, Optional[str]], Dict[str, Any]] = {}

    # entities -------------------------------------------------------------------------------------------------
    def _entities(self) -> None:
        raw: List[_Ent] = []
        for kind, lst in (("compound", self.m.compounds), ("protein", self.m.proteins), ("complex", self.m.complexes),
                          ("nucleic_acid", self.m.nucleic_acids)):
            for i, o in enumerate(lst or []):
                name = _str(o.name)
                if name is None:
                    self.omissions.append({"kind": kind, "name": None, "reason": "entity without a name", "source": "sbml_export"})
                elif "unknown_sentinel" in o.flags or (kind == "protein" and name.strip().casefold() == "unknown"):
                    self.sentinel_names.add(_lkey(name))
                    self.omissions.append({"kind": kind, "name": name, "reason": "PathBank Unknown placeholder, not an entity",
                                           "source": "sbml_export"})
                else:
                    raw.append(_Ent(kind, o, name, i))
        order = {"compound": 0, "protein": 1, "complex": 2, "nucleic_acid": 3}
        raw.sort(key=lambda e: (order[e.kind], e.name.casefold(), e.idx))
        self.ents: List[_Ent] = []
        self.by_name: Dict[str, _Ent] = {}
        self.by_fold: Dict[str, _Ent] = {}
        for e in raw:
            self._register(e)
        self.placed: Dict[int, List[str]] = {}  # entity n -> location keys in placement order
        for rows in self.m.placements.values():
            for pl in rows:
                e = self.entity(_str(pl.entity), create=False)
                loc = self.sloc.get(_str(pl.state))
                if e is not None and loc:
                    ks = self.placed.setdefault(e.n, [])
                    if _lkey(loc) not in ks:
                        ks.append(_lkey(loc))

    def _register(self, e: _Ent) -> None:
        e.n = len(self.ents) + 1
        e.eid = "e_%d" % e.n
        self.ents.append(e)
        self.by_name.setdefault(e.name, e)
        self.by_fold.setdefault(_lkey(e.name), e)

    def entity(self, name: Any, create: bool = True) -> Optional[_Ent]:
        if not isinstance(name, str) or not name.strip():
            return None
        if _is_sentinel(name) or _lkey(name) in self.sentinel_names:  # every reference type: never a species/entry/modifier
            self.sentinel_refs[name] = self.sentinel_refs.get(name, 0) + 1
            return None
        e = self.by_name.get(name) or self.by_fold.get(_lkey(name))
        if e is None and create:
            e = _Ent("undeclared", None, name, -1)
            self._register(e)
        return e

    # compartments ---------------------------------------------------------------------------------------------
    def _compartments(self) -> None:
        names: Dict[str, str] = {}

        def seen(n: Any) -> None:
            if isinstance(n, str) and n.strip():
                names.setdefault(_lkey(n), re.sub(r"\s+", " ", n).strip())

        for lo in self.m.locations or []:
            seen(_str(lo.name))
            seen(_str(lo.parent))
        for v in self.sloc.values():
            seen(v)
        for rx in self.m.reactions or []:
            for r in (rx.reactants or []) + (rx.products or []):
                seen(_str(r.location))
        for t in self.m.transports or []:
            seen(_str(t.from_location))
            seen(_str(t.to_location))
        self.comps: Dict[str, Dict[str, Any]] = {}
        used: set = set()
        for k in sorted(names):
            cid = "c_" + slug(names[k])
            base, i = cid, 2
            while cid in used or cid == UNSPEC_ID:
                cid, i = "%s_%d" % (base, i), i + 1
            used.add(cid)
            self.comps[k] = {"id": cid, "name": names[k], "obj": None}
        for lo in self.m.locations or []:
            n = _str(lo.name)
            if n and self.comps[_lkey(n)]["obj"] is None:
                self.comps[_lkey(n)]["obj"] = lo

    # species --------------------------------------------------------------------------------------------------
    def pick(self, e: _Ent, hint: Optional[str]) -> Optional[str]:
        if hint:
            return _lkey(hint)
        locs = self.placed.get(e.n, [])
        return locs[0] if len(locs) == 1 else None

    def sp(self, e: _Ent, lk: Optional[str]) -> Dict[str, Any]:
        key = (e.n, lk)
        if key not in self.species:
            cid = self.comps[lk]["id"] if lk else UNSPEC_ID
            self.species[key] = {"id": "s_%d_%s" % (e.n, cid[2:]), "ent": e, "lk": lk, "cid": cid}
        return self.species[key]

    # annotation pieces ----------------------------------------------------------------------------------------
    def ids_for(self, kind: str, name: Optional[str], obj: Any) -> _Ids:
        ids = _Ids()
        for fld in OBJ_FIELDS.get(kind, ()):
            f = getattr(obj, fld, None) if obj is not None else None
            if f is None and obj is not None and fld in getattr(obj, "extras", {}):
                ids.add(FIELD_IDTYPE[(kind, fld)], obj.extras[fld], {"label": "db", "source": "payload"}, fld)
            elif f is not None and not f.absent:
                ids.add(FIELD_IDTYPE[(kind, fld)], f.value, f.prov, fld)
        for it in self.layer["items"]:
            if it.get("kind") == kind and it.get("entity") == name and (kind, it.get("field")) in FIELD_IDTYPE:
                ids.add(FIELD_IDTYPE[(kind, it["field"])], it.get("value"), it.get("prov") or {}, it["field"])
        return ids

    def layer_elems(self, kind: str, name: Optional[str]) -> List[El]:
        out: List[El] = []
        for it in self.layer["items"]:
            if it.get("kind") == kind and it.get("entity") == name and (kind, it.get("field")) not in FIELD_IDTYPE:
                out.append(El("t2pw:dbvalue", dict(_prov_attrs(it.get("prov") or {}), field=it.get("field"), accession=(it.get("prov") or {}).get("accession")),
                              text=json.dumps(it.get("value"), sort_keys=True, ensure_ascii=False)))
        for it in self.layer["suggestions"]:
            if it.get("kind") == kind and it.get("entity") == name:
                out.append(El("t2pw:suggestion", dict(_prov_attrs(it.get("prov") or {}), field=it.get("field"), usage="not an identifier"),
                              [El("t2pw:candidate", text=json.dumps(c, sort_keys=True, ensure_ascii=False)) for c in it.get("candidates") or []]))
        for it in self.layer["conflicts"]:
            if it.get("kind") == kind and it.get("entity") == name:
                out.append(El("t2pw:conflict", dict(_prov_attrs(it.get("prov") or {}), field=it.get("field")),
                              [El("t2pw:frozen", text=json.dumps(it.get("frozen"), sort_keys=True, ensure_ascii=False)),
                               El("t2pw:db", text=json.dumps(it.get("db"), sort_keys=True, ensure_ascii=False))]))
        for it in self.layer["refused"]:
            if it.get("kind") == kind and it.get("entity") == name:
                out.append(El("t2pw:refused", {"field": it.get("field"), "reason": it.get("reason")},
                              text=json.dumps(it.get("detail"), sort_keys=True, ensure_ascii=False)))
        return out

    def annotation(self, metaid: str, ids: Optional[_Ids], extra: List[El], more_rdf: Optional[List[Tuple[str, str]]] = None) -> Optional[El]:
        rdf_items: List[Tuple[str, str]] = list(more_rdf or [])
        t2: List[El] = []
        if ids is not None:
            for t, i, pv in ids.valid:
                rdf_items.append((IDTYPES[t][2], "https://identifiers.org/%s/%s" % (IDTYPES[t][0], i)))
                t2.append(El("t2pw:id", dict(pv, qualifier=IDTYPES[t][2], db=t, id=i)))
            for t, i, pv in ids.invalid:
                t2.append(El("t2pw:id", dict(pv, db=t, id=i, valid="false", usage="not written as an identifier")))
            for f, v, pv in ids.inferred:
                t2.append(El("t2pw:inferred", dict(pv, field=f, usage="not an identifier"), text=json.dumps(v, ensure_ascii=False)))
        kids: List[El] = []
        if rdf_items:
            quals: Dict[str, List[str]] = {}
            for q, u in rdf_items:
                if u not in quals.setdefault(q, []):
                    quals[q].append(u)
            desc = El("rdf:Description", {"rdf:about": "#" + metaid},
                      [El(q, kids=[El("rdf:Bag", kids=[El("rdf:li", {"rdf:resource": u}) for u in us])]) for q, us in quals.items()])
            kids.append(El("rdf:RDF", {"xmlns:rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                                      "xmlns:bqbiol": "http://biomodels.net/biology-qualifiers/",
                                      "xmlns:bqmodel": "http://biomodels.net/model-qualifiers/"}, [desc]))
        if t2 or extra:  # SBML allows one top-level element per namespace inside an annotation
            kids.append(El("t2pw:info", kids=t2 + extra))
        return El("annotation", kids=kids) if kids else None

    # compartments / species / reactions -----------------------------------------------------------------------
    def compartment_el(self, k: Optional[str]) -> El:
        if k is None:
            return El("compartment", {"id": UNSPEC_ID, "metaid": "m_" + UNSPEC_ID, "name": "unspecified location", "constant": True,
                                      "sboTerm": "SBO:0000290"},
                      [notes(["The paper did not state a location for the entities placed here."])])
        c = self.comps[k]
        lo = c["obj"]
        paras = ["Location name as written in the paper: %s." % c["name"]]
        if lo is not None and _str(lo.evidence):
            paras.append('Evidence: "%s"' % _str(lo.evidence))
        ctx = sorted({"%s%s%s" % (_str(s.species) or "", (", cell type " + _str(s.cell_type)) if _str(s.cell_type) else "",
                                  (", tissue " + _str(s.tissue)) if _str(s.tissue) else "")
                      for n, s in self.state_ctx.items() if _lkey(self.sloc.get(n) or "") == k and (_str(s.species) or _str(s.cell_type) or _str(s.tissue))})
        if ctx:
            paras.append("Context stated for this location: " + "; ".join(ctx) + ".")
        ids, extra, more = _Ids(), [], []
        if lo is not None:
            ids = self.ids_for("location", c["name"], lo)
            extra = self.layer_elems("location", c["name"])
            if _str(lo.kind):
                extra.append(El("t2pw:kind", _prov_attrs(lo.kind.prov), text=_str(lo.kind)))
            par = _str(lo.parent)
            if par and _lkey(par) in self.comps and _lkey(par) != k:
                more.append(("bqbiol:isPartOf", "#" + "m_" + self.comps[_lkey(par)]["id"]))
                extra.append(El("t2pw:parent", _prov_attrs(lo.parent.prov), text=par))
        return El("compartment", {"id": c["id"], "metaid": "m_" + c["id"], "name": c["name"], "constant": True, "sboTerm": "SBO:0000290"},
                  [notes(paras), self.annotation("m_" + c["id"], ids, extra, more)])

    def species_el(self, s: Dict[str, Any]) -> El:
        e: _Ent = s["ent"]
        o = e.obj
        paras = ["%s: %s (provenance of the name: %s)." % (e.kind.replace("_", " "), e.name, o.name.prov.label if o is not None else "paper")]
        if e.kind == "undeclared":
            paras.append("Named in a process but not declared in the model entities; its type is not stated.")
        if s["lk"] is None:
            paras.append("Location not stated for this species; it sits in the unspecified-location compartment.")
        extra: List[El] = [El("t2pw:entity", {"entityId": e.eid, "kind": e.kind, "label": o.name.prov.label if o is not None else "paper"})]
        for fld in ("gene_name", "synonyms"):
            v = _val(getattr(o, fld, None)) if o is not None else None
            if v:
                v = v if isinstance(v, list) else [v]
                paras.append("%s: %s." % (fld.replace("_", " ").capitalize(), ", ".join(map(str, v))))
                extra += [El("t2pw:" + ("geneName" if fld == "gene_name" else "synonym"), text=str(x)) for x in v]
        if e.kind == "complex":
            members = [m for m in (o.members or [])]
            known = [m for m in members if not m.unknown and isinstance(m.entity.value, str)]
            if not known:
                paras.append("Members of this complex are unknown (the paper did not name them).")
            else:
                paras.append("Members: %s%s." % (", ".join(m.entity.value for m in known),
                                                 "; %d further member(s) unknown" % (len(members) - len(known)) if len(members) > len(known) else ""))
            for m in members:
                if m.unknown or not isinstance(m.entity.value, str):
                    extra.append(El("t2pw:member", {"unknown": True}))
                else:
                    me = self.entity(m.entity.value, create=False)
                    extra.append(El("t2pw:member", {"name": m.entity.value, "entityId": me.eid if me else None}))
        ids = _Ids()
        if e.kind in ("compound", "protein"):
            ids = self.ids_for(e.kind, e.name, o)
            extra += self.layer_elems(e.kind, e.name)
            for fld in ("inchikey", "formula", "smiles"):
                if o is not None and _val(getattr(o, fld, None)) is not None:
                    extra.append(El("t2pw:" + fld, _prov_attrs(getattr(o, fld).prov), text=str(getattr(o, fld).value)))
            if e.kind == "protein":
                if _val(o.inferred_suggestion) is not None:
                    extra.append(El("t2pw:suggestion", dict(_prov_attrs(o.inferred_suggestion.prov), field="uniprot", usage="not an identifier"),
                                    text=json.dumps(o.inferred_suggestion.value, sort_keys=True, ensure_ascii=False)))
                if _str(o.identity_status):
                    extra.append(El("t2pw:identityStatus", text=_str(o.identity_status)))
        mid = "m_" + s["id"]
        return El("species", {"id": s["id"], "metaid": mid, "name": e.name, "compartment": s["cid"], "hasOnlySubstanceUnits": False,
                              "boundaryCondition": False, "constant": False, "sboTerm": S.ENTITY_SBO[e.kind]},
                  [notes(paras), self.annotation(mid, ids, extra)])

    @staticmethod
    def _stated(f: Any) -> bool:
        return f is not None and not f.absent and f.prov.label == "paper"

    def _spref(self, tag: str, sp: Dict[str, Any], ref: Any) -> El:
        st, kids = None, []
        if self._stated(ref.stoichiometry) and isinstance(ref.stoichiometry.value, (int, float)) and not isinstance(ref.stoichiometry.value, bool):
            st = float(ref.stoichiometry.value)
        else:
            kids.append(El("annotation", kids=[El("t2pw:stoichiometry", {"state": "unstated"})]))
        return El(tag, {"species": sp["id"], "stoichiometry": st, "constant": True}, kids)

    def _modifier(self, sp: Dict[str, Any], ref: Any, seen: set, default: str = "modifier") -> Optional[El]:
        if sp["id"] in seen:
            return None
        seen.add(sp["id"])
        role = (_str(ref.role) or default).casefold()
        return El("modifierSpeciesReference", {"species": sp["id"], "sboTerm": S.ROLE_SBO.get(role, "SBO:0000019")})

    def _direction(self, rx: Any) -> Tuple[bool, bool]:
        """(reversible value, stated). Only a paper-labelled value counts as stated."""
        for f in (rx.reversible, rx.direction):
            if self._stated(f):
                v = f.value
                if v in (True, "reversible", "<->", "bidirectional", "both"):
                    return True, True
                if v in (False, "irreversible", "->", ">", "forward"):
                    return False, True
        return False, False

    def reaction_el(self, i: int, rx: Any) -> El:
        rid = "r_%d" % i
        st = _str(rx.state)
        hint = self.sloc.get(st) if st else None
        paras: List[str] = []
        if _str(rx.evidence):
            paras.append('Evidence: "%s"' % _str(rx.evidence))
        if hint:
            paras.append("Location (from the paper's biological state %s): %s." % (st, hint))
        else:
            paras.append("Location of this reaction not stated.")
        kids: List[El] = []
        parts = {}
        for tag, lst in (("listOfReactants", rx.reactants), ("listOfProducts", rx.products)):
            els = []
            for ref in lst or []:
                e = self.entity(ref.entity.value) if isinstance(ref.entity.value, str) and not ref.unknown else None
                if e is None or ("unknown_sentinel" in (e.obj.flags if e.obj is not None else [])):
                    paras.append("A participant was named only as Unknown or not at all; it is not written.")
                    continue
                lk = _lkey(ref.location.value) if _val(ref.location) else (_lkey(hint) if hint else None)
                els.append(self._spref("speciesReference", self.sp(e, lk if lk else self.pick(e, None)), ref))
            parts[tag] = els
        mods, seen = [], set()
        for ref, dflt in [(r, "catalyst") for r in rx.catalysts or []] + [(r, "modifier") for r in rx.modifiers or []]:
            e = self.entity(ref.entity.value) if isinstance(ref.entity.value, str) and not ref.unknown else None
            if e is None or (e.obj is not None and "unknown_sentinel" in e.obj.flags):
                paras.append("A catalyst or modifier was named only as Unknown; it is not written.")
                continue
            if _str(ref.evidence) and ref.evidence.value != _str(rx.evidence):
                paras.append('Evidence for %s: "%s"' % (e.name, _str(ref.evidence)))
            m = self._modifier(self.sp(e, self.pick(e, hint)), ref, seen, dflt)
            if m is not None:
                mods.append(m)
        rev, rev_stated = self._direction(rx)
        spon = "true" if self._stated(rx.spontaneous) and rx.spontaneous.value is True else "false" if self._stated(rx.spontaneous) and rx.spontaneous.value is False else "unstated"
        mid = "m_" + rid
        ids = self.ids_for("reaction", _str(rx.name), rx)
        extra = [El("t2pw:directionStated", text="true" if rev_stated else "false"), El("t2pw:spontaneous", text=spon),
                 El("t2pw:provenance", {"label": rx.name.prov.label if not rx.name.absent else "unknown"})]
        if not rev_stated:
            extra.append(El("t2pw:reversibleNote", text="reversible=false is the required SBML placeholder; the paper did not state a direction"))
        if _val(rx.step_index) is not None:
            extra.append(El("t2pw:stepIndex", _prov_attrs(rx.step_index.prov), text=str(rx.step_index.value)))
        extra += self.layer_elems("reaction", _str(rx.name))
        return El("reaction", {"id": rid, "metaid": mid, "name": _str(rx.name), "reversible": rev, "sboTerm": "SBO:0000176"},
                  [notes(paras), self.annotation(mid, ids, extra),
                   El("listOfReactants", kids=parts["listOfReactants"]) if parts["listOfReactants"] else None,
                   El("listOfProducts", kids=parts["listOfProducts"]) if parts["listOfProducts"] else None,
                   El("listOfModifiers", kids=mods) if mods else None])

    def transport_el(self, i: int, t: Any) -> Optional[El]:
        name = _str(t.name)
        cargo = self.entity(t.cargo.value) if isinstance(_val(t.cargo), str) else None
        if cargo is None:
            self.omissions.append({"kind": "transport", "name": name, "reason": "no cargo stated", "source": "sbml_export"})
            return None
        fr, to = _str(t.from_location), _str(t.to_location)
        paras = []
        if _str(t.evidence):
            paras.append('Evidence: "%s"' % _str(t.evidence))
        if not fr or not to:
            paras.append("The paper-derived record has no %s; the cargo is written in %s." % (
                "from-location and no to-location" if not fr and not to else "from-location" if not fr else "to-location",
                "the unspecified-location compartment" if not fr and not to else "the unspecified-location compartment on that side"))
        paras.append("Mechanism: %s." % _str(t.mechanism) if _str(t.mechanism) else "")
        ref_src, ref_dst = self.sp(cargo, _lkey(fr) if fr else None), self.sp(cargo, _lkey(to) if to else None)
        mods, seen = [], set()
        for ref in t.transporters or []:
            e = self.entity(ref.entity.value) if isinstance(ref.entity.value, str) and not ref.unknown else None
            if e is None or (e.obj is not None and "unknown_sentinel" in e.obj.flags):
                paras.append("A transporter was named only as Unknown; it is not written.")
                continue
            m = self._modifier(self.sp(e, self.pick(e, None)), ref, seen)
            if m is not None:
                m.attrs["sboTerm"] = "SBO:0000013"
                mods.append(m)
        tid, mid = "t_%d" % i, "m_t_%d" % i
        plain = _Plain()
        return El("reaction", {"id": tid, "metaid": mid, "name": name, "reversible": False, "sboTerm": "SBO:0000655"},
                  [notes(paras),
                   self.annotation(mid, None, [El("t2pw:directionStated", text="false"), El("t2pw:spontaneous", text="unstated"),
                                               El("t2pw:reversibleNote", text="reversible=false is the required SBML placeholder; the paper did not state a direction"),
                                               El("t2pw:provenance", {"label": t.name.prov.label if not t.name.absent else "unknown"})]),
                   El("listOfReactants", kids=[self._spref("speciesReference", ref_src, plain)]),
                   El("listOfProducts", kids=[self._spref("speciesReference", ref_dst, plain)]),
                   El("listOfModifiers", kids=mods) if mods else None])

    def interaction_el(self, ix: Any) -> El:
        kind = _str(ix.kind) or "other_unsigned"
        sign = _str(ix.sign) or "unsigned"
        sbo = S.KIND_SBO.get(kind, S.GENERIC_INTERACTION)
        if sign == "unsigned" and kind in ("activation", "inhibition"):
            sbo = S.GENERIC_INTERACTION
        s, t = _str(ix.source), _str(ix.target)
        es, et = self.entity(s, create=False), self.entity(t, create=False)
        kids = []
        q = _str(ix.evidence)
        note = " ".join(x for x in ((("Relationship as written: %s." % ix.relationship.value) if _val(ix.relationship) is not None else ""),
                                    ('Evidence: "%s"' % q) if q else "") if x)
        if note:
            kids.append(El("t2pw:note", text=note))
        return El("t2pw:interaction", {"source": es.eid if es else None, "target": et.eid if et else None, "sourceName": s, "targetName": t,
                                        "kind": kind, "sign": sign, "sbo": sbo, "relationship": _str(ix.relationship),
                                        "label": ix.kind.prov.label if not ix.kind.absent else "inferred",
                                        "undeclaredEndpoint": True if (es is None or et is None) else None}, kids)

    # document -------------------------------------------------------------------------------------------------
    def build(self) -> El:
        m, pw = self.m, self.m.pathway
        reactions = [self.reaction_el(i, rx) for i, rx in enumerate(m.reactions or [], 1)]
        transports = [x for x in (self.transport_el(i, t) for i, t in enumerate(m.transports or [], 1)) if x is not None]
        for e in self.ents:  # every declared entity gets at least one species
            if e.kind == "undeclared" or any(k[0] == e.n for k in self.species):
                continue
            locs = self.placed.get(e.n) or [None]
            for lk in locs:
                self.sp(e, lk)
        for e in self.ents:
            for lk in self.placed.get(e.n, []):
                self.sp(e, lk)
        sps = sorted(self.species.values(), key=lambda s: (s["ent"].n, s["cid"]))
        used_unspec = any(s["lk"] is None for s in sps)
        comp_els = [self.compartment_el(k) for k in sorted(self.comps)]
        if used_unspec or not comp_els:
            comp_els.append(self.compartment_el(None))
        sp_els = [self.species_el(s) for s in sps]

        # model-level
        pm, pmc, doi = _str(pw.pmid), _str(pw.pmcid), _str(pw.doi)
        title = _str(pw.title)
        rdf: List[Tuple[str, str]] = []
        if pm and pw.pmid.prov.label in ("paper", "db") and re.fullmatch(r"\d+", pm.strip()):  # a guessed pmid stays in t2pw:meta
            rdf.append(("bqmodel:isDescribedBy", "https://identifiers.org/pubmed/" + pm.strip()))
        if pmc and re.fullmatch(r"PMC\d+", pmc.strip()):
            rdf.append(("bqmodel:isDescribedBy", "https://identifiers.org/pmc/" + pmc.strip()))
        if doi and re.fullmatch(r"10\.\d{4,9}/\S+", doi.strip()):
            rdf.append(("bqmodel:isDescribedBy", "https://identifiers.org/doi/" + doi.strip()))
        tax = _str(pw.taxon_id) or (str(pw.taxon_id.value) if _val(pw.taxon_id) is not None else None)
        if tax and re.fullmatch(r"\d+", tax):
            rdf.append(("bqbiol:hasTaxon", "https://identifiers.org/taxonomy/" + tax))
        meta = []
        for k, f in (("title", pw.title), ("shortName", pw.short_name), ("pmid", pw.pmid), ("pmcid", pw.pmcid), ("doi", pw.doi),
                     ("authors", pw.authors), ("year", pw.year), ("organism", pw.organism), ("taxonId", pw.taxon_id),
                     ("pathwayClass", pw.pathway_class), ("tool", pw.tool), ("toolVersion", pw.tool_version),
                     ("runDate", pw.run_date), ("mode", pw.mode)):
            if _val(f) is not None:
                v = f.value
                meta.append(El("t2pw:meta", dict(_prov_attrs(f.prov), key=k), text="; ".join(map(str, v)) if isinstance(v, list) else str(v)))
        lic = [x.value for x in pw.licences if isinstance(x.value, str)]
        omis = [El("t2pw:omission", {"kind": r.kind, "name": r.name, "reason": r.reason, "source": r.source}) for r in m.removed]
        omis += [El("t2pw:omission", {k: o.get(k) for k in ("kind", "name", "reason", "source")}) for o in self.omissions]
        omis += [El("t2pw:omission", {"kind": "reference", "name": n, "source": "sbml_export",
                                      "reason": "PathBank Unknown placeholder referenced %d time(s) as participant, catalyst, transporter or endpoint; not written" % c})
                 for n, c in sorted(self.sentinel_refs.items())]
        omis += [El("t2pw:omission", {"kind": r.get("kind"), "name": r.get("entity"), "reason": "database value refused: %s (%s)" % (r.get("reason"), r.get("field")),
                                      "source": "db_layer"}) for r in self.layer["refused"]]
        ixs = [self.interaction_el(ix) for ix in m.interactions or []]
        extra = meta + [El("t2pw:licence", text=x) for x in lic]
        extra.append(El("t2pw:omissions", {"count": len(omis)}, omis))
        extra.append(El("t2pw:interactions", {"count": len(ixs), "note": "SBML core has no interaction object; these are not reactions"}, ixs))
        mid = "m_model"
        pcs = [
            "Exported from the T2PW clean model. Values the paper did not state are omitted or marked unstated, never filled with defaults.",
            "No initial amount or concentration is given: the paper states none.",
            "Reaction direction: reversible=\"false\" is the SBML-required placeholder unless t2pw:directionStated is true.",
            "Interactions are listed in the t2pw annotation of this model because SBML core has no interaction object.",
        ]
        pcs += ["Database attribution: " + x for x in lic] or ["No database attribution lines are recorded in the model."]
        model_name = title or pmc
        model = El("model", {"id": "t2pw_" + (slug(pmc) if pmc else "model"), "metaid": mid, "name": model_name},
                   [notes(pcs), self.annotation(mid, None, extra, rdf),
                    El("listOfCompartments", kids=comp_els), El("listOfSpecies", kids=sp_els) if sp_els else None,
                    El("listOfReactions", kids=reactions + transports) if reactions or transports else None])
        return El("sbml", {"xmlns": SBML_NS, "xmlns:t2pw": T2PW_NS, "level": 3, "version": 2}, [model])


def sbml_text(model: CleanModel) -> str:
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + _Exporter(model).build().render()


def write_sbml(model: CleanModel, path: str) -> str:
    text = sbml_text(model)
    with open(path, "wb") as fh:
        fh.write(text.encode("utf-8"))
    return path


def sbml_for_run(run_dir: str) -> str:
    """Exporters never look anything up (D-003): call ``fill_model`` on the model first if wanted."""
    from ..builder import build_from_run
    return sbml_text(build_from_run(run_dir))
