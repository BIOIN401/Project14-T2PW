"""Fill a CleanModel's database layer from public sources, with provenance, never touching frozen/paper values.

Where a value goes: typed slots that have no payload key (Compound.inchikey/formula/smiles, Protein.inferred_suggestion,
Location.go_id/parent/kind, Reaction.ec_number/rhea_id, Pathway fields) and ``model.db_layer``
(``items`` / ``suggestions`` / ``conflicts`` / ``refused``). ``to_payload`` never reads either, so it is unchanged.
"""

from __future__ import annotations

import dataclasses
import datetime
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from . import sources as S
from .cache import Cache, CachedFetcher
from .http import FetchError, UrllibFetcher
from ..model import CleanModel, Field, Prov

ALL_SOURCES = ("uniprot", "chebi", "go", "ncbi", "rhea")
LICENCES = {
    "uniprot": "UniProt: CC BY 4.0, https://www.uniprot.org/help/license",
    "chebi": "ChEBI: CC BY 4.0, https://www.ebi.ac.uk/chebi/",
    "rhea": "Rhea: CC BY 4.0, https://www.rhea-db.org/",
    "go": "Gene Ontology / QuickGO: CC BY 4.0, https://geneontology.org/docs/go-citation-policy/",
    "ncbi": "NCBI (E-utilities, Taxonomy, PubMed/PMC metadata): public domain, https://www.ncbi.nlm.nih.gov/home/about/policies/",
}
ID_FIELDS = {"chebi_id", "kegg_id", "kegg_gene", "hmdb_id", "pubchem_cid", "cas", "drugbank_id", "ensembl", "ensembl_gene", "uniprot",
             "go_id", "taxonomy_id", "taxon_id", "ec_number", "rhea_id", "doi", "pmid", "pmcid"}
_ACC = re.compile(r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$")
_CLASSY = re.compile(r"\b(family|superfamily|class|classes|group|receptors|channels|subunits)\b", re.I)
# frozen compound id field -> ChEBI xref key
_XREF = {"kegg_id": "kegg_compound", "hmdb_id": "hmdb", "pubchem_cid": "pubchem", "cas": "cas", "drugbank_id": "drugbank"}


@dataclasses.dataclass
class FillReport:
    fields_filled: int = 0
    ids_filled: int = 0
    by_source: Dict[str, int] = dataclasses.field(default_factory=dict)
    by_match_type: Dict[str, int] = dataclasses.field(default_factory=dict)
    by_field: Dict[str, int] = dataclasses.field(default_factory=dict)
    lookups_failed: Dict[str, int] = dataclasses.field(default_factory=dict)  # fetch errors, by source
    lookups_no_match: Dict[str, int] = dataclasses.field(default_factory=dict)
    lookups_ambiguous: int = 0  # several candidates -> a suggestion, not an id
    refused: Dict[str, int] = dataclasses.field(default_factory=dict)  # reason (collision, ...) -> count
    conflicts: int = 0
    requests: int = 0
    cache_hits: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _bump(d: Dict[str, int], k: str, n: int = 1) -> None:
    d[k] = d.get(k, 0) + n


def _text(f: Field) -> Optional[str]:
    return f.value if isinstance(f.value, str) and not f.absent else None


def _norm(v: Any) -> str:
    s = re.sub(r"[^a-z0-9]", "", str(v).casefold())
    s = re.sub(r"^(chebi|hmdb|pw)", "", s)
    return s.lstrip("0")


def _num(v: Any) -> Optional[str]:
    m = re.search(r"\d+", str(v))
    return m.group(0) if m else None


class _Ctx:
    def __init__(self, model: CleanModel, f: Any, today: str, enabled: Tuple[str, ...]) -> None:
        self.model, self.f, self.today, self.enabled = model, f, today, enabled
        self.date = today  # retrieval date of the latest lookup: the oldest response it used (a cache hit keeps its own date)
        self.rep = FillReport()
        self.used: set = set()
        self.model.db_layer = {"items": [], "suggestions": [], "conflicts": [], "refused": []}
        self.seen: set = set()

    def prov(self, source: str, accession: Any, match_type: str, label: str = "db") -> Prov:
        return Prov(label=label, source=source, accession=None if accession is None else str(accession),
                    retrieved=self.date, match_type=match_type)

    def put(self, kind: str, entity: Optional[str], fld: str, value: Any, prov: Prov) -> bool:
        key = (kind, entity, fld, repr(value))
        if key in self.seen:
            return False
        self.seen.add(key)
        self.used.add(prov.source)
        self.model.db_layer["items"].append({"kind": kind, "entity": entity, "field": fld, "value": value,
                                             "prov": dataclasses.asdict(prov)})
        self.rep.fields_filled += 1
        self.rep.ids_filled += fld in ID_FIELDS
        _bump(self.rep.by_source, prov.source)
        _bump(self.rep.by_match_type, (prov.match_type or "").split("+")[0])
        _bump(self.rep.by_field, f"{kind}.{fld}")
        return True

    def slot(self, obj: Any, attr: str, kind: str, entity: Optional[str], value: Any, prov: Prov) -> None:
        """Fill a payload-free typed slot (when still empty) and record the item."""
        if self.put(kind, entity, attr, value, prov) and getattr(obj, attr).absent:
            setattr(obj, attr, Field(value, prov))

    def conflict(self, kind: str, entity: Optional[str], fld: str, frozen_v: Any, db_v: Any, prov: Prov) -> None:
        self.used.add(prov.source)
        self.model.db_layer["conflicts"].append({"kind": kind, "entity": entity, "field": fld, "frozen": frozen_v,
                                                 "db": db_v, "prov": dataclasses.asdict(prov)})
        self.rep.conflicts += 1

    def refuse(self, reason: str, kind: str, entity: Optional[str], fld: str, detail: Any) -> None:
        self.model.db_layer["refused"].append({"kind": kind, "entity": entity, "field": fld, "reason": reason,
                                               "detail": detail})
        _bump(self.rep.refused, reason)

    def suggest(self, kind: str, entity: Optional[str], fld: str, candidates: List[Any], prov: Prov) -> None:
        self.used.add(prov.source)
        self.model.db_layer["suggestions"].append({"kind": kind, "entity": entity, "field": fld,
                                                   "candidates": candidates, "prov": dataclasses.asdict(prov)})
        self.rep.lookups_ambiguous += 1

    def call(self, source: str, fn: Any, *a: Any) -> Any:
        """Run one source lookup; a FetchError is counted and returns None (a missing value is never invented)."""
        dates = getattr(self.f, "dates", None)
        if dates is not None:
            del dates[:]
        try:
            r = fn(self.f, *a)
        except FetchError:
            _bump(self.rep.lookups_failed, source)
            return None
        self.date = min(dates) if dates else self.today
        if r is None or r == [] or r == {}:
            _bump(self.rep.lookups_no_match, source)
        return r

    def compare(self, kind: str, entity: Optional[str], fld: str, frozen_f: Field, db_vals: List[Any], prov: Prov) -> bool:
        """True when the field was already settled by a frozen value (equal -> nothing to add; differs -> conflict)."""
        if frozen_f.absent or frozen_f.value in (None, ""):
            return False
        if db_vals and _norm(frozen_f.value) not in {_norm(v) for v in db_vals}:
            self.conflict(kind, entity, fld, frozen_f.value, db_vals, prov)
        return True


# ---- compounds ----------------------------------------------------------------------------------------------------

def _chebi_details(c: _Ctx, comp: Any, name: str, info: Dict[str, Any], mt: str) -> None:
    pv = c.prov("ChEBI", "CHEBI:" + info["id"], mt)
    for attr in ("formula", "inchikey", "smiles"):
        if info.get(attr):
            c.slot(comp, attr, "compound", name, info[attr], pv)
    xp = c.prov("ChEBI", "CHEBI:" + info["id"], "cross_reference")
    for fld, key in _XREF.items():
        vals = info["xrefs"].get(key) or []
        held = getattr(comp, fld)
        if fld == "drugbank_id" and held.absent:
            held = comp.drugbank_mapped  # the id the payload carries under mapped_ids
        if c.compare("compound", name, fld, held, vals, xp):
            continue
        for v in vals:
            c.put("compound", name, fld, v, xp)


def _fill_compounds(c: _Ctx) -> None:
    comps = c.model.compounds or []
    ids: Dict[str, set] = {}
    for comp in comps:
        n = _num(comp.chebi_id.value) if not comp.chebi_id.absent else None
        if n:
            ids.setdefault(n, set()).add(_text(comp.name))
    for comp in comps:
        name = _text(comp.name)
        if not name:
            continue
        if not comp.chebi_id.absent and _num(comp.chebi_id.value):
            info = c.call("chebi", S.chebi_compound, _num(comp.chebi_id.value))
            if info:
                _chebi_details(c, comp, name, info, "accession")
            continue
        if any(not getattr(comp, a).absent and getattr(comp, a).value not in (None, "")
               for a in ("kegg_id", "hmdb_id", "pubchem_cid")):
            continue
        hits = c.call("chebi", S.chebi_by_name, name)
        if not hits:
            continue
        if len(hits) > 1:
            c.suggest("compound", name, "chebi_id", [{"id": "CHEBI:" + h["id"], "label": h["label"],
                                                       "match_type": h["match_type"], "charged": h["charged"]}
                                                      for h in hits], c.prov("ChEBI", None, hits[0]["match_type"], "inferred"))
            continue
        h = hits[0]
        others = ids.get(h["id"], set()) - {name}
        if others:  # D-109: a class name that resolves to a member's id must not take that id
            c.refuse("collision", "compound", name, "chebi_id", {"chebi_id": "CHEBI:" + h["id"], "used_by": sorted(map(str, others))})
            continue
        ids.setdefault(h["id"], set()).add(name)
        mt = h["match_type"] + ("+charged_form" if h["charged"] else "")
        c.put("compound", name, "chebi_id", "CHEBI:" + h["id"], c.prov("ChEBI", "CHEBI:" + h["id"], mt))
        info = c.call("chebi", S.chebi_compound, h["id"])
        if info:
            _chebi_details(c, comp, name, info, "accession")


# ---- proteins -----------------------------------------------------------------------------------------------------

def _model_taxon(c: _Ctx, p: Any) -> Optional[str]:
    if not p.taxonomy_id.absent and _num(p.taxonomy_id.value):
        return _num(p.taxonomy_id.value)
    sp = c.model.species or []
    if len(sp) == 1 and not sp[0].taxonomy_id.absent:
        return _num(sp[0].taxonomy_id.value)
    return None


def _fill_proteins(c: _Ctx) -> Dict[str, Dict[str, Any]]:
    entries: Dict[str, Dict[str, Any]] = {}
    for p in c.model.proteins or []:
        name = _text(p.name)
        if not name or p.flags:
            continue
        acc = _text(p.uniprot)
        if acc and _ACC.match(acc):
            e = c.call("uniprot", S.uniprot_entry, acc)
            if not e:
                continue
            entries[name] = e
            pv = c.prov("UniProt", e["accession"], "accession")
            xp = c.prov("UniProt", e["accession"], "cross_reference")
            if e["protein_name"]:
                c.put("protein", name, "protein_name", e["protein_name"], pv)
            if not c.compare("protein", name, "gene_name", p.gene_name, e["gene_names"], pv):
                for g in e["gene_names"]:
                    c.put("protein", name, "gene_name", g, pv)
            for s in e["synonyms"]:
                c.put("protein", name, "synonym", s, pv)
            have_ec = [x for f in (p.ec_number, p.ec_numbers) if not f.absent
                       for x in (f.value if isinstance(f.value, list) else [f.value])]
            if have_ec:
                if e["ec_numbers"] and not set(map(str, have_ec)) & set(e["ec_numbers"]):
                    c.conflict("protein", name, "ec_number", have_ec, e["ec_numbers"], pv)
            else:
                for ec in e["ec_numbers"]:
                    c.put("protein", name, "ec_number", ec, pv)
            mt = _model_taxon(c, p)
            if e["taxon_id"]:
                if mt and mt != e["taxon_id"]:
                    c.conflict("protein", name, "taxon_id", mt, e["taxon_id"], pv)  # strain-level ids differ from species-level
                else:
                    c.put("protein", name, "taxon_id", e["taxon_id"], pv)
            for fld in ("kegg_gene", "ensembl", "ensembl_gene"):
                for v in e["xrefs"][fld]:
                    c.put("protein", name, fld, v, xp)
        elif not acc and not p.gene_name.absent and isinstance(p.gene_name.value, str) \
                and not _CLASSY.search(name) and (p.identity_status.value != "placeholder"):
            tax = _model_taxon(c, p)
            if not tax:
                continue
            cands = c.call("uniprot", S.uniprot_by_gene, p.gene_name.value, tax)
            if cands:
                cands = sorted(cands, key=lambda x: not x["reviewed"])[:5]
                pv = c.prov("UniProt", None, "gene_symbol_taxon", "inferred")
                c.suggest("protein", name, "uniprot", cands, pv)
                if p.inferred_suggestion.absent:  # a proposal slot, never an id field (P-20)
                    p.inferred_suggestion = Field(cands, pv)
    return entries


# ---- locations ----------------------------------------------------------------------------------------------------

def _fill_locations(c: _Ctx) -> None:
    gos: Dict[str, Any] = {}
    for loc in c.model.locations or []:
        name = _text(loc.name)
        if not name:
            continue
        hits = c.call("go", S.go_cc_by_name, name)
        if not hits:
            continue
        if len(hits) > 1:
            c.suggest("location", name, "go_id", hits, c.prov("GO", None, hits[0]["match_type"], "inferred"))
            continue
        h = hits[0]
        c.slot(loc, "go_id", "location", name, h["id"], c.prov("GO", h["id"], h["match_type"]))
        gos[name] = (loc, h)
        if re.search(r"membrane$", h["name"]):
            c.slot(loc, "kind", "location", name, "membrane", c.prov("GO", h["id"], h["match_type"], "inferred"))
    if len(gos) < 2:
        return
    anc = c.call("go", S.go_ancestors, sorted({h["id"] for _, h in gos.values()})) or {}
    by_go = {h["id"]: n for n, (_, h) in gos.items()}
    for name, (loc, h) in gos.items():
        parents = [g for g in anc.get(h["id"], []) if g in by_go]
        if parents:  # nearest model ancestor = the one with the most ancestors itself
            g = max(parents, key=lambda x: len(anc.get(x, [])))
            c.slot(loc, "parent", "location", name, by_go[g], c.prov("GO", g, "cross_reference"))


# ---- reactions ----------------------------------------------------------------------------------------------------

def _fill_reactions(c: _Ctx, entries: Dict[str, Dict[str, Any]]) -> None:
    for rx in c.model.reactions or []:
        name = _text(rx.name)
        sets = []
        for ref in rx.catalysts or []:
            e = entries.get(ref.entity.value) if isinstance(ref.entity.value, str) else None
            if e is not None:
                sets.append(tuple(sorted(e["ec_numbers"])))
        if not sets or len(set(sets)) != 1 or len(sets[0]) != 1:
            continue
        ec = sets[0][0]
        prev = rx.extras.get("ec_number")
        if prev not in (None, "") and _norm(prev) != _norm(ec):
            c.conflict("reaction", name, "ec_number", prev, ec, c.prov("UniProt", None, "catalyst_ec", "inferred"))
            continue
        c.slot(rx, "ec_number", "reaction", name, ec, c.prov("UniProt", None, "catalyst_ec", "inferred"))
        if "rhea" in c.enabled:
            ids = c.call("rhea", S.rhea_by_ec, ec)
            if ids and len(ids) == 1:
                prev = rx.extras.get("rhea_id")
                if prev not in (None, "") and _norm(prev) != _norm(ids[0]):
                    c.conflict("reaction", name, "rhea_id", prev, ids[0], c.prov("Rhea", ids[0], "catalyst_ec", "inferred"))
                else:
                    c.slot(rx, "rhea_id", "reaction", name, ids[0], c.prov("Rhea", ids[0], "catalyst_ec", "inferred"))
            elif ids:
                c.rep.lookups_ambiguous += 1  # several reactions share the EC: no id


# ---- pathway and species ------------------------------------------------------------------------------------------

def _fill_pathway(c: _Ctx) -> None:
    pw = c.model.pathway
    a = None
    if "ncbi" in c.enabled and (not pw.pmcid.absent or not pw.pmid.absent):
        a = c.call("ncbi", S.ncbi_article, pw.pmcid.value if not pw.pmcid.absent else None,
                   pw.pmid.value if not pw.pmid.absent else None)
    if a:
        acc = a["pmcid"] if not pw.pmcid.absent else a["pmid"]
        pv = c.prov("NCBI", acc, "accession")
        for fld, key in (("title", "title"), ("authors", "authors"), ("year", "year"), ("doi", "doi"),
                         ("pmid", "pmid"), ("pmcid", "pmcid")):
            val = a.get(key)
            if val in (None, "", []):
                continue
            cur = getattr(pw, fld)
            if not cur.absent and cur.value not in (None, ""):
                same = (re.sub(r"\W", "", str(cur.value).casefold()) == re.sub(r"\W", "", str(val).casefold())
                        if not isinstance(val, list) else True)
                if not same:
                    c.conflict("pathway", None, fld, cur.value, val, pv)
            elif c.put("pathway", None, fld, val, pv):
                setattr(pw, fld, Field(val, pv))
    if "ncbi" in c.enabled:
        for sp in c.model.species or []:
            n = _text(sp.name)
            if n and sp.taxonomy_id.absent:
                t = c.call("ncbi", S.ncbi_taxids, n)
                if t and len(t) == 1:
                    c.put("species", n, "taxonomy_id", t[0], c.prov("NCBI Taxonomy", t[0], "exact_label"))
                    if pw.taxon_id.absent and pw.organism.value == n:
                        pw.taxon_id = Field(t[0], c.prov("NCBI Taxonomy", t[0], "exact_label"))
                elif t:
                    c.rep.lookups_ambiguous += 1
        if pw.taxon_id.absent and not pw.organism.absent and isinstance(pw.organism.value, str) and not (c.model.species or []):
            t = c.call("ncbi", S.ncbi_taxids, pw.organism.value)
            if t and len(t) == 1:
                c.put("pathway", None, "taxon_id", t[0], c.prov("NCBI Taxonomy", t[0], "exact_label"))
                pw.taxon_id = Field(t[0], c.prov("NCBI Taxonomy", t[0], "exact_label"))


def fill_model(model: CleanModel, fetcher: Any = None, cache: Optional[Cache] = None, today: Optional[str] = None,
               sources: Tuple[str, ...] = ALL_SOURCES, infer_reaction_ec: bool = False) -> FillReport:
    """``infer_reaction_ec`` (off by default): copy a lone catalyst's UniProt EC (and its unique Rhea id) to the reaction.
    Opt-in because the catalyst EC is never checked against the substrates (measured: wrong fills)."""
    today = today or datetime.date.today().isoformat()
    cf = CachedFetcher(fetcher or UrllibFetcher(), cache if cache is not None else Cache(), today)
    c = _Ctx(model, cf, today, tuple(sources))
    entries: Dict[str, Dict[str, Any]] = {}
    if "chebi" in sources:
        _fill_compounds(c)
    if "uniprot" in sources:
        entries = _fill_proteins(c)
    if "go" in sources:
        _fill_locations(c)
    if "uniprot" in sources and infer_reaction_ec:
        _fill_reactions(c, entries)
    _fill_pathway(c)
    have = {x.value for x in model.pathway.licences if isinstance(x.value, str)}
    names = {"UniProt": "uniprot", "ChEBI": "chebi", "Rhea": "rhea", "GO": "go", "NCBI": "ncbi", "NCBI Taxonomy": "ncbi"}
    for key in sorted({names[s] for s in c.used if s in names}):  # sorted: bytes must not depend on PYTHONHASHSEED
        if LICENCES[key] not in have:
            model.pathway.licences.append(Field(LICENCES[key], Prov(label="db", source=key, retrieved=today,
                                                                   match_type="accession")))
    c.rep.requests, c.rep.cache_hits = cf.requests, cf.hits
    return c.rep
