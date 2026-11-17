"""parse the clean-model SBML export into the a3 model format"""
import sys
sys.path.insert(0, "C:/t/corr-assess/scripts")
from lxml import etree
import a3_models as M
nn, nc, nl, fs = M.nn, M.nc, M.nl, M.fs
T2 = "http://www.t2pw.org/sbml"


def from_ours_sbml(path):
    root = etree.parse(path).getroot()
    ns = etree.QName(root).namespace
    q = lambda t: "{%s}%s" % (ns, t)
    mod = root.find(q("model"))
    m = M.new()
    comps = {c.get("id"): c.get("name") for c in (mod.find(q("listOfCompartments")) if mod.find(q("listOfCompartments")) is not None else [])}
    m["locs"] = {nl(v) for v in comps.values() if v and v != "unspecified location"}
    sp = {}
    kindmap = {}
    for s in (mod.find(q("listOfSpecies")) if mod.find(q("listOfSpecies")) is not None else []):
        kind = None
        for e in s.iter():
            if etree.QName(e).localname == "entity" and e.get("kind"): kind = e.get("kind")
        sp[s.get("id")] = dict(name=s.get("name"), kind=kind, comp=comps.get(s.get("compartment")), cid=s.get("compartment"))
        tgt = {"compound": m["comps"], "protein": m["prots"], "complex": m["cplx"], "nucleic_acid": m["nuc"], "undeclared": m["comps"]}.get(kind)
        if tgt is not None:
            tgt.add(nc(s.get("name")) if kind == "complex" else nn(s.get("name")))
    m["notes"]["n_species"] = len(sp)
    m["notes"]["undeclared_species"] = [v["name"] for v in sp.values() if v["kind"] == "undeclared"]
    m["notes"]["species_by_entity"] = {}
    for v in sp.values():
        m["notes"]["species_by_entity"].setdefault(v["name"], set()).add(v["comp"])
    def nm(sid):
        v = sp[sid]
        return (nc if v["kind"] == "complex" else nn)(v["name"])
    rl = mod.find(q("listOfReactions"))
    for r in (rl if rl is not None else []):
        g = lambda t: [x.get("species") for x in (r.find(q(t)) if r.find(q(t)) is not None else [])]
        L, R = g("listOfReactants"), g("listOfProducts")
        mods = [(x.get("species"), x.get("sboTerm")) for x in (r.find(q("listOfModifiers")) if r.find(q("listOfModifiers")) is not None else [])]
        rev = r.get("reversible")
        if r.get("sboTerm") == "SBO:0000655":
            m["trn"].append(dict(cargo=fs(nm(x) for x in L + R), frm=fs(nl(sp[x]["comp"]) for x in L), to=fs(nl(sp[x]["comp"]) for x in R),
                                 name=r.get("name"), transporters=fs(nm(x) for x, _ in mods), noop=(set(L) == set(R)), Lcomp=[sp[x]["comp"] for x in L], Rcomp=[sp[x]["comp"] for x in R]))
        else:
            locs = {sp[x]["comp"] for x in L + R}
            m["rxn"].append(dict(L=fs(nm(x) for x in L), R=fs(nm(x) for x in R), E=fs(nm(x) for x, _ in mods), loc=nl(next(iter(locs))) if len(locs) == 1 else None,
                                 dir="reversible" if rev == "true" else "irreversible", name=r.get("name"), roles=[(sp[x]["name"], s) for x, s in mods]))
    for e in root.iter("{%s}interaction" % T2) if False else root.iter():
        if etree.QName(e).localname == "interaction" and e.get("sourceName") is not None:
            m["itn"].append(dict(a=nc(e.get("sourceName")), b=nc(e.get("targetName")), type=e.get("kind"), sign=e.get("sign"), name=e.get("relationship"),
                                 undeclared=e.get("undeclaredEndpoint")))
    return m
