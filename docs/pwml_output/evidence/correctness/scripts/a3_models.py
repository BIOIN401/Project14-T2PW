"""A3: extract one normalised model per source format. Pure json/xml, no pipeline imports.
Model = dict(comps, prots, cplx, nuc : sets of norm names;
             rxn : list of dict(L,R: frozenset names, E: frozenset enzyme labels, loc, dir, name)
             trn : list of dict(cargo: frozenset, frm, to, name)
             itn : list of dict(a, b, type)
             locs: set of norm location names)
Run with C:/t/fmt-venv python.
"""
import json, re, sys, unicodedata
from lxml import etree

B = "C:/t/corr-assess"


def nn(s):
    """normalise a name: casefold, ASCII only, alnum only (non-ASCII is '?' in db.json so it is dropped everywhere)"""
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().casefold()
    s = s.replace("&rarr;", "")
    return re.sub(r"[^a-z0-9]", "", s)


def nc(s):
    """enzyme/complex key: strip a trailing word 'complex'"""
    s = str(s or "").strip()
    s = re.sub(r"\s+complex$", "", s, flags=re.I)
    return nn(s)


def nl(s):
    """location key: drop 'state', 'space', 'compartment' noise words"""
    s = str(s or "").casefold()
    s = re.sub(r"\b(state|space|compartment)\b", "", s)
    return nn(s)


def new():
    return dict(comps=set(), prots=set(), cplx=set(), nuc=set(), rxn=[], trn=[], itn=[], locs=set(), notes={})


def fs(it):
    return frozenset(x for x in it if x)


# ----------------------------------------------------------------------------------------------- payload
def from_payload(p):
    m = new()
    e = p.get("entities") or {}
    pr = p.get("processes") or {}
    for c in e.get("compounds", []): m["comps"].add(nn(c.get("name")))
    for c in e.get("proteins", []): m["prots"].add(nn(c.get("name")))
    for c in e.get("protein_complexes", []): m["cplx"].add(nc(c.get("name")))
    for c in e.get("nucleic_acids", []): m["nuc"].add(nn(c.get("name")))
    st = {}
    for s in p.get("biological_states", []) or []:
        st[s.get("name")] = s.get("subcellular_location") or s.get("name")
    loc_of = lambda sname: nl(st.get(sname, sname))
    for l in e.get("subcellular_locations", []) or []:
        m["locs"].add(nl(l.get("name")))
    cmembers = {}
    for c in e.get("protein_complexes", []):
        cmembers[nc(c.get("name"))] = [nn(x.get("name") if isinstance(x, dict) else x) for x in c.get("components", [])]
    m["notes"]["complex_members"] = cmembers
    for r in pr.get("reactions", []):
        enz = [x for x in r.get("enzymes", [])]
        mods = [x for x in r.get("modifiers", [])]
        enz_names = fs(nc(x.get("entity")) for x in enz)
        mod_roles = {}
        for x in mods:
            mod_roles[nc(x.get("entity"))] = x.get("role")
        m["rxn"].append(dict(L=fs(nn(x) for x in r.get("inputs", [])), R=fs(nn(x) for x in r.get("outputs", [])),
                             E=enz_names, MOD=mod_roles, loc=loc_of(r.get("biological_state")),
                             dir=None, name=r.get("name"), spont=r.get("spontaneous")))
    for t in pr.get("transports", []):
        els = t.get("elements_with_states") or []
        cargo = fs(nn(x.get("element")) for x in els) or fs([nn(t.get("cargo"))])
        m["trn"].append(dict(cargo=cargo, frm=loc_of(t.get("from_biological_state")), to=loc_of(t.get("to_biological_state")),
                             name=t.get("name"), transporters=fs(nc(x.get("entity") if isinstance(x, dict) else x) for x in (t.get("transporters") or []))))
    for i in pr.get("interactions", []):
        m["itn"].append(dict(a=nc(i.get("entity_1")), b=nc(i.get("entity_2")), type=i.get("relationship"), name=i.get("name")))
    return m


# ----------------------------------------------------------------------------------------------- IR
def from_ir(ir):
    m = new()
    ent = {}
    for kind, key in (("compound", "compounds"), ("protein", "proteins"), ("protein_complex", "protein_complexes"),
                      ("nucleic_acid", "nucleic_acids")):
        for e in ir["entities"].get(key, []):
            ent[(kind, e["key"])] = e
            nm = e.get("name")
            {"compound": m["comps"], "protein": m["prots"], "protein_complex": m["cplx"], "nucleic_acid": m["nuc"]}[kind].add(
                nc(nm) if kind == "protein_complex" else nn(nm))
    loc = {l["key"]: l["name"] for l in ir.get("subcellular_locations", [])}
    st = {s["key"]: nl(loc.get(s.get("subcellular_location_key"), s["name"])) for s in ir.get("biological_states", [])}
    for l in loc.values(): m["locs"].add(nl(l))

    def nm(x):
        e = ent.get((x["entity_type"], x["entity_key"]))
        return (nc if x["entity_type"] == "protein_complex" else nn)(e["name"]) if e else "?" + x["entity_key"]

    mem = {}
    for e in ir["entities"].get("protein_complexes", []):
        mem[nc(e["name"])] = [nn(ent[("protein", c["protein_key"])]["name"]) for c in e.get("components", []) if ("protein", c["protein_key"]) in ent]
    m["notes"]["complex_members"] = mem
    for r in ir["processes"].get("reactions", []):
        m["rxn"].append(dict(L=fs(nm(x) for x in r["left"]), R=fs(nm(x) for x in r["right"]),
                             E=fs(nm(x) for x in r.get("enzymes", [])), loc=st.get(r.get("biological_state_key")),
                             dir=r.get("direction"), name=r.get("name"), spont=r.get("spontaneous"),
                             roles=[x.get("role") for x in r.get("enzymes", [])]))
    for t in ir["processes"].get("transports", []):
        els = t.get("transport_elements", [])
        m["trn"].append(dict(cargo=fs(nm(x) for x in els), frm=fs(st.get(x.get("left_biological_state_key")) for x in els),
                             to=fs(st.get(x.get("right_biological_state_key")) for x in els), name=t.get("name"),
                             transporters=fs(nm(x) for x in t.get("transporters", []))))
    for i in ir["processes"].get("interactions", []):
        m["itn"].append(dict(a=nm(i["left"]), b=nm(i["right"]), type=i.get("interaction_type"), name=i.get("name"),
                             status=i.get("export_status")))
    return m


# ----------------------------------------------------------------------------------------------- PWML
def _t(e, tag):
    x = e.find(tag)
    return None if x is None else (x.text or "").strip()


def from_pwml(path):
    root = etree.parse(path).getroot()
    m = new()
    comp, prot, cx, nuc = {}, {}, {}, {}
    for e in root.iter("compound"):
        if e.getparent().tag == "compounds": comp[_t(e, "id")] = _t(e, "name")
    for e in root.iter("protein"):
        if e.getparent().tag == "proteins": prot[_t(e, "id")] = _t(e, "name")
    for e in root.iter("protein-complex"):
        if e.getparent().tag == "protein-complexes":
            cx[_t(e, "id")] = (_t(e, "name"), [x.findtext("protein-id") for x in e.iter("protein-complex-protein")])
    for e in root.iter("nucleic-acid"):
        if e.getparent().tag == "nucleic-acids": nuc[_t(e, "id")] = _t(e, "name")
    st = {}
    sl = {}
    for e in root.iter("subcellular-location"):
        if e.getparent().tag == "subcellular-locations": sl[_t(e, "id")] = _t(e, "name"); m["locs"].add(nl(_t(e, "name")))
    for e in root.iter("biological-state"):
        if e.getparent().tag == "biological-states":
            st[_t(e, "id")] = nl(sl.get(_t(e, "subcellular-location-id"), _t(e, "name")))
    m["notes"]["comp_order"] = list(comp.values())
    m["comps"] = {nn(v) for v in comp.values()}
    m["prots"] = {nn(v) for v in prot.values()}
    m["cplx"] = {nc(v[0]) for v in cx.values()}
    m["nuc"] = {nn(v) for v in nuc.values()}
    m["notes"]["complex_members"] = {nc(v[0]): [nn(prot.get(p)) for p in v[1]] for v in cx.values()}

    def el(x):
        t, i = _t(x, "element-type"), _t(x, "element-id")
        if t == "Compound": return nn(comp.get(i, "?" + i))
        if t == "Protein": return nn(prot.get(i, "?" + i))
        if t == "ProteinComplex": return nc(cx.get(i, ("?" + str(i),))[0])
        if t == "NucleicAcid": return nn(nuc.get(i, "?" + i))
        return "?" + str(t) + str(i)

    for r in root.iter("reaction"):
        if r.getparent().tag != "reactions": continue
        L = [el(x) for x in r.find("reaction-left-elements")]
        R = [el(x) for x in r.find("reaction-right-elements")]
        E = [nc(cx.get(_t(x, "protein-complex-id"), ("?" + str(_t(x, "protein-complex-id")),))[0]) for x in r.find("reaction-enzymes")]
        m["rxn"].append(dict(L=fs(L), R=fs(R), E=fs(E), loc=None, dir=_t(r, "direction"), name=_t(r, "pwr-id"),
                             spont=_t(r, "spontaneous")))
    for t in root.iter("transport"):
        if t.getparent().tag != "transports": continue
        els = list(t.find("transport-elements"))
        trs = [x for x in t.find("transport-transporters")] if t.find("transport-transporters") is not None else []
        m["trn"].append(dict(cargo=fs(el(x) for x in els), frm=fs(st.get(_t(x, "left-biological-state-id")) for x in els),
                             to=fs(st.get(_t(x, "right-biological-state-id")) for x in els), name=_t(t, "pwt-id"),
                             transporters=fs(el(x) for x in trs)))
    for t in root.iter("interaction"):
        if t.getparent().tag != "interactions": continue
        a = [el(x) for x in t.find("interaction-left-elements")]
        b = [el(x) for x in t.find("interaction-right-elements")]
        m["itn"].append(dict(a=a[0] if a else None, b=b[0] if b else None, type=_t(t, "interaction-type"), name=_t(t, "pwi-id")))
    return m


# ----------------------------------------------------------------------------------------------- PathWhiz DB
def from_db(d, cmap=None):
    m = new()
    cmap = cmap or {}
    comp = {x["id"]: cmap.get(x["id"], x["name"]) for x in d["compounds"]}
    m["notes"]["db_comp_names"] = {x["id"]: x["name"] for x in d["compounds"]}
    prot = {x["id"]: x["name"] for x in d["proteins"]}
    cx = {x["id"]: x["name"] for x in d["protein_complexes"]}
    m["comps"] = {nn(v) for v in comp.values()}
    m["prots"] = {nn(v) for v in prot.values()}
    m["cplx"] = {nc(v) for v in cx.values()}
    bs = {x["id"]: nl(x["location"]) for x in d["biological_states"]}
    m["locs"] = set(bs.values())
    rx = {}
    for x in d["reaction_elements"]:
        nmx = comp.get(x["element_id"], x["name"]) if x.get("element_type") == "Compound" else x["name"]
        rx.setdefault(x["reaction_id"], {"L": [], "R": []})["L" if x["type"] == "ReactionLeftElement" else "R"].append(nn(nmx))
    enz = {}
    for x in d["reaction_enzymes"]:
        enz.setdefault(x["reaction_id"], []).append(nc(x["complex_name"]))
    dirs = {x["id"]: x["direction"] for x in d["reactions"]}
    # reaction -> state via pathway_elements (ReactionEnzyme rows) is unreliable; leave loc None
    for rid, v in rx.items():
        m["rxn"].append(dict(L=fs(v["L"]), R=fs(v["R"]), E=fs(enz.get(rid, [])), loc=None, dir=dirs.get(rid), name=rid))
    te = {}
    for x in d["transport_elements"]:
        te.setdefault(x["transport_id"], []).append(x)
    ttr = {}
    for x in d.get("transport_transporters", []):
        ttr.setdefault(x.get("transport_id"), []).append(x)
    for tid, els in te.items():
        m["trn"].append(dict(cargo=fs(nn(comp.get(x["element_id"], x["name"]) if x.get("element_type") == "Compound" else x["name"]) for x in els), frm=fs(bs.get(x["left_state"]) for x in els),
                             to=fs(bs.get(x["right_state"]) for x in els), name=tid,
                             transporters=fs(nc(x.get("name") or x.get("complex_name")) for x in ttr.get(tid, []))))
    ie = {}
    for x in d["interaction_elements"]:
        ie.setdefault(x["interaction_id"], {})[x["type"]] = x
    ity = {x["id"]: x["interaction_type"] for x in d["interactions"]}
    for iid, v in ie.items():
        l = v.get("InteractionLeftElement"); r = v.get("InteractionRightElement")
        cn = lambda x: (nn(comp.get(x["element_id"], x["name"])) if x.get("element_type") == "Compound" else nc(x["name"])) if x else None
        m["itn"].append(dict(a=cn(l), b=cn(r), type=ity.get(iid), name=iid))
    return m


# ----------------------------------------------------------------------------------------------- PathWhiz SBML (dedup by pathwhiz:reaction_id)
NS = {"s": "http://www.sbml.org/sbml/level3/version1/core"}


def _local(e): return etree.QName(e).localname


def from_pw_sbml(path, cmap=None):
    cmap = cmap or {}
    root = etree.parse(path).getroot()
    ns = etree.QName(root).namespace
    q = lambda t: "{%s}%s" % (ns, t)
    mod = root.find(q("model"))
    m = new()
    sp = {}
    for s in mod.find(q("listOfSpecies")) if mod.find(q("listOfSpecies")) is not None else []:
        typ = None
        for a in s.iter():
            if _local(a) == "species" and a.get("{http://www.spmdb.ca/pathwhiz}species_type"):
                typ = a.get("{http://www.spmdb.ca/pathwhiz}species_type")
        nmx = s.get("name")
        mid = re.match(r"^Compound(\d+)$", s.get("id") or "")
        if typ == "compound" and mid:
            if nn(cmap.get(mid.group(1), nmx)) != nn(nmx): m["notes"].setdefault("label_diff", []).append((nmx, cmap.get(mid.group(1))))
            nmx = cmap.get(mid.group(1), nmx)
        sp[s.get("id")] = dict(name=nmx, type=typ, comp=s.get("compartment"))
        {"compound": m["comps"], "protein": m["prots"], "protein_complex": m["cplx"], "nucleic_acid": m["nuc"]}.get(typ, set()).add(
            nc(nmx) if typ == "protein_complex" else nn(nmx))
    comps = {c.get("id"): c.get("name") for c in mod.find(q("listOfCompartments")) or []}
    m["locs"] = {nl(re.sub(r"^.*?,\s*", "", v)) for v in comps.values() if v}
    seen = set()
    m["notes"]["species_n"] = len(sp)
    m["notes"]["species_comp"] = {k: v["comp"] for k, v in sp.items()}
    m["notes"]["n_listed_reactions"] = len(list(mod.find(q("listOfReactions")) or []))
    for r in mod.find(q("listOfReactions")) or []:
        rid = r.get("id")
        kind = ("transport" if rid.startswith("Transport") else "interaction" if rid.startswith("Interaction") else "reaction")
        pwid = None
        for a in r.iter():
            if _local(a) == "reaction" and a.get("{http://www.spmdb.ca/pathwhiz}reaction_id"):
                pwid = a.get("{http://www.spmdb.ca/pathwhiz}reaction_id")
        L = [x.get("species") for x in (r.find(q("listOfReactants")) or [])]
        R = [x.get("species") for x in (r.find(q("listOfProducts")) or [])]
        M = [x.get("species") for x in (r.find(q("listOfModifiers")) or [])]
        key = (kind, pwid)
        nmL = lambda ids: [(nc if sp[i]["type"] == "protein_complex" else nn)(sp[i]["name"]) for i in ids]
        rec = dict(L=fs(nmL(L)), R=fs(nmL(R)), E=fs(nmL(M)), loc=nl(re.sub(r"^.*?,\s*", "", comps.get(r.get("compartment"), ""))),
                   dir="reversible" if r.get("reversible") == "true" else "irreversible", name=r.get("name"), sbo=r.get("sboTerm"),
                   Lspecies=L, Rspecies=R)
        if kind == "reaction":
            m["rxn"].append(dict(rec, dup_of_visualization=key in seen)); seen.add(key)
        elif kind == "transport":
            if key in seen: continue
            seen.add(key)
            m["trn"].append(dict(cargo=rec["L"] | rec["R"], frm=frozenset([sp[L[0]]["comp"]]) if L else frozenset(),
                                 to=frozenset([sp[R[0]]["comp"]]) if R else frozenset(), name=r.get("name"), L=L, R=R))
        else:
            m["itn"].append(dict(a=next(iter(rec["L"]), None), b=next(iter(rec["R"]), None), type=(r.get("name") or ""), name=rid))
    return m


# ----------------------------------------------------------------------------------------------- PathWhiz SBGN
def from_pw_sbgn(path, cmap=None):
    cmap = cmap or {}
    root = etree.parse(path).getroot()
    ns = etree.QName(root).namespace
    q = lambda t: "{%s}%s" % (ns, t)
    mp = root.find(q("map"))
    m = new()
    g = {}
    for e in mp.iter(q("glyph")):
        lab = e.find(q("label"))
        labt = lab.get("text") if lab is not None else None
        mid = re.match(r"^Compound(\d+)\.", e.get("id") or "")
        if mid and e.get("class") == "simple chemical":
            if nn(cmap.get(mid.group(1), labt)) != nn(labt): m["notes"].setdefault("label_diff", []).append((labt, cmap.get(mid.group(1))))
            labt = cmap.get(mid.group(1), labt)
        g[e.get("id")] = dict(cls=e.get("class"), label=labt, comp=e.get("compartmentRef"))
    kind = {"simple chemical": "comps", "macromolecule": "prots", "nucleic acid feature": "nuc", "complex": "cplx"}
    for v in g.values():
        if v["cls"] in kind and v["label"]: m[kind[v["cls"]]].add(nn(v["label"]))
    proc = {}
    arcs = []
    for a in mp.iter(q("arc")):
        arcs.append(dict(cls=a.get("class"), src=a.get("source"), tgt=a.get("target"), id=a.get("id")))
    for gid, v in g.items():
        if v["cls"] == "process": proc[gid] = dict(L=[], R=[], E=[], other=[], glyph=v)

    def base(pid):  # arc end may refer to a port id like Reaction1.1.1
        return pid if pid in g else pid.rsplit(".", 1)[0]
    stray = []
    for a in arcs:
        s, t = base(a["src"]), base(a["tgt"])
        if a["cls"] == "consumption" and t in proc: proc[t]["L"].append(nn(g[s]["label"]))
        elif a["cls"] == "production" and s in proc: proc[s]["R"].append(nn(g[t]["label"]))
        elif a["cls"] in ("catalysis", "stimulation", "necessary stimulation", "modulation") and t in proc: proc[t]["E"].append(nn(g[s]["label"]))
        elif a["cls"] == "inhibition" and t in proc: proc[t]["other"].append(("inh", nn(g[s]["label"])))
        else: stray.append((a["cls"], a["src"], a["tgt"]))
    m["notes"]["stray_arcs"] = stray
    seen = set()
    for pid, v in proc.items():
        key = pid.split(".")[0]  # Reaction1 / Transport1 / Interaction1 ; one glyph per visualization
        if pid.startswith("Reaction"):
            m["rxn"].append(dict(L=fs(v["L"]), R=fs(v["R"]), E=fs(v["E"]), loc=nl(g.get(v["glyph"]["comp"], {}).get("label") or ""),
                                 dir=None, name=pid, ident=key))
        elif pid.startswith("Transport"):
            m["trn"].append(dict(cargo=fs(v["L"]) | fs(v["R"]), frm=None, to=None, name=pid, L=v["L"], R=v["R"], compL=None))
        else:
            srcs = [x[1] for x in v["other"]] + v["E"]
            m["itn"].append(dict(a=srcs[0] if srcs else None, b=(v["L"] or [None])[0], type=(v["other"][0][0] if v["other"] else "stim"), name=pid,
                                 b_prod=(v["R"] or [None])[0]))
    m["notes"]["glyph_classes"] = {c: sum(1 for v in g.values() if v["cls"] == c) for c in sorted({v["cls"] for v in g.values()})}
    m["notes"]["compartment_labels"] = sorted({v["label"] for v in g.values() if v["cls"] == "compartment"})
    return m


# ----------------------------------------------------------------------------------------------- web diagram (A2 prototype model)
def from_web(ir):
    sys.path.insert(0, B + "/a2/proto")
    import t2pw_diagrams as D
    D.OPTS.clear()
    mm = D.build(ir)
    m = new()
    for n in mm.nodes.values():
        if n.kind == "compound": m["comps"].add(nn(n.label))
        elif n.kind == "protein": m["prots"].add(nn(n.label))
        elif n.kind == "complex": m["cplx"].add(nc(n.label))
        elif n.kind == "nucleic": m["nuc"].add(nn(n.label))
    m["locs"] = {nl(c["name"]) for c in mm.comps.values() if c["nodes"] or True}
    for p in mm.procs:
        if p["kind"] == "reaction":
            J = p["node"]
            m["rxn"].append(dict(L=fs(nn(s.label) for s in J.subs), R=fs(nn(s.label) for s in J.prods),
                                 E=fs((nc if z.kind == "complex" else nn)(z.label) for z in J.enz), loc=nl(p["where"]), dir=None, name=p["tag"]))
        elif p["kind"] == "transport":
            J = p["node"]
            m["trn"].append(dict(cargo=fs(nn(s.label) for s in J.srcs), frm=fs(nl(mm.comps[a]["name"]) for _, a, b in J.pairs),
                                 to=fs(nl(mm.comps[b]["name"]) for _, a, b in J.pairs), name=p["tag"],
                                 transporters=fs((nc if t[0] == "protein_complex" else nn)(next((e["name"] for e in ir["entities"][D.ENT_KEYS[t[0]]] if e["key"] == t[1]), "")) for t in J.trn_ents)))
        else:
            e = p["edge"]
            m["itn"].append(dict(a=(nc if e.a.kind == "complex" else nn)(e.a.label), b=(nc if e.b.kind == "complex" else nn)(e.b.label),
                                 type=p["itype"], name=p["tag"], withheld=p.get("withheld")))
    m["notes"]["skipped_interactions"] = mm.skipped
    return m
