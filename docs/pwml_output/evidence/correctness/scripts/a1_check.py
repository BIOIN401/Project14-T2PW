"""A1 per-pathway checks: final_payload.json vs PWML (layout v2.1) vs PathWhiz DB snapshot.

usage: python a1_check.py            -> C:/t/corr-assess/a1/per_pathway.json
"""
import collections, json, os, re, sys
import xml.etree.ElementTree as ET

BASE = "C:/t/corr-assess/base"
PWML = "C:/t/corr-assess/replay/on_v2/pwml"
EXP = "C:/t/corr-assess/pathwhiz_exports"
OUT = "C:/t/corr-assess/a1/per_pathway.json"
IMPORT = "C:/t/corr-assess/a1/import_results.jsonl"

# location names PathWhiz/GO treat as the same compartment of a cell (used only to FLAG near-synonyms)
CURRENCY = {"atp", "adp", "amp", "nadh", "nad+", "nad", "nadph", "nadp+", "nadp", "h2o", "water", "coa", "coa-sh", "coenzyme a", "co2", "carbon dioxide", "h+", "hydrogen ion",
            "phosphate", "pi", "ppi", "pyrophosphate", "diphosphate", "sam", "s-adenosyl-l-methionine", "s-adenosylmethionine", "sah", "s-adenosylhomocysteine", "o2", "oxygen", "nh3", "ammonia",
            "udp", "gtp", "gdp", "fad", "fadh2", "atp"}
SYN = {"cell": "cell-interior", "cytoplasm": "cell-interior", "cytosol": "cell-interior", "intracellular": "cell-interior",
       "intracellular space": "cell-interior", "cytoplasmic": "cell-interior",
       "extracellular": "extracellular", "extracellular space": "extracellular", "extracellular region": "extracellular",
       "plasma membrane": "membrane", "cell membrane": "membrane", "membrane": "membrane", "cytoplasmic membrane": "membrane",
       "endoplasmic reticulum": "er", "synaptic vesicle": "synvesicle", "synaptic vesicles": "synvesicle",
       "plastid": "plastid", "chloroplast": "plastid", "etioplast": "plastid"}


def n(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def txt(e, tag, default=""):
    c = e.find(tag)
    return (c.text or "").strip() if c is not None and c.text else default


def parse_pwml(path):
    root = ET.parse(path).getroot()
    pv = root.find(".//pathway-visualization")
    P = {"canvas": (int(txt(pv, "width", "0")), int(txt(pv, "height", "0")))}
    P["species"] = {txt(s, "id"): txt(s, "name") for s in pv.find("species")}
    P["locs"] = {txt(s, "id"): txt(s, "name") for s in pv.find("subcellular-locations")}
    P["states"] = {txt(b, "id"): {"name": txt(b, "name"), "loc": P["locs"].get(txt(b, "subcellular-location-id"), ""),
                                  "species": P["species"].get(txt(b, "species-id"), "")} for b in pv.find("biological-states")}
    nm = {}
    for sec, et in (("compounds", "Compound"), ("nucleic-acids", "NucleicAcid"), ("element-collections", "ElementCollection"),
                    ("protein-complexes", "ProteinComplex"), ("proteins", "Protein")):
        s = pv.find(sec)
        for e in (s if s is not None else []):
            nm[(et, txt(e, "id"))] = txt(e, "name")
    P["names"] = nm
    P["complex_proteins"] = {txt(c, "id"): [nm.get(("Protein", txt(x, "protein-id")), "?") for x in c.iter("protein-complex-protein")]
                             for c in pv.find("protein-complexes")}
    rx = []
    for r in pv.find("reactions"):
        side = lambda tag: [(txt(e, "element-type"), txt(e, "element-id"), txt(e, "stoichiometry")) for e in r.find(tag)]
        rx.append({"id": txt(r, "id"), "direction": txt(r, "direction"), "spontaneous": txt(r, "spontaneous"),
                   "left": side("reaction-left-elements"), "right": side("reaction-right-elements"),
                   "enz": [txt(e, "protein-complex-id") for e in r.find("reaction-enzymes")]})
    P["reactions"] = rx
    P["rviz"] = {}
    for v in pv.find("reaction-visualizations"):
        P["rviz"][txt(v, "reaction-id")] = {
            "state": txt(v, "biological-state-id"),
            "cvis": [(txt(c, "compound-location-id"), txt(c, "side")) for c in v.iter("reaction-compound-visualization")]
            + [(txt(c, "nucleic-acid-location-id"), txt(c, "side")) for c in v.iter("reaction-nucleic-acid-visualization")]
            + [(txt(c, "element-collection-location-id"), txt(c, "side")) for c in v.iter("reaction-element-collection-visualization")],
            "evis": [txt(c, "protein-complex-visualization-id") for c in v.iter("reaction-enzyme-visualization")]}
    P["cloc"] = {}
    for l in pv.find("compound-locations"):
        P["cloc"][txt(l, "id")] = {"compound": txt(l, "compound-id"), "state": txt(l, "biological-state-id"), "x": int(txt(l, "x", "0")),
                                   "y": int(txt(l, "y", "0")), "w": int(float(txt(l, "width", "0"))), "h": int(float(txt(l, "height", "0"))),
                                   "hidden": txt(l, "hidden")}
    P["ploc"] = {txt(l, "id"): {"protein": txt(l, "protein-id"), "state": txt(l, "biological-state-id"), "x": int(txt(l, "x", "0")),
                                "y": int(txt(l, "y", "0")), "w": int(float(txt(l, "width", "0"))), "h": int(float(txt(l, "height", "0"))),
                                "hidden": txt(l, "hidden")} for l in pv.find("protein-locations")}
    P["pcv"] = {txt(v, "id"): {"complex": txt(v, "protein-complex-id"), "state": txt(v, "biological-state-id"),
                               "plocs": [txt(x, "protein-location-id") for x in v.iter("protein-complex-protein-visualization")]}
                for v in pv.find("protein-complex-visualizations")}
    tr = []
    for t in pv.find("transports"):
        tr.append({"id": txt(t, "id"),
                   "elements": [(txt(e, "element-type"), txt(e, "element-id"), txt(e, "left-biological-state-id"),
                                 txt(e, "right-biological-state-id"), txt(e, "direction")) for e in t.iter("transport-element")],
                   "transporters": [(txt(e, "protein-complex-id"), txt(e, "biological-state-id")) for e in t.iter("transport-transporter")]})
    P["transports"] = tr
    P["tviz"] = {txt(v, "transport-id"): {"cvis": [(txt(c, "compound-location-id"), txt(c, "side")) for c in v.iter("transport-compound-visualization")],
                                          "tvis": [txt(c, "protein-complex-visualization-id") for c in v.iter("transport-transporter-visualization")]}
                 for v in pv.find("transport-visualizations")}
    ints = []
    for i in pv.find("interactions"):
        ints.append({"id": txt(i, "id"), "type": txt(i, "interaction-type"),
                     "left": [(txt(e, "element-type"), txt(e, "element-id")) for e in i.find("interaction-left-elements")],
                     "right": [(txt(e, "element-type"), txt(e, "element-id")) for e in i.find("interaction-right-elements")]})
    P["interactions"] = ints
    P["edges"] = {txt(e, "id"): txt(e, "path") for e in pv.find("edges")}
    P["membranes"] = []
    for m in pv.find("membrane-visualizations"):
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", txt(m, "path"))]
        xs, ys = nums[0::2], nums[1::2]
        P["membranes"].append({"x0": min(xs), "x1": max(xs), "y0": min(ys), "y1": max(ys)} if xs else {})
    P["labels"] = [{"text": txt(l, "label-text"), "x": int(txt(l, "x", "0")), "y": int(txt(l, "y", "0"))} for l in pv.find("label-locations")]
    P["counts"] = {s: len(pv.find(s)) for s in ("compounds", "proteins", "protein-complexes", "reactions", "transports", "interactions",
                                                "compound-locations", "protein-locations", "edges", "biological-states", "membrane-visualizations", "label-locations")}
    return P


def payload_info(d, alias=None):
    pr = d["processes"]
    rx = []
    al = lambda x: (alias or {}).get(n(x), n(x))
    for r in pr["reactions"]:
        ents = list(r.get("enzymes") or []) + list(r.get("modifiers") or [])
        cat = {n(e["entity"]) for e in ents if e.get("role", "catalyst") == "catalyst"}
        noncat = {n(e["entity"]): e.get("role") for e in ents if e.get("role", "catalyst") != "catalyst"}
        rx.append({"name": r.get("name", ""), "state": r.get("biological_state"), "in": sorted({al(x) for x in r["inputs"]}), "out": sorted({al(x) for x in r["outputs"]}),
                   "enz": sorted(cat | set(noncat)), "cat": sorted(cat), "noncat": noncat})
    return {"reactions": rx, "transports": pr["transports"], "interactions": pr["interactions"], "states": d["biological_states"],
            "order": d.get("reaction_order", []), "quarantined": len(d.get("quarantined_locked_reactions", []))}


def dup_report(sigs):
    """sigs: list of (frozenset in, frozenset out, frozenset enz). exact duplicate / reverse duplicate / same-substrate-product pairs"""
    c = collections.Counter(sigs)
    exact = sum(v - 1 for v in c.values() if v > 1)
    rev = 0
    seen = set()
    for (i, o, e) in c:
        if (o, i, e) in c and (i, o, e) != (o, i, e) and ((o, i, e), (i, o, e)) not in seen:
            seen.add(((i, o, e), (o, i, e)))
            rev += 1
    chem = collections.Counter((i, o) for (i, o, e) in sigs)
    same_chem = sum(v - 1 for v in chem.values() if v > 1)
    revchem = sum(1 for (i, o) in chem if (o, i) in chem and i != o) // 2
    return {"exact_dup": exact, "reverse_dup_pairs_same_enzyme": rev, "same_inout_diff_enzyme": same_chem - exact, "reverse_pairs_any_enzyme": revchem}


def check(idx):
    res = {"idx": idx}
    d = json.load(open("%s/%d/final_payload.json" % (BASE, idx), encoding="utf-8"))
    P = parse_pwml("%s/%d.pwml" % (PWML, idx))
    nm = P["names"]
    # payload compound name -> PWML compound name (writer writes the DB name; match through db_id == PWML compound id)
    alias = {}
    for c in d["entities"]["compounds"]:
        w = nm.get(("Compound", str(c.get("db_id"))))
        if w:
            alias[n(c["name"])] = n(w)
    PL = payload_info(d, alias)
    pn = lambda et, i: n(nm.get((et, i), "?"))
    res["canvas"] = P["canvas"]
    # ---------- reactions: payload vs PWML
    pl_sigs = [(frozenset(r["in"]), frozenset(r["out"]), frozenset(r["enz"])) for r in PL["reactions"]]
    pw_sigs = []
    for r in P["reactions"]:
        pw_sigs.append((frozenset(pn(*(e[:2])) for e in r["left"]), frozenset(pn(*(e[:2])) for e in r["right"]),
                        frozenset(n(nm.get(("ProteinComplex", c), "?")) for c in r["enz"])))
    res["reactions"] = {"payload": len(pl_sigs), "pwml": len(pw_sigs), "pwml_viz": len(P["rviz"])}
    cp, cw = collections.Counter(pl_sigs), collections.Counter(pw_sigs)
    res["reactions"]["missing_in_pwml"] = [[sorted(k[0]), sorted(k[1]), sorted(k[2])] for k in (cp - cw)]
    res["reactions"]["extra_in_pwml"] = [[sorted(k[0]), sorted(k[1]), sorted(k[2])] for k in (cw - cp)]
    roles = collections.Counter()
    nc_as_enz = nc_total = 0
    nc_ex = []
    for r in PL["reactions"]:
        for ent, role in r["noncat"].items():
            roles[role] += 1
            nc_total += 1
            if any(s[0] == frozenset(r["in"]) and s[1] == frozenset(r["out"]) and ent in s[2] for s in pw_sigs):
                nc_as_enz += 1
                if len(nc_ex) < 4:
                    nc_ex.append([r["name"], ent, role])
    res["noncatalyst_modifiers"] = {"payload_roles": dict(roles), "total": nc_total, "written_as_reaction_enzyme": nc_as_enz, "examples": nc_ex}
    res["payload_dups"] = dup_report(pl_sigs)
    res["pwml_dups"] = dup_report(pw_sigs)
    # enzyme-less reactions
    res["reactions"]["payload_without_enzyme"] = sum(1 for s in pl_sigs if not s[2])
    res["reactions"]["pwml_without_enzyme"] = sum(1 for s in pw_sigs if not s[2])
    # directions
    res["reactions"]["direction_values"] = dict(collections.Counter(r["direction"] for r in P["reactions"]))
    # ---------- side vs role (viz side vs list)
    bad_side = ambiguous = total = missing_member = 0
    for r in P["reactions"]:
        v = P["rviz"].get(r["id"])
        if not v:
            missing_member += len(r["left"]) + len(r["right"])
            continue
        left_c = {e[1] for e in r["left"] if e[0] == "Compound"}
        right_c = {e[1] for e in r["right"] if e[0] == "Compound"}
        drawn = collections.Counter()
        for loc, side in v["cvis"]:
            c = P["cloc"].get(loc, {}).get("compound")
            if c is None:
                continue
            total += 1
            drawn[c] += 1
            if c in left_c and c in right_c:
                ambiguous += 1
            elif (c in left_c and side != "Left") or (c in right_c and side != "Right"):
                bad_side += 1
        for c in left_c | right_c:
            if not drawn[c]:
                missing_member += 1
    res["side_vs_role"] = {"members": total, "side_contradicts_list": bad_side, "on_both_lists": ambiguous, "elements_not_drawn": missing_member}
    # ---------- enzyme attachment: reaction -> enzyme viz -> complex == reaction-enzyme complex
    wrong_enz = no_enz_viz = 0
    for r in P["reactions"]:
        v = P["rviz"].get(r["id"])
        want = sorted(r["enz"])
        got = sorted(P["pcv"].get(x, {}).get("complex") for x in (v["evis"] if v else []))
        if want and not got:
            no_enz_viz += 1
        elif want != got:
            wrong_enz += 1
    res["enzymes"] = {"reaction_enzyme_rows": sum(len(r["enz"]) for r in P["reactions"]), "payload_enzyme_refs": sum(len(r["enz"]) for r in PL["reactions"]),
                      "viz_complex_mismatch": wrong_enz, "enzyme_not_drawn": no_enz_viz}
    # ---------- states / compartments
    used = collections.Counter()
    for v in P["rviz"].values():
        used[v["state"]] += 1
    for c in P["cloc"].values():
        used[c["state"]] += 1
    for p in P["ploc"].values():
        used[p["state"]] += 1
    for t in P["transports"]:
        for e in t["elements"]:
            used[e[2]] += 1
            used[e[3]] += 1
    states = P["states"]
    res["states"] = {"payload": len(PL["states"]), "pwml": len(states), "used_in_pwml": sorted(states[s]["name"] for s in used if s in states),
                     "defined_unused": sorted(states[s]["name"] for s in states if s not in used)}
    pl_names = [s["name"] for s in PL["states"]]
    res["states"]["payload_dup_names"] = sorted(k for k, v in collections.Counter(pl_names).items() if v > 1)
    # near-synonym groups among USED state locations
    byloc = collections.defaultdict(set)
    for s in used:
        if s in states:
            byloc[SYN.get(n(states[s]["loc"]), n(states[s]["loc"]))].add(states[s]["loc"] + "|" + states[s]["name"])
    res["states"]["synonym_groups_used"] = {k: sorted(v) for k, v in byloc.items() if len({x.split("|")[0] for x in v}) > 1 or len(v) > 1}
    labels = [l["text"] for l in P["labels"]]
    meta = json.load(open("%s/%d/meta.json" % (BASE, idx), encoding="utf-8"))
    res["species"] = {"organism": meta.get("organism"), "payload_species": [s.get("name") for s in d["entities"].get("species", [])],
                      "pwml_species": list(P["species"].values()), "state_species_in_pwml": sorted({s["species"] for s in states.values()})}
    res["states"]["null_state_reactions_payload"] = sum(1 for r in PL["reactions"] if not r["state"])
    res["compartment_boxes"] = {"boxes": len(P["membranes"]), "labels": labels, "dup_labels": sorted(k for k, v in collections.Counter(n(x) for x in labels).items() if v > 1)}
    sp = {states[s]["species"] for s in used if s in states}
    res["states"]["species_among_used"] = sorted(sp)
    strain = [l for l in labels if re.search(r"\b[A-Z]\. ?[a-z]+|\bstrain\b|[A-Z]{2,}\d+", l)]
    res["compartment_boxes"]["organism_or_strain_named"] = strain
    # states the payload reactions sit in vs states PWML draws them in
    plst = collections.Counter(r["state"] for r in PL["reactions"])
    pwst = collections.Counter(states.get(v["state"], {}).get("name") for v in P["rviz"].values())
    res["states"]["reaction_state_payload"] = dict(plst)
    res["states"]["reaction_state_pwml"] = dict(pwst)
    res["states"]["reaction_state_mismatch"] = int(sum(((collections.Counter({k: v for k, v in plst.items() if k}) - pwst) + (pwst - collections.Counter({k: v for k, v in plst.items() if k}))).values()) // 2) if all(plst) else None
    # ---------- transports
    tp = []
    for t in PL["transports"]:
        tp.append((n(t["cargo"]), n(t["from_biological_state"]), n(t["to_biological_state"]), len(t.get("transporters") or [])))
    tw = []
    for t in P["transports"]:
        for e in t["elements"]:
            tw.append((pn(e[0], e[1]), n(states.get(e[2], {}).get("name")), n(states.get(e[3], {}).get("name")), len(t["transporters"])))
    res["transports"] = {"payload": len(tp), "pwml": len(P["transports"]), "pwml_elements": len(tw),
                         "missing_in_pwml": [list(x) for x in (collections.Counter(tp) - collections.Counter(tw))],
                         "extra_in_pwml": [list(x) for x in (collections.Counter(tw) - collections.Counter(tp))],
                         "transporters_payload": sum(x[3] for x in tp), "transporters_pwml": sum(len(t["transporters"]) for t in P["transports"]),
                         "drawn": len(P["tviz"]),
                         "same_state_both_ends_pwml": sum(1 for x in tw if x[1] == x[2])}
    res["interactions"] = {"payload": len(PL["interactions"]), "pwml": len(P["interactions"])}
    # ---------- drawing: shared compounds redrawn, chain order
    per = collections.defaultdict(list)
    for lid, c in P["cloc"].items():
        per[(c["compound"], c["state"])].append(lid)
    drawn_locs = len(P["cloc"])
    distinct = len(per)
    multi = {k: v for k, v in per.items() if len(v) > 1}
    used_per_rx = collections.Counter()
    for v in P["rviz"].values():
        for loc, _ in v["cvis"]:
            used_per_rx[loc] += 1
    # shared compounds: appear in >1 reaction of the same state
    comp_rx = collections.defaultdict(set)
    for r in P["reactions"]:
        st = P["rviz"].get(r["id"], {}).get("state")
        for e in r["left"] + r["right"]:
            comp_rx[(e[1], st)].add(r["id"])
    shared = {k for k, v in comp_rx.items() if len(v) > 1}
    shared_redrawn = sum(1 for k in shared if len(per.get(k, [])) > 1)
    # drawn components (locations joined by reactions) vs logical components (reactions joined by compound sharing)
    par = {}

    def f(x):
        while par.setdefault(x, x) != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for r in P["reactions"]:
        v = P["rviz"].get(r["id"])
        if not v:
            continue
        locs = ["L" + l for l, _ in v["cvis"]] + ["E" + e for e in v["evis"]]
        for a in locs[1:]:
            par[f(a)] = f(locs[0])
    drawn_comp = len({f(x) for x in list(par)})
    par2 = {}

    def g(x):
        while par2.setdefault(x, x) != x:
            par2[x] = par2[par2[x]]
            x = par2[x]
        return x
    for r in P["reactions"]:
        for e in r["left"] + r["right"]:
            par2[g("C" + e[1])] = g("R" + r["id"])
    logical_comp = len({g(x) for x in list(par2)})
    res["drawing"] = {"compound_locations": drawn_locs, "distinct_compound_state": distinct, "compounds_with_multiple_locs": len(multi),
                      "shared_compound_state": len(shared), "shared_redrawn": shared_redrawn,
                      "drawn_components": drawn_comp, "logical_components": logical_comp,
                      "hidden_locs": sum(1 for c in P["cloc"].values() if c["hidden"] == "true")}
    # flow order: for A->B shared product/substrate within same state, x(A) < x(B)?
    def rx_x(r):
        v = P["rviz"].get(r["id"])
        if v and v["evis"]:
            pl = [P["ploc"][p] for e in v["evis"] for p in P["pcv"].get(e, {}).get("plocs", []) if p in P["ploc"]]
            if pl:
                return (sum(p["x"] + p["w"] / 2 for p in pl) / len(pl), sum(p["y"] + p["h"] / 2 for p in pl) / len(pl))
        if v and v["cvis"]:
            cs = [P["cloc"][l] for l, _ in v["cvis"] if l in P["cloc"]]
            return (sum(c["x"] + c["w"] / 2 for c in cs) / len(cs), sum(c["y"] + c["h"] / 2 for c in cs) / len(cs))
        return None
    prod = collections.defaultdict(list)
    cons = collections.defaultdict(list)
    for i, r in enumerate(P["reactions"]):
        for e in r["right"]:
            prod[e[1]].append(i)
        for e in r["left"]:
            cons[e[1]].append(i)
    prec = back = 0
    back_ex = []
    for c in set(prod) & set(cons):
        for a in prod[c]:
            for b in cons[c]:
                if a == b:
                    continue
                if P["rviz"].get(P["reactions"][a]["id"], {}).get("state") != P["rviz"].get(P["reactions"][b]["id"], {}).get("state"):
                    continue  # cross-state precedence is not a same-box ordering claim
                pa, pb = rx_x(P["reactions"][a]), rx_x(P["reactions"][b])
                if pa is None or pb is None:
                    continue
                if n(nm.get(("Compound", c), "")) in CURRENCY:
                    continue
                xa, xb = pa[0], pb[0]
                prec += 1
                # reading order = left to right; a wrap to a later row (>= 100 px lower) also counts as forward
                if not (xb > xa or pb[1] > pa[1] + 100):
                    back += 1
                    if len(back_ex) < 4:
                        back_ex.append([nm.get(("Compound", c), c), round(xa), round(xb)])
    res["flow"] = {"precedence_pairs": prec, "right_to_left_pairs": back, "examples": back_ex}
    # per-state extents, node overlap, canvas fill
    xs = [c["x"] for c in P["cloc"].values()] + [p["x"] for p in P["ploc"].values()]
    ys = [c["y"] for c in P["cloc"].values()] + [p["y"] for p in P["ploc"].values()]
    res["canvas_use"] = {"w": P["canvas"][0], "h": P["canvas"][1], "x_span": (max(xs) - min(xs)) if xs else 0, "y_span": (max(ys) - min(ys)) if ys else 0,
                         "fit_scale_1600x900": round(min(1600 / max(1, P["canvas"][0]), 900 / max(1, P["canvas"][1])), 3)}
    res["pwml_counts"] = P["counts"]
    # ---------- DB snapshot
    dbp = "%s/%d.db.json" % (EXP, idx)
    if os.path.exists(dbp):
        db = json.load(open(dbp, encoding="utf-8"))
        res["db"] = db_check(db, P, pw_sigs, pn, states, tw, tp)
    else:
        res["db"] = None
    return res


def db_check(db, P, pw_sigs, pn, states, tw, tp):
    out = {"counts": {k: v for k, v in db["counts"].items() if k in ("reactions", "reaction_elements", "reaction_enzymes", "compounds", "proteins", "protein_complexes",
                                                                     "biological_states", "subcellular_locations", "transports", "transport_elements", "transport_transporters",
                                                                     "interactions", "interaction_elements", "compound_locations", "protein_locations", "element_states", "pathway_elements")}}
    na = lambda s: "".join(ch if ord(ch) < 128 else "?" for ch in n(s))  # mysql client returned non-ascii as ?
    pw_sigs = [tuple(frozenset(na(x) for x in grp) for grp in sg) for sg in pw_sigs]
    el = collections.defaultdict(lambda: {"l": [], "r": []})
    for e in db["reaction_elements"]:
        el[e["reaction_id"]]["l" if "Left" in e["type"] else "r"].append(na(e["name"]))
    enz = collections.defaultdict(list)
    for e in db["reaction_enzymes"]:
        enz[e["reaction_id"]].append(na(e["complex_name"]))
    sigs = []
    for r in db["reactions"]:
        sigs.append((frozenset(el[r["id"]]["l"]), frozenset(el[r["id"]]["r"]), frozenset(enz[r["id"]])))
    cd, cw = collections.Counter(sigs), collections.Counter(pw_sigs)
    out["reactions_db"] = len(sigs)
    out["reactions_pwml_distinct"] = len(cw)
    out["reactions_merged_by_importer"] = len(pw_sigs) - len(sigs)
    out["reactions_pwml_not_in_db"] = [[sorted(k[0]), sorted(k[1]), sorted(k[2])] for k in (set(cw) - set(cd))]
    out["reactions_db_not_in_pwml"] = [[sorted(k[0]), sorted(k[1]), sorted(k[2])] for k in (set(cd) - set(cw))]
    # element row counts: expected from PWML
    out["reaction_elements_expected"] = sum(len(r["left"]) + len(r["right"]) for r in P["reactions"])
    out["reaction_enzymes_expected"] = sum(len(r["enz"]) for r in P["reactions"])
    # transports (DB): compare by location names
    bs = {b["id"]: n(b["location"]) for b in db["biological_states"]}
    dtw = [(na(e["name"]), bs.get(e["left_state"]), bs.get(e["right_state"])) for e in db["transport_elements"]]
    pww = [(na(a), n(states.get(l, {}).get("loc")), n(states.get(r, {}).get("loc"))) for t in P["transports"] for (et, ei, l, r, dd), a in
           [((e), pn(e[0], e[1])) for e in t["elements"]]]
    out["transports_db_vs_pwml_missing"] = [list(x) for x in (collections.Counter(pww) - collections.Counter(dtw))]
    out["transports_db_vs_pwml_extra"] = [list(x) for x in (collections.Counter(dtw) - collections.Counter(pww))]
    out["db_states"] = sorted({v for v in bs.values()})
    return out


def main():
    idxs = [s["idx"] for s in json.load(open("C:/t/corr-assess/sample.json", encoding="utf-8"))]
    imp = {}
    if os.path.exists(IMPORT):
        for l in open(IMPORT, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                imp[r["idx"]] = r
    allr = {}
    for i in idxs:
        try:
            r = check(i)
        except Exception as e:
            import traceback
            r = {"idx": i, "error": traceback.format_exc()[-600:]}
        im = imp.get(i)
        if im:
            r["import"] = {"ok": im.get("import_ok"), "errors": im.get("errors"), "pw_id": im.get("pw_id"), "created_main": im.get("created_main"),
                           "expected": im.get("expected"), "exports": im.get("exports"), "screenshot_ok": (im.get("screenshot") or {}).get("ok"),
                           "parser_exception": im.get("parser_exception")}
        allr[str(i)] = r
    json.dump(allr, open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote", OUT, len(allr))


if __name__ == "__main__":
    main()
