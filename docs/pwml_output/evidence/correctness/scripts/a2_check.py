"""A2: correctness of the web diagram vs pwml_ir.json / final_payload.json, plus clarity metrics.
usage: a2_check.py --proto DIR --out JSON [--opts '{"hide_reg":1}'] [--idx 7,9] [--style S]
Correctness is checked against the IR read independently (not via the prototype's own proc list)."""
import argparse, json, math, re, sys
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--proto", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--opts", default="{}")
ap.add_argument("--idx", default="")
ap.add_argument("--style", default="")
a = ap.parse_args()
sys.path.insert(0, a.proto)
import t2pw_diagrams as D  # noqa: E402
D.OPTS.update(json.loads(a.opts))

CA = "C:/t/corr-assess"
sample = json.load(open(CA + "/sample.json", encoding="utf-8"))
sel = [int(x) for x in a.idx.split(",")] if a.idx else [x["idx"] for x in sample]
REG = ("act", "inh", "assoc", "enz")
DIRECTIONAL = ("regulat", "modif", "substrate", "transport", "compet", "component")


def cf(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().casefold()


def plen(pts):
    return sum(math.dist(p, q) for p, q in zip(pts, pts[1:])) if pts and len(pts) > 1 else 0.0


def segs(e):
    return list(zip(e.pts, e.pts[1:])) if e.pts and len(e.pts) > 1 else []


def crossings(edges):
    H, V = [], []
    for i, e in enumerate(edges):
        for (x0, y0), (x1, y1) in segs(e):
            if abs(y0 - y1) < 0.5 and abs(x0 - x1) > 0.5:
                H.append((min(x0, x1), max(x0, x1), y0, i))
            elif abs(x0 - x1) < 0.5 and abs(y0 - y1) > 0.5:
                V.append((min(y0, y1), max(y0, y1), x0, i))
    n = 0
    for hx0, hx1, hy, i in H:
        for vy0, vy1, vx, j in V:
            if i != j and hx0 + 1 < vx < hx1 - 1 and vy0 + 1 < hy < vy1 - 1:
                n += 1
    return n


def payload_vs_ir(fp, ir):
    """Upstream: every payload process reaches the IR (names, sides, enzymes)."""
    P = fp.get("processes") or {}
    I = ir["processes"]
    nm = {}
    for t, k in D.ENT_KEYS.items():
        for e in ir["entities"].get(k, []):
            nm[(t, e["key"])] = e.get("name") or e["key"]
    irr = [(cf(r.get("name")), frozenset(cf(nm.get((x["entity_type"], x["entity_key"]))) for x in r["left"]),
            frozenset(cf(nm.get((x["entity_type"], x["entity_key"]))) for x in r["right"]),
            frozenset(cf(nm.get((x["entity_type"], x["entity_key"]))) for x in r.get("enzymes", []))) for r in I.get("reactions", [])]
    miss, side_diff, enz_diff = [], [], []
    for r in P.get("reactions") or []:
        ins = frozenset(cf(x if isinstance(x, str) else x.get("name") or x.get("entity")) for x in r.get("inputs") or [])
        outs = frozenset(cf(x if isinstance(x, str) else x.get("name") or x.get("entity")) for x in r.get("outputs") or [])
        enz = frozenset(cf(z.get("entity")) for z in r.get("enzymes") or [])
        hit = [x for x in irr if x[0] == cf(r.get("name"))]
        if not hit:
            miss.append(f"{r.get('name')}: {sorted(ins)} -> {sorted(outs)}")
            continue
        if not any(x[1] == ins and x[2] == outs for x in hit):
            side_diff.append(f"{r.get('name')}: payload {sorted(ins)}->{sorted(outs)} vs IR {[(sorted(x[1]), sorted(x[2])) for x in hit]}")
        elif not any(x[1] == ins and x[2] == outs and x[3] == enz for x in hit):
            enz_diff.append(f"{r.get('name')}: payload {sorted(enz)} vs IR {[sorted(x[3]) for x in hit if x[1] == ins and x[2] == outs]}")
    return dict(counts={k: [len(P.get(k) or []), len(I.get(k) or [])] for k in ("reactions", "transports", "interactions")},
                reactions_missing_in_ir=miss, reaction_sides_differ=side_diff, reaction_enzymes_differ=enz_diff)


def one(idx):
    ir = D.load(f"{CA}/ir/{idx}/pwml_ir.json")
    fp = json.load(open(f"{CA}/base/{idx}/final_payload.json", encoding="utf-8"))
    k = D.metrics(ir)
    style = a.style or D.pick_style(k)[0]
    svg, m, lay = D.render(ir, style, f"d{idx}")
    I = ir["processes"]
    reg_nodes = {id(n) for n in m.nodes.values()}
    issues = defaultdict(list)
    # ---- reactions
    J_by_key = {r["key"]: n for n in m.nodes.values() if n.kind == "junction" and getattr(n, "jkind", "") == "reaction" for r in getattr(n, "merged", [n.proc])}
    allJ = [n for n in m.nodes.values() if n.kind == "junction"]
    tags = Counter(n.label for n in allJ)
    for t, c in tags.items():
        if c > 1:
            issues["junction_tag_collision"].append(t)
    by_edge = defaultdict(lambda: defaultdict(Counter))
    for e in m.edges:
        if e.kind in ("sub", "cat", "xin"):
            by_edge[id(e.b)][e.kind][(e.a.id[0], e.a.id[1])] += 1
        if e.kind in ("prod", "xout"):
            by_edge[id(e.a)][e.kind][(e.b.id[0], e.b.id[1])] += 1
    stoich_hidden = 0
    for r in I.get("reactions", []):
        J = J_by_key.get(r["key"])
        if J is None or id(J) not in reg_nodes:
            issues["reaction_not_drawn"].append(r["key"])
            continue
        want_l = Counter((x["entity_type"], x["entity_key"]) for x in r["left"])
        want_r = Counter((x["entity_type"], x["entity_key"]) for x in r["right"])
        grp = getattr(J, "merged", [r])
        want_z = Counter((x["entity_type"], x["entity_key"]) for x in r.get("enzymes", [])) if len(grp) == 1 else Counter({(x["entity_type"], x["entity_key"]): 1 for rr in grp for x in rr.get("enzymes", [])})
        got = by_edge[id(J)]
        if got["sub"] != want_l:
            issues["substrates_differ"].append(f'{r["key"]}: want {dict(want_l)} got {dict(got["sub"])}')
        if got["prod"] != want_r:
            issues["products_differ"].append(f'{r["key"]}: want {dict(want_r)} got {dict(got["prod"])}')
        if got["cat"] != want_z:
            issues["enzymes_differ"].append(f'{r["key"]}: want {dict(want_z)} got {dict(got["cat"])}')
        if (r.get("direction") or ">") != ">":
            issues["non_forward_direction"].append(f'{r["key"]}:{r.get("direction")}')
        stoich_hidden += sum(1 for x in r["left"] + r["right"] if x.get("stoichiometry") not in (None, 1, "1", 1.0))
        exp_comp = m.state2comp.get(r.get("biological_state_key"), ("none",))
        for e in m.edges:
            if (e.b is J and e.kind in ("sub", "cat")) or (e.a is J and e.kind == "prod"):
                other = e.a if e.b is J else e.b
                if other.comp != exp_comp:
                    issues["participant_wrong_compartment"].append(f'{r["key"]}:{other.label}')
    # ---- transports
    T_by_key = {n.proc["key"]: n for n in m.nodes.values() if n.kind == "junction" and getattr(n, "jkind", "") == "transport"}
    for t in I.get("transports", []):
        J = T_by_key.get(t["key"])
        if J is None:
            issues["transport_not_drawn"].append(t["key"])
            continue
        got = by_edge[id(J)]
        want = Counter((x["entity_type"], x["entity_key"]) for x in t.get("transport_elements", []))
        if got["xin"] != want or got["xout"] != want:
            issues["transport_cargo_differs"].append(f'{t["key"]}: want {dict(want)} in {dict(got["xin"])} out {dict(got["xout"])}')
        for el in t.get("transport_elements", []):
            ca = m.state2comp.get(el.get("left_biological_state_key"), ("none",))
            cb = m.state2comp.get(el.get("right_biological_state_key"), ("none",))
            ins = [e.a.comp for e in m.edges if e.b is J and e.kind == "xin" and e.a.id[1] == el["entity_key"]]
            outs = [e.b.comp for e in m.edges if e.a is J and e.kind == "xout" and e.b.id[1] == el["entity_key"]]
            if ca not in ins or cb not in outs:
                issues["transport_side_compartment_wrong"].append(t["key"])
        trs = {(x["entity_type"], x["entity_key"]) for x in t.get("transporters", [])}
        pills = {(p.id[2], p.id[3]) for p in getattr(J, "pillnodes", [])}
        if trs != pills:
            issues["transporter_differs"].append(f'{t["key"]}: want {sorted(trs)} got {sorted(pills)}')
        if len(t.get("transporters", [])) != len(trs):
            issues["upstream_duplicate_transporter"].append(t["key"])
        if style == "spine":
            issues["transport_no_compartment_boundary_drawn"].append(t["key"])
    # ---- interactions
    hidden = bool(D.OPTS.get("hide_reg"))
    edges_by_tag = defaultdict(list)
    for e in m.edges:
        if getattr(e, "tag", None):
            edges_by_tag[e.tag].append(e)
    icls_count = Counter()
    for it in I.get("interactions", []):
        tag = "I" + re.sub(r"\D", "", it["key"])
        es = edges_by_tag.get(tag, [])
        cl = D.icls(it.get("interaction_type"))
        icls_count[cl] += 1
        if not es:
            if not hidden:
                issues["interaction_not_drawn"].append(it["key"])
            continue
        e = es[0]
        if (e.a.id[1], e.b.id[1]) != (it["left"]["entity_key"], it["right"]["entity_key"]):
            issues["interaction_ends_differ"].append(it["key"])
        if cl == "assoc" and any(w in cf(it.get("interaction_type")) for w in DIRECTIONAL):
            issues["directional_interaction_drawn_undirected"].append(f'{it["key"]}:{it.get("interaction_type")}')
        if it.get("export_status") == "withheld":
            issues["withheld_interaction_drawn_not_in_pwml"].append(f'{it["key"]}:{it.get("interaction_type")}')
    # ---- drawn edges
    undrawn = [e for e in m.edges if not e.pts or len(e.pts) < 2]
    if undrawn:
        issues["edge_without_path"].extend(f"{e.kind}:{e.a.label}->{e.b.label}" for e in undrawn)
    # svg counts
    nr = svg.count('class="jn reaction"')
    if nr != len({id(v) for v in J_by_key.values()}) or len(J_by_key) != len(I.get("reactions", [])):
        issues["svg_reaction_count_differs"].append(f"svg {nr} ir {len(I.get('reactions', []))}")
    if stoich_hidden:
        issues["stoichiometry_not_shown"].append(stoich_hidden)
    dupn = [n.label for n in allJ if n.note] + [e.tag for e in m.edges if e.badge]
    if dupn:
        issues["duplicate_marker_tags"].extend(dupn)
    trunc = [n.full for n in m.nodes.values() if getattr(n, "truncated", False)]
    if trunc:
        issues["label_truncated_tooltip_only"].extend(trunc)
    # ---- clarity metrics
    ents = [n for n in m.nodes.values() if n.kind != "junction"]
    lens = [plen(e.pts) for e in m.edges if e.pts]
    cat = [e for e in m.edges if e.kind == "cat" and not hasattr(e.a, "attached")]
    cat_d = sorted(math.dist((e.a.x, e.a.y), (e.b.x, e.b.y)) for e in cat)
    multi_enz = Counter(id(e.a) for e in cat)
    deg = Counter()
    for e in m.edges:
        if e.kind in ("sub", "prod"):
            for n in (e.a, e.b):
                if n.kind == "compound":
                    deg[n] += 1
    hub = max(deg.items(), key=lambda kv: kv[1]) if deg else (None, 0)
    procs_ids = {id(x) for e in m.edges if e.kind not in REG for x in (e.a, e.b)}
    reg_only = [n for n in ents if id(n) not in procs_ids]
    W, Hh = lay["W"], lay["H"]
    regs = getattr(m, "regions", [])
    cross_region = sum(1 for e in m.edges if e.a.comp != e.b.comp and e.kind not in ("xin", "xout", "cat"))
    chk = D.checks(m)
    met = dict(style=style, W=round(W), H=round(H_ := Hh), area_per_entity=round(W * H_ / max(1, len(ents))),
               n_entity_drawings=len(ents), n_unique_entities=len({(n.id[0], n.id[1]) for n in ents}),
               n_edges=len(m.edges), crossings=crossings(m.edges), bends=sum(max(0, len(e.pts) - 2) for e in m.edges if e.pts),
               edge_len_median=round(sorted(lens)[len(lens) // 2]) if lens else 0, edge_len_max=round(max(lens, default=0)),
               long_edges_gt600=sum(1 for x in lens if x > 600),
               enzyme_rxn_dist_median=round(cat_d[len(cat_d) // 2]) if cat_d else None,
               enzyme_rxn_dist_max=round(cat_d[-1]) if cat_d else None, enzyme_far_gt200=sum(1 for d in cat_d if d > 200),
               enzymes_serving_gt1_reaction=sum(1 for v in multi_enz.values() if v > 1),
               reg_edges=sum(1 for e in m.edges if e.kind in REG), reg_edge_share=round(sum(1 for e in m.edges if e.kind in REG) / max(1, len(m.edges)), 3),
               dashed_assoc_edges=sum(1 for e in m.edges if e.kind == "assoc"),
               reg_only_molecules=len(reg_only), hub_compound=(hub[0].label if hub[0] else None), hub_degree=hub[1],
               cross_region_edges=cross_region, lanes=len(regs), main_chain_len=k["chain"],
               dup_tags=len(dupn), truncated_labels=len(trunc), interaction_classes=dict(icls_count), **chk)
    if getattr(m, "hidden", None):
        met["hidden"] = m.hidden
    return dict(idx=idx, pathway=next(x["pathway_name"] for x in sample if x["idx"] == idx), metrics=met,
                diagram_vs_ir={k2: v for k2, v in issues.items()}, payload_vs_ir=payload_vs_ir(fp, ir))


res = {}
for idx in sel:
    try:
        res[idx] = one(idx)
    except Exception as ex:  # noqa: BLE001
        import traceback
        res[idx] = dict(idx=idx, error=repr(ex)[:300], tb=traceback.format_exc()[-900:])
    print(idx, res[idx].get("error") or res[idx]["metrics"]["style"], flush=True)
json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
