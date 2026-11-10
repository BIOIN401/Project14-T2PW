
def scan(fm_path):
    p = json.load(open(fm_path, encoding="utf-8"))
    ent = p.get("entities") or {}
    pr = p.get("processes") or {}
    res = {}
    b_name = b_id = b_alias = n_ent = 0
    b_examples = []
    for bucket in ("compounds", "proteins", "protein_complexes"):
        rows = [r for r in ent.get(bucket) or [] if isinstance(r, dict) and str(r.get("name") or "").strip()]
        n_ent += len(rows)
        c = Counter(nm(r["name"]) for r in rows)
        b_name += sum(v - 1 for v in c.values() if v > 1)
        if bucket in ("compounds", "proteins"):
            by = defaultdict(set)
            for i, r in enumerate(rows):
                for a in acc(r, bucket):
                    by[a].add(i)
            parent = list(range(len(rows)))
            def f(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            for s in by.values():
                s = sorted(s)
                for j in s[1:]:
                    parent[f(j)] = f(s[0])
            g = defaultdict(list)
            for i in range(len(rows)):
                g[f(i)].append(i)
            for grp in g.values():
                if len(grp) > 1:
                    b_id += len(grp) - 1
                    if len(b_examples) < 5:
                        b_examples.append([rows[i]["name"] for i in grp][:4])
        allnames = {nm(r["name"]): i for i, r in enumerate(rows)}
        for i, r in enumerate(rows):
            for s in r.get("synonyms") or []:
                j = allnames.get(nm(s))
                if j is not None and j != i:
                    b_alias += 1
                    break
    res.update(n_entities=n_ent, b_name_extra=b_name, b_id_extra=b_id, b_alias_overlap_rows=b_alias, b_examples=b_examples)
    im = ident_map(ent)
    def ik(name):
        return im.get(nm(name)) or ("?", nm(name))
    reacts = [r for r in pr.get("reactions") or [] if isinstance(r, dict)]
    trans = [t for t in pr.get("transports") or [] if isinstance(t, dict)]
    def rsig(r):
        ins = frozenset(ik(x) for x in names_of(r.get("inputs")))
        outs = frozenset(ik(x) for x in names_of(r.get("outputs")))
        en = frozenset(ik(x) for k in ("enzymes", "modifiers") for x in names_of(r.get(k)))
        return ins, outs, en, nm(r.get("biological_state"))
    def tsig(t):
        cargo = frozenset(ik(x) for x in names_of(t.get("cargo_complex") or t.get("cargo")))
        tr = frozenset(ik(x) for x in names_of(t.get("transporters")))
        return cargo, tr, nm(t.get("from_biological_state")), nm(t.get("to_biological_state"))
    rs = [rsig(r) for r in reacts]
    ts = [tsig(t) for t in trans]
    res["n_reactions"] = len(reacts)
    res["n_transports"] = len(trans)
    res["n_interactions"] = len(pr.get("interactions") or [])
    res["a_rx_exact"] = extra(rs)
    res["a_rx_same_enzymes_other_state"] = extra([s[:3] for s in rs]) - res["a_rx_exact"]
    res["a_rx_same_participants_other_enzymes"] = extra([s[:2] for s in rs]) - extra([s[:3] for s in rs])
    res["a_tr_exact"] = extra(ts)
    res["a_tr_same_cargo_transporter_other_state"] = extra([s[:2] for s in ts]) - res["a_tr_exact"]
    res["a_tr_same_cargo_state_other_transporter"] = extra([(s[0],) + s[2:] for s in ts]) - res["a_tr_exact"]
    res["a_rx_exact_nameonly"] = extra([(frozenset(nm(x) for x in names_of(r.get("inputs"))), frozenset(nm(x) for x in names_of(r.get("outputs"))),
                                         frozenset(nm(x) for k in ("enzymes", "modifiers") for x in names_of(r.get(k))), nm(r.get("biological_state"))) for r in reacts])
    slots = 0
    used = set()
    for r in reacts:
        part = {ik(x) for k in ("inputs", "outputs", "enzymes", "modifiers") for x in names_of(r.get(k))}
        slots += len(part)
        used |= part
    for t in trans:
        part = {ik(x) for x in names_of(t.get("cargo_complex") or t.get("cargo"))} | {ik(x) for x in names_of(t.get("transporters"))}
        slots += len(part)
        used |= part
    for i in pr.get("interactions") or []:
        if isinstance(i, dict):
            part = {ik(i.get("entity_1")), ik(i.get("entity_2"))}
            slots += len(part)
            used |= part
    res["c_tile_slots"] = slots
    res["c_distinct_entities_in_processes"] = len(used)
    res["c_tile_redraws"] = slots - len(used)
    el = p.get("element_locations") or {}
    dsame = 0
    nloc = 0
    ents_multi = 0
    for key in ("compound_locations", "protein_locations", "protein_complex_locations", "nucleic_acid_locations"):
        rows = [r for r in el.get(key) or [] if isinstance(r, dict)]
        nloc += len(rows)
        by = defaultdict(list)
        for r in rows:
            e = r.get("compound") or r.get("protein") or r.get("protein_complex") or r.get("nucleic_acid") or r.get("element")
            by[ik(e)].append(nm(r.get("biological_state")))
        for v in by.values():
            dsame += sum(c - 1 for c in Counter(v).values() if c > 1)
            if len(set(v)) > 1:
                ents_multi += 1
    res["d_location_rows"] = nloc
    res["d_entities_in_multi_states"] = ents_multi
    res["c_same_state_location_rows"] = dsame
    return res
