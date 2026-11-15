import json, sys, collections
r = json.load(open("C:/t/corr-assess/a3/compare.json"))
pairs = sys.argv[1:]
for pr in pairs:
    print("=====", pr)
    tot = collections.Counter()
    for i, v in r.items():
        p = v["pairs"].get(pr)
        if not p: continue
        line = []
        for k in ("comps", "prots", "cplx", "nuc"):
            if p[k]["only_a"] or p[k]["only_b"]: line.append(f'{k}:-{p[k]["only_a"]}/+{p[k]["only_b"]}')
        x = p["rxn"]
        if x["only_a"] or x["only_b"] or x["E_count_diff"]: line.append(f'rxn keys {x["common"]} ok, -{x["only_a"]}/+{x["only_b"]} Ecnt{x["E_count_diff"]} Ename{x["E_name_diff"]} n={x["n_a"]}/{x["n_b"]}')
        t = p["trn"]
        if t["only_a"] or t["only_b"]: line.append(f'trn n={t["n_a"]}/{t["n_b"]} -{t["only_a"]}/+{t["only_b"]}')
        it = p["itn"]
        if it["only_a"] or it["only_b"]: line.append(f'itn n={it["n_a"]}/{it["n_b"]} -{it["only_a"]}/+{it["only_b"]}')
        print(i, "; ".join(line) or "AGREE")
