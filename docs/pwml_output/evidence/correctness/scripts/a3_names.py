import json, sys, collections
sys.path.insert(0, "C:/t/corr-assess/scripts")
import a3_models as M
B = M.B
sample = [s["idx"] for s in json.load(open(f"{B}/sample.json", encoding="utf-8"))]
tot = collections.Counter(); rows = {}
for i in sample:
    W = M.from_pwml(f"{B}/replay/on_v2/pwml/{i}.pwml")
    db = json.load(open(f"{B}/pathwhiz_exports/{i}.db.json", encoding="utf-8"))
    mem = W["notes"]["complex_members"]
    ren = []; unk = []
    for x in db["reaction_enzymes"]:
        k = M.nc(x["complex_name"])
        ours = mem.get(k)
        if ours is None: continue
        pwp = M.nn(x["proteins"])
        if pwp == "unknown": unk.append((x["complex_name"], ours))
        elif len(ours) == 1 and ours[0] != pwp: ren.append((ours[0], x["proteins"]))
    ren = sorted(set(ren)); unk = sorted(set(map(str, unk)))
    rows[i] = dict(renamed_protein_names=ren[:8], n_renamed=len(ren), n_complex_shown_as_Unknown=len(unk), unknown_examples=unk[:3])
    tot["renamed"] += len(ren); tot["unknown_enzyme_complexes"] += len(unk)
    m = M.from_pw_sbgn(f"{B}/pathwhiz_exports/{i}.sbgn") if i != 58 else None
    if m:
        n = sum(1 for r in m["rxn"] if "unknown" in r["E"]); tot["sbgn_rxn_with_Unknown_enzyme"] += n; tot["sbgn_rxn"] += len(m["rxn"]); rows[i]["sbgn_rxn_unknown"] = n
json.dump(dict(totals=tot, rows=rows), open(f"{B}/a3/protein_names.json", "w"), indent=1)
print(dict(tot))
for i, r in rows.items(): print(i, r["n_renamed"], r["n_complex_shown_as_Unknown"], r.get("sbgn_rxn_unknown"), r["renamed_protein_names"][:2], r["unknown_examples"][:1])
