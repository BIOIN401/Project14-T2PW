import json, sys, collections
sys.path.insert(0, "C:/t/corr-assess/scripts")
import a3_compare as C
sample = C.sample
out = {}
for s in sample:
    i = s["idx"]
    mdl = C.load(i)
    row = {}
    for k, m in mdl.items():
        if k == "cmap_aligned": continue
        row[k] = dict(unknown_prot=("unknown" in m["prots"]), unknown_in_rxn_enzymes=sum(1 for r in m["rxn"] if any("unknown" in e for e in r["E"])),
                      n_rxn_noenz=sum(1 for r in m["rxn"] if not r["E"]), n_rxn=len(m["rxn"]))
    out[i] = row
json.dump(out, open("C:/t/corr-assess/a3/unknown_scan.json", "w"), indent=1)
for i, r in out.items():
    print(i, {k: (int(v["unknown_prot"]), v["unknown_in_rxn_enzymes"], v["n_rxn_noenz"]) for k, v in r.items()})
