"""Why do 8 of 97 SVG metrics differ from the CORRECTNESS-FIXES run? Compare today's IR (generate, default scope) with the
IR of that run (C:/t/render-corr/ir/<idx>/pwml_ir.json): entity names per type and process counts."""
import json
import sys

from t2pw.vision import generate as G
from t2pw.vision.payload import big_from_run

rl = [x for x in json.load(open("C:/t/ev-cm/runlist.json", encoding="utf-8")) if x.get("q")]
out = {}
for i in (3, 5, 7, 9, 17, 78, 79, 150):
    old = json.load(open("C:/t/render-corr/ir/%d/pwml_ir.json" % i, encoding="utf-8"))
    run = rl[i]["dir"]
    G._MEMO.clear()
    G.generate(big_from_run(run), G.parse_scope("default"), "pwml", export=G.export_options_from_run(run))
    new = next(iter(G._MEMO.values()))[1]
    G._MEMO.clear()
    rec = {}
    for kind in ("compounds", "proteins", "protein_complexes", "nucleic_acids", "element_collections"):
        a = sorted(e.get("name") for e in old["entities"].get(kind, []))
        b = sorted(e.get("name") for e in new["entities"].get(kind, []))
        if a != b:
            rec[kind] = {"old_only": [x for x in a if x not in b][:6], "new_only": [x for x in b if x not in a][:6], "n": [len(a), len(b)]}
    for kind in ("reactions", "transports", "interactions"):
        a, b = old["processes"].get(kind, []), new["processes"].get(kind, [])
        if len(a) != len(b):
            rec[kind] = [len(a), len(b)]
    rec["states"] = [len(old.get("biological_states", [])), len(new.get("biological_states", []))]
    rec["pathway_keys_equal"] = sorted(old) == sorted(new)
    rec["top_level_diff"] = [k for k in set(old) | set(new) if json.dumps(old.get(k), sort_keys=True) != json.dumps(new.get(k), sort_keys=True)]
    out[i] = rec
    print(i, json.dumps(rec)[:700], flush=True)
json.dump(out, open("C:/t/vis-runs/scratch/layout/irdiff.json", "w"), indent=1)
sys.exit(0)
