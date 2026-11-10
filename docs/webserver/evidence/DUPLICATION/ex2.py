import json, sys, os
d = sys.argv[1]
p = json.load(open(d + "/final_mapped.json", encoding="utf-8"))
from collections import defaultdict
g = defaultdict(list)
for r in p["processes"]["reactions"]:
    ins = tuple(sorted(str(x).lower() for x in r.get("inputs", [])))
    outs = tuple(sorted(str(x).lower() for x in r.get("outputs", [])))
    g[(ins, outs)].append((r.get("name"), r.get("biological_state"), [e.get("protein") or e.get("entity") for e in r.get("enzymes", [])], bool(r.get("locked_reaction_id"))))
n = 0
for k, v in g.items():
    if len(v) > 1 and n < 6:
        n += 1
        print(k)
        for x in v: print("   ", x)
f = d + "/ensemble_identity_merge_report.json"
if os.path.exists(f):
    j = json.load(open(f, encoding="utf-8"))
    print("merge report keys", list(j.keys()), "merges", len(j.get("merges", [])), "refused", len(j.get("refused", [])))
    for x in j.get("refused", [])[:6]: print("  refused", x)
