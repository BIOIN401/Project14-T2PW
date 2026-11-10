import json, sys
from collections import defaultdict
p = json.load(open(sys.argv[1], encoding="utf-8"))
g = defaultdict(list)
for t in p["processes"]["transports"]:
    tr = tuple(sorted(str(x.get("protein") or x.get("entity")) for x in t.get("transporters", [])))
    g[(str(t.get("cargo")).lower(), tr)].append((t.get("name"), t.get("from_biological_state"), t.get("to_biological_state"), [(e.get("element"), e.get("side"), e.get("biological_state")) for e in t.get("elements_with_states", [])]))
for k, v in g.items():
    print(k, len(v))
    for x in v: print("    ", x[:3])
