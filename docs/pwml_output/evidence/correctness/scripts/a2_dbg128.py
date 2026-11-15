"""A2: find which edge endpoint is not registered in m.nodes for idx 128 (cat_of crash)."""
import json, sys
from collections import Counter
sys.path.insert(0, "C:/t/corr-assess/a2/proto")
import t2pw_diagrams as D

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 128
ir = D.load(f"C:/t/corr-assess/ir/{idx}/pwml_ir.json")
P = ir["processes"]
print("reaction keys", [r["key"] for r in P.get("reactions", [])])
print("transport keys", [t["key"] for t in P.get("transports", [])])
for t in P.get("transports", []):
    print(" ", t["key"], "transporters", [(x["entity_type"], x["entity_key"], x.get("biological_state_key")) for x in t.get("transporters", [])])
m = D.build(ir)
ent = {}
for t, k in D.ENT_KEYS.items():
    for e in ir["entities"].get(k, []):
        ent[(t, e["key"])] = e
m._ent = ent
D.place_transports(m, "spine")
# replicate layout(): pill creation
ids = Counter()
for n in list(m.nodes.values()):
    if n.kind == "junction" and n.jkind == "transport":
        for et, ek, pck in n.pills:
            ids[("pill", n.id, et, ek)] += 1
print("duplicate pill ids:", {k: v for k, v in ids.items() if v > 1})
jt = Counter(("J", p["tag"]) for p in m.procs if p["kind"] in ("reaction", "transport"))
print("duplicate junction ids:", {k: v for k, v in jt.items() if v > 1})
reg = set(id(n) for n in m.nodes.values())
orph = [(e.kind, e.a.id, e.b.id) for e in m.edges if id(e.a) not in reg or id(e.b) not in reg]
print("edges with unregistered endpoint before pills:", orph)
