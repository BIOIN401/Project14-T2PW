"""A2: reactions repeated in the IR (same substrate/product names), within one compartment or across compartments."""
import json
from collections import defaultdict

CA = "C:/t/corr-assess"
out = {}
for x in json.load(open(CA + "/sample.json", encoding="utf-8")):
    i = x["idx"]
    ir = json.load(open(f"{CA}/ir/{i}/pwml_ir.json", encoding="utf-8"))
    nm = {}
    for k in ir["entities"]:
        for e in ir["entities"][k] if isinstance(ir["entities"][k], list) else []:
            nm[e["key"]] = (e.get("name") or "").casefold()
    loc = {l["key"]: l["name"] for l in ir.get("subcellular_locations", [])}
    st = {s["key"]: loc.get(s.get("subcellular_location_key"), s["name"]) for s in ir.get("biological_states", [])}
    g = defaultdict(list)
    for r in ir["processes"].get("reactions", []):
        sig = (tuple(sorted(nm.get(e["entity_key"], "") for e in r["left"])), tuple(sorted(nm.get(e["entity_key"], "") for e in r["right"])))
        g[sig].append((r["key"], st.get(r.get("biological_state_key"), "?"), sorted(nm.get(e["entity_key"], "") for e in r.get("enzymes", []))))
    same, cross = [], []
    for sig, rs in g.items():
        if len(rs) > 1:
            (cross if len({s for _, s, _ in rs}) > 1 else same).append(dict(sig=" + ".join(sig[0]) + " -> " + " + ".join(sig[1]), reactions=rs))
    if same or cross:
        out[i] = dict(n_reactions=len(ir["processes"].get("reactions", [])), same_compartment=same, across_compartments=cross)
json.dump(out, open(CA + "/a2/duplicate_reactions.json", "w", encoding="utf-8"), indent=1)
for i, v in out.items():
    print(i, v["n_reactions"], "same-comp groups", len(v["same_compartment"]), "cross-comp groups", len(v["across_compartments"]),
          sorted({s for gg in v["across_compartments"] for _, s, _ in gg["reactions"]}))
