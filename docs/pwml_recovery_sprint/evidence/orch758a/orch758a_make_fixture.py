"""Cut a minimal excerpt of the ORCH-755 sour-taste strict merged_payload.json.

Rows are copied verbatim; only rows outside the Ca++ / OTOP1 transports are dropped.
"""
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src, encoding="utf-8"))

KEEP_TRANSPORTS = [0, 1, 3, 5]  # 3 OTOP1 H+ transports + the Ca++ VGCC transport
transports = [d["processes"]["transports"][i] for i in KEEP_TRANSPORTS]
compounds = {"H+", "Ca++"}
proteins = {"OTOP1", "voltage-gated Ca ++ channels"}

states = set()
for t in transports:
    for k in ("from_biological_state", "to_biological_state"):
        if t.get(k):
            states.add(t[k])
    for e in t.get("elements_with_states", []):
        if e.get("biological_state"):
            states.add(e["biological_state"])

cl = [r for r in d["element_locations"]["compound_locations"] if r.get("compound") in compounds]
pl = [r for r in d["element_locations"]["protein_locations"] if r.get("protein") in proteins]
for r in cl + pl:
    if r.get("biological_state"):
        states.add(r["biological_state"])

ents = d["entities"]
out = {
    "entities": {
        "species": ents["species"],
        "cell_types": ents["cell_types"],
        "tissues": ents["tissues"],
        "subcellular_locations": ents["subcellular_locations"],
        "compounds": [r for r in ents["compounds"] if r["name"] in compounds],
        "proteins": [r for r in ents["proteins"] if r["name"] in proteins],
    },
    "biological_states": [s for s in d["biological_states"] if s.get("name") in states],
    "element_locations": {"compound_locations": cl, "protein_locations": pl},
    "processes": {"transports": transports, "interactions": [], "reactions": []},
}
with open(dst, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(out, fh, indent=2, ensure_ascii=False)
    fh.write("\n")
print("states", sorted(states))
print("compound_locations", len(cl), "protein_locations", len(pl))
