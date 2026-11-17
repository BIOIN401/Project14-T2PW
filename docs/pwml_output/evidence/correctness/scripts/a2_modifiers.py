"""A2: payload reaction modifiers with role activator/inhibitor that end up as IR reaction enzymes (drawn as catalysts)."""
import json
from collections import Counter

CA = "C:/t/corr-assess"
c, ex = Counter(), []
for x in json.load(open(CA + "/sample.json", encoding="utf-8")):
    i = x["idx"]
    p = json.load(open(f"{CA}/base/{i}/final_payload.json", encoding="utf-8"))["processes"]["reactions"]
    ir = json.load(open(f"{CA}/ir/{i}/pwml_ir.json", encoding="utf-8"))
    nm = {(t, e["key"]): (e.get("name") or "").casefold() for t, k in (("protein", "proteins"), ("protein_complex", "protein_complexes"), ("compound", "compounds"))
          for e in ir["entities"].get(k, [])}
    irr = {(r.get("name") or "").casefold(): r for r in ir["processes"]["reactions"]}
    for r in p:
        R = irr.get((r.get("name") or "").casefold())
        if not R:
            continue
        ze = {nm.get((z["entity_type"], z["entity_key"])) for z in R["enzymes"]}
        enz_names = {(z.get("entity") or "").casefold() for z in r.get("enzymes") or []}
        for md in r.get("modifiers") or []:
            role = md.get("role")
            n = (md.get("entity") or "").casefold()
            inir = n in ze
            c[(role, inir)] += 1
            if role not in ("catalyst", "enzyme") and inir and n not in enz_names:
                ex.append(dict(idx=i, reaction=r.get("name"), modifier=md.get("entity"), role=role))
out = dict(counts={f"{k[0]}|in_ir_enzymes={k[1]}": v for k, v in c.items()}, regulators_drawn_as_catalysts=ex)
json.dump(out, open(CA + "/a2/modifiers_as_enzymes.json", "w", encoding="utf-8"), indent=1)
print(json.dumps(out, indent=1))
