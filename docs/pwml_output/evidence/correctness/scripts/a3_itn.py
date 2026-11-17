import json, collections, sys
sys.path.insert(0, "C:/t/corr-assess/scripts")
import a3_models as M
B = M.B
sample = json.load(open(f"{B}/sample.json", encoding="utf-8"))
tot = collections.Counter(); rows = []
for s in sample:
    i = s["idx"]
    p = json.load(open(f"{B}/base/{i}/final_payload.json", encoding="utf-8"))
    ir = json.load(open(f"{B}/ir/{i}/pwml_ir.json", encoding="utf-8"))
    W = M.from_pwml(f"{B}/replay/on_v2/pwml/{i}.pwml")
    ents = {}
    for k, key in (("compound", "compounds"), ("protein", "proteins"), ("protein_complex", "protein_complexes"), ("nucleic_acid", "nucleic_acids")):
        for e in ir["entities"].get(key, []): ents[(k, e["key"])] = (k, e["name"])
    its = ir["processes"]["interactions"]
    st = collections.Counter(x.get("export_status") for x in its)
    wit = [x for x in its if x.get("export_status") == "withheld"]
    kinds = collections.Counter((ents[(x["left"]["entity_type"], x["left"]["entity_key"])][0], ents[(x["right"]["entity_type"], x["right"]["entity_key"])][0]) for x in wit)
    rows.append(dict(idx=i, payload=len(p["processes"]["interactions"]), ir=len(its), ir_status=dict(st), pwml=len(W["itn"]), withheld_endpoint_kinds={f"{a}->{b}": n for (a, b), n in kinds.items()},
                     withheld_examples=[x["name"] for x in wit[:3]]))
    tot["payload"] += len(p["processes"]["interactions"]); tot["ir"] += len(its); tot["pwml"] += len(W["itn"]); tot["withheld"] += len(wit)
json.dump(dict(totals=tot, rows=rows), open(f"{B}/a3/interactions_by_stage.json", "w"), indent=1)
print(dict(tot))
for r in rows:
    if r["payload"] or r["ir"]: print(r["idx"], "payload", r["payload"], "ir", r["ir"], r["ir_status"], "pwml", r["pwml"], r["withheld_endpoint_kinds"])
