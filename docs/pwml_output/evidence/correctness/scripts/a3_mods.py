import json, collections, sys
sys.path.insert(0, "C:/t/corr-assess/scripts")
import a3_models as M
B = M.B
sample = json.load(open(f"{B}/sample.json", encoding="utf-8"))
rows = []; tot = collections.Counter()
for s in sample:
    i = s["idx"]
    p = json.load(open(f"{B}/base/{i}/final_payload.json", encoding="utf-8"))
    ir = json.load(open(f"{B}/ir/{i}/pwml_ir.json", encoding="utf-8"))
    byname = {(r["name"] or "").casefold(): r for r in ir["processes"]["reactions"]}
    for r in p["processes"]["reactions"]:
        cat = {M.nc(x["entity"]) for x in r.get("enzymes", [])}
        nonc = [(x["entity"], x["role"]) for x in r.get("modifiers", []) if x.get("role") != "catalyst"]
        roles = collections.Counter(x["role"] for x in r.get("modifiers", []))
        tot.update({"roles:" + k: v for k, v in roles.items()})
        tot["rxn"] += 1
        if nonc:
            ir_r = byname.get((r["name"] or "").casefold())
            promoted = []
            if ir_r:
                ent = {e["key"]: e["name"] for e in ir["entities"]["protein_complexes"]}
                ent.update({e["key"]: e["name"] for e in ir["entities"]["proteins"]})
                irE = {M.nc(ent.get(x["entity_key"], "")) for x in ir_r["enzymes"]}
                promoted = [(n, rl) for n, rl in nonc if M.nc(n) in irE and M.nc(n) not in cat]
            rows.append(dict(idx=i, reaction=r["name"], payload_enzymes=sorted(cat), nonCatalystModifiers=nonc, promoted_to_enzyme_in_IR=promoted,
                             evidence=(r.get("evidence") or "")[:140]))
            tot["rxn_with_noncat_mod"] += 1
            tot["noncat_mods"] += len(nonc)
            tot["promoted"] += len(promoted)
            if not cat and promoted: tot["rxn_enzymeless_in_payload_gets_enzyme"] += 1
json.dump(dict(totals=tot, rows=rows), open(f"{B}/a3/modifiers_as_enzymes.json", "w", encoding="utf-8"), indent=1, default=str)
print(dict(tot))
for r in rows: print(r["idx"], r["reaction"], r["payload_enzymes"], r["nonCatalystModifiers"], "->IR enzymes", r["promoted_to_enzyme_in_IR"])
