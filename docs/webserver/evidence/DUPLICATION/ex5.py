import json, sys
d = sys.argv[1]
for f in ("stage1_payload.json", "merged_payload.json", "final_mapped.json"):
    p = json.load(open(d + "/" + f, encoding="utf-8"))
    print(f, "transports", [t["name"] for t in p["processes"]["transports"]], "reactions", [r["name"] for r in p["processes"]["reactions"]])
q = json.load(open(d + "/final_mapped.json", encoding="utf-8"))["quarantined_locked_reactions"]
print([(x["reaction_name"], x["reason"]) for x in q])
pi = json.load(open(d + "/pwml_ir.json", encoding="utf-8"))
print("IR compounds", [c.get("name") for c in pi["entities"]["compound"]] if isinstance(pi["entities"], dict) and "compound" in pi["entities"] else list(pi["entities"])[:5])
print("IR processes keys", list(pi["processes"].keys()) if isinstance(pi["processes"], dict) else len(pi["processes"]))
