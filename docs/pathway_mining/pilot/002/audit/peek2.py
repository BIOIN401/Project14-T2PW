import json
r = json.load(open("C:/t/mining-pilot-runs2/audit/pilot_002_audit.json", encoding="utf-8"))
s = r["search"]
for k, v in s.items():
    if k not in ("queries", "items"):
        print(k, json.dumps(v, ensure_ascii=False))
for it in s["items"]:
    print(it["round"], it["kind"], it["target"][:50], it["attempts"], it["results"], it["windows"], it["new_discovered"], it["new_included"], it["outcome"])
for q in s["queries"]:
    print(q["n"], q["round"], q["kind"][:10], q["attempt"], q["deeper"], "win", q["window"], "res", q["result_count"], q["window_filled"], "new", q["new_discovered"], q["new_by_source"], q["cumulative_discovered"], q["executed_at"][11:19])
print(json.dumps(r["seed_named_entities"], ensure_ascii=False))
for x in r["reactions"]:
    print(x["reaction_id"], x["graph_neighbours"])
