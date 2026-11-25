import json, sys
E = "C:/t/fix-s2scope/docs/webserver/evidence/FIX-S2SCOPE/"
rows = {(x["run"].replace("/", "\\"), x["index"]): x for x in json.load(open(E + "oos_rows.json", encoding="utf-8"))}
rep = json.load(open(E + "replay_heme.json", encoding="utf-8"))
for r in rep["runs"]:
    for d in r["drops"]:
        x = rows.get((r["run"].replace("/", "\\"), d["stage1_index"]))
        print(r["paper"], d["stage2"][:45], "bio_state=", x and x["biological_state"], "| gold_matching_run=", r.get("dropped_rows_matching_gold"))
