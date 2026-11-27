import json, sys
ir = json.load(open(sys.argv[1], encoding="utf-8"))
for c in ir["entities"]["compounds"]:
    if str(c.get("pathbank_compound_id")) in sys.argv[2:]:
        d = {k: v for k, v in c.items() if k not in ("description", "element_states")}
        print(json.dumps(d, ensure_ascii=False)[:2500]); print("---")
