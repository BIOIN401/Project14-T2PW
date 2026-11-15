import json, sys
r = json.load(open("C:/t/corr-assess/a3/compare.json"))
pr = sys.argv[1]
cats = sys.argv[2].split(",")
for i in (sys.argv[3].split(",") if len(sys.argv) > 3 else r):
    p = r[i]["pairs"].get(pr)
    if not p: continue
    for c in cats:
        x = p[c]
        if c == "rxn":
            if x["only_a"] or x["only_b"] or x["E_count_diff"]:
                print(i, c, "A-only:", x["ex_a"], "B-only:", x["ex_b"], "Ecnt:", x["ex_Ecnt"])
        elif x.get("only_a") or x.get("only_b"):
            print(i, c, "A-only:", x["ex_a"], "B-only:", x["ex_b"])
