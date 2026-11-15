import csv, collections, json
rows = list(csv.DictReader(open("C:/t/ev-oa/funnel/out/tidy_runs.csv", encoding="utf-8")))
pc = list(csv.DictReader(open("C:/t/ev-oa/funnel/out/paper_classes.csv", encoding="utf-8")))
by = collections.defaultdict(list)
for p in pc: by[p["class"]].append(p)
for cl, ps in by.items():
    print("\n", cl, len(ps), "papers,", sum(int(p["n_canon_runs"]) for p in ps), "canonical runs")
    print("   ", "; ".join(f"{p['paper']} ({p['n_canon_runs']})" for p in ps))
    dis = [p for p in ps if p["name_keyword_class"] not in ("other",) and p["name_keyword_class"] != cl]
    print("    name-keyword disagreements:", [(p["paper"], p["pathway_name_mode"], p["name_keyword_class"]) for p in dis])
c = collections.defaultdict(collections.Counter)
for r in rows:
    if r["qok"] == "True":
        c[r["paper_class"]][(r["group"])] += 1
for k, v in c.items(): print(k, dict(v))
# pooled retention by kind, pathwhiz canon
d = json.load(open("C:/t/ev-oa/funnel/out/class_kind.json"))
for kind in ("reactions", "transports", "interactions"):
    s1 = sum(v[kind]["S1"] for k, v in d.items() if k.startswith("P|"))
    f = sum(v[kind]["F_paper_stated"] for k, v in d.items() if k.startswith("P|"))
    pw = sum(v[kind]["PWML"] for k, v in d.items() if k.startswith("P|"))
    print(kind, s1, f, f"{f/s1:.1%}", pw, f"{pw/s1:.1%}")
