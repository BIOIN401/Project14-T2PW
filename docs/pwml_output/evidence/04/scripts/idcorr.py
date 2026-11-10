import sys, json, csv, collections
from pathlib import Path
sys.path.insert(0, "C:/t/oa/src")
from t2pw.bench.eval_v2 import reactions as R
ref = R.build_reference_index(Path("C:/t/oa"))
rows = list(csv.DictReader(open("C:/t/oa/docs/pwml_output/evidence/01/tidy_runs.csv")))
Rt = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/"
tot = collections.Counter(); bybucket = collections.defaultdict(collections.Counter); bymode = collections.defaultdict(collections.Counter)
wrong = []
for x in rows:
    if x["qok"] != "True": continue
    p = x["path"] if x["path"][1:2] == ":" else Rt + x["path"]
    d = json.load(open(p, encoding="utf-8"))
    ents = d.get("entities") or {}
    res = R.id_correctness(d, ref)
    # re-derive per-entity verdicts: id_correctness returns only wrong list + counts
    tot.update(res["counts"])
    bymode[x["mode"]].update(res["counts"])
    for w in res["wrong"]:
        wrong.append(dict(paper=x["paper"], mode=x["mode"], date=x["date"], bucket=w["bucket"], name=w["name"], contradicting=w.get("contradicting_ids"), independent=w.get("independent"),
                          per_id=[(q["id"], q["verdict"], q.get("reference_names", [])[:4]) for q in w["per_id"] if q["verdict"] == "wrong"]))
        bybucket[w["bucket"]]["wrong"] += 1
print("canonical runs", sum(1 for x in rows if x["qok"] == "True"))
print("TOTAL", dict(tot)); print("BYMODE", {k: dict(v) for k, v in bymode.items()})
print("WRONG by bucket", {k: dict(v) for k, v in bybucket.items()})
json.dump(wrong, open("C:/t/ev-oa/accvar/id_wrong.json", "w"), indent=1)
# unique (paper,bucket,name) wrong units
u = {(w["paper"], w["bucket"], w["name"].casefold()) for w in wrong}
print("wrong rows", len(wrong), "unique (paper,bucket,name)", len(u), "papers", len({a for a,_,_ in u}))
print("independent wrong", sum(1 for w in wrong if w["independent"]))
