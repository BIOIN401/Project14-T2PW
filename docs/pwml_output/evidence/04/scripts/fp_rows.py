import sys, json, csv
sys.path.insert(0, "C:/t/oa/src")
from t2pw.bench.eval_v2 import reactions as R
from t2pw.bench.eval_v2.gold import load_supplement
sup = load_supplement()
rows = list(csv.DictReader(open("C:/t/oa/docs/pwml_output/evidence/01/tidy_runs.csv")))
Rt = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/"
res=[]
for x in rows:
    c = sup.get(x["paper"])
    if not c or not c.complete or x["qok"]!="True": continue
    p = x["path"] if x["path"][1:2] == ":" else Rt + x["path"]
    d = json.load(open(p, encoding="utf-8"))
    m = R.score_leg(c, d, [], None, expected_species=c.species)
    for r in m["rows"]:
        if r["class"] not in ("target_paper_supported",) and r["duplicate_of"] is None:
            res.append(dict(paper=x["paper"],mode=x["mode"],date=x["date"],cls=r["class"],inputs=r["inputs"],outputs=r["outputs"],enzymes=r["enzymes"],gold=r["gold_ids"],evid=r["evidence"],missing=r.get("missing_defining")))
for r in res: print(r["paper"],r["mode"],r["date"],r["cls"],r["inputs"],"->",r["outputs"],r["enzymes"],r.get("missing"))
json.dump(res,open("C:/t/ev-oa/accvar/fp_rows.json","w"),indent=1)
