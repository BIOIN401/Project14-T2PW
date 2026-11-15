import sys, json, csv, os
sys.path.insert(0, "C:/t/oa/src")
from t2pw.bench.eval_v2 import reactions as R
from t2pw.bench.eval_v2.gold import load_supplement
sup = load_supplement()
rows = list(csv.DictReader(open("C:/t/oa/docs/pwml_output/evidence/01/tidy_runs.csv")))
Rt = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/"
out=[]
def sc(c,fp):
    return R.score_leg(c, json.load(open(fp,encoding="utf-8")), [], None, expected_species=c.species)
for x in rows:
    c = sup.get(x["paper"])
    if not c or not c.complete or x["qok"] != "True": continue
    p = x["path"] if x["path"][1:2] == ":" else Rt + x["path"]; d = os.path.dirname(p)
    s1 = sc(c, d+"/stage1_payload.json"); m = sc(c, d+"/merged_payload.json")
    k1 = {(tuple(sorted(r["inputs"])), tuple(sorted(r["outputs"]))) for r in s1["rows"]}
    for r in m["rows"]:
        if r["duplicate_of"] is not None: continue
        key=(tuple(sorted(r["inputs"])), tuple(sorted(r["outputs"])))
        if key not in k1:
            out.append((x["paper"],x["mode"],x["date"],r["class"],r["inputs"],r["outputs"],r["enzymes"]))
import collections
print(len(out), collections.Counter(o[3] for o in out))
for o in out: print(o)
