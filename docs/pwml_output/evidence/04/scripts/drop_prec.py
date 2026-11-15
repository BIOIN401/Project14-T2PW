import sys, json, csv, os, collections
sys.path.insert(0, "C:/t/oa/src")
from t2pw.bench.eval_v2 import reactions as R
from t2pw.bench.eval_v2.gold import load_supplement
sup = load_supplement()
rows = list(csv.DictReader(open("C:/t/oa/docs/pwml_output/evidence/01/tidy_runs.csv")))
Rt = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/"
S = collections.defaultdict(collections.Counter)
for x in rows:
    c = sup.get(x["paper"])
    if not c or not c.complete or x["qok"] != "True": continue
    p = x["path"] if x["path"][1:2] == ":" else Rt + x["path"]
    d = os.path.dirname(p)
    for stage, fn in (("S1", "stage1_payload.json"), ("M", "merged_payload.json"), ("F", "final_mapped.json")):
        fp = os.path.join(d, fn)
        if not os.path.exists(fp): continue
        m = R.score_leg(c, json.load(open(fp, encoding="utf-8")), [], None, expected_species=c.species)
        t1 = m["table1_paper_extraction"]
        k = (x["mode"], stage)
        S[k]["runs"] += 1; S[k]["rows"] += t1["rows_claimed_as_paper_extraction"]; S[k]["tp"] += t1["true_positive_rows"]
        S[k]["gold"] += t1["gold_reactions"]; S[k]["gm"] += len(t1["gold_reactions_matched"])
for k, v in sorted(S.items()):
    print(k, dict(v), "precision", round(v["tp"] / v["rows"], 3) if v["rows"] else None, "recall", round(v["gm"] / v["gold"], 3))
json.dump({f"{a}|{b}": dict(v) for (a, b), v in S.items()}, open("C:/t/ev-oa/accvar/stage_precision.json", "w"), indent=1)
