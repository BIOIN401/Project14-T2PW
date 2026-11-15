import sys, json, csv, collections, os
sys.path.insert(0, "C:/t/oa/src")
from pathlib import Path
from t2pw.bench import semantic as S
from t2pw.bench.goldset import load_gold_set
gs = load_gold_set(Path("C:/t/oa/src/t2pw/bench/gold/pinned_v1.json"))
cases = {c.paper_id: c for c in gs.cases}
rows = list(csv.DictReader(open("C:/t/oa/docs/pwml_output/evidence/01/tidy_runs.csv")))
Rt = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/"
agg = collections.defaultdict(collections.Counter); per = collections.defaultdict(collections.Counter)
for x in rows:
    c = cases.get(x["paper"])
    if not c or x["qok"] != "True": continue
    p = x["path"] if x["path"][1:2] == ":" else Rt + x["path"]
    d = os.path.dirname(p)
    for st, fn in (("S1", "stage1_payload.json"), ("F", "final_mapped.json")):
        fp = os.path.join(d, fn)
        if not os.path.exists(fp): continue
        voc = S._payload_vocabulary(json.load(open(fp, encoding="utf-8")))
        for lab, terms in (("enzymes", c.expected_enzymes), ("substrates", c.expected_substrates), ("products", c.expected_products)):
            if not terms: continue
            cv = S._cover(lab, terms, voc)
            agg[(st, lab)]["found"] += cv.found; agg[(st, lab)]["exp"] += cv.expected
            per[(x["paper"], st, lab)]["found"] += cv.found; per[(x["paper"], st, lab)]["exp"] += cv.expected
for k, v in sorted(agg.items()): print(k, dict(v), round(v["found"]/v["exp"], 3))
print("per paper (F)")
for k, v in sorted(per.items()):
    if k[1] == "F": print(k, dict(v), round(v["found"]/v["exp"], 3))
json.dump({f"{a}|{b}": dict(v) for (a, b), v in agg.items()}, open("C:/t/ev-oa/accvar/entity_cov.json", "w"), indent=1)
