import sys, json, csv, collections
from pathlib import Path
sys.path.insert(0, "C:/t/oa/src")
from t2pw.bench.eval_v2 import reactions as R
ref = R.build_reference_index(Path("C:/t/oa"))
rows = list(csv.DictReader(open("C:/t/oa/docs/pwml_output/evidence/01/tidy_runs.csv")))
Rt = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/"
c = collections.defaultdict(collections.Counter)
for x in rows:
    if x["qok"] != "True": continue
    p = x["path"] if x["path"][1:2] == ":" else Rt + x["path"]
    d = json.load(open(p, encoding="utf-8"))
    for b in ("compounds", "proteins", "protein_complexes"):
        for e in (d.get("entities") or {}).get(b) or []:
            if not isinstance(e, dict) or not e.get("name"): continue
            meta = e.get("mapping_meta") if isinstance(e.get("mapping_meta"), dict) else {}
            rule = str(e.get("chosen_rule") or meta.get("chosen_rule") or "none")
            res = R.id_correctness({"entities": {b: [e]}}, ref)
            v = next(iter(res["counts"]), "none")
            c[(b, rule)][v] += 1
out = []
for (b, rule), cc in sorted(c.items(), key=lambda kv: -sum(kv[1].values())):
    n = sum(cc.values()); ev = cc["correct_identity"] + cc["wrong_identity"]
    out.append(dict(bucket=b, rule=rule, n=n, correct=cc["correct_identity"], wrong_flag=cc["wrong_identity"], unresolved=cc["unresolved_or_novel"], no_ref=cc["not_evaluable_no_reference"],
                    flag_rate=round(cc["wrong_identity"] / ev, 3) if ev else None))
with open("C:/t/ev-oa/accvar/id_by_rule.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); [w.writerow(r) for r in out]
for r in out[:22]: print(r)
