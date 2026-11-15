import sys, json, csv, collections, os, re
sys.path.insert(0, "C:/t/oa/src")
from t2pw.bench.eval_v2 import reactions as R
from t2pw.bench.eval_v2.gold import load_supplement
sup = load_supplement()
print({k:(v.complete,len(v.reactions)) for k,v in sup.items()})
rows = list(csv.DictReader(open("C:/t/oa/docs/pwml_output/evidence/01/tidy_runs.csv")))
Rt = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/"
out = []
for x in rows:
    c = sup.get(x["paper"])
    if not c: continue
    p = x["path"] if x["path"][1:2] == ":" else Rt + x["path"]
    d = json.load(open(p, encoding="utf-8"))
    m = R.score_leg(c, d, [], None, expected_species=c.species)
    t1 = m["table1_paper_extraction"]
    rc = m["row_classes"]
    out.append(dict(path=x["path"], paper=x["paper"], mode=x["mode"], date=x["date"], qok=x["qok"], complete=c.complete,
        t1_rows=t1["rows_claimed_as_paper_extraction"], tp=t1["true_positive_rows"], n_gold=t1["gold_reactions"], gold_matched=len(t1["gold_reactions_matched"]),
        unique=m["unique_reactions"], dups=m["duplicate_rows_collapsed"], classes=json.dumps(rc),
        sub_acc=m["entity_accuracy"]["substrate_accuracy"], prod_acc=m["entity_accuracy"]["product_accuracy"], species=json.dumps(m["entity_accuracy"]["species_verdicts"])))
json.dump(out, open("C:/t/ev-oa/accvar/precision_runs.json", "w"), indent=1)
with open("C:/t/ev-oa/accvar/precision_runs.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); [w.writerow(r) for r in out]
def agg(sel,label):
    t1=sum(r["t1_rows"] for r in sel); tp=sum(r["tp"] for r in sel); g=sum(r["n_gold"] for r in sel); gm=sum(r["gold_matched"] for r in sel)
    cl=collections.Counter()
    for r in sel: cl.update(json.loads(r["classes"]))
    print(label,"runs",len(sel),"precision",tp,"/",t1, round(tp/t1,3) if t1 else None,"recall",gm,"/",g, round(gm/g,3) if g else None,"classes",dict(cl))
for qok in (True,):
  for comp in (True,):
    base=[r for r in out if r["qok"]=="True" and r["complete"]]
    agg(base,"complete-gold canonical ALL")
    for mode in ("pathwhiz","research"): agg([r for r in base if r["mode"]==mode],"  mode "+mode)
    for paper in sorted({r["paper"] for r in base}): agg([r for r in base if r["paper"]==paper],"  paper "+paper)
    late=[r for r in base if r["date"]>="2026-10-20"]; agg(late,"  late-era (>=Oct20)")
    early=[r for r in base if r["date"]<"2026-10-20" and r["date"]]; agg(early,"  Aug-Sep")
agg([r for r in out if r["qok"]!="True" and r["complete"]],"complete-gold NOT-canonical")
