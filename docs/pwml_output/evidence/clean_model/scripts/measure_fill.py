"""Re-fill the fill_sample picks (cache hits), dump items, check payload, export SBML before/after fill.
Run with the scratch venv python (needs libsbml for the validation step in measure_fill_validate.py; this one is stdlib only):
  PY -I -X utf8 measure_fill.py --seed 20261110 --runs 10
"""
import argparse, json, os, re, sys, traceback
sys.path.insert(0, "C:/t/cm/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2pw.clean_model import build_from_run, to_payload
from t2pw.clean_model.fill import fill_model
from t2pw.clean_model.export.sbml import sbml_text
import fill_sample

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=20261110)
ap.add_argument("--runs", type=int, default=10)
ap.add_argument("--out", default="C:/t/ev-cm/sbml_out_fill")
ap.add_argument("--json", default="C:/t/cm/docs/pwml_output/evidence/clean_model/measure_fill.json")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
IS = re.compile(r"<bqbiol:is>(.*?)</bqbiol:is>", re.S)
def n_is(t): return sum(len(re.findall(r'rdf:resource="', b)) for b in IS.findall(t))
res = []
for i, r in enumerate(fill_sample.pick(fill_sample.RUNLIST if hasattr(fill_sample, "RUNLIST") else "C:/t/ev-cm/runlist.json", a.runs, a.seed)):
    rec = {"i": i, "dir": r["dir"]}
    try:
        m = build_from_run(r["dir"]); before = to_payload(m); t0 = sbml_text(m)
        rep = fill_model(m)
        rec["payload_equal"] = to_payload(m) == before
        rec["requests"], rec["cache_hits"] = rep.requests, rep.cache_hits
        t1 = sbml_text(m)
        rec["is_before"], rec["is_after"] = n_is(t0), n_is(t1)
        rec["bytes_before"], rec["bytes_after"] = len(t0), len(t1)
        open(os.path.join(a.out, "%02d_fill.xml" % i), "wb").write(t1.encode("utf-8"))
        L = m.db_layer
        rec["items"] = [{k: it[k] for k in ("kind", "entity", "field", "value")} | {"src": it["prov"]["source"], "acc": it["prov"]["accession"], "mt": it["prov"]["match_type"]} for it in L["items"]]
        rec["suggestions"] = [{"kind": s["kind"], "entity": s["entity"], "field": s["field"], "candidates": s["candidates"][:5], "n": len(s["candidates"])} for s in L["suggestions"]]
        rec["conflicts"] = [{k: c[k] for k in ("kind", "entity", "field", "frozen", "db")} | {"acc": c["prov"]["accession"]} for c in L["conflicts"]]
        rec["refused"] = L["refused"]
        rec["lookups_failed"] = rep.lookups_failed
    except Exception:
        rec["exception"] = traceback.format_exc()[-1200:]
    res.append(rec)
    print(i, rec.get("payload_equal"), rec.get("is_before"), rec.get("is_after"), rec.get("exception", "")[-300:])
json.dump(res, open(a.json, "w", encoding="utf-8"), indent=1, default=str)
