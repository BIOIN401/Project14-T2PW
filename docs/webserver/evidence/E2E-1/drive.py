import json, sys, time
from pathlib import Path
from t2pw.rag.eligibility import RequestedScope
from t2pw.service import organisms
from t2pw.service.pathway import PathwayRequest
from t2pw.web.jobs import JobStore
from t2pw.web.worker import DEFAULT_RUN_FN, Worker

ROOT = Path("C:/t/e2e")
t0 = time.time()
store = JobStore(ROOT / "jobs.sqlite")
req = PathwayRequest(
    pmids=("32185015",), pdf_paths=(),
    scope=RequestedScope(requested_pathway="Taste transduction: sour",
                         requested_organism=organisms.get_by_label("Mus musculus").scope),
    export_mode="strict")
print("run_fn", DEFAULT_RUN_FN, "scope", req.scope, flush=True)
sid = store.submit([req], consent=True, email=None)
print("submission", sid, flush=True)
w = Worker(store, ROOT / "work")
print("recovered", w.start(), flush=True)
try:
    while True:
        i = w.run_one()
        if i is None:
            break
        it = store.item(i)
        print("ran item", i, "attempt", it["attempt"], "state", it["state"], "status", it["status"],
              "kind", it["failure_kind"], "t=%.0fs" % (time.time() - t0), flush=True)
finally:
    w.close()
sub = store.submission(sid)
for it in sub["items"]:
    print("FINAL item", it["id"], it["state"], it["status"], it["failure_kind"], "attempts", it["attempt"])
    (ROOT / "report").mkdir(exist_ok=True)
    (ROOT / "report" / "job_result.json").write_text(json.dumps(it["result"], indent=1), encoding="utf-8")
    (ROOT / "report" / "attempts.json").write_text(json.dumps(it["attempts"], indent=1, default=str), encoding="utf-8")
print("wall_s %.1f" % (time.time() - t0))
