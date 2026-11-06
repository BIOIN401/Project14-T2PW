import json, sqlite3, sys
sys.path.insert(0, "C:/t/ws-e2e/src")
from t2pw.web.jobs import JobStore
s = JobStore("C:/t/e2e2/jobs.sqlite")
sub = s.submission("ef042ea73c004af094e4e23d813540fc")
r = json.loads(json.dumps(sub, default=str))
for it in r["items"]:
    it.pop("result", None) if False else None
open("C:/t/e2e2/report/jobstore_state.json", "w", encoding="utf-8").write(json.dumps(r, indent=1))
print(json.dumps({k: v for k, v in r.items() if k != "items"}, indent=1)[:1500])
for it in r["items"]:
    print({k: (v if k != "result" else "...") for k, v in it.items()})
con = sqlite3.connect("C:/t/e2e2/jobs.sqlite")
for (n,) in con.execute("select name from sqlite_master where type='table'"):
    print(n, con.execute(f"select count(*) from {n}").fetchone())
