import json,glob,sys,collections
L=sys.argv[1]
rs={}
for f in glob.glob(f"C:/t/ev-fp/replay/{L}/results_*.jsonl"):
    for l in open(f,encoding="utf-8"):
        r=json.loads(l); rs[r["idx"]]=r
print(len(rs), "live",sum(r.get("live_requests",0) for r in rs.values()),"errs",sum(r.get("fetch_errors",0) for r in rs.values()),
 "fill_error",[(i,r["fill_error"]) for i,r in rs.items() if r.get("fill_error")],
 "harness",[(i,r["stage"]) for i,r in rs.items() if r["stage"].startswith("harness")])
print("budget-cut candidates", [i for i,r in rs.items() if r.get("seconds",0)>280])
