import json,os,glob,collections
P="C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/"
roots=[P+r for r in("runs_papers","runs_validation","runs_verify","runs_smoke","runs")]+["C:/t/e2e4-run","C:/t/e2e5-run"]
rows=[]
for r in roots:
    for dp,dn,fn in os.walk(r):
        if "final_mapped.json" in fn:
            q=None
            qp=os.path.join(dp,"quarantine_report.json")
            if os.path.exists(qp):
                q=json.load(open(qp,encoding="utf-8"))
            rows.append((dp.replace("\\","/"),q))
print(len(rows))
c=collections.Counter()
for dp,q in rows:
    c[(None if q is None else (q.get("ok"),q.get("export_mode"),json.dumps(q.get("research_mode"))[:40]))]+=1
print(c)
# other quarantine-named files
for dp,q in rows[:0]: pass
