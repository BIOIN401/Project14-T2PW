import os,datetime as dt,glob
runs={
 "E2E-2":("C:/t/ws-e2e","C:/t/e2e2","run-3b9c541c09aa"),
 "E2E-3":("C:/t/ws-E2E3c","C:/t/e2e3c-run2","run-c4d6293e764b"),
 "E2E-4":("C:/t/ws-E2E4","C:/t/e2e4-run","run-733acf326b98"),
}
def m(p): return os.path.getmtime(p)
out={}
for k,(w,r,rid) in runs.items():
    a=r+"/work/item-1/attempt-1/"
    t0=m(a+"request.json")
    pts=[
     ("Stage0+Stage1 ensemble done", w+"/tmp/stage1_raw_extraction.json"),
     ("Stage2 inference done (RAG loop starts)", w+"/tmp/rag_rounds/round_00/checkpoint.json"),
     ("RAG rd0 + merge + first ID map done", w+"/tmp/rag_rounds/round_00/draft_graph.json"),
     ("post-pipeline normalization/mapping done", w+"/tmp/reaction_preservation_after_normalization.json"),
     ("audit+gap rounds done", w+"/tmp/extraction_diagnostics/%s/audit_iteration_summary.json"%rid),
     ("curator+remap done (identity merge report)", w+"/tmp/extraction_diagnostics/%s/ensemble_identity_merge_report.json"%rid),
     ("enrichment+quarantine+export done", w+"/outputs/pathway.pwml" if k!="E2E-2" else w+"/tmp/final.mapped.json"),
     ("result.json written (end)", a+"result.json"),
    ]
    prev=t0; rows=[]
    for n,p in pts:
        t=m(p); rows.append((n,(t-prev)/60,(t-t0)/60,dt.datetime.fromtimestamp(t).strftime("%H:%M:%S"))); prev=t
    out[k]=rows
    print(k,"start",dt.datetime.fromtimestamp(t0).strftime("%Y-%m-%d %H:%M:%S"))
    for r_ in rows: print(f"   {r_[0]:48s} +{r_[1]:6.2f} min  (cum {r_[2]:6.2f})  @{r_[3]}")
# png/pwml gap for E2E-4
a="C:/t/e2e4-run/work/item-1/attempt-1/out/"
for f in ["pathway.review_required.pwml","pathway.png","run_outcome.json","../result.json"]:
    print(f, dt.datetime.fromtimestamp(m(a+f)).strftime("%H:%M:%S.%f"))
print("ws pathway.pwml", dt.datetime.fromtimestamp(m("C:/t/ws-E2E4/outputs/pathway.pwml")).strftime("%H:%M:%S.%f"))
