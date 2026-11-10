import json,glob,statistics as st
root="C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_papers"
allc=[]  # (stage, lat, status, fr, chars, leg, endelapsed, reason)
for f in sorted(glob.glob(root+"/*/legs/*/*/papers/*/*/LEG_TRACE.jsonl")):
    p=f.replace("\\","/").split("/")
    leg=p[p.index("legs")+1]+"@"+p[p.index("legs")+2]; orch=p[p.index("runs_papers")+1]
    prev=0.0
    for l in open(f,encoding="utf-8"):
        e=json.loads(l)
        if e["kind"]=="model_attempt":
            allc.append((e["stage"],e["elapsed_seconds"]-prev,e["status"],e.get("finish_reason"),e.get("content_chars"),orch+" "+leg,e["elapsed_seconds"],(e.get("reason") or "")[:70].replace("\n"," ")))
        prev=e["elapsed_seconds"]
def pct(a,q): a=sorted(a); return a[min(len(a)-1,int(q*len(a)))]
for stage in ["preprocessor","Stage 1 extraction","Stage 2 inference","audit loop","gap resolver","chat","rag_prose_extraction"]:
    for lab,flt in [("ok/stop",lambda c:c[2]=="ok" and c[3]=="stop"),("not ok/stop",lambda c:not(c[2]=="ok" and c[3]=="stop"))]:
        a=[c[1] for c in allc if c[0]==stage and flt(c)]
        if a: print(f"{stage:22s} {lab:12s} n={len(a):4d} p50={pct(a,.5)/60:5.2f}m p90={pct(a,.9)/60:5.2f}m max={max(a)/60:5.2f}m sum={sum(a)/60:6.1f}m")
print("\nStage1 calls >5 min:")
for c in sorted([c for c in allc if c[0]=="Stage 1 extraction" and c[1]>300],key=lambda c:-c[1]):
    print(f"  {c[1]/60:5.1f}m {c[2]:6s} fr={c[3]!s:7s} chars={c[4]} {c[5]} | {c[7]}")
print("\nreasons of non-ok:")
import collections
cnt=collections.Counter((c[0],c[2],c[3],c[7][:50]) for c in allc if c[2]!="ok")
for k,v in cnt.most_common(25): print(v,k)
