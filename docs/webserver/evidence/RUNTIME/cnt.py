import json,glob,collections
root="C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_papers"
c=collections.Counter(); over=0; n=0; over_stage=collections.Counter()
for f in glob.glob(root+"/*/legs/*/*/papers/*/*/LEG_TRACE.jsonl"):
    prev=0
    for l in open(f,encoding="utf-8"):
        e=json.loads(l)
        if e["kind"]=="model_attempt":
            n+=1; c[e["status"]]+=1
            if e["elapsed_seconds"]-prev>300: over+=1; over_stage[e["stage"]]+=1
        prev=e["elapsed_seconds"]
print(n,dict(c),"gaps>300s:",over,dict(over_stage))
