import subprocess, os, sys, json, collections
paths=[l.strip() for l in open("payloads.txt") if l.strip()]
runs=json.load(open("var_runs.json"))
R="C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/"
seen=set(paths)
for r in runs:
    p=r["path"] if r["path"][1:2]==":" else R+r["path"]
    q=os.path.join(os.path.dirname(p),"stage1_payload.json")
    if q not in seen and os.path.exists(q) and len(seen)<22 and r["paper"] in ("PMC7059786","PMC12782028","PMC12856317","PMC12180156","PMC9544450","PMC12096016","PMC12452463"):
        seen.add(q); paths.append(q)
out=[]
for q in paths:
    if not os.path.exists(q): continue
    hs=[];orph=[]
    for s in range(1,7):
        env=dict(os.environ,PYTHONHASHSEED=str(s))
        o=subprocess.run([sys.executable,"qa_seed.py",q],env=env,capture_output=True,text=True,timeout=300).stdout.strip().splitlines()
        if not o: hs.append("ERR"); continue
        hs.append(o[-1].split()[0]); orph.append(o[-1])
    out.append(dict(payload=q.replace(R,""),distinct=len(set(hs)),seeds=len(hs),detail=orph[0] if orph else ""))
    print(out[-1]["distinct"],"/",out[-1]["seeds"],out[-1]["payload"][-70:],out[-1]["detail"])
json.dump(out,open("qa_seed_results.json","w"),indent=1)
print("payloads",len(out),"with output differing across hash seeds:",sum(1 for o in out if o["distinct"]>1))
