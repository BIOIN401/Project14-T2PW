import sys, json, hashlib, subprocess, os
if len(sys.argv)>2 and sys.argv[2]=="child":
    sys.path.insert(0,"C:/t/oa/src")
    from t2pw.pipeline.pipeline import build_qa_feedback
    r=build_qa_feedback(json.load(open(sys.argv[1],encoding="utf-8")))
    print(json.dumps({k:hashlib.sha1(json.dumps(v).encode()).hexdigest()[:6] for k,v in r.items()}|{"nmiss":len(r["missing_links_suspected"]),"setmiss":hashlib.sha1(json.dumps(sorted(map(json.dumps,r["missing_links_suspected"]))).encode()).hexdigest()[:6]}))
else:
    for q in json.load(open("qa_seed_unique.json")):
        if q["distinct"]<2: continue
        p="C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/"+q["payload"]
        res=[json.loads(subprocess.run([sys.executable,__file__,p,"child"],env=dict(os.environ,PYTHONHASHSEED=str(s)),capture_output=True,text=True).stdout.strip().splitlines()[-1]) for s in range(1,7)]
        diff=[k for k in res[0] if len({r[k] for r in res})>1]
        print(q["payload"][-60:],"fields differing:",diff,"n_missing_links",res[0]["nmiss"])
