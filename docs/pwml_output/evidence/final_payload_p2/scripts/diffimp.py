import json,re,collections
def load(s):
    d={}
    for l in open(f"C:/t/fp/docs/pwml_output/evidence/final_payload_p2/import/{s}/results.jsonl",encoding="utf-8"):
        r=json.loads(l); d[int(re.search(r'(\d+)\.pwml$',r['file']).group(1))]=r
    return d
O,N=load("off5"),load("on5")
def msgs(r):
    c=collections.Counter()
    for k,v in r["errors"].items():
        kk=k.rsplit("_",1)[0] if k.rsplit("_",1)[-1].isdigit() else k
        if isinstance(v,str): c[kk+": "+v[:150]]+=1
        else:
            for m in v: c[kk+": "+m]+=1
    return c
for nm,D in (("off5",O),("on5",N)):
    for i,r in sorted(D.items()):
        if not r["success"] or "exception" in r["errors"]: print(nm,i,"success",r["success"],{k:(v if isinstance(v,str) else v[:2]) for k,v in r["errors"].items() if "exception" in k or "pathway" in k}.__repr__()[:400], {k:v for k,v in r.items() if k not in("errors","created","created_main","expected")}.__repr__()[:300])
new=collections.Counter(); gone=collections.Counter()
for i in O:
    a,b=msgs(O[i]),msgs(N[i])
    for m in b-a: new[m]+=b[m]-a[m]
    for m in a-b: gone[m]+=a[m]-b[m]
print("NEW in on5 vs off5"); [print(v,m) for m,v in new.most_common(25)]
print("GONE"); [print(v,m) for m,v in gone.most_common(15)]
