import json
g=json.load(open("src/t2pw/bench/gold/pinned_v1.json",encoding="utf-8"))
print(g.keys())
c=g["cases"][0]
print({k:(v if not isinstance(v,(list,dict)) else type(v).__name__+str(len(v))) for k,v in c.items()})
for k in c:
    if "reaction" in k or "must" in k or "forbid" in k:
        print(k, json.dumps(c[k])[:700])
