import collections, json, sys
for a in sys.argv[1:]:
    name, path = a.split("=",1)
    rs=[json.loads(l) for l in open(path,encoding="utf-8")]
    c=collections.Counter(); f=collections.defaultdict(set); cr=collections.Counter(); ex=collections.Counter()
    for i,r in enumerate(rs):
        for k,v in r["errors"].items():
            kk=k.rsplit("_",1)[0] if k.rsplit("_",1)[-1].isdigit() else k
            for m in v: c[kk+": "+m]+=1; f[kk+": "+m].add(i)
        for k,v in (r.get("created") or {}).items(): cr[k]+=v
        for k,v in (r.get("expected") or {}).items(): ex[k]+=v
    print("==",name,len(rs),"files; clean",sum(1 for r in rs if not r["errors"]),"success",sum(bool(r["success"]) for r in rs),"msgs",sum(c.values()))
    for k,v in c.most_common(): print("  %4d msgs %3d files  %s"%(v,len(f[k]),k))
    print("  expected",dict(ex)); print("  created ",dict(cr))
