import json,collections,re
exec(open("c2.py").read().split("print(dict(inst))")[0])
CX={}
for dp,q in rows:
    if not(q and q.get("ok")): continue
    pp=pid_of(dp)
    d=json.load(open(dp+"/final_mapped.json",encoding="utf-8"))
    for e in d["entities"]["protein_complexes"]:
        ph=e.get("identity_status")=="placeholder"
        acc=bool(e.get("pathbank_complex_id") or e.get("pathbank_protein_complex_id") or e.get("mapped_ids"))
        CX.setdefault((pp,norm(e["name"])),[]).append("placeholder" if ph else ("ok" if acc else "unmapped"))
print("complex units",len(CX),collections.Counter(("ok" if all(s=="ok" for s in v) else "ph" if all(s=="placeholder" for s in v) else "unm/mixed") for v in CX.values()))
both=set(P)|set(CX)
print("union units (proteins+complexes, by key)",len(both), "overlap",len(set(P)&set(CX)))
tot=len(P)+len(CX)
fail=sum(all(s!="ok" for s in v) for v in P.values())+sum(all(s!="ok" for s in v) for v in CX.values())
print("sum units",tot,"fail",fail)
