import json,os,glob,collections
exec(open("survey.py").read().split("print(len(rows))")[0])
can=[(dp,q) for dp,q in rows if q and q.get("ok") is True]
print(len(can))
tot=collections.Counter(); nop=[];hasp=0
loc=collections.Counter()
for dp,q in can:
    if q["export_mode"]!="pathwhiz": continue
    here=glob.glob(dp+"/*.pwml")
    anyp=[os.path.join(a,f) for a,_,fs in os.walk(dp) for f in fs if f.endswith(".pwml")]
    par=glob.glob(os.path.dirname(dp)+"/*.pwml")
    loc[(bool(here),bool(anyp),bool(par))]+=1
    d=json.load(open(dp+"/final_mapped.json",encoding="utf-8"))
    pr=d["processes"]
    n={k:len(v) for k,v in pr.items() if isinstance(v,list)}
    if not here:
        nop.append((dp,n))
        for k,v in n.items(): tot[k]+=v
    else: hasp+=1
print(loc); print("pathwhiz",loc and sum(loc.values()),"with pwml",hasp,"without",len(nop)); print(tot, sum(tot.values()))
ks=collections.Counter(k for _,n in nop for k in n);print(ks)
