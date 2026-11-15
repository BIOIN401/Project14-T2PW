import json,collections,re
exec(open("survey.py").read().split("print(len(rows))")[0])
def norm(s): return re.sub(r"[^a-z0-9]+"," ",(s or "").lower()).strip()
def pid_of(dp):
    m=re.findall(r"PMC\d+",dp)
    return m[-1] if m else "NOPAPER:"+dp
P={}  # unit -> list of (state)
C={}
inst=collections.Counter()
for dp,q in rows:
    if not(q and q.get("ok")): continue
    pp=pid_of(dp)
    d=json.load(open(dp+"/final_mapped.json",encoding="utf-8"))
    for e in d["entities"]["proteins"]:
        mm=e.get("mapping_meta") or {}
        ph=(e.get("identity_status")=="placeholder" or (mm.get("identity_status")=="placeholder") or e.get("pathbank_protein_id")==9659 or norm(e["name"])=="unknown")
        acc=bool(e.get("uniprot") or e.get("uniprot_id") or e.get("pathbank_protein_id") or e.get("pw_protein_id") or e.get("mapped_ids"))
        st="placeholder" if ph else ("ok" if acc else "unmapped")
        P.setdefault((pp,norm(e["name"])),[]).append(st); inst["p_"+st]+=1
    for e in d["entities"]["compounds"]:
        acc=bool(e.get("pathbank_compound_id") or e.get("db_id") or e.get("pwc_id") or e.get("pw_compound_id") or any(e.get(k) for k in("hmdb_id","kegg_id","chebi_id","pubchem_cid","cas","chemspider_id","biocyc_id","drugbank_id")) or e.get("mapped_ids"))
        C.setdefault((pp,norm(e["name"])),[]).append("ok" if acc else "noid"); inst["c_"+("ok" if acc else "noid")]+=1
print(dict(inst))
def rep(U,bad):
    n=len(U); 
    anyf=sum(any(s in bad for s in v) for v in U.values())
    allf=sum(all(s in bad for s in v) for v in U.values())
    # best state per unit
    return n,anyf,allf
print("papers",len({k[0] for k in P}))
print("protein units, n/any-fail/all-fail", rep(P,{"placeholder","unmapped"}))
print(" unmapped-only any", sum(any(s=="unmapped" for s in v) for v in P.values()), "ph any", sum(any(s=="placeholder" for s in v) for v in P.values()))
print(" all-fail breakdown", collections.Counter(v[0] for v in P.values() if all(s!="ok" for s in v) and len(set(v))==1))
print("compound units", rep(C,{"noid"}))
