"""Adds canonical (id-collapsed) key sets C for final_mapped. canon(name)=mapped id (uniprot/hmdb/pathbank_compound/chebi) else name."""
import csv, json, os, re
R = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/"
runs = json.load(open("C:/t/ev-oa/accvar/var_runs.json"))
def nm(s): return re.sub(r"\s+", " ", str(s or "").strip().casefold())
def absp(p): return p if re.match(r"^[A-Za-z]:", p) else R + p
EB=("compounds","proteins","protein_complexes","nucleic_acids")
def eid(b,r):
    if b=="proteins": return r.get("uniprot") or r.get("pathbank_protein_id")
    if b=="compounds": return r.get("hmdb_id") or r.get("pathbank_compound_id") or r.get("chebi_id")
for r in runs:
    p=json.load(open(absp(r["path"]),encoding="utf-8"))
    E=p.get("entities") or {}; m={}
    C={k:set() for k in ("compounds","proteins","protein_complexes","nucleic_acids","reactions","transports","interactions")}
    mapped=0; total=0
    for b in EB:
        for e in E.get(b) or []:
            if not isinstance(e,dict) or not e.get("name"): continue
            i=eid(b,e); n=nm(e["name"])
            c=f"{b[0]}:{i}" if i not in (None,"") else "n:"+n
            m.setdefault(n,c); C[b].add(c)
    cn=lambda x: m.get(nm(x),"n:"+nm(x))
    P=p.get("processes") or {}
    for x in P.get("reactions") or []:
        a=tuple(sorted(cn(i) for i in x.get("inputs") or [])); b=tuple(sorted(cn(i) for i in x.get("outputs") or []))
        C["reactions"].add(repr(tuple(sorted([a,b]))))
    for x in P.get("transports") or []:
        t=tuple(sorted(cn(y.get("protein") or y.get("entity") if isinstance(y,dict) else y) for y in (x.get("transporters") or [])))
        C["transports"].add(repr((cn(x.get("cargo")),t)))
    for x in P.get("interactions") or []:
        rel=nm(x.get("relationship"))
        if rel=="same_as": continue
        C["interactions"].add(repr((tuple(sorted([cn(x.get("entity_1")),cn(x.get("entity_2"))])),rel)))
    r["C"]={k:sorted(v) for k,v in C.items()}
json.dump(runs,open("C:/t/ev-oa/accvar/var_runs.json","w"))
print("ok")
