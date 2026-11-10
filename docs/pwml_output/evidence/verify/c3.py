import json,glob,collections,re,sys
sys.path.insert(0,"C:/t/oa/src")
from t2pw.pwml.qa import run_pwml_qa
exec(open("survey.py").read().split("print(len(rows))")[0])
def cls(e):
    if "XML parse" in e: return "xml_parse"
    if "interaction" in e.lower() or "visualization" in e: return "interaction"
    if "spontaneous" in e: return "spontaneous+enzyme"
    if "stoichiometry" in e: return "stoichiometry"
    if "Orphaned" in e: return "orphan_compound_location"
    return "other:"+e[:60]
npass=nfail=0;ncls=collections.Counter();runsby=collections.Counter();only_int=0;nfiles=collections.Counter()
for dp,q in rows:
    if not(q and q.get("ok")): continue
    fs=glob.glob(dp+"/*.pwml"); nfiles[len(fs)]+=1
    for f in fs[:1]:
        r=run_pwml_qa(open(f,"rb").read())
        if r["ok"]: npass+=1
        else:
            nfail+=1
            cs={cls(e) for e in r["errors"]}
            for c in cs: runsby[c]+=1
            for e in r["errors"]: ncls[cls(e)]+=1
            if cs=={"interaction"}: only_int+=1
print("files per run",dict(nfiles));print("pass",npass,"fail",nfail,"fail-only-interaction",only_int)
print("errors by class",dict(ncls));print("failing runs by class",dict(runsby))
