import json,hashlib,glob
from pathlib import Path
E=Path("C:/t/fp/docs/pwml_output/evidence")
def load(p):
    o={}
    for l in open(p,encoding="utf-8"):
        r=json.loads(l);o[r["idx"]]=r
    return o
old=load(E/"final_payload_p2/replay6/off6/results_0.jsonl")
off=load(E/"final_payload_p3/replay/off/results_0.jsonl")
on=load(E/"final_payload_p3/replay/on/results_0.jsonl")
on6=load(E/"final_payload_p2/replay6/on6/results_0.jsonl")
h=lambda d,i:hashlib.sha256((E/d/"pwml"/f"{i}.pwml").read_bytes()).hexdigest()
w=lambda r:{i for i,x in r.items() if x["pwml_written"]}
wo,wf,wn=w(old),w(off),w(on)
print("written old-off/new-off/on/p2on6",len(wo),len(wf),len(wn),len(w(on6)))
print("same set",wo==wf)
same=sum(1 for i in wo&wf if h("final_payload_p2/replay6/off6",i)==h("final_payload_p3/replay/off",i))
print("identical",same,"different",len(wo&wf)-same)
print("on vs on6 sha same",sum(1 for i in wn&w(on6) if h("final_payload_p2/replay6/on6",i)==h("final_payload_p3/replay/on",i)))
print("errors",[ (i,x["error"][:80]) for i,x in on.items() if x["error"]][:5])
Path("C:/t/fp/docs/pwml_output/evidence/final_payload_p3/import").mkdir(exist_ok=True)
lo=E/"final_payload_p3/import"
open(lo/"list_on.txt","w").write("\n".join(str(E/"final_payload_p3/replay/on/pwml"/f"{i}.pwml").replace("\\","/") for i in sorted(wn))+"\n")
