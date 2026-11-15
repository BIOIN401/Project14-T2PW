"""Gold reaction recall by stage. Gold = eval_v2 supplement (4 cases) + pinned_v1 supported_reactions (other papers).
Read-only. Output out/gold_recall.jsonl
"""
import sys, json, os, collections
sys.path.insert(0, "C:/t/oa/src")
from pathlib import Path
from t2pw.bench.eval_v2 import gold as G, reactions as RX
from t2pw.bench.eval_v2.gold import Term, Reaction

R = "C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/"
v2 = G.load_supplement()
cases = {pid: {"complete": c.complete, "src": "eval_v2", "reactions": list(c.reactions)} for pid, c in v2.items()}
pin = json.load(open("C:/t/oa/src/t2pw/bench/gold/pinned_v1.json", encoding="utf-8"))["cases"]
def T(x):
    return Term(str(x["name"]), tuple(x.get("aliases") or ()), "defining")
for c in pin:
    pid = c["paper_id"]
    if pid in cases or not c.get("supported_reactions"):
        continue
    rxs = []
    for i, r in enumerate(c["supported_reactions"]):
        try:
            rxs.append(Reaction(id=f"p{i}", label=r.get("label", ""), inputs=tuple(T(t) for t in r["inputs"]), outputs=tuple(T(t) for t in r["outputs"]),
                                enzyme=None, reversible=False, quote=r.get("quote", "x")))
        except Exception as ex:
            print("skip", pid, i, ex)
    cases[pid] = {"complete": False, "src": "pinned_v1", "reactions": rxs}
print({k: (v["src"], v["complete"], len(v["reactions"])) for k, v in cases.items()})
rows = [json.loads(l) for l in open("C:/t/ev-oa/funnel/out/runs_funnel.jsonl")]
out = open("C:/t/ev-oa/funnel/out/gold_recall.jsonl", "w", encoding="utf-8")
for r in rows:
    pid = r["paper"]
    if pid not in cases:
        continue
    d = os.path.dirname(R + r["path"] if not r["path"].startswith("C:") else r["path"])
    pay = {s: json.load(open(f"{d}/{n}.json", encoding="utf-8")) for s, n in (("s1", "stage1_payload"), ("m", "merged_payload"), ("f", "final_mapped"))}
    rec = {"path": r["path"], "paper": pid, "qok": r["qok"], "mode": r["mode"], "gold_src": cases[pid]["src"], "gold_complete": cases[pid]["complete"],
           "n_gold": len(cases[pid]["reactions"]), "pwml": bool(r.get("pwml"))}
    for s, p in pay.items():
        rx = p["processes"].get("reactions", [])
        hit = []
        part = []
        for g in cases[pid]["reactions"]:
            kinds = [RX.match_reaction(g, row)[0] for row in rx]
            hit.append("full" in kinds)
            part.append(("full" in kinds) or ("partial" in kinds))
        rec[s] = sum(hit); rec[s + "_partial"] = sum(part); rec[s + "_hits"] = hit
    # quarantine: gold reactions present in M but not F
    out.write(json.dumps(rec) + "\n")
out.close()
