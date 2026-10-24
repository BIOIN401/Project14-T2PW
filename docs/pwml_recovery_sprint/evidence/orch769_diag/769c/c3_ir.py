import json
d = r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/runs_papers/ORCH-759/legs/wnt_pcp-strict/2026-10-22_1432/papers/PMC10425814/strict"
ir = json.load(open(f"{d}/pwml_ir.json", encoding="utf-8"))
print(type(ir), list(ir.keys())[:30] if isinstance(ir, dict) else len(ir))
for k, v in (ir.items() if isinstance(ir, dict) else []):
    if isinstance(v, list): print(k, len(v))
    elif isinstance(v, dict): print(k, {kk: (len(vv) if isinstance(vv, (list, dict)) else vv) for kk, vv in list(v.items())[:20]})
rep = json.load(open(f"{d}/pwml_ir_report.json", encoding="utf-8"))
print("report counts", rep.get("counts"))
print("warnings", [w.get("message", w) if isinstance(w, dict) else w for w in rep.get("warnings", [])][:10])
