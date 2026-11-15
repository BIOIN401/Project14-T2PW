import json, sys, difflib
sys.path.insert(0, "C:/t/fp/docs/pwml_output/evidence/final_payload_p5/writerA/scripts")
from pathlib import Path
E = Path("C:/t/fp/docs/pwml_output/evidence")
D3 = E / "final_payload_p4/d3/replay/on"; WA = Path("C:/t/ev-fp/p5/replay/writerA_on"); OFF = E / "final_payload_p4/d3/replay/off"
def load(p): return {json.loads(l)["idx"]: json.loads(l) for l in open(p, encoding="utf-8")}
d3, wa = load(D3/"results_0.jsonl"), load(WA/"results_0.jsonl")
ch = [3, 4, 8, 10, 16, 18, 20, 23, 27, 31, 37, 46, 58, 66, 77, 78, 88, 91, 98, 102, 109, 119, 126, 141, 142, 146, 150, 151]
for i in ch:
    a, b = d3[i].get("layout_debug_counts") or {}, wa[i].get("layout_debug_counts") or {}
    mv = {k: (a.get(k, 0), b.get(k, 0)) for k in set(a) | set(b) if a.get(k, 0) != b.get(k, 0)}
    A = (D3/"pwml"/f"{i}.pwml").read_text(encoding="utf-8").splitlines(); B = (WA/"pwml"/f"{i}.pwml").read_text(encoding="utf-8").splitlines()
    nd = sum(1 for l in difflib.unified_diff(A, B, lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---"))
    print(i, "difflines", nd, {k: v for k, v in mv.items() if "dropped" in k or "kept" in k or "wildcard" in k}, {k: v for k, v in list(mv.items())[:4] if not ("dropped" in k or "kept" in k or "wildcard" in k)})
