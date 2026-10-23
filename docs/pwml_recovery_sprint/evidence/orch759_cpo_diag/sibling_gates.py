"""Stdlib-only: name_gate / rejected_mapped_ids / lineage of the bare enzymes in the 4 shape legs."""
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW")
CASES = [
    ("runs_papers/ORCH-759/legs/chlorophyll-strict/2026-10-22_1231/papers/PMC5130987/strict", "CPO"),
    ("runs_verify/2026-08-21_2057/papers/PMC12096016/strict", "lactate dehydrogenase"),
    ("runs_verify/2026-09-06_1425/papers/PMC11405693/strict", "AtCYP73A5"),
    ("runs_verify/2026-09-06_1425/papers/PMC12376012/strict", "ceramide synthase"),
]
for leg, name in CASES:
    d = ROOT / leg
    p = json.loads((d / "final_mapped.json").read_text(encoding="utf-8"))
    r = next(x for x in p["entities"]["proteins"] if x.get("name") == name)
    mm = r.get("mapping_meta") or {}
    ng = mm.get("name_gate") or {}
    iv = (mm.get("identity_verdict") or {}).get("name_gate") or {}
    print(f"{leg} :: {name}")
    print("   ids:", r.get("mapped_ids"), "| rejected:", mm.get("rejected_mapped_ids"), "| chosen_rule:", mm.get("chosen_rule"), "| provider/source:", mm.get("provider"), mm.get("source"), "| candidates:", len(mm.get("candidates") or []))
    print("   stored name_gate:", ng.get("verdict"), ng.get("reason"), "| compared:", ng.get("compared_names"))
    print("   identity_verdict.name_gate:", iv.get("verdict"), iv.get("reason"), iv.get("matched_symbol"))
    print("   lineage:", [ (l.get("stage"), (l.get("reason") or "")[:70]) for l in r.get("provenance_lineage") or []])
    g = d / "pwml_required_field_gate_report.json"
    if g.exists():
        gj = json.loads(g.read_text(encoding="utf-8"))
        print("   gate errors:", [(e.get("code"), e.get("pointer")) for e in gj.get("errors") or []])
    rt = d / "RESULT.txt"
    if rt.exists():
        lines = rt.read_text(encoding="utf-8", errors="replace").splitlines()
        print("   RESULT:", [l for l in lines if l.startswith("RESULT") or l.strip().startswith("message")][:2])
