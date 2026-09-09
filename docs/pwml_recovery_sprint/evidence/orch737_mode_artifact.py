"""Is REV-121 FINDING 1's extra pair a MODE artifact of the reviewer's own probe?

The two legs it names are both `/research/` legs. quarantine_and_close's own
docstring says research mode "runs every decision and applies none of them". Driving
a research leg through STRICT quarantine is not what production does. Tested here.
"""
import json, sys
from pathlib import Path

TREE = Path(sys.argv[1])
ARCHIVE = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW")
sys.path.insert(0, str(TREE / "src"))
import t2pw
print("TREE:", t2pw.__file__)
from t2pw.pipeline.strict_quarantine import quarantine_and_close

LEGS = [
    "runs_verify/2026-08-22_2147/papers/PMC12452463/research/final_mapped.json",
    "runs_verify/2026-08-24_1428/papers/PMC12782028/research/final_mapped.json",
]
for rel in LEGS:
    payload = json.loads((ARCHIVE / rel).read_text(encoding="utf-8"))
    print(f"\n### {rel}")
    for mode in ("research", "pathwhiz"):
        res = quarantine_and_close(payload, strict_db=True, mode=mode)
        p = res.payload if hasattr(res, "payload") else {}
        states = sorted(str(r.get("name")) for r in (p.get("biological_states") or []) if isinstance(r, dict))
        pr = p.get("processes") or {}
        print(f"   mode={mode:9} states={states} rx={len(pr.get('reactions') or [])} ok={getattr(res,'ok',None)}")
