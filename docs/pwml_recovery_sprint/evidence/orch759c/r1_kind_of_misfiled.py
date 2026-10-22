"""ORCH-759c round 1: for each location row the per-bucket rule removes on a non-target leg,
which entity buckets declare that name, and does that entity keep another location row?

Stdlib only; reads the legs read-only. Prints only.
"""
import json
import re
from pathlib import Path

P = Path(r"C:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW")
CASES = [
    (P / ".claude/worktrees/orch759c/runs/2026-07-28_0919/papers/PMC12444477__the-regulation-of-lipid-a-biosynthesis/strict", ["FtsH-LapB complex"]),
    (P / "runs_papers/ORCH-759/legs/wnt_pcp-strict/2026-10-22_1432/papers/PMC10425814/strict", ["complex of Dvl, Prickle, and inturned"]),
    (P / "runs_verify/2026-08-25_1216/papers/PMC12856317/strict", ["LONP1"]),
    (P / "runs_verify/2026-08-28_1816/papers/PMC12856317/strict", ["ALAS2 homodimer"]),
    (P / "runs_verify/2026-09-02_2052/papers/PMC12444477/strict", ["FtsH/YciM complex"]),
    (P / "runs_verify/2026-09-02_2052/papers/PMC12856317/strict", ["ALAS2 homodimer"]),
    (P / "runs_verify/2026-09-06_1425/papers/PMC11172790/strict", ["PvdA", "PvdI", "PvdJ"]),
    (P / "runs_papers/ORCH-759/legs/chlorophyll-strict/2026-10-22_1231/papers/PMC5130987/strict", ["Mg-chelatase complex", "Mg-Proto ME cyclase complex"]),
]


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


for leg, names in CASES:
    d = json.loads((leg / "final_mapped.json").read_text(encoding="utf-8"))
    ents, locs = d.get("entities") or {}, d.get("element_locations") or {}
    print(f"## {leg.relative_to(P)}")
    for name in names:
        kinds = [b for b, rows in ents.items() if isinstance(rows, list)
                 for r in rows if isinstance(r, dict) and norm(name) in {norm(r.get("name"))} | {norm(s) for s in r.get("synonyms") or [] if isinstance(s, str)}]
        rows = [(b, r.get("biological_state")) for b, rr in locs.items() if isinstance(rr, list)
                for r in rr if isinstance(r, dict) and any(norm(v) == norm(name) for v in r.values() if isinstance(v, str))]
        print(f"   {name!r}: declared in {kinds}; location rows {rows}")
