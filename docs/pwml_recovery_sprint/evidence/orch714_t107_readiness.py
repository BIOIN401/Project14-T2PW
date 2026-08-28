"""T-107 readiness: score the most recent full run at the MERGED tip and emit
every row the readiness table needs. Offline, deterministic, no live model."""
from __future__ import annotations
import json, sys
from pathlib import Path
TREE = Path(sys.argv[1]).resolve(); RUN = Path(sys.argv[2])
sys.path.insert(0, str(TREE / "src"))
from t2pw.bench.acceptance import score_run  # noqa: E402
from t2pw.bench.goldset import load_gold_set, pinned_gold_set_path  # noqa: E402
print("MEASURED_TREE t2pw =", sys.modules["t2pw"].__file__)
print("RUN =", RUN)

gold = load_gold_set(pinned_gold_set_path())
rep = score_run(RUN, gold)
d = rep.to_dict()

print("\n" + "=" * 78)
print("PRIORITIES AT THE MERGED TIP")
print("=" * 78)
for p in d.get("acceptance_priorities") or d.get("priorities") or []:
    print(f"\nrank {p.get('rank')}: {p.get('name')}")
    for k in ("ok", "evaluated", "observed", "counted", "raw", "accepted",
              "accepted_status", "papers", "not_evaluated_papers",
              "not_evaluated_legs", "not_evaluated_reasons",
              "contract_adjusted", "requested_core_coverage"):
        if k in p:
            v = json.dumps(p[k], default=str)
            print(f"    {k:26s} = {v[:600]}")

print("\n" + "=" * 78)
print("COMPLETION / DENOMINATORS")
print("=" * 78)
for k in ("completion", "denominators", "legs_attempted", "legs_scored",
          "is_complete", "coverage_reconciliation_corpus", "coverage_reconciliation"):
    if k in d:
        print(f"{k} = {json.dumps(d[k], default=str)[:1200]}")

print("\n" + "=" * 78)
print("PRIORITY-1 ROW COMPOSITION (raw)")
print("=" * 78)
rows = d.get("priority1_rows") or []
if not rows:
    for pap in d.get("papers", []):
        for mode, leg in (pap.get("legs") or {}).items():
            for f in (leg.get("semantic") or {}).get("findings", []) or []:
                if "false_real" in json.dumps(f):
                    print("  ", pap.get("paper_id"), mode, json.dumps(f)[:200])
else:
    for r in rows:
        print("  ", json.dumps(r, default=str)[:260])

print("\n" + "=" * 78)
print("LpxH CHECK on PMC12444477")
print("=" * 78)
for pap in d.get("papers", []):
    if pap.get("paper_id") != "PMC12444477":
        continue
    for mode, leg in (pap.get("legs") or {}).items():
        sem = leg.get("semantic") or {}
        names = []
        for f in sem.get("findings", []) or []:
            n = f.get("name") if isinstance(f, dict) else None
            if n: names.append(n)
        if names:
            print(f"  {mode}: {len(names)} findings -> {json.dumps(sorted(set(names)))}")
            print(f"     LpxH present: {'LpxH' in names}")
