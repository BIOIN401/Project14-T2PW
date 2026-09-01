"""REV-106 independent re-derivation of the four census-derived quantities.

Written from the two test bodies directly, NOT from the author's probe. Prints a
per-run attribution table so 62->72, 92->97 and 23->26 can be checked leg by leg.
Read-only.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REPO / "src"))

from t2pw.bench.acceptance import contract_accepted_coverage
from t2pw.bench.goldset import load_gold_set, pinned_gold_set_path

GOLD = {c.paper_id: c for c in load_gold_set(pinned_gold_set_path()).cases}

listed = subprocess.run(
    ["git", "ls-files", "*quarantine_report.json"],
    cwd=str(REPO), capture_output=True, text=True, encoding="utf-8", check=True,
)
paths = sorted(l.strip() for l in listed.stdout.splitlines() if l.strip())
print(f"git ls-files *quarantine_report.json : {len(paths)} paths")

legs = withheld = with_matched_forbidden = 0
skipped_no_case = skipped_none = 0
affected: dict[str, int] = {}
cleared: list[str] = []
per_run = defaultdict(lambda: {"legs": 0, "withheld": 0, "mf": 0})

for rel in paths:
    leg_dir = (REPO / rel).parent
    case = GOLD.get(leg_dir.parent.name)
    if case is None:
        skipped_no_case += 1
        continue
    coverage = json.load(io.open(REPO / rel, encoding="utf-8")).get("coverage") or {}
    out = contract_accepted_coverage(case, coverage)
    if out is None:
        skipped_none += 1
        continue
    run = rel.split("/")[1] if rel.startswith("runs_verify/") else rel.split("/")[0]
    legs += 1
    per_run[run]["legs"] += 1
    if out["excluded_count"]:
        affected[case.paper_id] = affected.get(case.paper_id, 0) + 1
        withheld += out["excluded_count"]
        per_run[run]["withheld"] += out["excluded_count"]
    if out["cleared_by_reconciliation"]:
        cleared.append(f"{case.paper_id}:{leg_dir.name}")
    if any(e["matched_in_raw"] for e in out["excluded_terms"]):
        with_matched_forbidden += 1
        per_run[run]["mf"] += 1

print(f"skipped: no gold case={skipped_no_case}  contract returned None={skipped_none}")
print()
print(f"{'run':<40} {'legs':>5} {'withheld':>9} {'matched_forb':>13}")
for run in sorted(per_run):
    d = per_run[run]
    print(f"{run:<40} {d['legs']:>5} {d['withheld']:>9} {d['mf']:>13}")
print("-" * 70)
print(f"{'TOTAL':<40} {legs:>5} {withheld:>9} {with_matched_forbidden:>13}")
print()
TARGET = "2026-08-28_1816"
d = per_run.get(TARGET, {"legs": 0, "withheld": 0, "mf": 0})
print(f"{TARGET} contributes: legs={d['legs']} withheld={d['withheld']} matched_forb={d['mf']}")
print(f"all OTHER runs sum to: legs={legs-d['legs']} withheld={withheld-d['withheld']} "
      f"matched_forb={with_matched_forbidden-d['mf']}")
print()
print(f"REV-106 DERIVED: legs={legs} checked={legs} withheld={withheld} "
      f"with_matched_forbidden={with_matched_forbidden}")
print(f"affected - F132_PAPERS = "
      f"{sorted(set(affected) - {'PMC12096016','PMC12312563','PMC12444477','PMC12452463','PMC12782028','PMC12856317'})}")
print(f"cleared = {cleared}")
