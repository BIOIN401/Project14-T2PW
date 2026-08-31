"""ORCH-717: measure EVERY derived quantity tests 10 and 13 pin, not just the census.

F-151 reports the breakage as ``assert 72 == 62``. That is only the FIRST
assertion in each test. Both tests carry further derived pins after it --
``withheld == 92``, ``with_matched_forbidden == 23``, and a set-equality on the
papers outside ``F132_PAPERS`` -- and none of them has ever executed against the
grown corpus, because the census assert aborts the test before reaching them.

So "re-pin 62 -> 72" is a hypothesis, not a fix. This probe runs both loops to
completion with the census assert removed and prints what the pins WOULD have to
become, plus the per-paper attribution that justifies moving them. It asserts
nothing and changes nothing.

Predictions were recorded first, in ``orch717_census_predictions.md``.

Usage::

    <python> orch717_census_probe.py <repo-root>
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REPO / "src"))

from t2pw.bench.acceptance import contract_accepted_coverage      # noqa: E402
from t2pw.bench.goldset import load_gold_set, pinned_gold_set_path  # noqa: E402

F132_PAPERS = (
    "PMC12096016", "PMC12312563", "PMC12444477",
    "PMC12452463", "PMC12782028", "PMC12856317",
)

GOLD = {case.paper_id: case for case in load_gold_set(pinned_gold_set_path()).cases}

listed = subprocess.run(
    ["git", "ls-files", "*quarantine_report.json"],
    cwd=str(REPO), capture_output=True, text=True, encoding="utf-8", check=True,
)
paths = sorted(line.strip() for line in listed.stdout.splitlines() if line.strip())

print(f"P1  tracked quarantine_report.json : {len(paths)}")

# Which runs do they come from, and which are new since the 62-era pin?
runs: dict[str, int] = {}
for rel in paths:
    parts = Path(rel).parts
    run = "/".join(parts[:2]) if len(parts) > 2 else "?"
    runs[run] = runs.get(run, 0) + 1
print("\nper-run population:")
for run in sorted(runs):
    print(f"    {run:44s} {runs[run]:>3}")

legs = checked = withheld = with_matched_forbidden = 0
affected_papers: dict[str, int] = {}
cleared: list[str] = []
# attribution: which leg contributed which excluded terms, grouped by run
per_run_legs: dict[str, int] = {}
per_run_withheld: dict[str, int] = {}
per_run_matched: dict[str, int] = {}

for rel in paths:
    leg_dir = (REPO / rel).parent
    case = GOLD.get(leg_dir.parent.name)
    if case is None:
        continue
    coverage = json.load(io.open(REPO / rel, encoding="utf-8")).get("coverage") or {}
    out = contract_accepted_coverage(case, coverage)
    if out is None:
        continue
    parts = Path(rel).parts
    run = "/".join(parts[:2]) if len(parts) > 2 else "?"
    legs += 1
    checked += 1
    per_run_legs[run] = per_run_legs.get(run, 0) + 1
    if out["excluded_count"]:
        affected_papers[case.paper_id] = affected_papers.get(case.paper_id, 0) + 1
        withheld += out["excluded_count"]
        per_run_withheld[run] = per_run_withheld.get(run, 0) + out["excluded_count"]
    if out["cleared_by_reconciliation"]:
        cleared.append(f"{case.paper_id}:{leg_dir.name}")
    if any(e["matched_in_raw"] for e in out["excluded_terms"]):
        with_matched_forbidden += 1
        per_run_matched[run] = per_run_matched.get(run, 0) + 1

print(f"\nP2  test 10  legs                   : {legs}      (pinned == 62)")
print(f"P3  test 13  checked                : {checked}      (pinned == 62)")
print(f"P4  test 10  withheld               : {withheld}      (pinned == 92)")
print(f"P5  test 13  with_matched_forbidden : {with_matched_forbidden}      (pinned == 23)")
print(f"P7  test 10  cleared                : {cleared}   (pinned == [])")

outside = set(affected_papers) - set(F132_PAPERS)
print(f"\nP6  affected_papers OUTSIDE F132_PAPERS : {sorted(outside)}")
print(f"    (pinned == {{'PMC13231680'}})")
print(f"    F132_PAPERS all present : {set(F132_PAPERS) <= set(affected_papers)}")
print(f"    affected_papers         : {dict(sorted(affected_papers.items()))}")

print("\nATTRIBUTION -- what each run contributes to the three moving pins:")
print(f"    {'run':44s} {'legs':>5} {'withheld':>9} {'matched':>8}")
for run in sorted(per_run_legs):
    print(f"    {run:44s} {per_run_legs[run]:>5} "
          f"{per_run_withheld.get(run, 0):>9} {per_run_matched.get(run, 0):>8}")
