"""C-102 / D-072: the pre-change vs post-change A/B on the exact F-132 population.

OFFLINE. It reads only committed ``quarantine_report.json`` artifacts and the
pinned gold set. No leg is re-run, no cohort repeated, no model call spent --
re-running the corpus is forbidden this wave, and the raw side of the A/B is
fully determined by what those artifacts already froze.

PRE-CHANGE is the frozen ``coverage_ratio`` the pipeline recorded and the
instrument reported unchanged. POST-CHANGE is
``t2pw.bench.acceptance.contract_accepted_coverage``, the shipped function --
not a local reimplementation of it, so this probe cannot agree with the scorer
by having been written twice.

Usage::

    <python> c102_f132_coverage_ab.py <repo-root-holding-runs_verify>

with ``PYTHONPATH`` selecting WHICH tree is measured. Pointed at a base tree it
exits 3 with ``POST-CHANGE UNAVAILABLE``; that is the pre-change half.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()

from t2pw.bench.goldset import load_gold_set, pinned_gold_set_path  # noqa: E402

try:
    from t2pw.bench.acceptance import contract_accepted_coverage
except ImportError as exc:  # pragma: no cover - exercised by the base leg
    print(f"POST-CHANGE UNAVAILABLE on this tree: {exc}")
    print("This is the PRE-CHANGE half of the A/B: the instrument has one coverage")
    print("answer per leg, computed over a denominator it never reconciled.")
    raise SystemExit(3)

#: The six papers the recovered ORCH-702 probe measured. Named so a future run
#: over a larger corpus is visibly a different population, not a silent one.
F132_PAPERS = (
    "PMC12096016", "PMC12312563", "PMC12444477",
    "PMC12452463", "PMC12782028", "PMC12856317",
)

gold = load_gold_set(pinned_gold_set_path())
listed = subprocess.run(
    ["git", "ls-files", "*quarantine_report.json"],
    cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
)

legs_with_coverage = legs_with_forbidden = 0
terms_total = terms_excluded = 0
cleared: list[str] = []
still_below: list[str] = []
undefined: list[str] = []
per_paper: dict[str, dict[str, int]] = {p: {"legs": 0, "terms": 0} for p in F132_PAPERS}
rows: list[dict] = []

for line in sorted(l.strip() for l in listed.stdout.splitlines() if l.strip()):
    leg_dir = (ROOT / line).parent
    paper, mode = leg_dir.parent.name, leg_dir.name
    case = gold.by_id(paper)
    if case is None:
        continue
    coverage = (json.loads((ROOT / line).read_text(encoding="utf-8")).get("coverage") or {})
    recon = contract_accepted_coverage(case, coverage)
    if recon is None:
        continue
    legs_with_coverage += 1
    terms_total += recon["raw_denominator"]
    key = f"{leg_dir.parents[1].parent.name}/{paper}:{mode}"
    if recon["accepted_state"] == "undefined_every_term_forbidden":
        undefined.append(key)
    if not recon["excluded_count"]:
        continue
    legs_with_forbidden += 1
    terms_excluded += recon["excluded_count"]
    per_paper.setdefault(paper, {"legs": 0, "terms": 0})
    per_paper[paper]["legs"] += 1
    per_paper[paper]["terms"] += recon["excluded_count"]
    if recon["cleared_by_reconciliation"]:
        cleared.append(key)
    if recon["accepted_below_minimum"] is True:
        still_below.append(key)
    rows.append({"key": key, **recon})
    raw = "n/a" if recon["raw_ratio"] is None else f"{recon['raw_ratio']:.3f}"
    acc = "UNDEFINED" if recon["accepted_ratio"] is None else f"{recon['accepted_ratio']:.3f}"
    print(
        f"{key:52s} raw {recon['raw_matched']:>2}/{recon['raw_denominator']:<2}={raw}"
        f"   accepted {recon['accepted_matched']:>2}/{recon['accepted_denominator']:<2}={acc}"
        f"   below_min raw={recon['raw_below_minimum']} accepted={recon['accepted_below_minimum']}"
        + ("   *** CLEARS ***" if recon["cleared_by_reconciliation"] else "")
    )
    for entry in recon["excluded_terms"]:
        print(
            f"      withheld {entry['term']!r}  kind={entry['forbidden_kind']}"
            f"  matched_in_raw={entry['matched_in_raw']}"
        )

print()
print("================ C-102 A/B, pre-change vs post-change ================")
print(f"legs with a coverage block            : {legs_with_coverage}")
print(f"legs carrying >=1 gold-forbidden term : {legs_with_forbidden}")
print(f"requested-core terms drawn, total     : {terms_total}")
print(f"gold-forbidden terms withheld         : {terms_excluded}")
print(f"legs CLEARED by the reconciliation    : {len(cleared)}  {cleared}")
print(f"legs still below the unchanged minimum: {len(still_below)}")
print(f"legs with an UNDEFINED accepted rate   : {len(undefined)}  {undefined}")
print()
print("per paper (the six F-132 papers first, then any other paper that shows one):")
for paper, counts in sorted(per_paper.items(), key=lambda kv: (kv[0] not in F132_PAPERS, kv[0])):
    flag = "" if paper in F132_PAPERS else "   <- NOT in the ORCH-702 population"
    print(f"  {paper:14s} legs={counts['legs']:>2}  forbidden_terms={counts['terms']:>2}{flag}")
print()
print("ratio deltas, every affected leg, pre -> post:")
for row in rows:
    acc = "UNDEFINED" if row["accepted_ratio"] is None else f"{row['accepted_ratio']:.4f}"
    pre = "n/a" if row["raw_ratio"] is None else f"{row['raw_ratio']:.4f}"
    print(f"  {row['key']:52s} {pre} -> {acc}")
