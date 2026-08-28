"""Verify REV-102's two numbers on the corpus, independently: how many legs carry a
MATCHED forbidden term, and what a denominator-only exclusion would report."""
import io, json, subprocess
from pathlib import Path
from t2pw.bench.goldset import load_gold_set, pinned_gold_set_path
from t2pw.bench.acceptance import contract_accepted_coverage

ROOT = Path("C:/t/c102")
gold = {c.paper_id: c for c in load_gold_set(pinned_gold_set_path()).cases}
out = subprocess.run(["git", "ls-files", "*quarantine_report.json"], cwd=str(ROOT),
                     capture_output=True, text=True, encoding="utf-8").stdout
legs = with_matched = 0
over_one = []
worst = []
for line in sorted(l.strip() for l in out.splitlines() if l.strip()):
    p = ROOT / line
    case = gold.get(p.parent.parent.name)
    if case is None:
        continue
    cov = json.loads(p.read_text(encoding="utf-8")).get("coverage") or {}
    r = contract_accepted_coverage(case, cov)
    if r is None:
        continue
    legs += 1
    matched_forbidden = sum(1 for e in r["excluded_terms"] if e["matched_in_raw"])
    if matched_forbidden:
        with_matched += 1
    # What a DENOMINATOR-ONLY exclusion would report: numerator keeps the
    # forbidden matches, denominator loses them.
    if r["accepted_denominator"]:
        den_only = r["raw_matched"] / r["accepted_denominator"]
        if den_only > 1.0:
            over_one.append((f"{p.parents[3].name}/{p.parent.parent.name}:{p.parent.name}",
                             round(den_only, 4), r["raw_matched"], r["accepted_denominator"]))
        if matched_forbidden:
            worst.append((f"{p.parent.parent.name}:{p.parent.name}", round(den_only, 4),
                          r["accepted_ratio"], matched_forbidden))
print(f"legs scored                              : {legs}")
print(f"legs carrying >=1 MATCHED forbidden term : {with_matched}")
print(f"denominator-only ratios ABOVE 1.0        : {len(over_one)}")
for row in over_one:
    print(f"    {row[0]:52s} {row[1]}   ({row[2]}/{row[3]})")
print(f"\nlegs where denominator-only differs from both-sides: {len(worst)}")
for row in worst[:30]:
    print(f"    {row[0]:28s} den_only={row[1]:<8} both_sides={row[2]}  matched_forbidden={row[3]}")
