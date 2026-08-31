"""ORCH-717 / F-148: classify T-107's three timeouts from COMMITTED ARTIFACTS ONLY.

The charter requires F-148's timeouts to be separated into: ordinary stochastic
timeout, wrapper/deadline behaviour, provider failure, pipeline non-termination,
paper-specific pathological expansion, retry amplification, or absence of a
payload caused by CLEANUP rather than by pipeline failure.

**T-107 is not rerun and no leg of it is repeated.** Everything below is read out
of `runs_verify/2026-08-28_1816`, which that run already produced. This probe
asserts nothing and changes nothing; it prints the evidence each classification
rests on so the reasoning can be checked against the artifacts rather than taken
on the Lead's word.

The measurement that decides it is the TIMING DISTRIBUTION of the seventeen legs
that finished, against the leg ceiling the run actually used. A timeout is only
"stochastic bad luck" if the ceiling was comfortable for everything else.

Usage::

    <python> orch717_f148_timeout_probe.py <repo-root>
"""

from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
RUN = REPO / "runs_verify/2026-08-28_1816"

rows = [
    json.loads(line)
    for line in io.open(RUN / "manifest.jsonl", encoding="utf-8")
    if line.strip()
]

# ---------------------------------------------------------------------------
# 1. The three timeouts, and the TWO DIFFERENT mechanisms behind them.
# ---------------------------------------------------------------------------
print("=" * 78)
print("1. THE THREE TIMEOUTS -- two mechanisms, not one")
print("=" * 78)
timeouts = [r for r in rows if r.get("termination_reason")]
for r in timeouts:
    print(f"\n  {r['paper_id']}/{r['mode']}")
    print(f"      termination_reason  : {r.get('termination_reason')}")
    print(f"      stage               : {r.get('stage')}")
    print(f"      operational_failure : {r.get('operational_failure')}")
    print(f"      files               : {r.get('files')}")
    print(f"      counts              : {r.get('counts')}")
    budget = r.get("budget")
    if budget:
        for key in ("leg_timeout_seconds", "leg_timeout_default_seconds",
                    "leg_timeout_overridden", "leg_timeout_override_reason",
                    "leg_timeout_override_source", "elapsed_seconds",
                    "remaining_seconds", "finalization_reserve_seconds",
                    "child_deadline_seconds"):
            print(f"      {key:28s}: {budget.get(key)!r}")
    else:
        print("      budget              : ABSENT")
        print(f"      budget_unrecorded   : {'budget_unrecorded' in r}")

# ---------------------------------------------------------------------------
# 2. The timing distribution -- was the ceiling comfortable for everything else?
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("2. TIMING DISTRIBUTION of the legs that FINISHED, against the ceiling")
print("=" * 78)

summary = io.open(RUN / "SUMMARY.txt", encoding="utf-8", errors="replace").read()
pat = re.compile(r"^\s+(strict|research)\s+:\s+(\S+)[^|]*\|\s*stage=(\S+)\s*\|\s*time=([\d.]+)s",
                 re.MULTILINE)
legs = [(m.group(1), m.group(2), m.group(3), float(m.group(4))) for m in pat.finditer(summary)]

CEILING = 1800.0
finished = [t for (_, verdict, _, t) in legs if verdict != "TIMEOUT"]
timed_out = [t for (_, verdict, _, t) in legs if verdict == "TIMEOUT"]
finished.sort(reverse=True)

print(f"\n  legs parsed        : {len(legs)}   finished: {len(finished)}   timed out: {len(timed_out)}")
print(f"  leg ceiling used   : {CEILING:.0f}s   (default is 3600s -- this run HALVED it)")
print(f"\n  slowest legs that FINISHED, as a fraction of the ceiling:")
for t in finished[:6]:
    bar = "#" * int(round(60 * t / CEILING))
    print(f"      {t:8.1f}s  {100 * t / CEILING:5.1f}%  {bar}")
print(f"\n  max finished       : {max(finished):.1f}s = {100 * max(finished) / CEILING:.1f}% of ceiling")
print(f"  median finished    : {sorted(finished)[len(finished) // 2]:.1f}s")
print(f"  headroom on slowest: {CEILING - max(finished):.1f}s")
print(f"\n  Against the DEFAULT 3600s ceiling the slowest finished leg is "
      f"{100 * max(finished) / 3600:.1f}%.")

# ---------------------------------------------------------------------------
# 3. Is the pipeline non-terminating on these papers?
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("3. NON-TERMINATION vs BUDGET -- does the other leg of the same paper finish?")
print("=" * 78)
by_paper: dict[str, list] = {}
for r in rows:
    by_paper.setdefault(r["paper_id"], []).append(r)
for paper in ("PMC12444477", "PMC12096016"):
    print(f"\n  {paper}")
    for r in by_paper.get(paper, []):
        term = r.get("termination_reason")
        if term:
            print(f"      {r['mode']:9s} TIMEOUT ({term})")
        else:
            counts = r.get("counts") or {}
            print(f"      {r['mode']:9s} FINISHED  counts={ {k: counts[k] for k in list(counts)[:4]} }")

# ---------------------------------------------------------------------------
# 4. Retry amplification -- can it even be ruled out from what survived?
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("4. RETRY AMPLIFICATION -- what evidence survived to test it with?")
print("=" * 78)
log = io.open(RUN / "batch.log", encoding="utf-8", errors="replace").read()
print(f"\n  batch.log lines            : {len(log.splitlines())}")
for term in ("retry", "retrying", "attempt", "backoff", "rate limit", "429"):
    print(f"  occurrences of {term!r:12s}: {len(re.findall(term, log, re.I))}")
print("\n  attempt records preserved on the three timed-out legs:")
for r in timeouts:
    keys = [k for k in r if "attempt" in k or "retry" in k or "call" in k]
    print(f"      {r['paper_id']}/{r['mode']:9s} -> {keys or 'NONE'}")

# ---------------------------------------------------------------------------
# 5. The override that halved the budget, and what it recorded about itself.
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("5. THE OVERRIDE -- recorded as a fact, unrecorded as a decision")
print("=" * 78)
for r in timeouts:
    budget = r.get("budget")
    if not budget:
        continue
    print(f"\n  {r['paper_id']}/{r['mode']}")
    print(f"      overridden : {budget.get('leg_timeout_overridden')}   "
          f"{budget.get('leg_timeout_default_seconds')} -> {budget.get('leg_timeout_seconds')}")
    print(f"      reason     : {budget.get('leg_timeout_override_reason')!r}")
    print(f"      source     : {budget.get('leg_timeout_override_source')!r}")
    reserve = budget.get("finalization_reserve_seconds")
    child = budget.get("child_deadline_seconds")
    elapsed = budget.get("elapsed_seconds")
    print(f"      finalization reserve : {reserve}s, child deadline {child}s")
    print(f"      elapsed              : {elapsed}s  -> overran the child deadline by "
          f"{elapsed - child:.2f}s and consumed the ENTIRE reserve")
