"""ORCH-718: census every recorded leg duration, to choose the T-108 leg ceiling deliberately.

T108-READINESS.md section 2 records the blocker this probe serves:

    T-107 ran with leg_timeout_overridden: true, 3600.0 -> 1800.0, and BOTH
    leg_timeout_override_reason and leg_timeout_override_source EMPTY. The
    slowest leg that finished used 92.1% of that ceiling. At that budget three
    timeouts is the expected outcome, not an anomaly.

PRODUCT_CONTRACT section 9 requires a per-leg override to be explicit and recorded in
the run manifest. On T-107 the FACT and the VALUE are recorded; the DECISION is not.
This probe supplies the measurement a deliberate decision needs, and nothing else.

WHAT THIS IS NOT
----------------
This is a read-only census of ALREADY-COMMITTED artifacts. It runs no leg, scores
nothing, re-runs nothing and mutates nothing. **T-107 is immutable: its verdict is
NOT ACCEPTED and this probe neither restates nor reinterprets it.** A duration is a
fact about how long a process ran; it is not a verdict about what the process produced.

NAMING THE RUN TREES EXPLICITLY
-------------------------------
`runs/` and `runs_verify/` are BOTH live and both hold papers/*/*/ artifacts, so
"the pinned run" is ambiguous and has cost a full rescan before. This probe therefore
enumerates every manifest under BOTH roots and labels each by its own directory,
rather than accepting any caller's idea of which tree is meant.

Usage::

    <venv-python> orch718_leg_duration_census.py <repo-root>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
ROOTS = ["runs", "runs_verify"]

# A leg that TIMED OUT tells us only that it exceeded the ceiling in force -- its
# duration is censored at the budget and is evidence about the budget, never about
# how long the work needed. Only legs that FINISHED bound the requirement from below.
# Conflating the two is how a ceiling gets "justified" by the very timeouts it caused.
#
# CORRECTION, attempt 2. Attempt 1 classified a CRASHED leg as "finished", because it
# tested only for timeout and operational_failure and let everything else fall through.
# That swept in all 56 legs of runs/2026-07-27_2135, every one of which is
#
#     "failure_kind": "crash", "seconds": 0.0,
#     ModuleNotFoundError: No module named 'streamlit'
#
# -- F-143 itself, preserved in a committed run tree: a bare `python` outside the venv.
# It is an INFRASTRUCTURE FAILURE, not a test result, and a leg that crashed at init in
# 0.0s carries no information about how long the work takes. Fifty-six zeros dragged the
# pooled median and p90 down while leaving the max untouched, so the CEILING conclusion
# was unaffected and the DISTRIBUTION was wrong. Attempt 1's log is preserved beside this
# file as orch718_leg_duration_census.attempt1-crashes-counted-as-finished.log.
CRASH_KINDS = {"crash", "error"}


def classify(row):
    status = str(row.get("status") or "").lower()
    if status == "timeout" or row.get("failure_kind") == "timeout":
        return "timeout"
    if status in CRASH_KINDS or str(row.get("failure_kind") or "").lower() in CRASH_KINDS:
        return "crash"
    if row.get("operational_failure") is True:
        return "operational"
    return "finished"


def main() -> int:
    manifests = []
    for r in ROOTS:
        base = ROOT / r
        if base.is_dir():
            manifests.extend(sorted(base.glob("*/manifest.jsonl")))

    if not manifests:
        print("NO MANIFESTS FOUND -- this is a measurement failure, not a result")
        return 1

    print("leg-duration census  |  roots: %s" % ", ".join(ROOTS))
    print("=" * 100)

    all_finished = []
    for m in manifests:
        rows = []
        for line in m.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    print("  !! unparseable line in %s -- preserved as a defect, not skipped" % m)
        if not rows:
            continue

        tree = m.parent.relative_to(ROOT).as_posix()
        buckets = {"finished": [], "timeout": [], "operational": [], "crash": []}
        ceiling = None
        for row in rows:
            kind = classify(row)
            secs = row.get("seconds")
            budget = row.get("budget") or {}
            if budget.get("leg_timeout_seconds") is not None:
                ceiling = budget["leg_timeout_seconds"]
            if not isinstance(secs, (int, float)):
                continue
            buckets[kind].append(float(secs))
        fin, to, op, cr = (buckets["finished"], buckets["timeout"],
                           buckets["operational"], buckets["crash"])

        all_finished.extend((tree, s) for s in fin)
        print()
        print("TREE %s" % tree)
        print("  legs=%-3d finished=%-3d timeout=%-3d operational=%-3d crash=%-3d  ceiling_in_force=%s"
              % (len(rows), len(fin), len(to), len(op), len(cr), ceiling))
        if cr:
            print("  %d CRASHED leg(s) EXCLUDED from the distribution -- infrastructure, not duration"
                  % len(cr))
        if fin:
            fin_sorted = sorted(fin)
            print("  FINISHED durations: min=%.1f  median=%.1f  max=%.1f"
                  % (fin_sorted[0], fin_sorted[len(fin_sorted) // 2], fin_sorted[-1]))
            if ceiling:
                print("  slowest finisher used %.1f%% of the %.0fs ceiling (%.0fs headroom)"
                      % (100.0 * fin_sorted[-1] / ceiling, ceiling, ceiling - fin_sorted[-1]))
        if to:
            print("  TIMED OUT at: %s   <-- CENSORED at the ceiling; bounds nothing from above"
                  % ", ".join("%.1f" % s for s in sorted(to)))

    print()
    print("=" * 100)
    print("POOLED ACROSS EVERY TREE -- finished legs only")
    if not all_finished:
        print("  none")
        return 0
    durations = sorted(s for _, s in all_finished)
    n = len(durations)
    print("  n=%d  min=%.1f  median=%.1f  p90=%.1f  max=%.1f"
          % (n, durations[0], durations[n // 2], durations[int(n * 0.9)], durations[-1]))
    slowest = max(all_finished, key=lambda t: t[1])
    print("  slowest finisher anywhere: %.1fs in %s" % (slowest[1], slowest[0]))
    print()
    print("  Headroom that each candidate ceiling would have left the slowest OBSERVED finisher:")
    for cand in (1800.0, 2400.0, 3000.0, 3600.0):
        used = 100.0 * slowest[1] / cand
        print("    %6.0fs ceiling -> slowest finisher uses %5.1f%%  (%.0fs headroom, "
              "child deadline %.0fs after the 120s reserve)" % (cand, used, cand - slowest[1], cand - 120.0))
    print()
    print("  NOTE: every timed-out leg is CENSORED -- it proves only that the work needed")
    print("  MORE than the ceiling, never how much more. The true requirement is therefore")
    print("  at least the pooled max above, and possibly much larger. A ceiling chosen to")
    print("  just clear the observed max is a ceiling chosen to fail on the next slow leg.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
