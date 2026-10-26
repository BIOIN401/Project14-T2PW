"""ORCH-773 phase 2: per-leg PWML coverage, base vs tip, from two ORCH-768 corpus records.

Uses the harness's own ``leg_percent`` and ``coverage_summary`` (imported, not rebuilt). Prints every leg,
then the legs that write a PWML only at the tip, then the aggregates on 91 and 81 legs.

    python coverage_table.py <base corpus.json> <tip corpus.json>
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1] / "orch768_replay_corpus.py"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    spec = importlib.util.spec_from_file_location("orch768_replay_corpus_773c", HARNESS)
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    base, tip = (json.loads(Path(p).read_text(encoding="utf-8")) for p in sys.argv[1:3])
    b_rows, t_rows = ({r["id"]: r for r in doc["legs"]} for doc in (base, tip))

    def cov(row):
        c = (row["replay"] or {}).get("pwml_coverage") or {}
        return f"{c.get('drawn')}/{c.get('total')} (by design {c.get('by_design_withheld')})" if c else "n/a"

    def pct(row, ex=False):
        v = H.leg_percent(row, ex)
        return "  n/a" if v is None else f"{v:5.1f}"

    print(f"{'leg':58s} {'base class':12s} {'tip class':12s} {'base %':>6s} {'tip %':>6s}  base cov -> tip cov")
    for leg_id, t in t_rows.items():
        b = b_rows[leg_id]
        print(f"{leg_id:58s} {b['class'][:12]:12s} {t['class'][:12]:12s} {pct(b):>6s} {pct(t):>6s}  {cov(b)} -> {cov(t)}")
    fixed = [i for i, t in t_rows.items()
             if (t["replay"] or {}).get("pwml_written") and not (b_rows[i]["replay"] or {}).get("pwml_written")]
    print(f"\n== legs that write a PWML only at the tip: {len(fixed)} ==")
    tip_values = []
    for i in fixed:
        t = t_rows[i]
        tip_values.append(H.leg_percent(t))
        print(f"  {i:58s} {pct(b_rows[i])} -> {pct(t)}%  (ex-by-design {pct(t, True)}%)  {cov(t)}")
    known = [v for v in tip_values if v is not None]
    if known:
        print(f"  these {len(known)}: mean {statistics.fmean(known):.1f}%  median {statistics.median(known):.1f}%  "
              f">= 75%: {sum(v >= 75 for v in known)}/{len(known)}  drawing no process: {sum(v == 0 for v in known)}")
    for name, doc in (("base", base), ("tip", tip)):
        print(f"\n== {name}: coverage_summary (harness) ==")
        summary = H.coverage_summary(doc["legs"])
        for key in ("strict_legs_with_an_outcome.headline", "strict_legs_with_an_outcome.excluding_by_design",
                    "replayed.headline", "replayed.excluding_by_design"):
            s = summary[key]
            print(f"  {key:48s} legs {s['legs']:3d}  mean {s['mean']:6.2f}%  median {s['median']:6.2f}%  "
                  f">= 75%: {s['share_at_or_above_target']}")
        print(f"  by known issue: {summary['omitted_processes_by_known_issue']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
