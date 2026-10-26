"""ORCH-772b proof: base and tip corpus records differ ONLY in the reason / KI text of processes lost with a no-PWML leg.

    python orch772b_compare_coverage.py <base.json> <tip.json>

Per leg it checks: the replay record apart from ``pwml_coverage`` (every verdict, PWML sha256 and output-file digest)
is identical; every coverage number (drawn, total, per kind, KI-008 count, merged duplicates, not counted, not
recorded, pwml_written) is identical; every omission row of a leg that wrote a PWML, and every pre-freeze row of a
leg that did not, is identical; the rows of processes lost with a no-PWML leg keep their count and differ only in KI
id and reason text. It prints the moved KI labels and exits 1 on any other difference. Stdlib only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

NUMBERS = ("drawn", "total", "by_kind", "by_design_withheld", "merged_duplicates", "not_counted", "not_recorded",
           "pwml_written")


def main() -> int:
    base, tip = (json.loads(Path(p).read_text(encoding="utf-8")) for p in sys.argv[1:3])
    b, t = ({r["id"]: r for r in doc["legs"]} for doc in (base, tip))
    bad, moved = [], {}
    if set(b) != set(t):
        bad.append(f"leg sets differ: {sorted(set(b) ^ set(t))}")
    for leg in sorted(set(b) & set(t)):
        rb, rt = b[leg]["replay"] or {}, t[leg]["replay"] or {}
        strip = lambda r: json.dumps({k: v for k, v in r.items() if k != "pwml_coverage"}, sort_keys=True)  # noqa: E731
        if strip(rb) != strip(rt) or {k: v for k, v in b[leg].items() if k != "replay"} != {
                k: v for k, v in t[leg].items() if k != "replay"}:
            bad.append(f"{leg}: replay record or row differs apart from pwml_coverage")
        cb, ct = rb.get("pwml_coverage"), rt.get("pwml_coverage")
        if cb is None or ct is None:
            if cb is not ct:
                bad.append(f"{leg}: coverage present on one side only")
            continue
        for key in NUMBERS:
            if cb[key] != ct[key]:
                bad.append(f"{leg}: {key} {cb[key]} -> {ct[key]}")
        keep = lambda rows: sorted(r for r in rows if r[2] != "no_pwml")  # noqa: E731
        lost = lambda rows: sorted(r[0] for r in rows if r[2] == "no_pwml")  # noqa: E731
        if keep(cb["omitted_by_reason"]) != keep(ct["omitted_by_reason"]):
            bad.append(f"{leg}: an omission row other than a no-PWML row changed")
        if lost(cb["omitted_by_reason"]) != lost(ct["omitted_by_reason"]):
            bad.append(f"{leg}: the count of processes lost with the leg changed")
        for rows, side in ((cb["omitted_by_reason"], "base"), (ct["omitted_by_reason"], "tip")):
            for n, ki, group, _reason in rows:
                if group == "no_pwml":
                    moved.setdefault(leg, {}).setdefault(side, []).append(f"{n}x {ki}")
    for leg, sides in sorted(moved.items()):
        print(f"{leg:58s} base {', '.join(sides.get('base', []))}  ->  tip {', '.join(sides.get('tip', []))}")
    print(f"legs compared: {len(set(b) & set(t))}; no-PWML legs whose lost processes were relabelled: {len(moved)}")
    print("other differences:", bad or "none")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
