"""F-167 — is the anchor/subprocess disagreement CHRONIC, or is it a T-108 artifact?

Read-only census over **every committed `coverage_summary.json`** under both `runs/`
and `runs_verify/` — named explicitly, because both roots are live and "the pinned
run" is ambiguous. No leg is run, re-run, re-scored or mutated. A coverage verdict
is a fact about what a leg recorded, never a verdict about what it produced.

For each leg that declared a requested core, it asks the F-167 question:

  of the anchors that matched no admitted process, how many are named in
  Stage-0's OWN `main_subprocesses` list?

F-167 measured 0 of 10 on T-108's two strict-denominator legs. This asks whether
that ratio is the historical norm.

It also separates the two failure shapes F-167 distinguishes:

  * an unmatched anchor that IS in the payload  -> extracted but not wired into
    any admitted process;
  * an unmatched anchor that is NOT in the payload -> never extracted at all.

Usage:  f167_history_census.py <repo>
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def main() -> int:
    repo = os.path.abspath(sys.argv[1])
    roots = [os.path.join(repo, "runs"), os.path.join(repo, "runs_verify")]

    legs = []
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            if "coverage_summary.json" in files:
                legs.append(os.path.join(dirpath, "coverage_summary.json"))
    legs.sort()
    print(f"committed coverage_summary.json found : {len(legs)}")
    print(f"roots scanned                         : runs/ and runs_verify/, both named")
    print()

    tot = Counter()
    per_run = {}
    rows = []

    for path in legs:
        try:
            cov = json.load(open(path, encoding="utf-8"))
        except Exception:
            tot["unreadable"] += 1
            continue
        if not cov.get("requested_core_declared"):
            tot["no_declared_core"] += 1
            continue
        tot["legs_with_declared_core"] += 1

        unmatched = list(cov.get("unmatched_terms") or [])
        ctx = cov.get("requested_context") or {}
        subs_list = ctx.get("main_subprocesses") or []
        subs = " | ".join(str(x) for x in subs_list).lower()
        has_ctx = bool(subs_list)

        rel = os.path.relpath(path, repo).replace("\\", "/")
        parts = rel.split("/")
        run_id = "/".join(parts[:2])
        leg_id = "/".join(parts[-3:-1]) if len(parts) >= 3 else rel

        if not unmatched:
            tot["legs_fully_matched"] += 1
            per_run.setdefault(run_id, Counter())["fully_matched"] += 1
            continue

        tot["legs_with_unmatched"] += 1
        per_run.setdefault(run_id, Counter())["with_unmatched"] += 1

        if not has_ctx:
            tot["legs_unmatched_but_no_context"] += 1
            continue
        tot["legs_testable"] += 1

        # payload names, for the extracted-vs-wired split
        leg_dir = os.path.dirname(path)
        names_norm = set()
        fm = os.path.join(leg_dir, "final_mapped.json")
        if os.path.isfile(fm):
            try:
                pay = json.load(open(fm, encoding="utf-8"))
                for k, v in (pay.get("entities") or {}).items():
                    if isinstance(v, list):
                        for e in v:
                            if isinstance(e, dict) and e.get("name"):
                                names_norm.add(norm(e["name"]))
            except Exception:
                pass

        in_sub = in_pay = 0
        for a in unmatched:
            tot["anchors_unmatched"] += 1
            if a.lower() in subs:
                in_sub += 1
                tot["anchors_in_subprocesses"] += 1
            if norm(a) in names_norm or any(norm(a) and norm(a) in n for n in names_norm):
                in_pay += 1
                tot["anchors_in_payload"] += 1
        rows.append((run_id, leg_id, len(unmatched), in_sub, in_pay,
                     cov.get("coverage_ratio"), unmatched))

    print("=" * 92)
    print("PER-LEG — every leg with at least one unmatched anchor and a Stage-0 context")
    print("=" * 92)
    print(f"  {'run':26s} {'leg':26s} {'unm':>4s} {'inSub':>6s} {'inPay':>6s} {'cov':>7s}")
    print("  " + "-" * 88)
    for run_id, leg_id, n, s, p, cov, _u in rows:
        flag = "" if s == 0 else "   <-- some anchor IS a named subprocess"
        c = f"{cov:.3f}" if isinstance(cov, (int, float)) else "-"
        print(f"  {run_id:26s} {leg_id:26s} {n:>4d} {s:>6d} {p:>6d} {c:>7s}{flag}")

    print()
    print("=" * 92)
    print("VERDICT")
    print("=" * 92)
    for k in ("legs_with_declared_core", "legs_fully_matched", "legs_with_unmatched",
              "legs_unmatched_but_no_context", "legs_testable", "no_declared_core",
              "unreadable"):
        print(f"  {k:34s} = {tot[k]}")
    a = tot["anchors_unmatched"]
    print()
    print(f"  unmatched anchors examined         : {a}")
    print(f"  named in Stage-0's OWN subprocesses : {tot['anchors_in_subprocesses']}"
          f"  ({(100.0*tot['anchors_in_subprocesses']/a):.1f}%)" if a else "")
    print(f"  present in the extracted payload    : {tot['anchors_in_payload']}"
          f"  ({(100.0*tot['anchors_in_payload']/a):.1f}%)" if a else "")
    print()
    if a and tot["anchors_in_subprocesses"] == 0:
        print("  CHRONIC AND ABSOLUTE: across every committed leg that declared a requested")
        print("  core and recorded a Stage-0 context, NOT ONE unmatched anchor is named in")
        print("  Stage-0's own main_subprocesses. F-167 is not a T-108 artifact.")
    elif a:
        print("  NOT absolute -- some unmatched anchors ARE named subprocesses. Those legs are")
        print("  the interesting ones: there the pipeline failed to match something Stage 0")
        print("  itself called a step, which is a DIFFERENT defect from the anchor-derivation")
        print("  one. They are flagged per-leg above.")
    print()
    print(f"  F167_HISTORY: legs_testable={tot['legs_testable']} anchors={a} "
          f"in_subprocesses={tot['anchors_in_subprocesses']} in_payload={tot['anchors_in_payload']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
