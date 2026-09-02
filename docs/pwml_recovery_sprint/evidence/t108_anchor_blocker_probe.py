"""T-108 — localise what actually blocks Priorities 4 and 5. Read-only, offline.

T-108 established that Priority 5's `0/2` is now entirely a coverage matter: both
`strict_exportable` legs executed, cleared the strict technical gates, PASSED
semantic evaluation, produced valid PWML, and were capped at `review_required`
by unmatched requested-core anchors.

This probe asks the next question: **where do the unmatched anchors come from,
and could the pipeline have matched them at all?** It reads only committed run
artifacts and the gold set. It changes nothing and judges nothing biological.

Three checks per strict_exportable leg:

  A. Is each unmatched anchor present in the paper's SOURCE TEXT?
     -> separates "the paper never said it" from "we did not extract it".
  B. Is each unmatched anchor present in the leg's EXTRACTED payload?
     -> separates an extraction gap from a wiring/matching gap.
  C. Is each unmatched anchor named in STAGE 0's OWN `main_subprocesses`?
     -> this is the load-bearing one. The anchors are Stage-0's
        `key_compounds` + `key_proteins`; `main_subprocesses` is Stage-0's own
        account of what the pathway DOES. An anchor absent from that list is one
        Stage 0 itself did not treat as a pathway step.

Usage:  t108_anchor_blocker_probe.py <repo> <run-dir-relative>
"""

from __future__ import annotations

import json
import os
import re
import sys


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def main() -> int:
    repo = os.path.abspath(sys.argv[1])
    rel = sys.argv[2]
    run = os.path.join(repo, rel)
    sys.path.insert(0, os.path.join(repo, "src"))
    from t2pw.bench.goldset import load_gold_set, pinned_gold_set_path  # noqa: E402

    gold = load_gold_set(pinned_gold_set_path())
    targets = [c for c in gold.cases
               if str(getattr(c, "expected_export", "")) == "strict_exportable"]
    print(f"strict_exportable papers (Priority 5's whole denominator): "
          f"{[getattr(c,'paper_id','?') for c in targets]}")

    totals = {"in_source": 0, "in_payload": 0, "in_subprocesses": 0, "anchors": 0}

    for case in targets:
        pid = getattr(case, "paper_id", "?")
        leg = os.path.join(run, "papers", pid, "strict")
        cov_p = os.path.join(leg, "coverage_summary.json")
        if not os.path.isfile(cov_p):
            print(f"\n{pid}/strict: no coverage_summary.json — skipped")
            continue
        cov = json.load(open(cov_p, encoding="utf-8"))
        ctx = cov.get("requested_context") or {}
        unmatched = list(cov.get("unmatched_terms") or [])
        subs = " | ".join(ctx.get("main_subprocesses") or []).lower()

        src_p = os.path.join(run, "papers", pid, "01_source_text.txt")
        source = open(src_p, encoding="utf-8", errors="replace").read().lower() \
            if os.path.isfile(src_p) else ""

        names = []
        fm = os.path.join(leg, "final_mapped.json")
        if os.path.isfile(fm):
            pay = json.load(open(fm, encoding="utf-8"))
            for k, v in (pay.get("entities") or {}).items():
                if isinstance(v, list):
                    for e in v:
                        if isinstance(e, dict) and e.get("name"):
                            names.append(str(e["name"]))
        names_norm = {norm(n) for n in names}

        print()
        print("=" * 78)
        print(f"{pid}/strict")
        print("=" * 78)
        print(f"  coverage_ratio          : {cov.get('coverage_ratio')}   "
              f"min_core_coverage: {(cov.get('thresholds') or {}).get('min_core_coverage')}")
        print(f"  core_accepted_processes : {cov.get('core_accepted_processes')}")
        print(f"  requested_core_source   : {cov.get('requested_core_source')}")
        print(f"  Stage-0 key_compounds   : {ctx.get('key_compounds')}")
        print(f"  Stage-0 key_proteins    : {ctx.get('key_proteins')}")
        print(f"  Stage-0 main_subprocesses ({len(ctx.get('main_subprocesses') or [])}):")
        for s in ctx.get("main_subprocesses") or []:
            print(f"      - {s}")
        print()
        print(f"  UNMATCHED ANCHORS ({len(unmatched)}):")
        print(f"    {'anchor':16s} {'in source':>10s} {'in payload':>11s} {'in Stage-0 subprocesses':>24s}")
        print("    " + "-" * 66)
        for a in unmatched:
            in_src = a.lower() in source
            in_pay = norm(a) in names_norm or any(norm(a) in n for n in names_norm)
            in_sub = a.lower() in subs
            totals["anchors"] += 1
            totals["in_source"] += int(in_src)
            totals["in_payload"] += int(in_pay)
            totals["in_subprocesses"] += int(in_sub)
            print(f"    {a:16s} {str(in_src):>10s} {str(in_pay):>11s} {str(in_sub):>24s}")

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    t = totals
    print(f"  unmatched anchors examined            : {t['anchors']}")
    print(f"  present in the paper's source text    : {t['in_source']}")
    print(f"  present in the extracted payload      : {t['in_payload']}")
    print(f"  named in Stage-0's OWN subprocesses   : {t['in_subprocesses']}")
    print()
    if t["anchors"] and t["in_subprocesses"] == 0:
        print("  EVERY unmatched anchor is absent from Stage-0's own main_subprocesses.")
        print("  The anchors are Stage-0's key_compounds + key_proteins; the pipeline is")
        print("  graded on matching each of them to an ADMITTED PROCESS's core_terms")
        print("  (strict_quarantine.py:989-996), and ONE unmatched anchor is enough to")
        print("  cap release_ready at review_required (the INCOMPLETE-CORE CAP, F-094,")
        print("  release_status.py:921-930).")
        print()
        print("  So Stage 0 nominates anchors it does not itself list as pathway steps,")
        print("  and the cap then requires every one of them. This is a LOCALISATION,")
        print("  not a classification: whether a cofactor or a regulator BELONGS in a")
        print("  requested core is a biological judgement and belongs to the auditor.")
    print()
    print(f"  T108_ANCHOR_PROBE: anchors={t['anchors']} in_source={t['in_source']} "
          f"in_payload={t['in_payload']} in_subprocesses={t['in_subprocesses']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
