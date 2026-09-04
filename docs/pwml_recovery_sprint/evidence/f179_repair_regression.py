"""F-179 repair regression — census replay (§11) and false-positive protection (§12).

EVALUATION-ONLY and READ-ONLY. Re-runs no pipeline leg and takes no fresh LLM draw. It
replays archived canonical payloads through the PRODUCTION rule
(:mod:`t2pw.pipeline.reaction_support`) — the same code the pre-export contract calls —
so the numbers are the repaired gate's own behaviour rather than a reimplementation of
it.

TWO QUESTIONS, KEPT APART:

  §11 CENSUS REPLAY. Of the 28 rows the discovery census flagged (terminal product AND
      no paper/RAG attribution), how many sit in a leg the repaired gate now refuses,
      how many are legitimately allowed because a defensible core exists, and how many
      are indeterminate? **Not all 28 are assumed fabricated** — the census was a
      detection heuristic and the production rule is not.

  §12 FALSE-POSITIVE PROTECTION. Over every archived canonical leg: how many are
      allowed, blocked, indeterminate, and which previously-exported legs would no
      longer export. The two gold ``strict_exportable`` cases are reported by name.

POPULATIONS. ``committed`` and ``preserved_untracked`` are counted apart and never
summed (F-178).

Usage:
  python f179_repair_regression.py <repo-root> [--census <census.json>] [--json OUT]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

COMMITTED = "committed"
PRESERVED = "preserved_untracked"
POPULATIONS = (COMMITTED, PRESERVED)

#: The two gold cases whose ``expected_export`` is ``strict_exportable``. Named ONLY for
#: reporting, never consulted by the production rule -- §12 requires they be inspected
#: explicitly, and this is that inspection.
STRICT_EXPORTABLE_PAPERS = ("PMC12096016", "PMC12782028")


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("repo_root")
    ap.add_argument("--census", default="docs/pwml_recovery_sprint/evidence/rd093_shortcut_census.json")
    ap.add_argument("--json", dest="json_path", default=None)
    args = ap.parse_args(argv)

    root = Path(args.repo_root).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))
    from t2pw.pipeline.reaction_support import (       # noqa: E402
        evaluate_reaction_support, reaction_support_class,
    )

    tracked = set(subprocess.run(["git", "ls-files"], capture_output=True, text=True,
                                 encoding="utf-8", errors="replace").stdout.split())
    payloads = sorted({p.replace("\\", "/") for p in
                       glob.glob("runs/**/final_mapped.json", recursive=True) +
                       glob.glob("runs_verify/**/final_mapped.json", recursive=True)}
                      | {f for f in tracked if f.endswith("final_mapped.json")})

    leg_verdict: Dict[str, Dict[str, Any]] = {}
    counts = {p: collections.Counter() for p in POPULATIONS}
    rows_counts = {p: collections.Counter() for p in POPULATIONS}
    blocked_exports: List[Dict[str, Any]] = []
    strict_rows: List[Dict[str, Any]] = []

    for rel in payloads:
        if not os.path.isfile(rel):
            continue
        pop = COMMITTED if rel in tracked else PRESERVED
        leg = os.path.dirname(rel)
        try:
            doc = json.load(open(rel, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        report = evaluate_reaction_support(doc)
        pwml = sorted(os.path.basename(x) for x in glob.glob(leg + "/*.pwml"))
        leg_verdict[leg] = {"verdict": report["verdict"], "population": pop,
                            "pwml_files": pwml, "report": report}
        counts[pop][report["verdict"]] += 1
        rows_counts[pop]["reactions"] += report["reactions"]
        rows_counts[pop]["supported"] += report["supported"]
        rows_counts[pop]["unattributed"] += report["unattributed"]
        if report["verdict"] == "no_defensible_core" and pwml:
            blocked_exports.append({"leg": leg, "population": pop, "pwml": pwml,
                                    "reactions": report["reactions"]})
        m = re.search(r"/papers/(PMC\d+)", rel)
        if m and m.group(1) in STRICT_EXPORTABLE_PAPERS:
            strict_rows.append({"leg": leg, "population": pop,
                                "verdict": report["verdict"],
                                "supported": report["supported"],
                                "reactions": report["reactions"],
                                "pwml": pwml})

    # ---- §11 census replay ------------------------------------------------
    census_summary: Dict[str, Any] = {"available": False}
    cpath = root / args.census
    if cpath.is_file():
        census = json.loads(cpath.read_text(encoding="utf-8"))
        target = [h for h in census.get("hits", [])
                  if "precursor_terminal_shortcut" in h.get("criteria", [])
                  and "no_paper_and_no_rag" in h.get("criteria", [])]
        per_row = collections.Counter()
        row_support = collections.Counter()
        for h in target:
            v = leg_verdict.get(h["leg_dir"], {}).get("verdict", "leg_not_evaluated")
            per_row[v] += 1
            # Row-level support under the production rule, for the "not all 28 are
            # fabricated" check: a flagged row inside an otherwise supported leg may
            # itself carry support.
            try:
                doc = json.load(open(h["leg_dir"] + "/final_mapped.json", encoding="utf-8"))
                rows = (doc.get("processes") or {}).get("reactions") or []
                row = rows[h["row_index"]] if 0 <= h["row_index"] < len(rows) else None
                row_support[str(reaction_support_class(row) if row else "row_missing")] += 1
            except (OSError, ValueError, KeyError):
                row_support["unreadable"] += 1
        census_summary = {
            "available": True,
            "intersection_rows": len(target),
            "distinct_papers": len({h.get("paper") for h in target}),
            "distinct_runs": len({h.get("run") for h in target}),
            "leg_verdict_for_row": dict(per_row),
            "row_level_support_class": dict(row_support),
            "rows_in_legs_that_had_exported": sum(
                1 for h in target if leg_verdict.get(h["leg_dir"], {}).get("pwml_files")),
            "rows_in_legs_that_had_exported_and_are_now_blocked": sum(
                1 for h in target
                if leg_verdict.get(h["leg_dir"], {}).get("pwml_files")
                and leg_verdict.get(h["leg_dir"], {}).get("verdict") == "no_defensible_core"),
        }

    report = {
        "instrument": "f179_repair_regression",
        "evaluation_only": True, "reran_nothing": True,
        "rule": "t2pw.pipeline.reaction_support (the production rule itself)",
        "legs_by_population": {p: dict(counts[p]) for p in POPULATIONS},
        "reaction_rows_by_population": {p: dict(rows_counts[p]) for p in POPULATIONS},
        "previously_exported_now_blocked": blocked_exports,
        "strict_exportable_papers": strict_rows,
        "census_replay": census_summary,
    }

    print("F-179 REPAIR REGRESSION -- evaluation-only, re-ran nothing")
    print("populations counted APART and never summed\n")
    for p in POPULATIONS:
        c = counts[p]
        tot = sum(c.values())
        print(f"== {p}: {tot} canonical legs ==")
        for v in ("supported", "indeterminate", "no_defensible_core"):
            print(f"   {v:22s} {c[v]:4d}")
        r = rows_counts[p]
        print(f"   reaction rows: {r['reactions']}  supported {r['supported']}  "
              f"unattributed {r['unattributed']}")
        print()

    print(f"previously EXPORTED legs now blocked: {len(blocked_exports)}")
    for b in blocked_exports:
        print(f"   {b['leg']:62s} rx={b['reactions']:2d} {b['pwml']}")
    print()
    print("gold strict_exportable papers -- MUST remain exportable:")
    bad = [s for s in strict_rows if s["verdict"] == "no_defensible_core"]
    for s in strict_rows:
        flag = "  <-- BLOCKED" if s["verdict"] == "no_defensible_core" else ""
        print(f"   {s['leg']:62s} {s['verdict']:18s} sup={s['supported']}/{s['reactions']}{flag}")
    print(f"   => {len(bad)} of {len(strict_rows)} strict-exportable legs blocked")
    print()
    if census_summary.get("available"):
        cs = census_summary
        print(f"census replay: {cs['intersection_rows']} intersection rows "
              f"({cs['distinct_papers']} papers, {cs['distinct_runs']} runs)")
        print(f"   leg verdict for those rows : {cs['leg_verdict_for_row']}")
        print(f"   row-level support class    : {cs['row_level_support_class']}")
        print(f"   in legs that had exported  : {cs['rows_in_legs_that_had_exported']}")
        print(f"   ...now blocked             : {cs['rows_in_legs_that_had_exported_and_are_now_blocked']}")

    if args.json_path:
        out = Path(args.json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
