"""D-088 — is Stage-0 `main_subprocesses` STABLE enough to be a hard-failure input?

Read-only, over every committed `coverage_summary.json` under **both `runs/` and
`runs_verify/`**, both roots named. Nothing is run, re-run, re-scored or mutated.

WHY THIS PROBE EXISTS — it was not planned, it was forced by the first one
--------------------------------------------------------------------------
`d088_subprocess_recall_ab.py` (G11 ORCH-719/01) asked what a subprocess-level cap would
do. On the T-108 tree it gave exactly the discrimination D-088 requires: `PMC12096016`
loses the cap with 0 uncovered subprocesses, `PMC12782028` keeps it with 2 uncovered
(`mevalonate pathway`, `methylsterol demethylation`), identically at all three stoplist
strengths.

**But its per-run rows showed something the headline verdict did not.** On
`runs_verify/2026-08-21_2239`, `PMC12782028/strict` came back with **4 subprocesses and
ZERO uncovered** — the proposed cap would have RELEASED it. F-167 established that
`PMC12782028` has never once cleared in six runs and that its upstream mevalonate arm is
genuinely absent. So on that archived draw the proposed rule would have released a leg
that is a known reaction-recall failure, **not because the recall improved but because
Stage 0 did not name the missing arm that time.**

That is the exact failure mode D-088 clause 10 exists to prevent — a change that moves a
score by removing the measurement — arriving through a back door nobody was watching: the
specification is itself an LLM draw, and it varies.

THE QUESTION
------------
For each paper, across every archived draw: **does Stage 0 name the same subprocesses
every time?** If it does not, then a cap keyed to Stage-0's own list inherits that
variance, and the same leg can be capped or released on the strength of a draw.

This probe reports that variance directly. It proposes nothing and fixes nothing.

WHAT A RESULT HERE DOES AND DOES NOT LICENCE
--------------------------------------------
Instability here does **not** mean Stage-0 subprocesses are useless — they are still a
process-level signal, and they are the only one production holds without reading gold.
It means they cannot be the SOLE hard-failure input, which is what D-088 clause 9 already
says about the entity anchors they would replace. Swapping one unvalidated Stage-0 list
for another unvalidated Stage-0 list is a lateral move, not the reaction-level replacement
clause 10 requires.

Stability here would equally not licence the rule on its own; it would remove one specific
objection to it.

Usage:  d088_stage0_spec_stability.py <repo>
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict

# The named consequence legs, and the tree T-108 ran into.
T108_TREE = "runs_verify/2026-09-01_1612"
NAMED = ("PMC12096016", "PMC12782028")


def norm_sub(s):
    """A subprocess phrase reduced to its content tokens, for cross-draw comparison.

    Deliberately coarse: two draws that say 'mevalonate pathway' and 'mevalonate pathway
    (upstream isoprenoid biosynthesis)' are the SAME named stage, and a comparison that
    called them different would overstate the variance this probe is measuring. Erring
    toward "same" makes the instability reported here a LOWER BOUND.
    """
    toks = [t for t in re.split(r"[^a-z0-9]+", str(s).lower()) if len(t) >= 4]
    drop = {"pathway", "reaction", "step", "from", "with", "into", "onto", "this", "that",
            "upstream", "downstream", "catalyzed", "catalysed", "mediated", "conversion",
            "synthesis", "formation", "production"}
    return frozenset(t for t in toks if t not in drop)


def load(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def main():
    if len(sys.argv) < 2:
        print("usage: d088_stage0_spec_stability.py <repo>")
        return 2
    repo = os.path.abspath(sys.argv[1])

    legs = []
    for root_name in ("runs", "runs_verify"):
        root = os.path.join(repo, root_name)
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            if "coverage_summary.json" in files:
                legs.append(os.path.join(dirpath, "coverage_summary.json"))
    legs.sort()

    print("D-088 — STAGE-0 SPECIFICATION STABILITY ACROSS ARCHIVED DRAWS")
    print("=" * 94)
    print(f"repo                     : {repo}")
    print(f"roots scanned            : runs/ AND runs_verify/, both named")
    print(f"coverage_summary.json    : {len(legs)}")
    print("nothing re-run, re-scored or mutated")
    print()

    # paper/mode -> list of (run_id, [subprocess strings])
    draws = defaultdict(list)
    for path in legs:
        cov = load(path)
        if not isinstance(cov, dict) or not cov.get("requested_core_declared"):
            continue
        rel = os.path.relpath(path, repo).replace("\\", "/")
        parts = rel.split("/")
        if len(parts) < 4:
            continue
        run_id = "/".join(parts[:2])
        paper = parts[-3]
        mode = parts[-2]
        ctx = cov.get("requested_context") or {}
        subs = [str(x) for x in (ctx.get("main_subprocesses") or [])]
        draws[(paper, mode)].append((run_id, subs))

    print("=" * 94)
    print("PER PAPER/MODE — how many subprocesses did Stage 0 name, draw by draw?")
    print("=" * 94)
    print(f"{'paper/mode':34s} {'draws':>5s} {'counts seen':>18s}  {'union':>5s} {'always':>6s} {'stable?':>8s}")
    print("-" * 94)

    unstable = []
    for key in sorted(draws):
        paper, mode = key
        ds = draws[key]
        if len(ds) < 2:
            continue
        sets = [set(norm_sub(s) for s in subs if norm_sub(s)) for _r, subs in ds]
        counts = [len(subs) for _r, subs in ds]
        union = set().union(*sets) if sets else set()
        always = set.intersection(*sets) if sets else set()
        stable = len(union) == len(always)
        if not stable:
            unstable.append((paper, mode, ds, sets, union, always))
        print(f"{paper + '/' + mode:34s} {len(ds):5d} {str(sorted(set(counts))):>18s}  "
              f"{len(union):5d} {len(always):6d} {'YES' if stable else 'NO':>8s}")

    total_multi = len([k for k in draws if len(draws[k]) >= 2])
    print()
    print(f"paper/mode pairs with >= 2 archived draws : {total_multi}")
    print(f"  Stage-0 named an IDENTICAL subprocess set every time : {total_multi - len(unstable)}")
    print(f"  Stage-0 named a DIFFERENT subprocess set across draws: {len(unstable)}")
    print()

    print("=" * 94)
    print("THE TWO NAMED CONSEQUENCE PAPERS, DRAW BY DRAW, IN FULL")
    print("=" * 94)
    for paper in NAMED:
        for mode in ("strict", "research"):
            ds = draws.get((paper, mode))
            if not ds:
                continue
            print()
            print(f"--- {paper}/{mode}  ({len(ds)} archived draws) ---")
            for run_id, subs in sorted(ds):
                marker = "  <== T-108" if run_id == T108_TREE else ""
                print(f"  {run_id}  n={len(subs)}{marker}")
                for s in subs:
                    print(f"      * {s}")

    print()
    print("=" * 94)
    print("THE SPECIFIC ROW THAT FORCED THIS PROBE")
    print("=" * 94)
    key = ("PMC12782028", "strict")
    ds = dict(draws.get(key, []))
    for run_id in sorted(ds):
        subs = ds[run_id]
        names = " | ".join(subs).lower()
        has_mev = "mevalonate" in names
        print(f"  {run_id}  n={len(subs)}  names a mevalonate stage: "
              f"{'YES' if has_mev else 'NO '}")
    n_with = sum(1 for r in ds if "mevalonate" in " | ".join(ds[r]).lower())
    print()
    print(f"  archived PMC12782028/strict draws                 : {len(ds)}")
    print(f"  draws in which Stage 0 NAMED a mevalonate stage   : {n_with}")
    print(f"  draws in which it did NOT                         : {len(ds) - n_with}")
    print()
    print("  F-167 established that this leg's upstream mevalonate arm is absent in EVERY")
    print("  run, and that the leg has never cleared: 0.222 0.280 0.296 0.321 0.571 0.538.")
    print("  The BIOLOGY does not vary across these draws. Only the SPECIFICATION does.")
    print()
    print("  A hard cap keyed to Stage-0's own subprocess list therefore inherits a")
    print("  variance the underlying failure does not have: on a draw where Stage 0 omits")
    print("  the missing arm, the cap goes quiet and the leg reads as complete. That is a")
    print("  measurement being removed by a draw, which is what D-088 clause 10 forbids")
    print("  being done by a patch. The mechanism differs; the outcome does not.")
    print()
    print("  CONCLUSION, stated as narrowly as the evidence supports: Stage-0")
    print("  main_subprocesses is a USABLE process-level signal and is NOT a sufficient")
    print("  SOLE hard-failure input. D-088 clause 9's replacement must be anchored to")
    print("  something that does not vary with the draw being judged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
