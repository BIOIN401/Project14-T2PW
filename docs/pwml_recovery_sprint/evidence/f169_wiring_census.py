"""F-169 — how many unmatched anchors are ALREADY WIRED into an admitted core process?

Read-only census over every committed `coverage_summary.json` under **both `runs/` and
`runs_verify/`**, both roots named. Nothing is run, re-run, re-scored or mutated.

WHY THIS EXISTS
---------------
The `pwml-bio-auditor`, adjudicating F-169, established two things this probe answers.

**First, F-167's `90/374` "present in the extracted payload" is a LOWER BOUND, biased down
by exactly the defect under adjudication.** `f167_history_census.py:91-113` matches each
anchor against `final_mapped.json` ENTITY NAMES by one-directional containment on a
punctuation-stripped string. On `PMC12096016/strict` it reports `in_payload = 1` (EntD) and
**does not count ATP**, because `"atp"` is not a substring of `"adenosinetriphosphate"` —
the same synonym blindness that produced the finding.

**Second, and this is the measurement that does not exist:** the census measures
ENTITY-LIST PRESENCE, not WIRING. D-088 clause 6 says an extracted entity does not satisfy
coverage merely by existing in the payload — it must be **connected to an admitted
process**. So within the 314/374 (84%) population that D-088 clauses 1-2 would reclassify
to warnings, nothing tells us how many are genuinely unwired ancillary participants
(correctly downgraded) and how many are ATP-shaped: **participants that ARE wired into an
admitted core process and are being falsely reported as missing.**

Until that split is measured, "one row on one leg" is the only defensible statement about
the scale of what a cofactor downgrade would hide.

THE EQUIVALENCE SOURCE, AND WHY IT IS NOT TUNED
-----------------------------------------------
The synonym question is where a probe like this can quietly become an argument. So this one
authors **no synonyms at all**. It imports two tables that already exist in production, for
their own reasons, committed long before this finding:

  * `pwml.compound_templates.COMPOUND_NAME_TO_TEMPLATE` — 103 names over 36 multi-name
    groups. Two names sharing a template id are **the codebase's own assertion that they
    are one compound**. `"adenosine triphosphate"` and `"atp"` are both 42.
  * `pipeline.process_normalizer.BIOCHEMICAL_ALIAS_MAP` — the canonicalisation map, folded
    in both directions.

**Nothing is added to either.** If a synonym is missing from both, this probe does not see
it, and the number it reports is therefore itself a **lower bound** — stated here rather
than discovered later. That is the honest direction for this measurement to err: it can
understate how much a downgrade would hide, never overstate it.

WHAT A RESULT DOES AND DOES NOT LICENCE
---------------------------------------
A high count would NOT licence "fix the matcher". The auditor was explicit that the
matcher's substring rule and its match-against-ADMITTED-PROCESSES design are both correct
and must not be touched — D-088 clause 6, and F-167's trap 3, which warns that matching
anchors against the entity list destroys the cap. The defect is upstream, in
canonicalisation. This probe sizes the problem; it does not locate the fix, and it proposes
none.

A near-zero count would mean F-169 is one row on one leg, and should be recorded rather
than chartered. **That is a real possible outcome and the probe is written to report it
plainly.**

Usage:  f169_wiring_census.py <repo>
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "src"))

CORE_ACCEPTED = "core_accepted"


def load_equivalences():
    """name -> set of equivalent names, from PRODUCTION tables only. Nothing authored."""
    groups = defaultdict(set)
    try:
        from t2pw.pwml.compound_templates import COMPOUND_NAME_TO_TEMPLATE as M
    except Exception as exc:                                    # pragma: no cover
        print(f"  *** could not import COMPOUND_NAME_TO_TEMPLATE: {exc}")
        M = {}
    by_tpl = defaultdict(set)
    for name, tpl in M.items():
        by_tpl[tpl].add(str(name).lower())
    for _tpl, names in by_tpl.items():
        for n in names:
            groups[n] |= names

    try:
        from t2pw.pipeline.process_normalizer import BIOCHEMICAL_ALIAS_MAP as A
    except Exception as exc:                                    # pragma: no cover
        print(f"  *** could not import BIOCHEMICAL_ALIAS_MAP: {exc}")
        A = {}
    for key, canon in A.items():
        k, c = str(key).lower(), str(canon).lower()
        groups[k] |= {k, c}
        groups[c] |= {k, c}
    return groups, len(M), len(A)


def norm(s):
    """The module's own normalisation, reproduced: lowercase, strip non-alphanumerics."""
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def term_matches(term_norm, term_set):
    """strict_quarantine._term_matches, reproduced exactly: substring either way."""
    if not term_norm:
        return False
    return any(term_norm == o or term_norm in o or o in term_norm for o in term_set)


def load(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def main():
    if len(sys.argv) < 2:
        print("usage: f169_wiring_census.py <repo>")
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

    groups, n_tpl, n_alias = load_equivalences()

    print("F-169 — ARE UNMATCHED ANCHORS ALREADY WIRED INTO AN ADMITTED CORE PROCESS?")
    print("=" * 94)
    print(f"repo                              : {repo}")
    print(f"roots scanned                     : runs/ AND runs_verify/, both named")
    print(f"coverage_summary.json found       : {len(legs)}")
    print(f"equivalence source                : PRODUCTION tables only, nothing authored")
    print(f"  COMPOUND_NAME_TO_TEMPLATE names : {n_tpl}")
    print(f"  BIOCHEMICAL_ALIAS_MAP entries   : {n_alias}")
    print(f"  names with >= 1 equivalent      : {sum(1 for k, v in groups.items() if len(v) > 1)}")
    print()

    tot = Counter()
    hits = []

    for path in legs:
        cov = load(path)
        if not isinstance(cov, dict) or not cov.get("requested_core_declared"):
            continue
        unmatched = [str(x) for x in (cov.get("unmatched_terms") or [])]
        if not unmatched:
            continue

        leg_dir = os.path.dirname(path)
        rep = load(os.path.join(leg_dir, "quarantine_report.json"))
        if not isinstance(rep, dict) or not isinstance(rep.get("admissions"), list):
            tot["legs_without_quarantine_report"] += 1
            continue
        tot["legs_examined"] += 1

        core_sets = []
        for row in rep["admissions"]:
            if isinstance(row, dict) and row.get("state") == CORE_ACCEPTED:
                core_sets.append({norm(t) for t in (row.get("core_terms") or [])})

        rel = os.path.relpath(path, repo).replace("\\", "/")
        parts = rel.split("/")
        run_id = "/".join(parts[:2])
        leg_id = "/".join(parts[-3:-1]) if len(parts) >= 3 else rel

        for a in unmatched:
            tot["anchors"] += 1
            a_norm = norm(a)

            # 1. The EXISTING rule. Must be False -- these anchors were declared
            #    unmatched by exactly this rule. A True here means the archived
            #    verdict disagrees with a faithful replay of its own matcher, which
            #    would be a much more serious finding than the one being measured.
            direct = any(term_matches(a_norm, s) for s in core_sets)
            if direct:
                tot["anchors_direct_match_REPLAY_DISAGREES"] += 1

            # 2. The same rule, with the anchor expanded to its production-declared
            #    equivalents. A True here is the ATP shape: wired, falsely reported.
            variants = {a_norm}
            for key, eq in groups.items():
                if norm(key) == a_norm:
                    variants |= {norm(e) for e in eq}
            wired = any(term_matches(v, s) for v in variants if v for s in core_sets)

            if wired and not direct:
                tot["anchors_wired_via_synonym"] += 1
                covering = sorted({t for s in core_sets for v in variants
                                   if v and term_matches(v, s) for t in s if v in t or t in v})
                hits.append((run_id, leg_id, a, sorted(variants - {a_norm}), covering[:4]))
            elif not wired:
                tot["anchors_not_wired"] += 1

    print("=" * 94)
    print("RESULT")
    print("=" * 94)
    for k in ("legs_examined", "legs_without_quarantine_report", "anchors",
              "anchors_direct_match_REPLAY_DISAGREES",
              "anchors_wired_via_synonym", "anchors_not_wired"):
        print(f"  {k:42s}: {tot[k]}")
    print()

    if tot["anchors_direct_match_REPLAY_DISAGREES"]:
        print("  *** A REPLAY OF THE ARCHIVED MATCHER DISAGREES WITH THE ARCHIVED VERDICT ON")
        print(f"  *** {tot['anchors_direct_match_REPLAY_DISAGREES']} ANCHOR(S). That is a bigger finding than this probe's")
        print("  *** own question and must be investigated before anything here is used.")
        print()

    print("=" * 94)
    print("EVERY ANCHOR THAT IS UNMATCHED BUT WIRED, under production's own compound identity")
    print("=" * 94)
    if not hits:
        print("  NONE.")
        print()
        print("  F-169 is then ONE ROW ON ONE LEG. It should be RECORDED, not chartered, and")
        print("  the scale argument in its favour does not exist. That is a real outcome and")
        print("  this probe was written to report it as readily as the other one.")
    else:
        for run_id, leg_id, anchor, variants, covering in hits:
            print(f"  {run_id:28s} {leg_id:28s}")
            print(f"      anchor    : {anchor!r}")
            print(f"      equivalents used (from production tables): {variants}")
            print(f"      matched core_terms: {covering}")

    print()
    print("=" * 94)
    print("WHAT THIS DOES AND DOES NOT SAY")
    print("=" * 94)
    print("  This is a LOWER BOUND on both sides of the question:")
    print("    * a synonym absent from BOTH production tables is invisible here, so the")
    print("      wired count can only be understated;")
    print("    * F-167's 90/374 'present in the payload' is likewise a lower bound, biased")
    print("      down by the same blindness -- it did not count ATP on PMC12096016/strict.")
    print()
    print("  It does NOT licence a matcher change. The auditor ruled the matcher's")
    print("  substring rule and its match-against-ADMITTED-PROCESSES design are both")
    print("  correct (D-088 clause 6; F-167 trap 3), and located the defect UPSTREAM in")
    print("  canonicalisation. This probe sizes the problem and proposes nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
