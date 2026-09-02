"""D-088 — what would a SUBPROCESS-LEVEL hard cap do, measured on archived artifacts?

Read-only A/B across **every committed `coverage_summary.json` under both `runs/` and
`runs_verify/`** — both roots named explicitly, because both are live and "the pinned
run" is ambiguous. **No leg is run, re-run, re-scored or mutated.** Every number here is
a fact about what a leg already recorded.

WHY THIS EXISTS, BEFORE ANY CARD IS CHARTERED
---------------------------------------------
D-088 clause 9 says the untyped Stage-0 entity-anchor list must not remain the sole
hard-failure input of the incomplete-core cap, and clause 10 forbids relaxing that cap
"without replacing it with reaction-level coverage". The candidate replacement is the one
process-level specification production already has at the coverage seam: Stage 0's own
``main_subprocesses``, carried into ``coverage_summary.json`` inside ``requested_context``
and therefore available without reading gold and without a Stage-0 redesign.

This probe measures that candidate on the archived corpus **before** it is built, so the
rule is chosen against 83 legs rather than against the two legs whose failure prompted it.
That is the explicit lesson of F-167's own amendment: a rule measured only on the case in
front of you is the one least likely to have been tested against the cases that are not.

WHAT IT DECIDES NOTHING ABOUT
-----------------------------
It proposes no code change, edits nothing, and scores no milestone. A cap that looks good
here is a candidate, not a merge.

THE TWO REQUIRED CONSEQUENCES (D-088, HANDOFF 5.2a step 6)
----------------------------------------------------------
  * ``PMC12096016`` must lose ONLY the false entity-anchor cap;
  * ``PMC12782028`` must REMAIN a reaction-recall failure.
A rule that clears both has removed the measurement rather than improved recall, and the
probe reports that outcome as ``REJECT`` in as many words rather than leaving a reader to
notice it.

THE THREE POPULATIONS THAT MUST SURVIVE (D-088 clauses 7 and 8)
---------------------------------------------------------------
The census established 60/374 subprocess-aligned and 90/374 payload-present-but-unwired
unmatched anchors. This probe recomputes both under the proposed classification and
reports them SEPARATELY, so it can be shown that reclassifying what hard-fails did not
erase what is recorded.

SENSITIVITY, NOT A SINGLE NUMBER
--------------------------------
The subprocess/process match needs a generic-word stoplist, and a stoplist is exactly the
kind of knob that can be tuned until the answer is pleasing. So the probe runs the whole
A/B at THREE stoplist strengths and prints all three. If the verdict on the two named
papers is not stable across all three, the rule is tuned rather than measured, and the
probe says so.

Usage:  d088_subprocess_recall_ab.py <repo>
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter

# Generic process/English words that carry no biological identity. A subprocess and an
# admitted process sharing only these has not been shown to be the same step.
#
# MINIMAL is English structure only. STANDARD adds the verbs and nouns that describe ANY
# enzymatic step. AGGRESSIVE additionally drops the reaction-class nouns, which is the
# strictest reading and the one most likely to call a genuinely covered subprocess
# uncovered. The rule is reported at all three; nothing is chosen here.
STOP_MINIMAL = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "of", "on", "onto",
    "or", "the", "to", "via", "with", "its", "their",
}
STOP_STANDARD = STOP_MINIMAL | {
    "reaction", "reactions", "step", "steps", "pathway", "pathways", "process",
    "processes", "subprocess", "conversion", "synthesis", "formation", "production",
    "complex", "cycle", "cycles", "activity", "mediated", "catalyzed", "catalysed",
    "dependent", "final", "first", "second", "third", "last", "initial", "terminal",
}
STOP_AGGRESSIVE = STOP_STANDARD | {
    "reduction", "oxidation", "hydrolysis", "isomerization", "isomerisation",
    "transfer", "assembly", "loading", "binding", "release", "uptake", "transport",
    "secretion", "export", "import", "cleavage", "condensation", "addition",
    "removal", "modification", "activation", "inhibition", "regulation",
    "biosynthesis", "degradation", "metabolism", "enzyme", "enzymes", "protein",
    "proteins", "substrate", "product", "domain", "subunit",
}

STOPLISTS = [
    ("minimal", STOP_MINIMAL),
    ("standard", STOP_STANDARD),
    ("aggressive", STOP_AGGRESSIVE),
]

# The two legs D-088 names as the required discrimination.
MUST_CLEAR = "PMC12096016"
MUST_REMAIN = "PMC12782028"

ACCEPTED_CORE_STATE = "core_accepted"


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def tokens(s, stop):
    """Content tokens of a phrase: alphanumeric runs, stopwords and 1-char noise gone.

    Greek letters and the punctuation inside '2,3-diDHB' or '14alpha' are folded by the
    same rule that produced ``core_terms`` in the first place, so both sides of the
    comparison are normalised identically. A token shorter than 3 characters is dropped:
    'of', 'to' are already stopwords, but 'b1' or 'ii' would otherwise match promiscuously.
    """
    raw = re.split(r"[^a-z0-9]+", str(s).lower())
    out = set()
    for t in raw:
        if len(t) >= 3 and t not in stop:
            out.add(t)
    return out


def subprocess_covered(sub, core_term_token_sets, stop):
    """Is this Stage-0 subprocess covered by at least one ADMITTED CORE process?

    Covered means: some core_accepted process's core_terms share at least one content
    token with the subprocess phrase. Deliberately generous on the COVERED side, because
    the number this feeds is a HARD FAILURE and a false 'uncovered' would refuse a
    correct pathway. The counterweight is the stoplist sensitivity above: a rule that only
    discriminates at one stoplist strength is not discriminating on biology.

    NOTE the asymmetry with the entity-anchor matcher this replaces. That one asks whether
    a flat entity name equals a process's core_terms, which a cofactor can never satisfy.
    This asks whether a NAMED STEP is represented among the steps that survived, which is
    the question D-088 clause 4 says the hard decision should be based on.
    """
    st = tokens(sub, stop)
    if not st:
        return None  # nothing to test on; reported separately, never counted as covered
    for cts in core_term_token_sets:
        if st & cts:
            return True
    return False


def load(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def payload_names(leg_dir):
    """Normalised entity names in the leg's final_mapped.json, for the unwired split."""
    out = set()
    pay = load(os.path.join(leg_dir, "final_mapped.json"))
    if not isinstance(pay, dict):
        return out
    ents = pay.get("entities")
    if isinstance(ents, dict):
        for _k, v in ents.items():
            if isinstance(v, list):
                for e in v:
                    if isinstance(e, dict) and e.get("name"):
                        out.add(norm(e["name"]))
    return out


def core_term_sets(leg_dir, stop):
    """Token sets of the core_terms of every CORE-ACCEPTED process in this leg.

    Read from quarantine_report.json, which is the artifact that records the admission
    states. A leg without one cannot be tested and is reported as such, never as covered.
    """
    rep = load(os.path.join(leg_dir, "quarantine_report.json"))
    if not isinstance(rep, dict):
        return None
    adms = rep.get("admissions")
    if not isinstance(adms, list):
        return None
    sets = []
    for row in adms:
        if not isinstance(row, dict) or row.get("state") != ACCEPTED_CORE_STATE:
            continue
        ts = set()
        for term in row.get("core_terms") or []:
            ts |= tokens(term, stop)
        label = row.get("label")
        if isinstance(label, str):
            ts |= tokens(label, stop)
        sets.append(ts)
    return sets


def collect_legs(repo):
    legs = []
    for root_name in ("runs", "runs_verify"):
        root = os.path.join(repo, root_name)
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            if "coverage_summary.json" in files:
                legs.append(os.path.join(dirpath, "coverage_summary.json"))
    legs.sort()
    return legs


def run_pass(repo, legs, stop_name, stop):
    """One full A/B at one stoplist strength. Returns (totals, rows)."""
    tot = Counter()
    rows = []
    for path in legs:
        cov = load(path)
        if not isinstance(cov, dict):
            tot["unreadable"] += 1
            continue
        if not cov.get("requested_core_declared"):
            tot["no_declared_core"] += 1
            continue
        tot["legs_with_declared_core"] += 1

        leg_dir = os.path.dirname(path)
        rel = os.path.relpath(path, repo).replace("\\", "/")
        parts = rel.split("/")
        run_id = "/".join(parts[:2])
        leg_id = "/".join(parts[-3:-1]) if len(parts) >= 3 else rel
        paper = parts[-3] if len(parts) >= 3 else ""

        unmatched = [str(x) for x in (cov.get("unmatched_terms") or [])]
        ctx = cov.get("requested_context") or {}
        subs = [str(x) for x in (ctx.get("main_subprocesses") or [])]

        # ---- OLD cap: any unmatched entity anchor removes release_ready.
        old_capped = bool(unmatched)
        if old_capped:
            tot["old_cap_legs_capped"] += 1

        # ---- NEW cap: any Stage-0-named subprocess with no admitted core process.
        cts = core_term_sets(leg_dir, stop)
        if not subs:
            tot["legs_without_subprocesses"] += 1
            new_capped = None
        elif cts is None:
            tot["legs_without_quarantine_report"] += 1
            new_capped = None
        else:
            tot["legs_testable_under_new_cap"] += 1
            uncovered = []
            untestable = 0
            for s in subs:
                c = subprocess_covered(s, cts, stop)
                if c is None:
                    untestable += 1
                elif not c:
                    uncovered.append(s)
            new_capped = bool(uncovered)
            if new_capped:
                tot["new_cap_legs_capped"] += 1
            tot["subprocesses_examined"] += len(subs)
            tot["subprocesses_uncovered"] += len(uncovered)
            tot["subprocesses_untestable"] += untestable

        # ---- The three anchor populations D-088 clauses 7/8 require to stay visible.
        names = payload_names(leg_dir)
        sub_blob = " | ".join(subs).lower()
        in_sub = in_pay = neither = 0
        for a in unmatched:
            tot["anchors_unmatched"] += 1
            aligned = bool(sub_blob) and a.lower() in sub_blob
            na = norm(a)
            present = bool(na) and (na in names or any(na in n for n in names))
            if aligned:
                in_sub += 1
                tot["anchors_subprocess_aligned"] += 1
            if present:
                in_pay += 1
                tot["anchors_payload_present_unwired"] += 1
            if not aligned and not present:
                neither += 1
                tot["anchors_neither"] += 1

        if old_capped or new_capped:
            rows.append({
                "run": run_id, "leg": leg_id, "paper": paper,
                "coverage_ratio": cov.get("coverage_ratio"),
                "old": old_capped, "new": new_capped,
                "n_unmatched": len(unmatched), "n_subs": len(subs),
                "n_uncovered": (len(uncovered) if new_capped is not None else None),
                "uncovered": (uncovered if new_capped is not None else []),
                "unmatched": unmatched,
                "in_sub": in_sub, "in_pay": in_pay, "neither": neither,
            })
    return tot, rows


def verdict_for(rows, paper, want_new_capped):
    """What did the proposed cap do to every strict leg of one named paper?"""
    hits = [r for r in rows if r["paper"] == paper and r["leg"].endswith("strict")]
    if not hits:
        return "NO STRICT LEG FOUND", []
    bad = [r for r in hits if r["new"] is not None and bool(r["new"]) != want_new_capped]
    ok = not bad
    return ("HOLDS" if ok else "VIOLATED"), hits


def main():
    if len(sys.argv) < 2:
        print("usage: d088_subprocess_recall_ab.py <repo>")
        return 2
    repo = os.path.abspath(sys.argv[1])
    legs = collect_legs(repo)

    print("D-088 SUBPROCESS-RECALL A/B — archived artifacts only, nothing re-run")
    print("=" * 92)
    print(f"repo                                  : {repo}")
    print(f"roots scanned                         : runs/ AND runs_verify/, both named")
    print(f"committed coverage_summary.json found : {len(legs)}")
    print()

    named_verdicts = {}

    for stop_name, stop in STOPLISTS:
        tot, rows = run_pass(repo, legs, stop_name, stop)
        print("=" * 92)
        print(f"STOPLIST = {stop_name}   ({len(stop)} words)")
        print("=" * 92)
        for k in ("legs_with_declared_core", "no_declared_core", "unreadable",
                  "legs_without_subprocesses", "legs_without_quarantine_report",
                  "legs_testable_under_new_cap",
                  "old_cap_legs_capped", "new_cap_legs_capped",
                  "subprocesses_examined", "subprocesses_uncovered",
                  "subprocesses_untestable",
                  "anchors_unmatched", "anchors_subprocess_aligned",
                  "anchors_payload_present_unwired", "anchors_neither"):
            print(f"  {k:38s}: {tot[k]}")

        testable = [r for r in rows if r["new"] is not None]
        flips = [r for r in testable if r["old"] and not r["new"]]
        holds = [r for r in testable if r["old"] and r["new"]]
        newly = [r for r in testable if not r["old"] and r["new"]]
        print()
        print(f"  legs capped OLD and NOT capped NEW (released) : {len(flips)}")
        print(f"  legs capped OLD and STILL capped NEW          : {len(holds)}")
        print(f"  legs capped NEW but NOT capped OLD (stricter) : {len(newly)}")

        v1, h1 = verdict_for(rows, MUST_CLEAR, want_new_capped=False)
        v2, h2 = verdict_for(rows, MUST_REMAIN, want_new_capped=True)
        named_verdicts[stop_name] = (v1, v2)
        print()
        print(f"  D-088 consequence 1 — {MUST_CLEAR} strict loses the cap  : {v1}")
        for r in h1:
            print(f"      {r['run']} {r['leg']}  old={r['old']} new={r['new']} "
                  f"subs={r['n_subs']} uncovered={r['n_uncovered']} {r['uncovered']}")
        print(f"  D-088 consequence 2 — {MUST_REMAIN} strict stays capped  : {v2}")
        for r in h2:
            print(f"      {r['run']} {r['leg']}  old={r['old']} new={r['new']} "
                  f"subs={r['n_subs']} uncovered={r['n_uncovered']} {r['uncovered']}")
        print()

        if stop_name == "standard":
            print("  " + "-" * 88)
            print("  EVERY leg the proposed cap would RELEASE (was capped, no longer is)")
            print("  " + "-" * 88)
            for r in sorted(flips, key=lambda r: (r["run"], r["leg"])):
                print(f"    {r['run']:28s} {r['leg']:34s} cov={r['coverage_ratio']} "
                      f"subs={r['n_subs']} unmatched={r['n_unmatched']}")
                print(f"        anchors downgraded to warnings: {r['unmatched']}")
                print(f"        of those: subprocess-aligned={r['in_sub']} "
                      f"payload-present-unwired={r['in_pay']} neither={r['neither']}")
            print()

    print("=" * 92)
    print("STABILITY OF THE TWO REQUIRED CONSEQUENCES ACROSS STOPLIST STRENGTH")
    print("=" * 92)
    for name, (v1, v2) in named_verdicts.items():
        print(f"  {name:12s}  {MUST_CLEAR}={v1:9s}  {MUST_REMAIN}={v2}")
    vals = set(named_verdicts.values())
    if len(vals) == 1 and list(vals)[0] == ("HOLDS", "HOLDS"):
        print()
        print("  BOTH CONSEQUENCES HOLD AT EVERY STOPLIST STRENGTH.")
        print("  The discrimination is a property of the biology in these legs, not of the")
        print("  stoplist. That is the only reading under which this rule is MEASURED and")
        print("  not TUNED -- and it is still only a candidate, not a merge.")
    else:
        print()
        print("  *** THE VERDICT IS NOT STABLE ACROSS STOPLIST STRENGTH. ***")
        print("  The rule is sensitive to a knob rather than to the biology. Under D-088")
        print("  clause 10 that is exactly the shape of a change that moves a score by")
        print("  removing a measurement. Do NOT charter it in this form.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
