"""REV-108 round 1: NEW grammar against the inverted agent-noun contra.

Written after reading the repair but WITHOUT reusing either the four spans I
supplied at round 0 or the ten the author added -- those are the spans the fix
was built against. Sections T and A below are new constructions, and section S
tests the round-1 load-bearing claim directly:

    "base fired on every occurrence of the stem inside an agent noun, and this
     fires on a SUBSET of those, so relative to base the tip can only refuse
     LESS. No over-refusal against base is reachable through here."

Three code roots, one row per span, so every verdict is base / round0 / round1.

Usage: <python> rev108_r1_attack.py <base-root> <r0-root> <r1-root>
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOTS = [Path(a).resolve() for a in sys.argv[1:4]]
LABELS = ["base", "r0", "r1"]


def load(root):
    for name in [m for m in list(sys.modules) if m.startswith("t2pw")]:
        del sys.modules[name]
    sys.path.insert(0, str(root / "src"))
    mod = importlib.import_module("t2pw.curation.apply_audit_patch")
    assert str(root).lower().replace("\\", "/") in mod.__file__.lower().replace("\\", "/"), \
        (str(root), mod.__file__)
    return mod


def verdict(mod, container, actor, evidence, bucket="reactions"):
    proc = {"name": "A to B", "inputs": ["A"], "outputs": ["B"],
            "evidence": "A is converted in the gut", container: []}
    payload = {"entities": {"compounds": [{"name": "A"}, {"name": "B"}],
                            "proteins": [{"name": actor}], "protein_complexes": [],
                            "nucleic_acids": []},
               "processes": {bucket: [proc]}}
    op = {"op": "add", "path": "/processes/%s/0/%s/-" % (bucket, container),
          "value": actor, "confidence": 1.0, "evidence": evidence}
    _r, rep = mod.apply_patch_with_policy(payload, [op], stage="probe")
    return rep["summary"]["accepted_count"] == 1


E, T, R = "enzymes", "transporters", "reactions"

# (section, container, bucket, actor, span, want)
CASES = [
    # == T: TARGET-DIRECTED. The actor is the thing being shut down: REFUSE. ==
    # None of these is one of my round-0 four or the author's ten.
    ("T-new", E, R, "P4X", "resistance to the inhibitor was lost in P4X, which catalyses A to B", False),
    ("T-new", E, R, "P4X", "P4X and its inhibitor were co-purified; P4X catalyses the conversion of A to B", False),
    ("T-new", E, R, "P4X", "the inhibitor, which targets P4X, was added while P4X catalyses A to B", False),
    ("T-new", E, R, "P4X", "P4X activity fell in the presence of its antagonist, yet P4X catalyses A to B", False),
    ("T-new", E, R, "P4X", "an antagonist raised in rabbits recognised P4X, which catalyses A to B", False),
    ("T-new", E, R, "P4X", "knockdown of the P4X repressor changed nothing, though P4X catalyses A to B", False),
    ("T-new", E, R, "P4X", "the inhibitor bound the P4X enzyme and catalysis of A to B ceased", False),
    ("T-new", E, R, "P4X", "inhibitor treated P4X still catalyses the conversion of A to B", False),
    ("T-new", E, R, "P4X", "the repressor sits upstream of P4X, which catalyses the conversion of A to B", False),
    ("T-new", E, R, "P4X", "co-incubation of P4X with a suppressor abolished nothing; P4X catalyses A to B", False),
    # == A: ATTACKS ON THE A2 DETERMINER SET, which is new in round 1. ==
    # "its"/"their" are POSSESSIVE and "this"/"that" are DEICTIC; none of the
    # four is guaranteed to mark an apposition with THIS actor.
    ("A2-attack", E, R, "P4X", "P4X, its inhibitor bound at the active site, converts A to B slowly", False),
    ("A2-attack", E, R, "P4X", "P4X, this inhibitor notwithstanding, catalyses the conversion of A to B", False),
    ("A2-attack", E, R, "P4X", "P4X, an inhibitor target in oncology, catalyses the conversion of A to B", False),
    ("A2-attack", E, R, "P4X", "P4X, the inhibitor binding site mapped, catalyses the conversion of A to B", False),
    ("A2-attack", E, R, "P4X", "P4X, their inhibitors profiled, catalyses the conversion of A to B", False),
    ("A2-attack", E, R, "P4X", "P4X, a known inhibitor target, catalyses the conversion of A to B", False),
    # == A1 attacks: can a modifier run reach across a non-appositive reading? ==
    ("A1-attack", E, R, "P4X", "the inhibitor sensitive P4X catalyses the conversion of A to B", False),
    ("A1-attack", E, R, "P4X", "the inhibitor small molecule P4X catalyses the conversion of A to B", True),
    ("A1-attack", E, R, "P4X", "the suppressor enzyme complex P4X catalyses the conversion of A to B", True),
    ("A1-attack", E, R, "P4X", "the inhibitor of the enzyme P4X was added while P4X catalyses A to B", False),
    # == P: PRESERVATIONS. Member (d) must not be regressed by the inversion. ==
    ("P-appositive", E, R, "P4X", "the repressor complex P4X catalyses the conversion of A to B", True),
    ("P-appositive", E, R, "P4X", "the suppressor protein P4X catalyses the conversion of A to B", True),
    ("P-appositive", E, R, "P4X", "the inhibitor protein P4X catalyses the conversion of A to B", True),
    ("P-appositive", E, R, "P4X", "P4X, a repressor, catalyses the conversion of A to B", True),
    ("P-appositive", E, R, "P4X", "the repressor P4X catalyses the conversion of A to B", True),
    ("P-appositive", E, R, "P4X", "the attenuator isoform P4X catalyses the conversion of A to B", True),
    # == S: THE SUBSET CLAIM. Every span here carries an agent noun in an
    # unusual morphology. If the claim holds, NOTHING may be REFUSE at r1 while
    # ACCEPT at base -- that would be an over-refusal reachable through the new
    # path, and it would be blocking in the other direction.
    ("S-subset", E, R, "P4X", "the co-inhibitor P4X catalyses the conversion of A to B", True),
    ("S-subset", E, R, "P4X", "the down-regulator P4X catalyses the conversion of A to B", True),
    ("S-subset", E, R, "P4X", "the pan-antagonist P4X catalyses the conversion of A to B", True),
    ("S-subset", E, R, "P4X", "the inhibitors P4X and Q catalyse the conversion of A to B", True),
    ("S-subset", E, R, "P4X", "the attenuator variant P4X catalyses the conversion of A to B", True),
    ("S-subset", E, R, "P4X", "P4X, the abolisher, catalyses the conversion of A to B", True),
    ("S-subset", E, R, "P4X", "the inactivator subunit P4X catalyses the conversion of A to B", True),
    # == X: the pinned properties and the other families. ==
    ("X-pinned", E, R, "P", "A significantly inhibited P activity in the assay", False),
    ("X-pinned", E, R, "NDM-1", "PSA significantly inhibited NDM-1 enzyme activity", False),
    ("X-pinned", E, R, "P", "the inhibitory effect on P activity abolished the catalysis of A to B", False),
    ("X-pinned", E, R, "P", "repression of P activity was observed while P catalyses A to B", False),
    ("X-pinned", T, R, "P", "P transports A across the inner membrane", True),
    ("X-pinned", T, R, "P", "add P as a transporter to resolve the structural inconsistency", False),
    # R-a and R-c, the two I registered at round 0.
    ("X-ra-rc", T, R, "P", "P channeled calcium into the cytosol", True),
    ("X-ra-rc", T, R, "P", "P channelled calcium into the cytosol", True),
    ("X-ra-rc", T, R, "P", "P is a substrate of the transporter TonB", False),
    ("X-ra-rc", T, R, "P", "P is a high affinity transporter for A", True),
    ("X-ra-rc", T, R, "P", "P acts as an inner membrane channel for A", True),
]


def main():
    results = []
    for label, root in zip(LABELS, ROOTS):
        mod = load(root)
        col = []
        for _sec, cont, bucket, actor, span, _want in CASES:
            col.append(verdict(mod, cont, actor, span, bucket))
        results.append(col)

    def w(v):
        return "ACCEPT" if v else "REFUSE"

    print("=" * 100)
    print("REV-108 ROUND-1 ATTACK   base=%s  r0=%s  r1=%s" % tuple(str(r) for r in ROOTS))
    print("=" * 100)
    blocking, over_refusal, still_open, fixed = [], [], [], []
    cur = None
    for i, (sec, cont, bucket, actor, span, want) in enumerate(CASES):
        b, r0, r1 = results[0][i], results[1][i], results[2][i]
        if sec != cur:
            print("\n--- %s ---" % sec)
            cur = sec
        tag = "OK"
        if r1 != want:
            if b == want:
                if want is False:
                    tag = "<< BLOCKING: gate weakened vs base"
                    blocking.append((sec, actor, span, b, r0, r1, want))
                else:
                    tag = "<< OVER-REFUSAL vs base"
                    over_refusal.append((sec, actor, span, b, r0, r1, want))
            else:
                tag = "<< still open at base too (not this card)"
                still_open.append((sec, actor, span, b, r0, r1, want))
        elif r0 != want and r1 == want:
            tag = "<< FIXED BY ROUND 1"
            fixed.append((sec, actor, span))
        print("  [%s] actor=%r" % (cont, actor))
        print("      %r" % (span[:112],))
        print("      base=%s  r0=%s  r1=%s  want=%s   %s" % (w(b), w(r0), w(r1), w(want), tag))

    print()
    print("=" * 100)
    print("REV108 R1 TOTALS  cases=%d  blocking=%d  over_refusal_vs_base=%d  "
          "open_at_base=%d  fixed_by_round1=%d"
          % (len(CASES), len(blocking), len(over_refusal), len(still_open), len(fixed)))
    print("=" * 100)
    for name, rows in (("BLOCKING", blocking), ("OVER-REFUSAL", over_refusal),
                       ("OPEN-AT-BASE", still_open)):
        for sec, actor, span, b, r0, r1, want in rows:
            print("%-13s %-11s actor=%s base=%s r0=%s r1=%s want=%s"
                  % (name, sec, actor, w(b), w(r0), w(r1), w(want)))
            print("              %r" % (span,))

    # The subset claim, stated as a machine check over EVERY case above.
    violations = [(CASES[i][4], results[0][i], results[2][i])
                  for i in range(len(CASES))
                  if results[2][i] is False and results[0][i] is True
                  and CASES[i][1] == "enzymes"]
    print()
    print("SUBSET CLAIM (catalysis rows only): spans REFUSED at r1 but ACCEPTED at base = %d"
          % len(violations))
    for span, _b, _t in violations:
        print("   !! %r" % (span,))
    return 0


sys.exit(main())
