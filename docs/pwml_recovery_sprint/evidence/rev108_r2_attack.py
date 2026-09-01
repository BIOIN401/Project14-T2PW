"""REV-108 round 2: NEW possessive/deictic spans, not the eight already fixed.

The defect class was caught twice by writing new grammar rather than re-running
the supplied list, so section N below is four constructions the author has not
seen. Section E re-checks the eight for closure. Section O is the merge-rule-6
direction: nothing may be REFUSED at round 2 that was ACCEPTED at base.

Three code roots: base / round1 / round2.

Usage: <python> rev108_r2_attack.py <base-root> <r1-root> <r2-root>
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOTS = [Path(a).resolve() for a in sys.argv[1:4]]
LABELS = ["base", "r1", "r2"]


def load(root):
    for name in [m for m in list(sys.modules) if m.startswith("t2pw")]:
        del sys.modules[name]
    sys.path.insert(0, str(root / "src"))
    mod = importlib.import_module("t2pw.curation.apply_audit_patch")
    here = str(root).lower().replace("\\", "/")
    assert here in mod.__file__.lower().replace("\\", "/"), (here, mod.__file__)
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
CASES = [
    # -- N: FOUR NEW possessive/deictic constructions. want REFUSE. --------
    ("N-new", E, R, "P4X", "P4X, its repressor characterised in 2019, catalyses the conversion of A to B", False),
    ("N-new", E, R, "P4X", "P4X, their antagonists well characterised, catalyses the conversion of A to B", False),
    ("N-new", E, R, "P4X", "P4X, that suppressor aside, catalyses the conversion of A to B", False),
    ("N-new", E, R, "P4X", "P4X, its protein inhibitor removed, catalyses the conversion of A to B", False),
    # -- E: the escapes I reported at round 1. want REFUSE. -----------------
    ("E-round1", E, R, "P4X", "P4X, its inhibitor bound at the active site, converts A to B slowly", False),
    ("E-round1", E, R, "P4X", "P4X, this inhibitor notwithstanding, catalyses the conversion of A to B", False),
    ("E-round1", E, R, "P4X", "P4X, their inhibitors profiled, catalyses the conversion of A to B", False),
    # R8(ii), registered and deliberately NOT fixed this round.
    ("E-r8ii", E, R, "P4X", "P4X, an inhibitor target in oncology, catalyses the conversion of A to B", False),
    ("E-r8ii", E, R, "P4X", "P4X, the inhibitor binding site mapped, catalyses the conversion of A to B", False),
    # -- O: the apposition a bare determiner marks MUST survive. want ACCEPT.
    ("O-preserve", E, R, "P4X", "P4X, the inhibitor, catalyses the conversion of A to B", True),
    ("O-preserve", E, R, "P4X", "P4X, a repressor, catalyses the conversion of A to B", True),
    ("O-preserve", E, R, "P4X", "P4X, an inactivator, catalyses the conversion of A to B", True),
    ("O-preserve", E, R, "P4X", "P4X, the attenuator, catalyses the conversion of A to B", True),
    ("O-preserve", E, R, "P4X", "P4X, a small molecule inhibitor, catalyses the conversion of A to B", True),
    ("O-preserve", E, R, "P4X", "the inhibitor protein P4X catalyses the conversion of A to B", True),
    ("O-preserve", E, R, "P4X", "the repressor complex P4X catalyses the conversion of A to B", True),
    ("O-preserve", E, R, "P4X", "the suppressor protein P4X catalyses the conversion of A to B", True),
    ("O-preserve", E, R, "P4X", "the potent inhibitor P4X catalyses the conversion of A to B", True),
    # -- X: pinned properties across the families. --------------------------
    ("X-pinned", E, R, "P", "A significantly inhibited P activity in the assay", False),
    ("X-pinned", E, R, "NDM-1", "PSA significantly inhibited NDM-1 enzyme activity", False),
    ("X-pinned", E, R, "P", "P catalyses the conversion of A to B", True),
    ("X-pinned", E, R, "P", "the inhibitory effect on P activity abolished the catalysis of A to B", False),
    ("X-pinned", T, R, "P", "P transports A across the inner membrane", True),
    ("X-pinned", T, R, "P", "P channeled calcium into the cytosol", True),
    ("X-pinned", T, R, "P", "add P as a transporter to resolve the structural inconsistency", False),
    ("X-pinned", E, R, "LpxC hydrolase", "LpxC hydrolase was quantified in the lysate", False),
]


def main():
    cols = []
    for root in ROOTS:
        mod = load(root)
        cols.append([verdict(mod, c, a, s, b) for _x, c, b, a, s, _w in CASES])

    def w(v):
        return "ACCEPT" if v else "REFUSE"

    print("=" * 96)
    print("REV-108 ROUND-2 ATTACK  base / r1 / r2")
    print("=" * 96)
    bad, over, fixed = [], [], []
    cur = None
    for i, (sec, cont, bucket, actor, span, want) in enumerate(CASES):
        b, r1, r2 = cols[0][i], cols[1][i], cols[2][i]
        if sec != cur:
            print("\n--- %s ---" % sec)
            cur = sec
        tag = "OK"
        if r2 != want and b == want and want is False:
            tag = "<< STILL WEAKER THAN BASE"
            bad.append((sec, span, b, r1, r2, want))
        elif r2 != want and b == want and want is True:
            tag = "<< OVER-REFUSAL vs base"
            over.append((sec, span, b, r1, r2, want))
        elif r2 != want:
            tag = "<< open at base too"
            bad.append((sec, span, b, r1, r2, want))
        elif r1 != want and r2 == want:
            tag = "<< CLOSED BY ROUND 2"
            fixed.append(span)
        print("  %r" % (span[:104],))
        print("     base=%s r1=%s r2=%s want=%s  %s" % (w(b), w(r1), w(r2), w(want), tag))

    viol = [CASES[i][4] for i in range(len(CASES))
            if cols[2][i] is False and cols[0][i] is True]
    print()
    print("=" * 96)
    print("TOTALS cases=%d  not_as_wanted=%d  over_refusal=%d  closed_by_round2=%d"
          % (len(CASES), len(bad), len(over), len(fixed)))
    print("SUBSET/MERGE-RULE-6: spans REFUSED at r2 but ACCEPTED at base = %d" % len(viol))
    for s in viol:
        print("   !! %r" % (s,))
    for sec, span, b, r1, r2, want in bad:
        print("NOT-AS-WANTED %s base=%s r1=%s r2=%s want=%s" % (sec, w(b), w(r1), w(r2), w(want)))
        print("   %r" % (span,))
    print("=" * 96)
    return 0


sys.exit(main())
