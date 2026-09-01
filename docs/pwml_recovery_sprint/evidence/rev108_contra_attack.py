"""REV-108's OWN adversarial probe. Not the author's list.

Runs the SAME cases through the real seam at TWO code roots, so every verdict is
base-vs-tip and no case is a single-SHA assertion. Sections S1, S2, S6 and S7 are
reviewer-designed rephrasings that deliberately avoid the F3/F4 frames the author
built, which is the only way to tell a grammatical fix from a lexical one.

Usage:  <python> rev108_contra_attack.py <base-root> <tip-root>
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

BASE = Path(sys.argv[1]).resolve()
TIP = Path(sys.argv[2]).resolve()


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


E = "enzymes"
T = "transporters"
R = "reactions"

# (section, container, bucket, actor, span, want)  want True = ACCEPT
CASES = [
    # -- S1  B8: the contra must still refuse. The actor is being shut down. ---
    ("S1-contra", E, R, "P", "A significantly inhibited P activity in the assay", False),
    ("S1-contra", E, R, "P4X", "the inhibitor of P4X was added before the assay", False),
    ("S1-contra", E, R, "P4X", "the P4X inhibitor was added before the assay", False),
    ("S1-contra", E, R, "P4X", "the P4X specific inhibitor abrogated the signal", False),
    ("S1-contra", E, R, "P4X", "an inhibitor selective for P4X was used and P4X catalyses A to B", False),
    # REPHRASINGS that avoid the F3/F4 frames. REVIEWER-DESIGNED.
    ("S1-rephrase", E, R, "P4X", "P4X is a target of the inhibitor and catalyses the conversion of A to B", False),
    ("S1-rephrase", E, R, "P4X", "P4X was subject to inhibitors during the assay, yet catalyses A to B", False),
    ("S1-rephrase", E, R, "P4X", "treatment with the inhibitor abolished the catalysis of A to B by P4X", False),
    ("S1-rephrase", E, R, "P4X", "P4X, whose inhibitor was characterised, catalyses the conversion of A to B", False),
    ("S1-rephrase", E, R, "P4X", "addition of the inhibitor reduced P4X catalysis of A to B", False),
    ("S1-rephrase", E, R, "P4X", "the repressor bound P4X and the catalysis of A to B stopped", False),
    ("S1-rephrase", E, R, "P4X", "a suppressor of the P4X reaction was added, though P4X catalyses A to B", False),
    ("S1-rephrase", E, R, "P4X", "P4X catalysis of A to B was blocked by the antagonist", False),
    ("S1-rephrase", E, R, "P4X", "P4X is inhibited and does not catalyse the conversion of A to B", False),
    # -- S2  anchor choice: inflections MUST still fire the contra ------------
    ("S2-anchor", E, R, "P", "the inhibitory effect on P activity abolished the catalysis of A to B", False),
    ("S2-anchor", E, R, "P", "repression of P activity was observed while P catalyses A to B", False),
    ("S2-anchor", E, R, "P", "inhibition of P catalysis of A to B was complete", False),
    ("S2-anchor", E, R, "P", "attenuating P activity stopped the catalysis of A to B", False),
    ("S2-anchor", E, R, "P", "suppression of P activity blocked the catalysis of A to B", False),
    # -- S3  appositive agent nouns must now ACCEPT (the (d) correction) ------
    ("S3-appositive", E, R, "P4X", "the repressor complex P4X catalyses the conversion of A to B", True),
    ("S3-appositive", E, R, "P4X", "the suppressor protein P4X catalyses the conversion of A to B", True),
    ("S3-appositive", E, R, "P4X", "the inhibitor protein P4X catalyses the conversion of A to B", True),
    ("S3-appositive", E, R, "P4X", "P4X, a repressor, catalyses the conversion of A to B", True),
    ("S3-appositive", E, R, "P4X", "the repressor P4X catalyses the conversion of A to B", True),
    # -- S4  (a) transport paraphrases ---------------------------------------
    ("S4-a-transport", T, R, "P", "add P as a transporter to resolve the structural inconsistency", False),
    ("S4-a-transport", T, R, "P", "P should be added as a transporter", False),
    ("S4-a-transport", T, R, "P", "P is added to the transporters", False),
    ("S4-a-transport", T, R, "P", "the transporter P resolves the inconsistency", False),
    ("S4-a-transport", T, R, "P", "P transports A across the inner membrane", True),
    ("S4-a-transport", T, R, "P", "P is the transporter for A", True),
    ("S4-a-transport", T, R, "P", "P is a substrate of the transporter TonB", False),
    # channel / carrier / pump: verb readings the base accepted
    ("S4-verbforms", T, R, "P", "P channeled calcium into the cytosol", True),
    ("S4-verbforms", T, R, "P", "P channels calcium into the cytosol", True),
    ("S4-verbforms", T, R, "P", "P carries A across the membrane", True),
    ("S4-verbforms", T, R, "P", "P is the carrier of A across the membrane", True),
    ("S4-verbforms", T, R, "P", "A is carried by P across the membrane", True),
    ("S4-verbforms", T, R, "P", "P pumps protons across the membrane", True),
    ("S4-verbforms", T, R, "P", "the channel P was added to the payload", False),
    # -- S5  (a) catalysis schema noun; C-105's four dead alternatives --------
    ("S5-a-catalysis", E, R, "P", "add P as a catalyst to resolve the structural inconsistency", False),
    ("S5-a-catalysis", E, R, "P", "P is the catalyst responsible for the conversion of A to B", True),
    ("S5-a-catalysis", E, R, "P", "P is a catalyst for the conversion of A to B", True),
    ("S5-a-catalysis", E, R, "P", "P is the catalyst of this reaction", True),
    ("S5-a-catalysis", E, R, "P", "P catalyses the conversion of A to B", True),
    ("S5-a-catalysis", E, R, "P", "the catalyst P was listed without evidence", False),
    # -- S6  (c) a name is not a claim, and its rephrasings -------------------
    ("S6-c-name", E, R, "LpxC hydrolase", "LpxC hydrolase was quantified in the lysate", False),
    ("S6-c-name", E, R, "LpxC hydrolase", "LpxC, a hydrolase, was quantified in the lysate", False),
    ("S6-c-name", E, R, "LpxC hydrolase", "LpxC (a hydrolase) was quantified in the lysate", False),
    ("S6-c-name", E, R, "LpxC hydrolase", "the hydrolase LpxC was quantified in the lysate", False),
    ("S6-c-name", E, R, "LpxC hydrolase", "LpxC hydrolase catalyses the conversion of A to B", True),
    ("S6-c-name", E, R, "LpxC hydrolase", "LpxC hydrolase is the enzyme responsible for the conversion", True),
    ("S6-c-trap", T, R, "inner membrane translocase", "P is the translocase of the inner membrane", True),
    ("S6-c-trap", E, R, "EntB isochorismatase", "chorismate is converted to 2,3-diDHB by EntB isochorismatase activity", True),
    # -- S7  B15 substring collisions ----------------------------------------
    ("S7-collide", E, R, "P", "P is a reductase that reduces A to B", True),
    ("S7-collide", E, R, "P", "the blocker of P was added and P catalyses A to B", False),
    ("S7-collide", E, R, "P", "the silencer element upstream of P was deleted, P catalyses A to B", True),
    ("S7-collide", E, R, "P", "interferon signalling rose while P catalyses A to B", True),
    ("S7-collide", E, R, "P", "the intermediate is consumed as P mediates the condensation", True),
    ("S7-collide", T, R, "P", "P is the transporter subunit; the transporters list is empty", False),
    # -- S8  B16 actor names containing role nouns ---------------------------
    ("S8-actorname", E, R, "inhibitor of apoptosis protein",
     "inhibitor of apoptosis protein catalyses the conversion of A to B", True),
    ("S8-actorname", E, R, "inhibitor of apoptosis protein",
     "inhibitor of apoptosis protein was quantified in the lysate", False),
    ("S8-actorname", T, R, "ABC transporter MsbA", "ABC transporter MsbA was detected in the membrane", False),
    ("S8-actorname", T, R, "ABC transporter MsbA", "ABC transporter MsbA transports lipid A across the membrane", True),
    # -- S9  B17 period-stripped multi-sentence ------------------------------
    ("S9-multisent", E, R, "P", "P catalyses the conversion of A to B. The inhibitor of P was added.", False),
    ("S9-multisent", E, R, "P",
     "P catalyses the conversion of A to B. " + ("x " * 60) + "The inhibitor of P was added.", True),
    ("S9-multisent", E, R, "P", "The inhibitor of P was added. P catalyses the conversion of A to B.", False),
    # -- S10 F-146 pinned ----------------------------------------------------
    ("S10-f146", E, R, "P",
     "add P as an enzyme to resolve the structural inconsistency where an inhibitor is listed without a target enzyme",
     False),
]


def main():
    base = load(BASE)
    rows = []
    for sec, cont, bucket, actor, span, want in CASES:
        rows.append([sec, cont, bucket, actor, span, want,
                     verdict(base, cont, actor, span, bucket)])
    tip = load(TIP)
    for row in rows:
        sec, cont, bucket, actor, span, want, _b = row
        row.append(verdict(tip, cont, actor, span, bucket))

    def word(v):
        return "ACCEPT" if v else "REFUSE"

    regressions = []
    open_at_both = []
    closed = []
    print("=" * 100)
    print("REV-108 OWN ADVERSARIAL PROBE  base=%s  tip=%s" % (BASE, TIP))
    print("=" * 100)
    cur = None
    for sec, cont, bucket, actor, span, want, b, t in rows:
        if sec != cur:
            print("\n--- %s ---" % sec)
            cur = sec
        if t != want and b == want:
            tag = "<< REGRESSION INTRODUCED BY THIS CARD"
            regressions.append((sec, actor, span, word(b), word(t), word(want)))
        elif t != want and b != want:
            tag = "<< still open at BOTH SHAs (not this card's regression)"
            open_at_both.append((sec, actor, span, word(want)))
        elif t == want and b != want:
            tag = "<< CLOSED BY THIS CARD"
            closed.append((sec, actor, span))
        else:
            tag = "OK"
        print("  [%s] actor=%r" % (cont, actor))
        print("      %r" % (span[:120],))
        print("      base=%s tip=%s want=%s  %s" % (word(b), word(t), word(want), tag))
    print()
    print("=" * 100)
    print("REV108 PROBE TOTALS  cases=%d  regressions=%d  open_at_both=%d  closed=%d"
          % (len(rows), len(regressions), len(open_at_both), len(closed)))
    print("=" * 100)
    for r in regressions:
        print("REGRESSION  %s  actor=%s  base=%s tip=%s want=%s" % (r[0], r[1], r[3], r[4], r[5]))
        print("            %r" % (r[2],))
    for r in open_at_both:
        print("OPEN-BOTH   %s  actor=%s want=%s" % (r[0], r[1], r[3]))
        print("            %r" % (r[2],))
    return 0


sys.exit(main())
