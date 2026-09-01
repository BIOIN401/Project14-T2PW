"""C-107 correction round 1: the four frames, the cofactor leaks, the fallback.

Reproduces REV-107's two blocking measurements and the B2 residual, through the
real seam, so the same script prints the BEFORE and the AFTER.

BLOCKING 1 -- each of the eleven near-synonyms in FOUR frames, not one:
    A  object present, within 40 chars    the frame C-107 section 1a quoted
    B  object ABSENT                      "the <w> of X is mediated by Y"
    C  object BEFORE the stem             "X activity showed <w> in the Y-mediated assay"
    D  object BEYOND 40 chars             45 characters of padding

BLOCKING 2 -- the cofactor family's loose terms, and the corollary that
_ANY_ROLE_CUE_RE is rebuilt from every _ROLE_CUE_RES value, so anything added to
the cofactor family widens the "other" fallback for EVERY unmapped role.

ITEM 3 -- the B2 residual: real chemistry that "level" and "function" collide with.

Usage::  <python> c107_r1_frames.py <repo-root>
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REPO / "src"))

from t2pw.curation.apply_audit_patch import apply_patch_with_policy  # noqa: E402


def seam(name, evidence, container="enzymes", bucket="reactions", role=None):
    value = name if role is None else {"entity": name, "role": role}
    proc = {"name": "A to B", "inputs": ["A"], "outputs": ["B"],
            "evidence": "chem only", container: []}
    payload = {"entities": {"compounds": [{"name": "A"}, {"name": "B"}],
                            "proteins": [{"name": name}],
                            "protein_complexes": [], "nucleic_acids": []},
               "processes": {bucket: [proc]}}
    op = {"op": "add", "path": f"/processes/{bucket}/0/{container}/-",
          "value": value, "confidence": 1.0}
    if evidence is not None:
        op["evidence"] = evidence
    _r, rep = apply_patch_with_policy(payload, [op], stage="probe")
    return rep["summary"]["accepted_count"] == 1


NEAR = ["blockade", "impairment", "disruption", "reduction", "loss", "silencing",
        "sequestration", "depletion", "ablation", "interference", "quenching"]

PAD = "x" * 45


def frames(w):
    return [
        ("A object present", f"the {w} of P activity is mediated by Q"),
        ("B object absent", f"the {w} of P is mediated by Q"),
        ("C object first", f"P activity showed {w} in the Q-mediated assay"),
        ("D object beyond 40", f"the {w} of P {PAD} enzymatic activity is mediated by Q"),
    ]


print("=" * 96)
print("BLOCKING 1 -- eleven near-synonyms x four frames. True == the defect is ADMITTED.")
print("=" * 96)
print(f"  {'word':16s} {'A':>7s} {'B':>7s} {'C':>7s} {'D':>7s}   open frames")
open_total = 0
for w in NEAR:
    verdicts = [seam("P", span) for _label, span in frames(w)]
    n = sum(verdicts)
    open_total += n
    print(f"  {w:16s} " + " ".join(f"{str(v):>7s}" for v in verdicts) + f"   {n}")
print(f"\n  TOTAL OPEN (want 0): {open_total} / {len(NEAR) * 4}")

print()
print("=" * 96)
print("BLOCKING 1 -- PRESERVATION. Redox chemistry, which must all license.")
print("=" * 96)
REDOX = [
    ("P", "NADH-dependent reduction of the substrate by P"),
    ("ferrochelatase", "ferrochelatase reduces the substrate in this step"),
    ("P", "P catalyses the reduction of the quinone to the quinol"),
    ("P", "the reduction of A to B is carried out by P"),
    ("P", "P reduces nitrite to nitric oxide"),
    ("P", "reducing equivalents are transferred by P during the reduction of the disulfide"),
    ("P", "P is the reductase for this step"),
    ("P", "P catalyses the reduction of the disulfide bond of the substrate protein"),
    ("P", "the two-electron reduction of the flavin is catalysed by P"),
]
bad = 0
for name, span in REDOX:
    ok = seam(name, span)
    if not ok:
        bad += 1
    print(f"  {'licensed' if ok else 'REFUSED <<':11s}  actor={name!r:18s} {span!r}")
print(f"\n  REFUSED (want 0): {bad} / {len(REDOX)}")

print()
print("=" * 96)
print("ITEM 3 -- the B2 residual: 'level' and 'function' colliding with real chemistry")
print("=" * 96)
B2_RESIDUAL = [
    ("P", "P catalyses the reduction of the substrate level in vitro"),
    ("ferrochelatase", "ferrochelatase reduces the cellular level of protoporphyrin"),
    ("P", "P-dependent reduction of flavin is required for enzyme function"),
]
bad3 = 0
for name, span in B2_RESIDUAL:
    ok = seam(name, span)
    if not ok:
        bad3 += 1
    print(f"  {'licensed' if ok else 'REFUSED <<':11s}  actor={name!r:18s} {span!r}")
print(f"\n  REFUSED (want 0): {bad3} / {len(B2_RESIDUAL)}")

print()
print("=" * 96)
print("BLOCKING 2 -- the cofactor family. True == an UNEVIDENCED patch passes.")
print("=" * 96)
LEAKS = [
    ("the reaction requires a cofactor, so P is added to resolve the "
     "structural inconsistency"),
    ("P depends on the payload structure being consistent"),
    ("add P as a cofactor to resolve the structural inconsistency"),
    ("P was purchased from a commercial supplier"),
    ("the reaction is required to be structurally consistent, so P is added"),
]
leaked = 0
for span in LEAKS:
    ok = seam("P", span, container="modifiers", role="cofactor")
    if ok:
        leaked += 1
    print(f"  {'ADMITTED <<' if ok else 'refused':12s}  {span!r}")
print(f"\n  ADMITTED (want 0): {leaked} / {len(LEAKS)}")

print()
print("  PRESERVATION -- real cofactor evidence, which must all license:")
COFACTOR_OK = [
    "P is a required cofactor for the step",
    "the reaction requires P as a cofactor",
    "the enzyme is dependent on P for activity",
    "the conversion proceeds only in the presence of P",
    "P is the coenzyme of this reaction",
    "P is the prosthetic group of the enzyme",
    "the reaction requires the cofactor P",
]
bad5 = 0
for span in COFACTOR_OK:
    ok = seam("P", span, container="modifiers", role="cofactor")
    if not ok:
        bad5 += 1
    print(f"  {'licensed' if ok else 'REFUSED <<':12s}  {span!r}")
print(f"\n  REFUSED (want 0): {bad5} / {len(COFACTOR_OK)}")

print()
print("=" * 96)
print("BLOCKING 2 COROLLARY -- did the 'other' fallback widen for an UNMAPPED role?")
print("=" * 96)
FALLBACK = [
    ("chaperone", "P is required for the reaction to proceed"),
    ("chaperone", "the reaction requires a chaperone, so P is added"),
    ("scaffold", "the assembly is dependent on the payload being consistent, so P is added"),
    ("adaptor", "P is present in the complex"),
]
widened = 0
for role, span in FALLBACK:
    ok = seam("P", span, container="modifiers", role=role)
    if ok:
        widened += 1
    print(f"  role={role:10s} {'ADMITTED <<' if ok else 'refused':12s}  {span!r}")
print(f"\n  ADMITTED (want 0 -- base behaviour): {widened} / {len(FALLBACK)}")

print()
print("=" * 96)
print("REGISTERED, NOT FIXED -- an actor whose own NAME carries an enzyme noun")
print("=" * 96)
for name, span in [("LpxC hydrolase", "LpxC hydrolase was quantified in the lysate"),
                   ("P", "P was quantified in the lysate")]:
    print(f"  licensed={str(seam(name, span)):5s}  actor={name!r:18s} {span!r}")
