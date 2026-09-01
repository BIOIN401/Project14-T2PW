"""REV-107 correction-round-1 probe. Runs at BASE, at r0 tip 9890770 and at r1
tip b569205; PYTHONPATH selects the tree.

A. the 11 x 4 = 44 frame matrix, reproduced independently
B. FRAMES THE AUTHOR DID NOT ENUMERATE -- the whole finding last round was that
   the fix closed exactly the frames it was written from
C. F1 left boundary: _ATTENUATION_STEM_SRC is unanchored inside actor_contra,
   and "reduc" lives inside "reductase"
D. redox preservation, the author cases and REV-107 additions
E. cofactor dependence and the unmapped-role fallback, with REV-107 roles/spans

Usage:  <python> rev107_probe4.py <label>
"""
from __future__ import annotations

import sys

from t2pw.curation.apply_audit_patch import (
    apply_patch_with_policy, _actor_role_family, _ROLE_CUE_RES, _ANY_ROLE_CUE_RE,
)
import t2pw.curation.apply_audit_patch as M

LABEL = sys.argv[1] if len(sys.argv) > 1 else "?"
print("code loaded from:", M.__file__)
print("LABEL:", LABEL)


def seam(container, value, evidence, bucket="reactions", name=None):
    nm = name
    if nm is None:
        nm = value if isinstance(value, str) else (
            value.get("entity") or value.get("protein") or "")
    proc = {"name": "A to B", "inputs": ["A"], "outputs": ["B"],
            "evidence": "A is converted in the gut", container: []}
    payload = {"entities": {"compounds": [{"name": "A"}, {"name": "B"}],
                            "proteins": [{"name": nm}], "protein_complexes": [],
                            "nucleic_acids": []},
               "processes": {bucket: [proc]}}
    op = {"op": "add", "path": f"/processes/{bucket}/0/{container}/-",
          "value": value, "confidence": 1.0}
    if evidence is not None:
        op["evidence"] = evidence
    _r, rep = apply_patch_with_policy(payload, [op], stage="rev107r1")
    return rep["summary"]["accepted_count"] == 1


NEAR = ["blockade", "impairment", "disruption", "reduction", "loss", "silencing",
        "sequestration", "depletion", "ablation", "interference", "quenching"]

print()
print("=" * 92)
print("A -- the 11 x 4 = 44 matrix. Every cell must REFUSE. Through the real seam.")
print("=" * 92)
FRAMES = [
    ("A object AFTER stem", "the {w} of NDM-1 activity is mediated by PSA"),
    ("B object ABSENT", "the {w} of NDM-1 is mediated by PSA"),
    ("C object BEFORE stem", "NDM-1 activity showed {w} in the PSA-mediated assay"),
    ("D object BEYOND 40",
     "the {w} of NDM-1 " + "x" * 45 + " enzymatic activity is mediated by PSA"),
]
open_cells = []
for fname, tmpl in FRAMES:
    cells = []
    for w in NEAR:
        got = seam("enzymes", "NDM-1", tmpl.replace("{w}", w))
        if got:
            open_cells.append((fname.split()[0], w))
        cells.append(("OPEN" if got else "ref ") + ":" + w[:5])
    print(f"  {fname:<22} " + " ".join(cells))
print(f"\n  MATRIX OPEN CELLS: {len(open_cells)} of 44")
for c in open_cells:
    print(f"     {c}")

print()
print("=" * 92)
print("B -- FRAMES THE AUTHOR DID NOT ENUMERATE (REV-107 own)")
print("=" * 92)
EXTRA = [
    ("E possessive", "NDM-1 activity underwent {w} after PSA-mediated exposure"),
    ("F stem first, actor far",
     "a marked {w} was observed, and it was of NDM-1, mediated by PSA"),
    ("G passive verb form",
     "NDM-1 was subject to {w} by PSA, and the hydrolysis is mediated by it"),
    ("H apposition", "the {w}, of the enzyme NDM-1, is mediated by PSA"),
    ("I two modifiers", "the {w} of the purified recombinant NDM-1 is mediated by PSA"),
    ("J six modifiers",
     "the {w} of the purified recombinant human intestinal bacterial membrane NDM-1 is mediated by PSA"),
    ("K cross-clause", "PSA-mediated catalysis was seen. The {w} of NDM-1 followed"),
    ("L object 100 after actor",
     "NDM-1 " + "y" * 100 + " activity showed {w} in the PSA-mediated assay"),
    ("M no of/in linker", "{w} NDM-1 was mediated by PSA"),
    ("N hyphen linker", "PSA-mediated {w}-of-NDM-1 was observed"),
]
extra_open = []
for fname, tmpl in EXTRA:
    hits = []
    for w in NEAR:
        if seam("enzymes", "NDM-1", tmpl.replace("{w}", w)):
            hits.append(w)
            extra_open.append((fname.split()[0], w))
    print(f"  {fname:<26} OPEN {len(hits):>2} of 11   {hits}")
print(f"\n  EXTRA-FRAME OPEN CELLS: {len(extra_open)} of {len(EXTRA) * 11}")

print()
print("=" * 92)
print("C -- F1 LEFT BOUNDARY. _ATTENUATION_STEM_SRC is unanchored inside")
print("    actor_contra, and 'reduc' lives inside 'reductase'. Every span below")
print("    is LEGITIMATE catalysis and must be ACCEPTED.")
print("=" * 92)
BOUNDARY = [
    ("reductase, name adjacent", "P4X",
     "the reductase P4X catalyses the conversion of A to B"),
    ("nitroreductase adjacent", "NfsB",
     "the nitroreductase NfsB catalyses the conversion of A to B"),
    ("oxidoreductase adjacent", "YkgC",
     "the oxidoreductase YkgC catalyses the conversion of A to B"),
    ("reductase is the own name", "aldo-keto reductase",
     "aldo-keto reductase catalyses the conversion of A to B"),
    ("reductase + modifier", "NfsB",
     "the purified reductase NfsB catalyses the conversion of A to B"),
    ("blocker adjacent", "P4X",
     "the blocker protein P4X catalyses the conversion of A to B"),
    ("silencer adjacent", "P4X",
     "the silencer complex P4X catalyses the conversion of A to B"),
    ("interferon adjacent", "IRF3",
     "interferon IRF3 catalyses the conversion of A to B"),
    ("quenchase nonsense control", "P4X",
     "the quenchase P4X catalyses the conversion of A to B"),
    ("CONTROL: no stem in noun", "P4X",
     "the hydrolase P4X catalyses the conversion of A to B"),
]
boundary_bad = []
for label, name, ev in BOUNDARY:
    got = seam("enzymes", name, ev)
    if not got:
        boundary_bad.append(label)
    print(f"  {'ACCEPT' if got else 'REFUSE  <<< FALSE REFUSAL'}   {label:<28} {ev[:58]!r}")
print(f"\n  FALSE REFUSALS: {len(boundary_bad)} of {len(BOUNDARY)}   {boundary_bad}")

print()
print("=" * 92)
print("D -- REDOX PRESERVATION. All must be ACCEPTED.")
print("=" * 92)
REDOX = [
    ("ferrochelatase", "NADH-dependent reduction of the substrate by ferrochelatase"),
    ("ferrochelatase", "ferrochelatase reduces protoporphyrin IX to heme"),
    ("nitroreductase", "nitroreductase catalyses the reduction of the nitro group"),
    ("thioredoxin", "the reduction of the disulfide bond is mediated by thioredoxin"),
    ("ferredoxin reductase", "reducing equivalents are transferred by ferredoxin reductase"),
    ("enzyme", "the enzyme catalyses the reduction of the substrate level in vitro"),
    ("ferrochelatase", "ferrochelatase reduces the cellular level of protoporphyrin"),
    ("flavin reductase", "NADPH-dependent reduction of flavin is required for enzyme function"),
    ("GshR", "GshR reduces oxidised glutathione using NADPH as electron donor"),
    ("MsrA", "MsrA catalyses the reduction of methionine sulfoxide residues"),
    ("TrxB", "the reduction of the substrate is catalysed by TrxB in the presence of NADPH"),
    ("NfsB", "NfsB, a nitroreductase, reduces the nitro group of the prodrug"),
    ("Fdx", "Fdx supplies reducing power for the hydroxylation step"),
    ("LpxC", "the disruption of the substrate ring is catalysed by LpxC"),
    ("MenD", "MenD catalyses the cleavage and loss of the pyruvate leaving group"),
]
redox_bad = []
for name, ev in REDOX:
    got = seam("enzymes", name, ev)
    if not got:
        redox_bad.append((name, ev[:56]))
    print(f"  {'ACCEPT' if got else 'REFUSE  <<< BROKEN'}   [{name}] {ev[:64]!r}")
print(f"\n  REDOX/CATALYSIS REFUSED: {len(redox_bad)} of {len(REDOX)}")
for r in redox_bad:
    print(f"     {r}")

print()
print("=" * 92)
print("E -- BLOCKING 2: cofactor dependence and the unmapped-role fallback")
print("=" * 92)
pat = _ROLE_CUE_RES["cofactor"].pattern if "cofactor" in _ROLE_CUE_RES else "(absent)"
print(f"  cofactor family pattern: {pat!r}")
print("  are the loose dependence terms still reachable from _ANY_ROLE_CUE_RE?")
for t in ["requires", "required for", "depends on", "dependent on",
          "in the presence of", "requirement for", "dependence on"]:
    print(f"     {t:<20} _ANY_ROLE_CUE_RE match: {bool(_ANY_ROLE_CUE_RE.search(t))}")

print()
print("  E1 -- the rationale spans REV-107 raised (role=cofactor).")
RATIONALE = [
    ("requires-a-cofactor rationale",
     "the reaction requires a cofactor, so NDM-1 is added to resolve the structural inconsistency"),
    ("payload-structure rationale",
     "NDM-1 depends on the payload structure being consistent"),
    ("bare required-for", "NDM-1 is required for the reaction to proceed"),
    ("bare dependent-on", "the reaction is dependent on NDM-1"),
    ("bare presence-of", "the decomposition proceeds in the presence of NDM-1"),
    ("as-a-cofactor rationale",
     "add NDM-1 as a cofactor to resolve the structural inconsistency"),
    ("REV-107 new: requires-to-resolve",
     "the reaction requires NDM-1 to resolve the structural inconsistency"),
    ("REV-107 new: dependent-added",
     "the structure is dependent on the enzyme NDM-1 being listed"),
]
for label, ev in RATIONALE:
    got = seam("modifiers", {"entity": "NDM-1", "role": "cofactor", "evidence": ev}, None)
    print(f"  {'ACCEPT' if got else 'refuse'}   {label:<32} {ev[:62]!r}")

print()
print("  E2 -- cofactor PRESERVATION. All must ACCEPT.")
COFPRES = [
    ("PLP", "PLP is the cofactor for ALAS2 in this condensation"),
    ("Zn2+", "the reaction requires Zn2+ as a cofactor"),
    ("PLP", "the enzyme is dependent on PLP for activity"),
    ("PLP", "the condensation proceeds only in the presence of PLP"),
    ("PLP", "PLP is the prosthetic group of the enzyme"),
    ("Zn2+", "the enzyme requires the divalent metal ion Zn2+"),
    ("PLP", "catalysis is dependent upon PLP"),
    ("NAD+", "NAD+ is a coenzyme for the dehydrogenase"),
    ("PLP", "the reaction has an absolute requirement for PLP"),
]
cof_bad = []
for nm, ev in COFPRES:
    got = seam("modifiers", {"entity": nm, "role": "cofactor", "evidence": ev}, None, name=nm)
    if not got:
        cof_bad.append(ev[:52])
    print(f"  {'ACCEPT' if got else 'REFUSE  <<<'}   [{nm}] {ev[:62]!r}")
print(f"  COFACTOR PRESERVATION REFUSED: {len(cof_bad)} of {len(COFPRES)}   {cof_bad}")

print()
print("  E3 -- THE UNMAPPED-ROLE FALLBACK, with REV-107 roles and REV-107 spans.")
MYROLES = ["chaperone", "scaffold", "adaptor", "regulator", "subunit",
           "allosteric modulator", "coenzyme", "metal", "substrate", "product"]
for r in MYROLES:
    print(f"     role={r!r:<24} family={_actor_role_family('modifiers', {'role': r})!r}")
MYSPANS = [
    "NDM-1 is required for the reaction to proceed",
    "the reaction proceeds in the presence of NDM-1",
    "the enzyme requires NDM-1",
    "the reaction is dependent on NDM-1",
    "catalysis depends on NDM-1",
    "NDM-1 depends on the payload structure being consistent",
    "the requirement for NDM-1 was demonstrated",
    "the dependence on NDM-1 is absolute",
    "NDM-1 was detected in the assay",
    "add NDM-1 as a chaperone to resolve the structural inconsistency",
    "NDM-1 is the cofactor for the reaction",
    "NDM-1 catalyses the hydrolysis of meropenem",
]
for role in ("chaperone", "scaffold", "allosteric modulator"):
    print(f"     --- role={role!r} (family 'other') ---")
    for ev in MYSPANS:
        got = seam("modifiers", {"entity": "NDM-1", "role": role, "evidence": ev}, None)
        print(f"       {'ACCEPT' if got else 'refuse'}   {ev[:70]!r}")

print()
print("REV107_PROBE4_DONE")
