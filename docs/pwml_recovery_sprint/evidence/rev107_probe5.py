"""REV-107 correction-round-2 probe. PYTHONPATH selects the tree.

A1. THE ALTERNATION-ORDERING CLAIM, TESTED DIRECTLY. Every inflection in
    _ATTENUATION_WORD_SRC and in the six anchored inhibition stems, matched
    against itself as a bare word, and against itself embedded in a longer word.
A2. the boundary battery, widened, plus C-105's own unanchored stems as controls
A3. attenuation words that ARE attenuation must still refuse
A4. the 44 matrix and REV-107's own 110 frames
A5. redox / legitimate catalysis preservation
B.  the cofactor route scope

Usage:  <python> rev107_probe5.py <label>
"""
from __future__ import annotations

import re
import sys

from t2pw.curation.apply_audit_patch import (
    apply_patch_with_policy, _ROLE_CUE_RES, _actor_role_family,
)
import t2pw.curation.apply_audit_patch as M

print("code loaded from:", M.__file__)
print("LABEL:", sys.argv[1] if len(sys.argv) > 1 else "?")


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
    _r, rep = apply_patch_with_policy(payload, [op], stage="rev107r2")
    return rep["summary"]["accepted_count"] == 1


# ---------------------------------------------------------------- A1
print()
print("=" * 94)
print("A1 -- THE ALTERNATION-ORDERING CLAIM, TESTED, NOT ARGUED.")
print("=" * 94)

WORDS = ("reduce reduces reduced reducing reduction reductions loss losses "
         "deplete depletes depleted depleting depletion depletions "
         "disrupt disrupts disrupted disrupting disruption disruptions "
         "quench quenches quenched quenching "
         "blockade blockades block blocks blocked blocking "
         "impair impairs impaired impairing impairment impairments "
         "silence silences silenced silencing "
         "sequester sequesters sequestered sequestering sequestration sequestrations "
         "ablate ablates ablated ablating ablation ablations "
         "interfere interferes interfered interfering interference").split()

SRC = getattr(M, "_ATTENUATION_WORD_SRC", None)
if SRC is None:
    SRC = getattr(M, "_ATTENUATION_STEM_SRC")
    print("  (this tree has _ATTENUATION_STEM_SRC, not _ATTENUATION_WORD_SRC)")
ATT = re.compile(SRC)

print(f"\n  {len(WORDS)} declared inflections, each matched against ITSELF:")
bad_self = []
for w in WORDS:
    m = ATT.fullmatch(w) or (ATT.match(w) if ATT.match(w) and ATT.match(w).group(0) == w else None)
    if not m:
        bad_self.append(w)
print(f"    inflections that do NOT match themselves as a whole word: "
      f"{len(bad_self)}  {bad_self}")

print("\n  each inflection EMBEDDED in a longer word (must NOT match):")
SUFFIXES = ("r", "rs", "ase", "ases", "s", "ed", "ing", "ment", "on", "ons",
            "ase complex", "y", "al")
leaks = []
for w in WORDS:
    for suf in SUFFIXES:
        if " " in suf:
            continue
        longer = w + suf
        if longer in WORDS:
            continue
        if ATT.search(longer):
            leaks.append(longer)
print(f"    longer words the pattern still matches inside: {len(leaks)}")
for x in sorted(set(leaks))[:30]:
    print(f"       {x}")

print("\n  the specific words REV-107 measured, and their real-world neighbours:")
NEIGHBOURS = ["reductase", "reductases", "oxidoreductase", "nitroreductase",
              "blocker", "blockers", "silencer", "silencers", "interferon",
              "interferons", "quenchase", "disruptor", "depletor",
              "sequestrant", "ablator", "impairer", "lossless", "blockage"]
for w in NEIGHBOURS:
    m = ATT.search(w)
    flag = "  <<< MATCHES INSIDE" if m else ""
    print(f"     {w:<16} match={m.group(0) if m else None!r}{flag}")

print("\n  the SIX anchored inhibition stems, same two tests:")
INH = _ROLE_CUE_RES["inhibition"]
INH_WORDS = ("blockade blockades impair impairs impaired impairing impairment "
             "impairments silence silences silenced silencing sequestration "
             "sequestrations sequestrate sequestrates sequestrated sequestrating "
             "ablate ablates ablated ablating ablation ablations "
             "interfere interferes interfered interfering interference").split()
missing = [w for w in INH_WORDS if not INH.search(w)]
print(f"    declared inhibition inflections not matched: {len(missing)}  {missing}")
inh_leaks = [w for w in ["silencer", "silencers", "blockader", "impairer",
                         "ablator", "interferon", "interferons", "sequestrant"]
             if INH.search(w)]
print(f"    inhibition pattern still matching inside a longer word: {inh_leaks}")
print(f"    C-105 stems (UNTOUCHED, expected to still leak): "
      f"{[w for w in ['inhibitor', 'suppressor', 'repressor', 'blockage'] if INH.search(w)]}")

# ---------------------------------------------------------------- A2
print()
print("=" * 94)
print("A2 -- THE BOUNDARY BATTERY, WIDENED. All legitimate catalysis: must ACCEPT.")
print("=" * 94)
BOUNDARY = [
    ("reductase adjacent", "P4X", "the reductase P4X catalyses the conversion of A to B"),
    ("nitroreductase adjacent", "NfsB", "the nitroreductase NfsB catalyses the conversion of A to B"),
    ("oxidoreductase adjacent", "YkgC", "the oxidoreductase YkgC catalyses the conversion of A to B"),
    ("reductase own name", "aldo-keto reductase", "aldo-keto reductase catalyses the conversion of A to B"),
    ("reductase + modifier", "NfsB", "the purified reductase NfsB catalyses the conversion of A to B"),
    ("blocker adjacent", "P4X", "the blocker protein P4X catalyses the conversion of A to B"),
    ("silencer adjacent", "P4X", "the silencer complex P4X catalyses the conversion of A to B"),
    ("interferon adjacent", "IRF3", "interferon IRF3 catalyses the conversion of A to B"),
    ("quenchase nonsense", "P4X", "the quenchase P4X catalyses the conversion of A to B"),
    ("CONTROL no stem", "P4X", "the hydrolase P4X catalyses the conversion of A to B"),
    # REV-107 additions this round
    ("reductases plural", "P4X", "the reductases P4X and Q7Y catalyse the conversion of A to B"),
    ("disruptor adjacent", "P4X", "the disruptor protein P4X catalyses the conversion of A to B"),
    ("sequestrant adjacent", "P4X", "the sequestrant P4X catalyses the conversion of A to B"),
    ("blockage adjacent", "P4X", "the blockage assay showed P4X catalyses the conversion of A to B"),
    # C-105's own unanchored stems -- NOT this card's, expected to still refuse
    ("C-105 repressor", "P4X", "the repressor complex P4X catalyses the conversion of A to B"),
    ("C-105 suppressor", "P4X", "the suppressor complex P4X catalyses the conversion of A to B"),
    ("C-105 inhibitor-noun", "P4X", "the inhibitor screen showed P4X catalyses the conversion of A to B"),
]
false_ref = []
for label, name, ev in BOUNDARY:
    got = seam("enzymes", name, ev)
    if not got:
        false_ref.append(label)
    print(f"  {'ACCEPT' if got else 'REFUSE  <<< FALSE REFUSAL'}   {label:<24} {ev[:56]!r}")
this_card = [x for x in false_ref if not x.startswith("C-105")]
print(f"\n  FALSE REFUSALS attributable to THIS CARD: {len(this_card)}  {this_card}")
print(f"  FALSE REFUSALS from C-105's untouched stems: "
      f"{[x for x in false_ref if x.startswith('C-105')]}")

# ---------------------------------------------------------------- A3
print()
print("=" * 94)
print("A3 -- ATTENUATION THAT IS REALLY ATTENUATION MUST STILL REFUSE.")
print("=" * 94)
INFL_FRAMES = [
    "the {w} of NDM-1 is mediated by PSA",
    "the {w} of NDM-1 activity is mediated by PSA",
    "NDM-1 activity showed {w} in the PSA-mediated assay",
]
SAMPLE = ["reduction", "reductions", "reduced", "reduces", "reducing", "reduce",
          "loss", "losses", "depletion", "depleted", "deplete", "depleting",
          "disruption", "disrupted", "disrupts", "quenching", "quenched",
          "blockade", "blockades", "blocked", "blocking", "blocks",
          "impairment", "impaired", "impairs", "silencing", "silenced",
          "sequestration", "sequestered", "ablation", "ablated",
          "interference", "interfering"]
admitted = []
for w in SAMPLE:
    for f in INFL_FRAMES:
        ev = f.replace("{w}", w)
        if seam("enzymes", "NDM-1", ev):
            admitted.append((w, f[:26]))
print(f"  cells: {len(SAMPLE)} words x {len(INFL_FRAMES)} frames = "
      f"{len(SAMPLE) * len(INFL_FRAMES)}")
print(f"  ADMITTED (should be 0): {len(admitted)}")
for a in admitted[:30]:
    print(f"     {a}")

# ---------------------------------------------------------------- A4
print()
print("=" * 94)
print("A4 -- the 44 matrix and REV-107's own 110 frames")
print("=" * 94)
NEAR = ["blockade", "impairment", "disruption", "reduction", "loss", "silencing",
        "sequestration", "depletion", "ablation", "interference", "quenching"]
M44 = [
    ("A object AFTER stem", "the {w} of NDM-1 activity is mediated by PSA"),
    ("B object ABSENT", "the {w} of NDM-1 is mediated by PSA"),
    ("C object BEFORE stem", "NDM-1 activity showed {w} in the PSA-mediated assay"),
    ("D object BEYOND 40",
     "the {w} of NDM-1 " + "x" * 45 + " enzymatic activity is mediated by PSA"),
]
m44_open = [(f.split()[0], w) for f, t in M44 for w in NEAR
            if seam("enzymes", "NDM-1", t.replace("{w}", w))]
print(f"  MATRIX OPEN CELLS: {len(m44_open)} of 44   {m44_open}")

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
    hits = [w for w in NEAR if seam("enzymes", "NDM-1", tmpl.replace("{w}", w))]
    extra_open += [(fname.split()[0], w) for w in hits]
    print(f"  {fname:<26} OPEN {len(hits):>2} of 11   {hits}")
print(f"  EXTRA-FRAME OPEN CELLS: {len(extra_open)} of {len(EXTRA) * 11}")

# ---------------------------------------------------------------- A5
print()
print("=" * 94)
print("A5 -- REDOX / LEGITIMATE CATALYSIS PRESERVATION. All must ACCEPT.")
print("=" * 94)
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
rb = [(n, e[:52]) for n, e in REDOX if not seam("enzymes", n, e)]
for n, e in REDOX:
    got = seam("enzymes", n, e)
    print(f"  {'ACCEPT' if got else 'REFUSE  <<< BROKEN'}   [{n}] {e[:62]!r}")
print(f"\n  REFUSED: {len(rb)} of {len(REDOX)}   {rb}")

# ---------------------------------------------------------------- B
print()
print("=" * 94)
print("B -- the cofactor route scope, re-checked with REV-107 roles and spans")
print("=" * 94)
SPANS = ["NDM-1 is required for the reaction to proceed",
         "the reaction proceeds in the presence of NDM-1",
         "the enzyme requires NDM-1",
         "the reaction is dependent on NDM-1",
         "NDM-1 depends on the payload structure being consistent",
         "the requirement for NDM-1 was demonstrated",
         "the dependence on NDM-1 is absolute",
         "the reaction requires a cofactor, so NDM-1 is added to resolve the structural inconsistency"]
for role in ("chaperone", "scaffold", "allosteric modulator", "subunit"):
    fam = _actor_role_family("modifiers", {"role": role})
    outs = ["A" if seam("modifiers", {"entity": "NDM-1", "role": role,
                                      "evidence": e}, None) else "r" for e in SPANS]
    print(f"  role={role!r:<22} family={fam!r:<8} verdicts={''.join(outs)}"
          f"   (all 'r' == back to base)")
COF = [("PLP", "PLP is the cofactor for ALAS2 in this condensation"),
       ("Zn2+", "the reaction requires Zn2+ as a cofactor"),
       ("PLP", "the enzyme is dependent on PLP for activity"),
       ("PLP", "the condensation proceeds only in the presence of PLP"),
       ("PLP", "PLP is the prosthetic group of the enzyme"),
       ("Zn2+", "the enzyme requires the divalent metal ion Zn2+"),
       ("PLP", "catalysis is dependent upon PLP"),
       ("NAD+", "NAD+ is a coenzyme for the dehydrogenase"),
       ("PLP", "the reaction has an absolute requirement for PLP")]
cb = [e[:46] for n, e in COF
      if not seam("modifiers", {"entity": n, "role": "cofactor", "evidence": e}, None, name=n)]
print(f"  cofactor preservation refused: {len(cb)} of {len(COF)}   {cb}")
for label, e in [("requires-a-cofactor rationale",
                  "the reaction requires a cofactor, so NDM-1 is added to resolve the structural inconsistency"),
                 ("payload-structure rationale",
                  "NDM-1 depends on the payload structure being consistent")]:
    got = seam("modifiers", {"entity": "NDM-1", "role": "cofactor", "evidence": e}, None)
    print(f"  {'ACCEPT <<<' if got else 'refuse'}   {label}")

print()
print("REV107_PROBE5_DONE")
