"""C-107: the 29-case battery, plus this card's own pinned gates.

SECTION A is REV-105's B1/B2/B3 battery, reproduced case-for-case from the
reviewer's own instrument so the number stays comparable across SHAs. C-105's
approved tip scores **1 mismatch of 29**. C-107 may not make that worse.

SECTION B is the F-146 pinned safety property, in the exact shape C-105's test
file pins it, run through the real seam. It must print REJECTED at every
intermediate state of this card.

SECTION C is C-107's own rejection/preservation battery: the eleven
near-synonyms in a shape that carries a REAL catalysis cue (the shape a
word-level probe misses), redox preservation, the passive agent, the -ase
stoplist, transport and cofactor.

Usage::  <python> c107_battery.py <repo-root>
Exit code is 0 always; read the printed counts.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REPO / "src"))

from t2pw.curation.apply_audit_patch import apply_patch_with_policy  # noqa: E402


def run(container, value, evidence, bucket="reactions", conf=1.0, name_for_registry=None):
    nm = name_for_registry
    if nm is None:
        nm = value if isinstance(value, str) else (
            value.get("entity") or value.get("protein") or value.get("protein_complex") or "")
    proc = {"name": "A to B", "inputs": ["A"], "outputs": ["B"],
            "evidence": "A is converted in the gut", container: []}
    payload = {"entities": {"compounds": [{"name": "A"}, {"name": "B"}],
                            "proteins": [{"name": nm}], "protein_complexes": [],
                            "nucleic_acids": []},
               "processes": {bucket: [proc]}}
    op = {"op": "add", "path": f"/processes/{bucket}/0/{container}/-",
          "value": value, "confidence": conf}
    if evidence is not None:
        op["evidence"] = evidence
    _r, rep = apply_patch_with_policy(payload, [op], stage="probe")
    ok = rep["summary"]["accepted_count"] == 1
    reason = rep["rejected"][0]["reason"] if rep["rejected"] else ""
    return ok, reason


# ---------------------------------------------------------------------------
# SECTION A -- REV-105's battery, verbatim in content and order.
# ---------------------------------------------------------------------------
CASES = [
 ("author's own catalysis span",           "enzymes", "P", "P catalyses the conversion of A to B under physiological conditions", True),
 ("hyphenated adjectival: P-catalyzed",    "enzymes", "P", "P-catalyzed conversion of A to B is the rate-limiting step", True),
 ("hyphenated adjectival: P-mediated",     "enzymes", "P", "P-mediated hydrolysis of A yields B", True),
 ("'is the enzyme responsible for'",       "enzymes", "P", "P is the enzyme responsible for the decomposition of A into B", True),
 ("passive: 'A is converted to B by P'",   "enzymes", "P", "A is converted to B by P in the intestine", True),
 ("passive: 'catalysed by P'",             "enzymes", "P", "The reaction is catalysed by P", True),
 ("'P converts A into B'",                 "enzymes", "P", "The enzyme P converts A into B", True),
 ("'P hydrolyses A'",                      "enzymes", "P", "P hydrolyses A to give B", True),
 ("'P acts on A to give B'",               "enzymes", "P", "P acts on A to give B", True),
 ("'P breaks down A'",                     "enzymes", "P", "P breaks down A into B", True),
 ("'P is an enzyme that acts upon A'",     "enzymes", "P", "P is an enzyme that acts upon A", True),
 ("'P was shown to catalyse this step'",   "enzymes", "P", "P was shown to catalyse this step", True),
 ("multi-word name, exact in span",        "enzymes", "DNA polymerase I", "DNA polymerase I catalyses the extension", True),
 ("registry name vs paper symbol",         "enzymes", "MenD complex", "MenD catalyses the first irreversible step", True),
 ("systematic name vs gene symbol",        "enzymes", "UDP-N-acetylglucosamine acyltransferase",
                                            "LpxA, the first enzyme in the pathway, catalyzes the reversible acylation of UDP-GlcNAc", True),
 ("' complex' suffix, symbol in span",     "enzymes", "ALAS2 complex", "ALAS2 mediates the condensation of glycine", True),
 ("hyphenated NAME, exact in span",        "enzymes", "NDM-1", "NDM-1 hydrolyses the beta-lactam ring", True),
 ("hyphenated NAME + hyphenated cue",      "enzymes", "NDM-1", "NDM-1-catalyzed hydrolysis of the beta-lactam ring", True),
 ("cue >80 chars from the name",           "enzymes", "P", "P " + ("x" * 100) + " catalyses the conversion", False),
 ("dict row w/ own evidence",              "enzymes", {"protein": "P", "evidence": "P catalyses the step"}, None, True),
 ("transport, plain",                      "transporters", "P", "P transports A across the inner membrane", True, "transports"),
 ("transport, 'P is the importer of A'",   "transporters", "P", "P is the importer of A", True, "transports"),
 ("modifier inhibitor, evidenced",         "modifiers", {"entity": "P", "role": "inhibitor", "evidence": "P inhibits the reaction"}, None, True),
 ("modifier activator, evidenced",         "modifiers", {"entity": "P", "role": "activator", "evidence": "P activates the pathway"}, None, True),
 ("modifier cofactor (role outside vocab)", "modifiers", {"entity": "P", "role": "cofactor", "evidence": "P is a required cofactor for the step"}, None, True),
 ("B2 DEFECT: inhibitor span for enzymes", "enzymes", "P", "A significantly inhibited P enzyme activity", False),
 ("B3 structural rationale",               "enzymes", "P", "add P as an enzyme to resolve the structural inconsistency where an inhibitor is listed without a target enzyme", False),
 ("no evidence at all",                    "enzymes", "P", None, False),
 ("span names a DIFFERENT protein",        "enzymes", "P", "Q catalyses the conversion of A to B", False),
]

print("=" * 100)
print("SECTION A -- REV-105's 29-case battery, through apply_patch_with_policy")
print("=" * 100)
bad = 0
mismatches = []
for c in CASES:
    label, cont, val, ev, expect = c[0], c[1], c[2], c[3], c[4]
    bucket = c[5] if len(c) > 5 else "reactions"
    ok, reason = run(cont, val, ev, bucket=bucket)
    flag = "  " if ok == expect else "<< MISMATCH"
    if ok != expect:
        bad += 1
        mismatches.append(label)
    print(f"{'ACCEPT' if ok else 'REFUSE':<7} (want {'ACCEPT' if expect else 'REFUSE':<6}) {flag}  {label}")
    if not ok:
        print(f"           reason: {reason[:150]}")
print()
print(f"BATTERY MISMATCHES: {bad} / {len(CASES)}")
print(f"BATTERY MISMATCH LABELS: {mismatches}")

# ---------------------------------------------------------------------------
# SECTION B -- the F-146 pinned safety property.
# ---------------------------------------------------------------------------
print()
print("=" * 100)
print("SECTION B -- F-146 PINNED PATCH.  Must print REJECTED.")
print("=" * 100)
F146_RATIONALE = ("add NDM-1 as an enzyme to the decomposition reaction to resolve the "
                  "structural inconsistency where an inhibitor is listed without a target enzyme")


def f146():
    payload = {
        "entities": {"compounds": [{"name": "phthalylsulfacetamide"}, {"name": "sulfacetamide"}],
                     "proteins": [{"name": "NDM-1"}], "protein_complexes": [], "nucleic_acids": []},
        "processes": {"reactions": [{
            "name": "phthalylsulfacetamide decomposition",
            "inputs": ["phthalylsulfacetamide"], "outputs": ["sulfacetamide"],
            "enzymes": [],
            "modifiers": [{"entity": "NDM-1", "role": "inhibitor",
                           "evidence": "PSA significantly inhibited NDM-1 enzyme activity"}],
            "evidence": "PSA is decomposed in the intestine, resulting in an antibacterial effect",
        }]},
    }
    op = {"op": "add", "path": "/processes/reactions/0/enzymes/-", "value": "NDM-1",
          "confidence": 1.0, "evidence": F146_RATIONALE}
    result, rep = apply_patch_with_policy(payload, [op], stage="audit")
    return rep, result


rep, result = f146()
acc = rep["summary"]["accepted_count"]
print(f"  accepted_count = {acc}   enzymes after = "
      f"{result['processes']['reactions'][0]['enzymes']!r}")
print(f"  reason         = {rep['rejected'][0]['reason'][:160] if rep['rejected'] else '(none)'}")
print(f"  F-146 VERDICT  : {'REJECTED  <-- REQUIRED' if acc == 0 else 'ADMITTED  <-- CARD FAILS'}")

# ---------------------------------------------------------------------------
# SECTION C -- C-107's own gates.
# ---------------------------------------------------------------------------
NEAR_SYNONYMS = ["blockade", "impairment", "disruption", "reduction", "loss",
                 "silencing", "sequestration", "depletion", "ablation",
                 "interference", "quenching"]

print()
print("=" * 100)
print("SECTION C1 -- 1a REJECTION: each near-synonym in a window that ALSO carries a")
print("real catalysis cue ('mediated'). At the C-106 base every one of these is ADMITTED.")
print("=" * 100)
c1_bad = 0
for w in NEAR_SYNONYMS:
    span = f"the {w} of NDM-1 activity is mediated by PSA"
    ok, reason = run("enzymes", "NDM-1", span)
    if ok:
        c1_bad += 1
    print(f"  {w:16s} {'ADMITTED <<' if ok else 'refused'}   {span!r}")
print(f"  C1 ADMITTED (want 0): {c1_bad} / {len(NEAR_SYNONYMS)}")

print()
print("=" * 100)
print("SECTION C2 -- 1a PRESERVATION: redox chemistry must keep licensing (REV-107 B2)")
print("=" * 100)
REDOX = [
    ("P", "NADH-dependent reduction of the substrate by P"),
    ("ferrochelatase", "ferrochelatase reduces the substrate in this step"),
    ("P", "P catalyses the reduction of the quinone to the quinol"),
    ("P", "the reduction of A to B is carried out by P"),
    ("P", "P reduces nitrite to nitric oxide"),
    ("P", "reducing equivalents are transferred by P during the reduction of the disulfide"),
]
c2_bad = 0
for name, span in REDOX:
    ok, reason = run("enzymes", name, span)
    if not ok:
        c2_bad += 1
    print(f"  {'licensed' if ok else 'REFUSED <<':11s}  {span!r}")
    if not ok:
        print(f"      reason: {reason[:140]}")
print(f"  C2 REFUSED (want 0): {c2_bad} / {len(REDOX)}")

print()
print("=" * 100)
print("SECTION C3 -- 1b passive-with-agent: the agent must BE the actor")
print("=" * 100)
C3 = [
    ("P", "A is converted to B by Q, and P was also detected in the assay", False),
    ("P", "A is produced by Q while P remained bound to the membrane", False),
    ("P", "B is formed by Q; P is unrelated to this step", False),
    ("P", "A is converted to B by P in the intestine", True),
    ("P", "A is produced by P during the second step", True),
    ("Serine hydroxymethyltransferase, mitochondrial",
     "the reaction is catalyzed by serine hydroxymethyltransferase", True),
    ("EntB", "isochorismate is converted to 2,3-dihydro-2,3-dihydroxybenzoate and pyruvate by EntB isochorismatase activity", True),
    ("NDM-1", "the beta-lactam ring is converted to the open form by NDM-1", True),
]
c3_bad = 0
for name, span, want in C3:
    ok, reason = run("enzymes", name, span)
    if ok != want:
        c3_bad += 1
    print(f"  {'ACCEPT' if ok else 'REFUSE'} (want {'ACCEPT' if want else 'REFUSE'}) "
          f"{'<< MISMATCH' if ok != want else '           '}  {span!r}")
print(f"  C3 MISMATCHES (want 0): {c3_bad} / {len(C3)}")

print()
print("=" * 100)
print("SECTION C4 -- 1c the -ase stoplist: plural bypass closed, real enzymes preserved")
print("=" * 100)
C4 = [
    ("LpxA", "LpxA appears in three purchases recorded in the supplement", False),
    ("LpxA", "LpxA appears beside two showcases in the exhibition", False),
    ("LpxA", "LpxA was photographed on the staircases of the institute", False),
    ("LpxA", "LpxA was left in one of the briefcases", False),
    ("LpxA", "LpxA pleases the reviewers of this manuscript", False),
    ("LpxA", "LpxA was found in the suitcases of the courier", False),
    ("LpxA", "LpxA was noted beside the grease on the bench", False),
    ("LpxA", "LpxA paraphrases the earlier report", False),
    ("LpxA", "LpxA was noted while the incidence of disease did not increase after release of the database", False),
    ("LpxA", "LpxA is the acyltransferase for this step", True),
    ("P", "P and its paralogues are hydrolases of the same family", True),
    ("P", "P belongs to the kinases described earlier", True),
    ("P", "P is the lyase for this step", True),
    ("EndA", "EndA is the DNase for this step", True),
]
c4_bad = 0
for name, span, want in C4:
    ok, _reason = run("enzymes", name, span)
    if ok != want:
        c4_bad += 1
    print(f"  {'ACCEPT' if ok else 'REFUSE'} (want {'ACCEPT' if want else 'REFUSE'}) "
          f"{'<< MISMATCH' if ok != want else '           '}  {span!r}")
print(f"  C4 MISMATCHES (want 0): {c4_bad} / {len(C4)}")

print()
print("=" * 100)
print("SECTION C5 -- 1d transport, and 1e cofactor")
print("=" * 100)
C5 = [
    ("transporters", "transports", "MsbA", "MsbA is the flippase for lipid A", True),
    ("transporters", "transports", "P", "P is the translocase of the inner membrane", True),
    ("transporters", "transports", "P", "P is the permease for this substrate", True),
    ("transporters", "transports", "P", "P transports A across the inner membrane", True),
    ("transporters", "transports", "P", "P was detected in the membrane fraction", False),
    ("transporters", "transports", "P", "add P as a transporter to resolve the structural inconsistency", False),
    ("transporters", "transports", "P", "A significantly inhibited P activity in the assay", False),
]
c5_bad = 0
for cont, bucket, name, span, want in C5:
    ok, _reason = run(cont, name, span, bucket=bucket)
    if ok != want:
        c5_bad += 1
    print(f"  {'ACCEPT' if ok else 'REFUSE'} (want {'ACCEPT' if want else 'REFUSE'}) "
          f"{'<< MISMATCH' if ok != want else '           '}  {span!r}")

COFACTOR = [
    ("P is a required cofactor for the step", True),
    ("the reaction requires P as a cofactor", True),
    ("the enzyme is dependent on P for activity", True),
    ("the conversion proceeds only in the presence of P", True),
    ("P is the coenzyme of this reaction", True),
    ("P was purchased from a commercial supplier", False),
    ("add P as a cofactor to resolve the structural inconsistency", False),
]
for span, want in COFACTOR:
    ok, _reason = run("modifiers", {"entity": "P", "role": "cofactor"}, span)
    if ok != want:
        c5_bad += 1
    print(f"  {'ACCEPT' if ok else 'REFUSE'} (want {'ACCEPT' if want else 'REFUSE'}) "
          f"{'<< MISMATCH' if ok != want else '           '}  cofactor: {span!r}")
print(f"  C5 MISMATCHES (want 0): {c5_bad} / {len(C5) + len(COFACTOR)}")

print()
print("=" * 100)
print("SECTION C6 -- 1f 'mediat' anchoring")
print("=" * 100)
C6 = [
    ("EntB", "EntB is an intermediate carrier in this pathway", False),
    ("EntB", "EntB accumulates as one of the intermediates", False),
    ("ALAS2 complex", "ALAS2 mediates the condensation of glycine and succinyl-CoA", True),
    ("P", "P-mediated hydrolysis of A yields B", True),
    ("P", "the step is mediated by P", True),
]
c6_bad = 0
for name, span, want in C6:
    ok, _reason = run("enzymes", name, span)
    if ok != want:
        c6_bad += 1
    print(f"  {'ACCEPT' if ok else 'REFUSE'} (want {'ACCEPT' if want else 'REFUSE'}) "
          f"{'<< MISMATCH' if ok != want else '           '}  {span!r}")
print(f"  C6 MISMATCHES (want 0): {c6_bad} / {len(C6)}")

print()
print("=" * 100)
print(f"TOTALS  battery={bad}/29  F146={'REJECTED' if acc == 0 else 'ADMITTED'}  "
      f"C1={c1_bad} C2={c2_bad} C3={c3_bad} C4={c4_bad} C5={c5_bad} C6={c6_bad}")
print("=" * 100)
