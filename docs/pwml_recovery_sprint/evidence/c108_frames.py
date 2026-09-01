"""C-108: the F-155 class probe, all five members, through the REAL seam.

Every case is driven through ``apply_patch_with_policy`` -- never against a
private regex -- so base and tip are comparable and a verdict here is a verdict
the production gate produces.

Usage::  <python> c108_frames.py <code-root>

Exit code is 0 always; read the printed table. Sections are labelled with the
F-155 member they measure, and each row prints WANT so a base run reads as a
list of what is broken and a tip run as a list of what is fixed.
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(CODE / "src"))

from t2pw.curation.apply_audit_patch import apply_patch_with_policy  # noqa: E402
import t2pw.curation.apply_audit_patch as _m  # noqa: E402

print("code loaded from:", _m.__file__, file=sys.stderr)


def run(container, value, evidence, bucket="reactions", name_for_registry=None):
    nm = name_for_registry
    if nm is None:
        nm = value if isinstance(value, str) else (
            value.get("entity") or value.get("protein")
            or value.get("protein_complex") or "")
    proc = {"name": "A to B", "inputs": ["A"], "outputs": ["B"],
            "evidence": "A is converted in the gut", container: []}
    payload = {"entities": {"compounds": [{"name": "A"}, {"name": "B"}],
                            "proteins": [{"name": nm}], "protein_complexes": [],
                            "nucleic_acids": []},
               "processes": {bucket: [proc]}}
    op = {"op": "add", "path": "/processes/%s/0/%s/-" % (bucket, container),
          "value": value, "confidence": 1.0}
    if evidence is not None:
        op["evidence"] = evidence
    _r, rep = apply_patch_with_policy(payload, [op], stage="c108")
    return rep["summary"]["accepted_count"] == 1


TOTALS = {}


def section(tag, title, cases):
    """cases: (container, bucket, registry_name, value, span, want)"""
    print()
    print("=" * 100)
    print("SECTION %s -- %s" % (tag, title))
    print("=" * 100)
    bad = 0
    for cont, bucket, nm, value, span, want in cases:
        ok = run(cont, value, span, bucket=bucket, name_for_registry=nm)
        if ok != want:
            bad += 1
        print("  %s (want %s) %s  [%s] actor=%r"
              % ("ACCEPT" if ok else "REFUSE",
                 "ACCEPT" if want else "REFUSE",
                 "<< MISMATCH" if ok != want else "           ",
                 cont, nm))
        print("        %r" % (span,))
    print("  %s MISMATCHES (want 0): %d / %d" % (tag, bad, len(cases)))
    TOTALS[tag] = bad
    return bad


def T(nm, span, want, cont="transporters", bucket="transports"):
    return (cont, bucket, nm, nm, span, want)


def E(nm, span, want):
    return ("enzymes", "reactions", nm, nm, span, want)


def M(nm, role, span, want):
    return ("modifiers", "reactions", nm, {"entity": nm, "role": role}, span, want)


# ---------------------------------------------------------------------------
# A -- member (a): does a bare TRANSPORT SCHEMA NOUN self-license?
#
# The promoted-rationale frame _normalize_patch_op actually produces. Every one
# of these is a rationale ARGUING FROM PAYLOAD SHAPE, so every one must REFUSE.
# The whole transport vocabulary is swept, not only the stem the card quotes.
# ---------------------------------------------------------------------------
TRANSPORT_VOCAB = [
    "transport", "transporter", "translocation", "translocator", "import",
    "importer", "export", "exporter", "efflux", "efflux pump", "influx",
    "uptake", "secretion", "shuttle", "permease", "symporter", "antiporter",
    "uniporter", "extruder", "channel", "carrier", "pump", "cargo",
    "cargo complex", "flippase",
]
A_CASES = [
    T("P", "add P as a %s to resolve the structural inconsistency" % w, False)
    for w in TRANSPORT_VOCAB
]

# ---------------------------------------------------------------------------
# A2 -- the same sweep against the CATALYSIS family, as the control C-105 built.
# "enzyme", "enzymatic" and "activity" are already excluded; this asks whether
# the neighbouring schema nouns are.
# ---------------------------------------------------------------------------
CATALYSIS_VOCAB = [
    "enzyme", "enzymatic component", "activity", "catalyst", "catalysts",
    "modifier", "biocatalyst", "hydrolase", "synthase",
]
A2_CASES = [
    E("P", "add P as a %s to resolve the structural inconsistency" % w, False)
    for w in CATALYSIS_VOCAB
]

# ---------------------------------------------------------------------------
# A3 -- member (a) PRESERVATION. The transport VERB is a legitimate cue and
# must keep licensing. If any of these flips, the fix went lexical-by-deletion.
# ---------------------------------------------------------------------------
A3_CASES = [
    T("P", "P transports A across the inner membrane", True),
    T("P", "P transported A into the periplasm", True),
    T("P", "P is transporting A across the membrane", True),
    T("P", "A is transported across the inner membrane by P", True),
    T("P", "P mediates the transport of A across the inner membrane", True),
    T("P", "the transport of A is carried out by P", True),
    T("P", "P is the importer of A", True),
    T("P", "P imports A into the cell", True),
    T("P", "P exports A from the cytoplasm", True),
    T("MsbA", "MsbA is the flippase for lipid A", True),
    T("P", "P is the translocase of the inner membrane", True),
    T("P", "P is the permease for this substrate", True),
    T("P", "P translocates A across the bilayer", True),
    T("P", "P pumps protons across the membrane", True),
    T("P", "P channels calcium into the cytosol", True),
    T("P", "P carries A to the periplasm", True),
    T("P", "P shuttles A between the two compartments", True),
    T("P", "P secretes A into the medium", True),
    T("P", "the efflux of A is driven by P", True),
    T("P", "P drives the uptake of A", True),
    T("P", "P is a symporter for A and sodium", True),
    T("P", "P is the channel through which A crosses the membrane", True),
    T("P", "P is the carrier protein for A in this step", True),
    T("P", "P is the efflux pump for A", True),
    T("P", "P was detected in the membrane fraction", False),
]

# ---------------------------------------------------------------------------
# C -- member (c): an actor whose own NAME contains an enzyme noun.
# ---------------------------------------------------------------------------
C_CASES = [
    E("LpxC hydrolase", "LpxC hydrolase was quantified in the lysate", False),
    E("LpxC synthase", "LpxC synthase was quantified in the lysate", False),
    E("LpxC transferase", "LpxC transferase was detected in the membrane fraction", False),
    E("LpxC hydrolase", "LpxC hydrolase was purchased from a commercial supplier", False),
    E("MurA synthase", "MurA synthase levels were unchanged in the mutant", False),
    E("P kinase", "P kinase was resolved on the gel", False),
    # PREDICATION must still license -- the same names, with a real claim
    E("LpxC hydrolase", "LpxC hydrolase catalyses the conversion of A to B", True),
    E("LpxC hydrolase", "LpxC hydrolase is a hydrolase", True),
    E("LpxC hydrolase", "LpxC hydrolase hydrolyses A to give B", True),
    E("LpxC synthase", "LpxC synthase is the enzyme responsible for this step", True),
    E("MurA synthase", "the conversion of A to B is catalysed by MurA synthase", True),
    E("LpxC hydrolase", "LpxC is the hydrolase for this step", True),
]

# C2 -- the OVER-REFUSAL trap the card names explicitly: an actor whose registry
# name shares a token with the span's only predicating phrase.
C2_CASES = [
    T("inner membrane translocase", "P is the translocase of the inner membrane", True),
    E("acyltransferase complex", "LpxA is the acyltransferase for this step", True),
    E("DNA polymerase I", "DNA polymerase I catalyses the extension", True),
    E("P hydrolase", "P is a hydrolase", True),
    E("P hydrolase", "P is the hydrolase for this step", True),
    E("serine hydroxymethyltransferase",
      "the reaction is catalyzed by serine hydroxymethyltransferase", True),
    E("UDP-N-acetylglucosamine acyltransferase",
      "LpxA, the first enzyme in the pathway, catalyzes the reversible acylation of UDP-GlcNAc",
      True),
    T("P permease", "P is the permease for this substrate", True),
]

# ---------------------------------------------------------------------------
# D -- member (d): C-105's unanchored attenuation stems in the contra.
# ---------------------------------------------------------------------------
D_CASES = [
    E("P4X", "the repressor complex P4X catalyses the conversion of A to B", True),
    E("P4X", "the suppressor protein P4X catalyses the conversion of A to B", True),
    E("P4X", "the inhibitor protein P4X catalyses the conversion of A to B", True),
    E("P4X", "the inhibitor P4X catalyses the conversion of A to B", True),
    E("P4X", "the antagonist P4X catalyses the conversion of A to B", True),
    E("P4X", "the downregulator P4X catalyses the conversion of A to B", True),
    E("P4X", "the inactivator P4X catalyses the conversion of A to B", True),
    E("P4X", "the attenuator protein P4X catalyses the conversion of A to B", True),
    E("P4X", "the abolisher P4X catalyses the conversion of A to B", True),
    E("P4X", "the blocker protein P4X catalyses the conversion of A to B", True),
]

# D2 -- member (d) PRESERVATION. The contra is a BIOLOGICAL GATE; every one of
# these must stay REFUSED, and the inhibition family must keep licensing.
D2_CASES = [
    E("P", "A significantly inhibited P activity in the assay", False),
    E("NDM-1", "PSA significantly inhibited NDM-1 enzyme activity", False),
    E("NDM-1", "PSA-mediated inhibition of NDM-1 activity", False),
    E("NDM-1", "the inhibition of NDM-1 is mediated by PSA", False),
    E("P", "the inhibitor of P was added to the assay", False),
    E("P", "an inhibitor of P blocks the mediated conversion", False),
    E("P", "the repression of P is mediated by Q", False),
    E("P", "the suppression of P activity is mediated by Q", False),
    E("P", "Q suppresses P and mediates the conversion of A to B", False),
    E("P", "Q represses P and mediates the conversion of A to B", False),
    E("P", "P was inactivated before the mediated conversion of A to B", False),
    E("P", "the downregulation of P is mediated by Q", False),
    E("P", "P is attenuated in the mutant, which mediates the conversion", False),
    E("P", "the antagonism of P is mediated by Q", False),
    E("P", "abolishing P activity is mediated by Q", False),
    E("P", "Q blocks P and mediates the conversion of A to B", False),
    M("P", "inhibitor", "P is an inhibitor of X", True),
    M("P", "inhibitor", "P inhibits the reaction", True),
    M("P", "inhibitor", "P is the repressor of the operon", True),
    M("P", "inhibitor", "P suppresses the conversion of A to B", True),
    M("P", "inhibitor", "P is a suppressor of this step", True),
    M("Fur", "inhibitor",
      "holo-Fur binds to the promoters of three ent gene clusters, silencing gene expression",
      True),
]

# ---------------------------------------------------------------------------
# S -- substring collisions the card names, plus the ones a fix can create.
# ---------------------------------------------------------------------------
S_CASES = [
    E("P", "P is the nitroreductase for this step", True),
    E("P", "P is an oxidoreductase acting on the substrate", True),
    E("P", "P is the reductase for this step", True),
    E("P", "P belongs to the reductases of this family", True),
    E("P", "P is a blocker of the channel and catalyses the conversion of A to B", True),
    E("P", "the silencer element upstream of P is mediated by Q", True),
    E("P", "interferon is produced while P catalyses the conversion of A to B", True),
    T("P", "P is the transporter for A in this step", True),
    T("P", "P is one of the transporters of the inner membrane", True),
    E("P", "P is an intermediate carrier in this pathway", False),
    E("P", "P accumulates as one of the intermediates", False),
    E("P", "the photoablation of P was measured while Q mediates the conversion", True),
    E("P", "counterinterference with P was noted while Q mediates the conversion", True),
    E("P", "the microablation of P was recorded while Q mediates the conversion", True),
    E("P", "nonimpairment of P was recorded while Q mediates the conversion", True),
]

# ---------------------------------------------------------------------------
# X -- period-stripped multi-sentence evidence (card item 7).
# ---------------------------------------------------------------------------
X_CASES = [
    E("P", "P catalyses the conversion of A to B. The inhibitor of P was added later.", False),
    E("P", "P catalyses the conversion of A to B. Separately, Q inhibits P in the assay.", False),
    E("P", "P catalyses the conversion of A to B. " + ("x " * 60)
      + "The inhibitor of P was added later.", True),
    T("P", "P was detected in the lysate. add P as a transporter to resolve the "
      "structural inconsistency", False),
]

section("A", "member (a): bare transport SCHEMA NOUN in a promoted rationale -- all must REFUSE",
        A_CASES)
section("A2", "member (a) control: the catalysis family schema nouns -- all must REFUSE",
        A2_CASES)
section("A3", "member (a) PRESERVATION: the transport VERB and real transporter predications",
        A3_CASES)
section("C", "member (c): an actor whose own NAME is an enzyme noun", C_CASES)
section("C2", "member (c) OVER-REFUSAL TRAP: name shares a token with the predicating phrase",
        C2_CASES)
section("D", "member (d): C-105 unanchored attenuation stems -- appositive agent nouns",
        D_CASES)
section("D2", "member (d) PRESERVATION: the contra is a biological gate", D2_CASES)
section("S", "substring collisions", S_CASES)
section("X", "period-stripped multi-sentence evidence", X_CASES)

print()
print("=" * 100)
print("C108 TOTALS  " + "  ".join("%s=%d" % (k, v) for k, v in TOTALS.items()))
print("=" * 100)
