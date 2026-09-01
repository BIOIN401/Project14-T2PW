"""C-107 correction round 2: the over-refusal battery and the family-scope battery.

BLOCKING A -- _ATTENUATION_STEM_SRC is unanchored on the left and its "[a-z]*"
swallows the rest of a different word, so "reduc" matches inside "reductase",
"block" inside "blocker", "silenc" inside "silencer" and "interfer" inside
"interferon". Legitimate evidenced CATALYSIS spans are then falsely refused --
against EC class 1, whose enzymes are literally named reductase. This is finding
1f's own defect class, reintroduced in the contra written to fix 1a, in the
OVER-REFUSAL direction: the direction C-105 round 1 was rejected for.

BLOCKING B -- nothing pins the SCOPE of the cofactor dependence route. Flipping
its family test to `if True` puts it back on every family with the whole suite
green, which is blocking 2's regression reintroducible for free.

Each case also reports WHICH guard refuses, so a refusal is attributed rather
than guessed at.

Usage::  <python> c107_r2_probe.py <repo-root>
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REPO / "src"))

import t2pw.curation.apply_audit_patch as M  # noqa: E402
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


def attribute(span, actor):
    """Which contra fires on the folded span: the family one, or the actor one."""
    folded = M._match_fold(span)
    inhibition = M._CATALYSIS_CONTRA_RE.search(folded)
    who = []
    if inhibition:
        who.append(f"inhibition-family:{inhibition.group(0)!r}")
    # rebuild the actor-anchored contra the way _span_licenses_actor does
    import re
    needles = M._identifying_match_tokens(actor) or [M._match_fold(actor)]
    stem = getattr(M, "_ATTENUATION_WORD_SRC", None) or M._ATTENUATION_STEM_SRC
    tail = "" if hasattr(M, "_ATTENUATION_WORD_SRC") else r"[a-z]*\b"
    for needle in needles:
        escaped = re.escape(needle)
        f1 = re.compile(
            stem + tail + r"(?:\s+(?:of|in))?"
            r"(?:\s+" + M._PASSIVE_AGENT_MODIFIERS_SRC + r"){0,4}\s+"
            + escaped + r"(?![a-z0-9])"
        )
        f2 = re.compile(
            r"(?<![a-z0-9])" + escaped + r"(?![a-z0-9])"
            r"[^.]{0," + str(M._ATTENUATION_GAP) + r"}?\b"
            + M._ATTENUATION_OBJECT_SRC + r"[a-z]*\b"
            r"[^.]{0," + str(M._ATTENUATION_GAP) + r"}?\b" + stem
        )
        m1, m2 = f1.search(folded), f2.search(folded)
        if m1:
            who.append(f"F1:{m1.group(0)!r}")
        if m2:
            who.append(f"F2:{m2.group(0)!r}")
    return "; ".join(who) or "(no contra -- refused for having no cue)"


# ---------------------------------------------------------------------------
print("=" * 96)
print("BLOCKING A -- legitimate evidenced CATALYSIS spans that must LICENSE")
print("=" * 96)
OVER_REFUSAL = [
    ("P4X", "the reductase P4X catalyses the conversion of A to B"),
    ("NfsB", "the nitroreductase NfsB catalyses the conversion of A to B"),
    ("YkgC", "the oxidoreductase YkgC catalyses the conversion of A to B"),
    ("P4X", "the blocker protein P4X catalyses the conversion of A to B"),
    ("P4X", "the silencer complex P4X catalyses the conversion of A to B"),
    ("IRF3", "interferon IRF3 catalyses the conversion of A to B"),
    ("P4X", "the disulfide reductase P4X catalyses the reduction of the substrate"),
    ("P4X", "P4X, a quinone oxidoreductase, catalyses the conversion of A to B"),
    ("P4X", "the ferredoxin-NADP reductase P4X reduces the flavin cofactor"),
    ("P4X", "the hydrolase P4X catalyses the conversion of A to B"),
]
refused = 0
for actor, span in OVER_REFUSAL:
    ok = seam(actor, span)
    if not ok:
        refused += 1
    print(f"  {'licensed' if ok else 'REFUSED <<':11s}  {span!r}")
    if not ok:
        print(f"      why: {attribute(span, actor)}")
print(f"\n  FALSELY REFUSED (want 0): {refused} / {len(OVER_REFUSAL)}")

print()
print("=" * 96)
print("BLOCKING A -- the refusals that must SURVIVE the repair")
print("=" * 96)
MUST_REFUSE = [
    ("P4X", "the reduction of P4X activity is mediated by Q"),
    ("P4X", "the reduction of P4X is mediated by Q"),
    ("P4X", "P4X activity showed reduction in the Q-mediated assay"),
    ("P4X", "the blockade of P4X activity is mediated by Q"),
    ("P4X", "the silencing of P4X is mediated by Q"),
    ("P4X", "the interference of P4X activity is mediated by Q"),
    ("P4X", "the blocking of P4X is mediated by Q"),
    ("P4X", "the loss of P4X is mediated by Q"),
    ("P4X", "the quenching of P4X is mediated by Q"),
    ("P4X", "the depletion of P4X is mediated by Q"),
    ("P4X", "the disruption of P4X is mediated by Q"),
]
admitted = 0
for actor, span in MUST_REFUSE:
    ok = seam(actor, span)
    if ok:
        admitted += 1
    print(f"  {'ADMITTED <<' if ok else 'refused':12s}  {span!r}")
print(f"\n  ADMITTED (want 0): {admitted} / {len(MUST_REFUSE)}")

print()
print("=" * 96)
print("BLOCKING B -- the cofactor dependence route must reach NO other family")
print("=" * 96)
#: Spans that ONLY the dependence route can license: no catalysis, activation,
#: inhibition or transport cue appears in any of them. Under V9 -- the family
#: test flipped to `if True` -- every one of these is ACCEPTED.
SCOPE = [
    ("catalysis via container", "enzymes", "reactions", None,
     "the enzyme is dependent on P for activity"),
    ("catalysis via container 2", "enzymes", "reactions", None,
     "the reaction requires P as a cofactor"),
    ("catalysis via role", "modifiers", "reactions", "catalyst",
     "the conversion proceeds only in the presence of P"),
    ("inhibition via role", "modifiers", "reactions", "inhibitor",
     "the reaction requires P as a cofactor"),
    ("activation via role", "modifiers", "reactions", "activator",
     "the enzyme is dependent on P for activity"),
    ("transport via container", "transporters", "transports", None,
     "the conversion proceeds only in the presence of P"),
    ("other, unmapped role", "modifiers", "reactions", "chaperone",
     "the reaction requires P as a cofactor"),
    ("other, unmapped role 2", "modifiers", "reactions", "scaffold",
     "the assembly is dependent on P for activity"),
]
leaked = 0
for label, cont, bucket, role, span in SCOPE:
    ok = seam("P", span, container=cont, bucket=bucket, role=role)
    if ok:
        leaked += 1
    print(f"  {label:26s} {'ADMITTED <<' if ok else 'refused':12s}  {span!r}")
print(f"\n  ADMITTED (want 0): {leaked} / {len(SCOPE)}")

print("\n  and the cofactor family itself must still license all of these:")
COFACTOR_OK = [
    "P is a required cofactor for the step",
    "the reaction requires P as a cofactor",
    "the enzyme is dependent on P for activity",
    "the conversion proceeds only in the presence of P",
    "P is the coenzyme of this reaction",
    "P is the prosthetic group of the enzyme",
    "the reaction requires the cofactor P",
]
bad = 0
for span in COFACTOR_OK:
    ok = seam("P", span, container="modifiers", role="cofactor")
    if not ok:
        bad += 1
    print(f"  {'licensed' if ok else 'REFUSED <<':12s}  {span!r}")
print(f"\n  REFUSED (want 0): {bad} / {len(COFACTOR_OK)}")

print()
print("=" * 96)
print("REGISTERED 3 / 4 -- residuals, measured and NOT fixed this round")
print("=" * 96)
for label, actor, span, cont, role in [
    ("R3 requires-subject", "NDM-1",
     "the reaction requires NDM-1 to resolve the structural inconsistency",
     "modifiers", "cofactor"),
    ("R3 dependent-subject", "NDM-1",
     "the structure is dependent on the enzyme NDM-1 being listed",
     "modifiers", "cofactor"),
    ("R4 shared token", "flavin reductase",
     "NADPH-dependent reduction of flavin is required for enzyme function",
     "enzymes", None),
]:
    ok = seam(actor, span, container=cont, role=role)
    print(f"  {label:22s} {'ACCEPTED' if ok else 'refused':9s}  actor={actor!r}")
    print(f"      {span!r}")
