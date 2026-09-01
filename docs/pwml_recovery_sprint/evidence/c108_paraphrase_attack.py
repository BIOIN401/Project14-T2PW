"""C-108: ATTACK THIS CARD'S OWN FIX WITH PARAPHRASES.

C-107 round 1 bound its fix to the wording its card quoted and left 15 of 44
routes open. The audit stage regenerates its rationale every round, so a rephrase
is not hypothetical. This probe rewrites every frame C-108 closes into the ways a
model would actually say the same thing, and prints the verdict at BOTH SHAs, so
a route that was already open is not reported as one this card opened.

Every case is driven through the real seam ``apply_patch_with_policy``.

Usage::  <python> c108_paraphrase_attack.py <base-code-root> <tip-code-root>

Read the table. ``want`` is what the class requires; ``base`` and ``tip`` are
measured. A row where base and tip both miss the want is a route this card did
NOT close and it is reported as such rather than omitted.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

BASE = Path(sys.argv[1]).resolve()
TIP = Path(sys.argv[2]).resolve()


def load(root, alias):
    path = root / "src" / "t2pw" / "curation" / "apply_audit_patch.py"
    saved = list(sys.path)
    sys.path.insert(0, str(root / "src"))
    for name in [m for m in list(sys.modules) if m.startswith("t2pw")]:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


BASE_M = load(BASE, "c108_base_mod")
TIP_M = load(TIP, "c108_tip_mod")
print("base:", BASE_M.__file__, file=sys.stderr)
print("tip :", TIP_M.__file__, file=sys.stderr)


def verdict(mod, container, bucket, name, value, span):
    proc = {"name": "A to B", "inputs": ["A"], "outputs": ["B"],
            "evidence": "A is converted in the gut", container: []}
    payload = {"entities": {"compounds": [{"name": "A"}, {"name": "B"}],
                            "proteins": [{"name": name}], "protein_complexes": [],
                            "nucleic_acids": []},
               "processes": {bucket: [proc]}}
    op = {"op": "add", "path": "/processes/%s/0/%s/-" % (bucket, container),
          "value": value, "confidence": 1.0, "evidence": span}
    _r, rep = mod.apply_patch_with_policy(payload, [op], stage="c108attack")
    return rep["summary"]["accepted_count"] == 1


def T(nm, span, want):
    return ("transporters", "transports", nm, nm, span, want)


def E(nm, span, want):
    return ("enzymes", "reactions", nm, nm, span, want)


def M(nm, role, span, want):
    return ("modifiers", "reactions", nm, {"entity": nm, "role": role}, span, want)


SECTIONS = [
    ("(a) the transport schema noun, rephrased", [
        T("P", "add P as the transporter to resolve the structural inconsistency", False),
        T("P", "add P as transporters to resolve the structural inconsistency", False),
        T("P", "P should be added as a transporter to resolve the inconsistency", False),
        T("P", "P is added as a transporter so the payload is consistent", False),
        T("P", "adding P as a transporter resolves the structural inconsistency", False),
        T("P", "listing P among the transporters resolves the inconsistency", False),
        T("P", "the payload lists no transporter, so P is added", False),
        T("P", "P, a transporter, is added to resolve the structural inconsistency", False),
        T("P", "assign P the transporter role to resolve the inconsistency", False),
        T("P", "register P as a channel to resolve the structural inconsistency", False),
        T("P", "include P as a carrier so the transport step has an actor", False),
        T("P", "P is required as a pump for the payload to validate", False),
        # the predications the same rephrasing must NOT break
        T("P", "P acts as a transporter of A across the inner membrane", True),
        T("P", "P functions as the channel for calcium entry", True),
        T("P", "P serves as a carrier for A in this step", True),
        T("P", "P was the transporter for A in the reconstituted system", True),
        T("P", "P is a transporter", True),
        T("P", "P is the efflux pump for A", True),
    ]),
    ("(a) the catalysis schema noun, rephrased", [
        E("P", "add P as the catalyst to resolve the structural inconsistency", False),
        E("P", "add P to the catalysts so the reaction has an actor", False),
        E("P", "P is proposed as a catalyst to resolve the inconsistency", False),
        E("P", "P is a catalyst", True),
        E("P", "P is the catalyst for this step", True),
        E("P", "P acts as a catalyst in the conversion of A to B", True),
    ]),
    ("(c) the actor name supplying its own cue, rephrased", [
        E("LpxC hydrolase", "the hydrolase LpxC was quantified in the lysate", False),
        E("LpxC hydrolase", "LpxC hydrolase levels were unchanged in the mutant", False),
        E("LpxC hydrolase", "we quantified LpxC hydrolase in the lysate", False),
        E("LpxC hydrolase", "the LpxC hydrolase band was excised from the gel", False),
        E("LpxC hydrolase", "LpxC hydrolase, LpxD and LpxA were all detected", False),
        E("LpxC hydrolase", "LpxC hydrolase is present in the complex", False),
        E("LpxC hydrolase", "purified LpxC hydrolase was stored at minus 80", False),
        E("LpxC hydrolase", "LpxC (a hydrolase) was quantified in the lysate", False),
        E("LpxC hydrolase", "LpxC, a hydrolase, was quantified in the lysate", False),
        # and the predications it must not break
        E("LpxC hydrolase", "LpxC hydrolase catalyses the conversion of A to B", True),
        E("LpxC hydrolase", "A is converted to B by LpxC hydrolase", True),
        E("LpxC hydrolase", "the conversion is catalysed by the LpxC hydrolase", True),
        E("LpxC hydrolase", "LpxC hydrolase is the enzyme responsible for this step", True),
    ]),
    ("(d) the appositive agent noun, rephrased", [
        E("P4X", "the inhibitor P4X catalyses the conversion of A to B", True),
        E("P4X", "P4X, the inhibitor, catalyses the conversion of A to B", True),
        E("P4X", "the potent inhibitor P4X catalyses the conversion of A to B", True),
        E("P4X", "inhibitor P4X catalyses the conversion of A to B", True),
        E("P4X", "the repressor P4X hydrolyses A to give B", True),
        E("P4X", "the suppressor P4X mediates the conversion of A to B", True),
        E("P4X", "the antagonist P4X converts A into B", True),
        # and the TARGET readings that must all stay refused
        E("P", "the inhibitor of P was added while Q mediates the conversion", False),
        E("P", "inhibitors of P were added while Q mediates the conversion", False),
        E("P", "the inhibitor for P was added while Q mediates the conversion", False),
        E("P", "an inhibitor selective for P was added while Q mediates", False),
        E("P", "the inhibitor targeting P was added while Q mediates", False),
        E("P", "the inhibitor directed against P was added while Q mediates", False),
        E("P", "the suppressor acting on P was added while Q mediates", False),
        E("P", "the P inhibitor was added while Q mediates the conversion", False),
        E("P", "the P specific inhibitor was added while Q mediates", False),
        E("P", "the P activity inhibitor was added while Q mediates", False),
        E("P", "the repressor of the P gene was deleted while Q mediates", False),
        E("P", "an inhibitory effect on P was seen while Q mediates the conversion", False),
        E("P", "P inhibitors were used while Q mediates the conversion", False),
        # and the inhibition CUE, which this card does not touch
        M("P", "inhibitor", "P is an inhibitor of X", True),
        M("P", "inhibitor", "P is a suppressor of the operon", True),
        M("P", "inhibitor", "P is the repressor of the operon", True),
        M("P", "inhibitor", "P is an antagonist of the receptor", True),
    ]),
    ("(b) period-stripped multi-sentence evidence", [
        E("P", "P catalyses the conversion of A to B. The inhibitor of P was added.", False),
        E("P", "The inhibitor of P was added. P catalyses the conversion of A to B.", False),
        E("P", "P catalyses the conversion of A to B.\nThe inhibitor of P was added.", False),
        E("P", "P catalyses the conversion of A to B. " + ("x " * 60)
          + "The inhibitor of P was added.", True),
        E("P", "P catalyses the conversion of A to B", True),
    ]),
    ("substring collisions, including any this card creates", [
        E("P", "P is the reductase for this step", True),
        E("P", "P is the nitroreductase for this step", True),
        E("P", "P is an oxidoreductase acting on the substrate", True),
        E("P", "the silencer element upstream of P is mediated by Q", True),
        E("P", "interferon is produced while P catalyses the conversion of A to B", True),
        E("P", "P has an inhibitory role and Q catalyses the conversion", False),
        E("P", "the attenuation of P is mediated by Q", False),
        E("P", "P is an intermediate carrier in this pathway", False),
        # the anchors this card ADDS: a stem must still reach its other inflections
        E("P", "the inhibition of P is mediated by Q", False),
        E("P", "the repression of P is mediated by Q", False),
        E("P", "the antagonism of P is mediated by Q", False),
        E("P", "P was inactivated before the mediated conversion", False),
        E("P", "the downregulation of P is mediated by Q", False),
        E("P", "abolishing P activity is mediated by Q", False),
        # and "transport" must not have lost its verb or its event noun
        T("P", "P transports A across the inner membrane", True),
        T("P", "the transport of A is carried out by P", True),
        T("P", "P translocates A across the bilayer", True),
        T("P", "the translocation of A is driven by P", True),
        T("P", "P imports A into the cell", True),
        T("P", "P exports A from the cytoplasm", True),
        T("P", "P channels calcium into the cytosol", True),
        T("P", "P pumps protons across the membrane", True),
    ]),
]

print()
opened_by_this_card = 0
still_open = 0
total = 0
for title, cases in SECTIONS:
    print("=" * 100)
    print("SECTION -- " + title)
    print("=" * 100)
    for cont, bucket, nm, value, span, want in cases:
        b = verdict(BASE_M, cont, bucket, nm, value, span)
        t = verdict(TIP_M, cont, bucket, nm, value, span)
        total += 1
        if t != want:
            if b == want:
                flag = "<< REGRESSION -- this card broke it"
                opened_by_this_card += 1
            else:
                flag = "<< STILL OPEN at base and tip -- NOT closed by this card"
                still_open += 1
        else:
            flag = "OK" if b == want else "<< CLOSED BY THIS CARD"
        print("  base=%-6s tip=%-6s want=%-6s  %s"
              % ("ACCEPT" if b else "REFUSE",
                 "ACCEPT" if t else "REFUSE",
                 "ACCEPT" if want else "REFUSE", flag))
        print("        [%s] actor=%r  %r" % (cont, nm, span))
    print()

print("=" * 100)
print("PARAPHRASE ATTACK TOTALS  cases=%d  regressions_introduced=%d  still_open=%d"
      % (total, opened_by_this_card, still_open))
print("=" * 100)
