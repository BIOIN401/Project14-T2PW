"""C-108 correction round 1: the REV-108 blocking finding, reproduced and pinned.

Four rephrasings in which THE ACTOR IS THE THING BEING SHUT DOWN went
base=REFUSE -> tip=ACCEPT at the round-0 tip. That is a weakened biological gate
and merge rule 6 forbids it.

WHY IT HAPPENED, and it is the card's own quoted lesson turned on its author:
round 0 rebuilt the agent-noun half of the catalysis contra as F3/F4, a BOUNDED
CLOSED LIST OF TARGET-DIRECTED FRAMES WITH ACCEPT AS THE DEFAULT OUTSIDE THEM.
Handoff lesson 3: a bounded closed list flips polarity between a cue and a
contra -- in a cue it under-accepts and is safe, in a contra it under-refuses
and is not. The list was not wrong. Its POLARITY was.

This probe measures four populations at both SHAs and never nets them:

  BLOCKING      the four spans. Must REFUSE at both SHAs.
  PARAPHRASE    the same grammar, written by this round to attack its own
                repair rather than fit it to the four spans handed over.
  APPOSITIVE    member (d) itself. Must ACCEPT at the tip and REFUSE at base --
                that is (d)'s base failure and it must not be regressed to fix
                the blocking finding.
  PINNED        the target readings round 0 already refused. Must stay REFUSED.

Usage::  <python> c108_r1_blocking.py <base-code-root> <tip-code-root>
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


BASE_M = load(BASE, "c108r1_base")
TIP_M = load(TIP, "c108r1_tip")
print("base:", BASE_M.__file__, file=sys.stderr)
print("tip :", TIP_M.__file__, file=sys.stderr)


def verdict(mod, name, span, container="enzymes", bucket="reactions", role=None):
    value = name if role is None else {"entity": name, "role": role}
    proc = {"name": "A to B", "inputs": ["A"], "outputs": ["B"],
            "evidence": "A is converted in the gut", container: []}
    payload = {"entities": {"compounds": [{"name": "A"}, {"name": "B"}],
                            "proteins": [{"name": name}], "protein_complexes": [],
                            "nucleic_acids": []},
               "processes": {bucket: [proc]}}
    op = {"op": "add", "path": "/processes/%s/0/%s/-" % (bucket, container),
          "value": value, "confidence": 1.0, "evidence": span}
    _r, rep = mod.apply_patch_with_policy(payload, [op], stage="c108r1")
    return rep["summary"]["accepted_count"] == 1


# REV-108 four, verbatim. In every one the actor is the TARGET of the
# attenuation, and the fourth is the sharpest: the span says the catalysis
# STOPPED.
BLOCKING = [
    ("P4X", "P4X is a target of the inhibitor and catalyses the conversion of A to B"),
    ("P4X", "P4X was subject to inhibitors during the assay, yet catalyses A to B"),
    ("P4X", "P4X, whose inhibitor was characterised, catalyses the conversion of A to B"),
    ("P4X", "the repressor bound P4X and the catalysis of A to B stopped"),
]

# Rephrasings of the same GRAMMAR, added by this round so the repair is attacked
# rather than fitted to the four spans that were handed to it.
BLOCKING_PARAPHRASE = [
    ("P4X", "P4X is the target of a suppressor and catalyses the conversion of A to B"),
    ("P4X", "P4X remained a target of the antagonist while catalysing A to B"),
    ("P4X", "the inhibitor was raised against P4X, which catalyses A to B"),
    ("P4X", "an inhibitor was co-crystallised with P4X, which catalyses A to B"),
    ("P4X", "P4X, for which an inhibitor exists, catalyses the conversion of A to B"),
    ("P4X", "the suppressor bound P4X and the catalysis of A to B stopped"),
    ("P4X", "the antagonist blocked P4X and the conversion of A to B stopped"),
    ("P4X", "the repressor acts on P4X, which catalyses the conversion of A to B"),
    ("P4X", "inhibitors were screened against P4X, which catalyses A to B"),
    ("P4X", "P4X sensitivity to the inhibitor was measured while it catalyses A to B"),
]

# CORRECTION ROUND 2 -- REV-108 R8(i). A POSSESSIVE marks the agent noun as
# BELONGING TO the actor, which is the TARGET reading and the exact opposite of
# an apposition; the demonstratives sit beside them because no paper writes
# "P4X, that inhibitor, ..." and leaving them in the A2 determiner set cost
# escapes for nothing. Every one of these must REFUSE, and every one REFUSES at
# base too -- so like the round-1 block above, this is a PRESERVATION control,
# not a base failure.
POSSESSIVE_AND_DEMONSTRATIVE = [
    ("P4X", "P4X, its inhibitor bound at the active site, converts A to B slowly"),
    ("P4X", "P4X, its inhibitor characterised earlier, catalyses the conversion of A to B"),
    ("P4X", "P4X, their inhibitor bound at the active site, catalyses A to B"),
    ("P4X", "P4X, their suppressor identified, catalyses the conversion of A to B"),
    ("P4X", "P4X, this inhibitor aside, catalyses the conversion of A to B"),
    ("P4X", "P4X, that antagonist notwithstanding, catalyses the conversion of A to B"),
    ("P4X", "P4X, its repressor deleted, catalyses the conversion of A to B"),
    ("P4X", "P4X, their antagonist co-purified, catalyses the conversion of A to B"),
]

# MEMBER (d) ITSELF. These must ACCEPT at the tip. Fixing the blocking finding by
# refusing these would be a regression of the very finding this card closes.
APPOSITIVE = [
    ("P4X", "the repressor complex P4X catalyses the conversion of A to B"),
    ("P4X", "the suppressor protein P4X catalyses the conversion of A to B"),
    ("P4X", "the inhibitor protein P4X catalyses the conversion of A to B"),
    ("P4X", "the inhibitor P4X catalyses the conversion of A to B"),
    ("P4X", "P4X, the inhibitor, catalyses the conversion of A to B"),
    ("P4X", "the potent inhibitor P4X catalyses the conversion of A to B"),
    ("P4X", "inhibitor P4X catalyses the conversion of A to B"),
    ("P4X", "the antagonist P4X catalyses the conversion of A to B"),
    ("P4X", "the downregulator P4X catalyses the conversion of A to B"),
    ("P4X", "the inactivator P4X catalyses the conversion of A to B"),
    ("P4X", "the attenuator protein P4X catalyses the conversion of A to B"),
    ("P4X", "the abolisher P4X catalyses the conversion of A to B"),
    ("P4X", "the blocker protein P4X catalyses the conversion of A to B"),
    ("P4X", "the repressor P4X hydrolyses A to give B"),
    ("P4X", "the suppressor P4X mediates the conversion of A to B"),
    ("P4X", "the antagonist P4X converts A into B"),
    ("P4X", "the selective inhibitor P4X catalyses the conversion of A to B"),
    ("P4X", "the small molecule inhibitor P4X catalyses the conversion of A to B"),
]

# The target readings round 0 already refused. They must stay refused.
PINNED = [
    ("P", "the inhibitor of P was added while Q mediates the conversion"),
    ("P", "inhibitors of P were added while Q mediates the conversion"),
    ("P", "the inhibitor for P was added while Q mediates the conversion"),
    ("P", "an inhibitor selective for P was added while Q mediates"),
    ("P", "the inhibitor targeting P was added while Q mediates"),
    ("P", "the inhibitor directed against P was added while Q mediates"),
    ("P", "the suppressor acting on P was added while Q mediates"),
    ("P", "the P inhibitor was added while Q mediates the conversion"),
    ("P", "the P specific inhibitor was added while Q mediates"),
    ("P", "the P activity inhibitor was added while Q mediates"),
    ("P", "the repressor of the P gene was deleted while Q mediates"),
    ("P", "P inhibitors were used while Q mediates the conversion"),
    ("P", "an inhibitor of P blocks the mediated conversion"),
    ("P", "an inhibitory effect on P was seen while Q mediates the conversion"),
    ("P", "A significantly inhibited P activity in the assay"),
    ("NDM-1", "PSA significantly inhibited NDM-1 enzyme activity"),
    ("NDM-1", "PSA-mediated inhibition of NDM-1 activity"),
    ("NDM-1", "the inhibition of NDM-1 is mediated by PSA"),
]


def block(title, cases, want):
    print()
    print("=" * 100)
    print("%s -- want %s" % (title, "ACCEPT" if want else "REFUSE"))
    print("=" * 100)
    nb = nt = 0
    for name, span in cases:
        b = verdict(BASE_M, name, span)
        t = verdict(TIP_M, name, span)
        if b != want:
            nb += 1
        if t != want:
            nt += 1
        print("  base=%-6s tip=%-6s  %s  %r"
              % ("ACCEPT" if b else "REFUSE", "ACCEPT" if t else "REFUSE",
                 "<< TIP WRONG" if t != want else "            ", span))
    print("  wrong at base: %d / %d      wrong at tip: %d / %d"
          % (nb, len(cases), nt, len(cases)))
    return nb, nt


b1, t1 = block("BLOCKING -- the four REV-108 spans, actor is the TARGET",
               BLOCKING, False)
b2, t2 = block("BLOCKING PARAPHRASE -- same grammar, this round own attack",
               BLOCKING_PARAPHRASE, False)
b5, t5 = block("POSSESSIVE / DEMONSTRATIVE -- round 2, R8(i): belonging is not "
               "apposition", POSSESSIVE_AND_DEMONSTRATIVE, False)
b3, t3 = block("APPOSITIVE -- member (d) itself, actor IS the attenuator",
               APPOSITIVE, True)
b4, t4 = block("PINNED -- target readings round 0 already refused", PINNED, False)

print()
print("=" * 100)
print("C108 LEFT   blocking_admitted=%d paraphrase_admitted=%d "
      "possessive_admitted=%d appositive_refused=%d pinned_leaked=%d"
      % (b1, b2, b5, b3, b4))
print("C108 RIGHT  blocking_admitted=%d paraphrase_admitted=%d "
      "possessive_admitted=%d appositive_refused=%d pinned_leaked=%d"
      % (t1, t2, t5, t3, t4))
print("=" * 100)
