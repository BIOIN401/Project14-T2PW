"""REV-108: why did MY R2 mutation come back GREEN when the author's textually
equivalent N17 came back RED, on the same tree, in the same session?

One of the two runs is wrong and I am not entitled to pick. This applies MY
mutation, reads the file back, prints the mutated line, rebuilds the pattern and
evaluates the five spans the author's coverage test uses -- so the answer comes
from the bytes on disk and the verdicts, not from either harness's summary.

Usage: <python> rev108_r1_r2_diagnostic.py <r1-root>
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from c102_mutation_attack import (  # noqa: E402
    apply_mutation, crlf_count, find_occurrences, restore_saved_bytes, sha256_of,
)

ROOT = Path(sys.argv[1]).resolve()
GUARD = ROOT / "src/t2pw/curation/apply_audit_patch.py"

R2_OLD = (
    '            + _PREDICATION_GAP_SRC + r"{0,3}" + noun + r"s?(?![a-z])"\n'
    '        )\n'
    '        # The copular-equivalent verbs.'
)
R2_NEW = (
    '            + _PREDICATION_GAP_SRC + r"{0,0}" + noun + r"s?(?![a-z])"\n'
    '        )\n'
    '        # The copular-equivalent verbs.'
)
SPANS = [
    "P is a high affinity transporter",
    "P is an inner membrane transporter",
    "P acts as an inner membrane channel",
    "P is the outer membrane transporter for A",
    "P was a well characterised sodium carrier",
]


def fresh():
    for name in [m for m in list(sys.modules) if m.startswith("t2pw")]:
        del sys.modules[name]
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    return importlib.import_module("t2pw.curation.apply_audit_patch")


def transports(mod, actor, evidence):
    proc = {"name": "A to B", "inputs": ["A"], "outputs": ["B"],
            "evidence": "A is converted in the gut", "transporters": []}
    payload = {"entities": {"compounds": [{"name": "A"}, {"name": "B"}],
                            "proteins": [{"name": actor}], "protein_complexes": [],
                            "nucleic_acids": []},
               "processes": {"transports": [proc]}}
    op = {"op": "add", "path": "/processes/transports/0/transporters/-",
          "value": actor, "confidence": 1.0, "evidence": evidence}
    _r, rep = mod.apply_patch_with_policy(payload, [op], stage="probe")
    return rep["summary"]["accepted_count"] == 1


def show(tag):
    mod = fresh()
    text = GUARD.read_text(encoding="utf-8")
    hits03 = text.count('+ _PREDICATION_GAP_SRC + r"{0,3}"')
    hits00 = text.count('+ _PREDICATION_GAP_SRC + r"{0,0}"')
    print("  --- %s ---" % tag)
    print("    occurrences of {0,3} on disk : %d" % hits03)
    print("    occurrences of {0,0} on disk : %d" % hits00)
    pat = mod._ROLE_CUE_RES["transport"].pattern
    idx = pat.find("is|are|was|were|remains|remain")
    print("    transport pattern, first predication alternative:")
    print("      %r" % pat[idx - 4:idx + 90])
    for span in SPANS:
        print("    %s  %r" % ("ACCEPT" if transports(mod, "P", span) else "REFUSE", span))


original = GUARD.read_bytes()
start_sha = sha256_of(original)
print("find_occurrences(R2_OLD) =", find_occurrences(GUARD, R2_OLD))
print()
print("BEFORE")
show("unmutated")

saved = apply_mutation(GUARD, R2_OLD, R2_NEW)
try:
    print()
    print("AFTER MY R2 MUTATION")
    show("mutated")
finally:
    restore_saved_bytes(GUARD, saved)

after = GUARD.read_bytes()
print()
print("restored byte-identical:", sha256_of(after) == start_sha)
print("CRLF preserved         :", crlf_count(after) == crlf_count(original))
