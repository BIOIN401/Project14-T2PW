"""REV-108 round 1: MEASURE the proposed narrowing of A2, do not propose it.

"A reviewer who proposes rather than measures sends the author down a path that
half works." So the candidate repair is applied through C-106's harness (D-084:
restore replays SAVED BYTES, proven by sha256 and CRLF count), measured against
the five A2 escapes AND the appositive preservations AND the focused suite, and
restored.

CANDIDATE: drop the POSSESSIVE and DEICTIC determiners from
_APPOSITIVE_DETERMINER_SRC, leaving only the articles.

    (?:the|a|an|this|that|its|their)  ->  (?:the|a|an)

"its"/"their" are possessive: "P4X, its inhibitor ..." says the inhibitor
BELONGS TO P4X, which is the TARGET reading, not an apposition. "this"/"that"
are deictic and point at something already named, which need not be the actor.
Narrowing an EXEMPTION can only refuse more, so by the author's own subset
argument this cannot introduce an over-refusal against base -- which is checked
below rather than assumed.

Usage: <python> rev108_r1_repair_probe.py <r1-root>
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from c102_mutation_attack import (  # noqa: E402
    apply_mutation, crlf_count, restore_saved_bytes, sha256_of,
)

ROOT = Path(sys.argv[1]).resolve()
GUARD = ROOT / "src/t2pw/curation/apply_audit_patch.py"
PY = "c:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe"
BASETEMP = "C:/t/bt108/r1repair"
TESTS = ["tests/test_c108_f155_class.py",
         "tests/test_c107_actor_cue_calibration.py",
         "tests/test_c105_actor_role_evidence.py"]

OLD = '_APPOSITIVE_DETERMINER_SRC = r"(?:the|a|an|this|that|its|their)"\n'
NEW = '_APPOSITIVE_DETERMINER_SRC = r"(?:the|a|an)"  # REV-108 CANDIDATE\n'

# The five A2 escapes -- must REFUSE after the narrowing.
ESCAPES = [
    "P4X, its inhibitor bound at the active site, converts A to B slowly",
    "P4X, this inhibitor notwithstanding, catalyses the conversion of A to B",
    "P4X, an inhibitor target in oncology, catalyses the conversion of A to B",
    "P4X, the inhibitor binding site mapped, catalyses the conversion of A to B",
    "P4X, their inhibitors profiled, catalyses the conversion of A to B",
]
# Member (d)'s appositives -- must still ACCEPT after the narrowing.
PRESERVE = [
    "the repressor complex P4X catalyses the conversion of A to B",
    "the suppressor protein P4X catalyses the conversion of A to B",
    "the inhibitor protein P4X catalyses the conversion of A to B",
    "P4X, a repressor, catalyses the conversion of A to B",
    "P4X, the inhibitor, catalyses the conversion of A to B",
    "the repressor P4X catalyses the conversion of A to B",
    "the inhibitor P4X catalyses the conversion of A to B",
    "the potent inhibitor P4X catalyses the conversion of A to B",
]


def fresh():
    for name in [m for m in list(sys.modules) if m.startswith("t2pw")]:
        del sys.modules[name]
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    return importlib.import_module("t2pw.curation.apply_audit_patch")


def verdict(mod, actor, evidence):
    proc = {"name": "A to B", "inputs": ["A"], "outputs": ["B"],
            "evidence": "A is converted in the gut", "enzymes": []}
    payload = {"entities": {"compounds": [{"name": "A"}, {"name": "B"}],
                            "proteins": [{"name": actor}], "protein_complexes": [],
                            "nucleic_acids": []},
               "processes": {"reactions": [proc]}}
    op = {"op": "add", "path": "/processes/reactions/0/enzymes/-",
          "value": actor, "confidence": 1.0, "evidence": evidence}
    _r, rep = mod.apply_patch_with_policy(payload, [op], stage="probe")
    return rep["summary"]["accepted_count"] == 1


def measure(tag):
    mod = fresh()
    print("  --- %s ---" % tag)
    esc = []
    for span in ESCAPES:
        v = verdict(mod, "P4X", span)
        print("    %s  %r" % ("ACCEPT" if v else "REFUSE", span[:88]))
        esc.append(v)
    pres = []
    for span in PRESERVE:
        v = verdict(mod, "P4X", span)
        print("    %s  %r" % ("ACCEPT" if v else "REFUSE", span[:88]))
        pres.append(v)
    print("    escapes still ACCEPTED : %d of %d" % (sum(esc), len(esc)))
    print("    appositives ACCEPTED   : %d of %d" % (sum(pres), len(pres)))
    return sum(esc), sum(pres)


def suite():
    cmd = [PY, "-m", "pytest", "-q", "--basetemp=" + BASETEMP, "-p", "no:randomly"] + TESTS
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    tail = [ln for ln in p.stdout.splitlines() if ln.strip()][-1:]
    red = [ln.split(" ")[1] for ln in p.stdout.splitlines() if ln.startswith("FAILED ")]
    return p.returncode, (tail[0] if tail else ""), red


original = GUARD.read_bytes()
start_sha, start_crlf = sha256_of(original), crlf_count(original)
print("target                :", GUARD)
print("target sha256 at start:", start_sha)
print()
print("BEFORE THE CANDIDATE (round 1 as committed)")
e0, p0 = measure("as committed")
rc0, tail0, _ = suite()
print("  focused suite: exit=%d  %s" % (rc0, tail0))

print()
print("AFTER THE CANDIDATE (%s -> %s)" % (OLD.strip(), NEW.strip()))
saved = apply_mutation(GUARD, OLD, NEW)
try:
    e1, p1 = measure("candidate applied")
    rc1, tail1, red1 = suite()
    print("  focused suite: exit=%d  %s" % (rc1, tail1))
    for t in red1[:12]:
        print("    RED:", t)
finally:
    restore_saved_bytes(GUARD, saved)

after = GUARD.read_bytes()
print()
print("target sha256 after   :", sha256_of(after))
print("byte-identical        :", sha256_of(after) == start_sha)
print("CRLF preserved        :", crlf_count(after) == start_crlf)
print()
print("=" * 92)
print("CANDIDATE RESULT  escapes ACCEPTED %d -> %d   appositives ACCEPTED %d -> %d"
      % (e0, e1, p0, p1))
print("=" * 92)
