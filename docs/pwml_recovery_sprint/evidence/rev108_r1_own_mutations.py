"""REV-108 round 1: MY OWN two round-0 survivors, re-expressed against the
round-1 source, plus one new mutation on the round-1 exemption.

R2 and R3 survived at round 0 -- they were green with the guard broken, so
nothing tested them. Round 1 claims both are now covered. This re-runs them in
MY spelling rather than reading the author's N16/N17, because a mutation is only
evidence if the person relying on it wrote it.

  R2r1  the predication modifier gap closes to zero
  R3r1  the transport verb inflections collapse to "channels|pumps" -- the hole
        that hid the missing "channeled"
  R5r1  NEW: the appositive determiner set loses its POSSESSIVE and DEICTIC
        members. Narrowing an EXEMPTION can only refuse more, so GREEN here is
        the expected and correct result; a RED would mean a test is pinning the
        hole open.

R1 and R4 from round 0 are not re-run: they targeted F3/F4, which round 1
deleted outright.

Restore is C-106's, D-084: replays SAVED BYTES, proven by sha256 AND CRLF count.

Usage::  <python> rev108_r1_own_mutations.py <worktree-root>
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from c102_mutation_attack import (  # noqa: E402
    apply_mutation, crlf_count, restore_saved_bytes, sha256_of,
)

PY = "c:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe"
GUARD = "src/t2pw/curation/apply_audit_patch.py"
TESTS = ["tests/test_c108_f155_class.py",
         "tests/test_c107_actor_cue_calibration.py",
         "tests/test_c105_actor_role_evidence.py"]
BASETEMP = "C:/t/bt108/revmutr1"

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
R3_OLD = (
    '        r"|channels|channeled|channelled|channeling|channelling"\n'
    '        r"|pumps|pumped|pumping"\n'
)
R3_NEW = '        r"|channels|pumps"\n'
R5_OLD = '_APPOSITIVE_DETERMINER_SRC = r"(?:the|a|an|this|that|its|their)"\n'
R5_NEW = '_APPOSITIVE_DETERMINER_SRC = r"(?:the|a|an)"\n'

MUTATIONS = [
    ("R2r1", "red",
     "(a) R2 re-run at round 1: the predication modifier gap closes to zero. "
     "SURVIVED at round 0.",
     R2_OLD, R2_NEW),
    ("R3r1", "red",
     "(a) R3 re-run at round 1, my spelling not the author's: the transport verb "
     "inflections collapse to channels|pumps. SURVIVED at round 0 and is the hole "
     "that hid the missing 'channeled'.",
     R3_OLD, R3_NEW),
    ("R5r1", "green",
     "(d) round 1, NEW: the appositive determiner set loses its possessive and "
     "deictic members. Narrowing an exemption can only refuse more, so GREEN is "
     "correct here; RED would mean a test pins the hole open.",
     R5_OLD, R5_NEW),
]


def run_suite(root):
    cmd = [PY, "-m", "pytest", "-q", "--basetemp=" + BASETEMP, "-p", "no:randomly"] + TESTS
    proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
    tail = [ln for ln in proc.stdout.splitlines() if ln.strip()][-1:]
    red = [ln.split(" ")[1] for ln in proc.stdout.splitlines() if ln.startswith("FAILED ")]
    return proc.returncode, (tail[0] if tail else ""), red


def main():
    root = Path(sys.argv[1]).resolve()
    target = root / GUARD
    original = target.read_bytes()
    start_sha, start_crlf = sha256_of(original), crlf_count(original)
    print("target sha256 at start :", start_sha)
    rc, tail, _ = run_suite(root)
    print("BASELINE (unmutated): exit=%d  %s" % (rc, tail))
    if rc != 0:
        print("!! baseline is not green; nothing below is interpretable")
        return 2

    failures = []
    for name, expect, why, old, new in MUTATIONS:
        print()
        print("=" * 92)
        print("%s [expect %s]  %s" % (name, expect.upper(), why))
        print("=" * 92)
        saved = apply_mutation(target, old, new)
        try:
            rc, tail, red = run_suite(root)
            got = "red" if rc != 0 else "green"
            ok = (got == expect)
            print("  result: %s  exit=%d  %s   %s"
                  % (got.upper(), rc, tail, "OK" if ok else "<< NOT AS EXPECTED"))
            for t in red[:10]:
                print("    RED:", t)
            if not ok:
                failures.append(name)
        finally:
            restore_saved_bytes(target, saved)
        assert sha256_of(target.read_bytes()) == start_sha, "restore did not replay saved bytes"

    after = target.read_bytes()
    print()
    print("target sha256 after    :", sha256_of(after))
    print("byte-identical         :", sha256_of(after) == start_sha)
    print("CRLF preserved         :", crlf_count(after) == start_crlf)
    print("REV108 R1 OWN MUTATIONS: %d run, %d not as expected %s"
          % (len(MUTATIONS), len(failures), failures))
    return 0


sys.exit(main())
