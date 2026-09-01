"""REV-108's OWN mutations. B13/B18 -- NOT the author's list.

The author's 13 mutations test what the author thought of. These four attack the
constants the author's mutations only ever remove WHOLESALE, so a surviving arm
here means the CALIBRATION inside a frame is untested even though the frame is.

  R1  F4's adjective allowance          (N10 removes the whole F4 frame; this
                                         only narrows it)
  R2  the predication modifier gap      (N4 empties the predication set; this
                                         only narrows the gap)
  R3  the transport verb inflections    (never mutated at all)
  R4  F3's non-"of" target heads        (N9 removes the whole F3 frame; this
                                         keeps "of" and the participials)

Restore is C-106's, D-084: replays SAVED BYTES, proven by sha256 AND CRLF count.
No ``git checkout --``, no porcelain check.

Usage::  <python> rev108_own_mutations.py <worktree-root>
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
BASETEMP = "C:/t/bt108/revmut"

MUTATIONS = [
    (
        "R1",
        "(d) F4: no adjective may stand between the actor and the agent noun. "
        "'the P4X specific inhibitor' stops being a target-directed frame.",
        "_ATTENUATION_AGENT_MAX_ADJ = 2\n",
        "_ATTENUATION_AGENT_MAX_ADJ = 0  # MUTATION R1\n",
    ),
    (
        "R2",
        "(a): the predication modifier gap closes to zero, so 'is a membrane "
        "transporter' stops being a predication. Also the gap that lets "
        "'is a substrate of the transporter' license.",
        '            r"(?:[a-z0-9]+ ){0,3}" + noun + r"s?(?![a-z])"\n'
        '        )\n'
        '        # The copular-equivalent verbs. "P acts as a transporter" is the same\n',
        '            r"(?:[a-z0-9]+ ){0,0}" + noun + r"s?(?![a-z])"  # MUTATION R2\n'
        '        )\n'
        '        # The copular-equivalent verbs. "P acts as a transporter" is the same\n',
    ),
    (
        "R3",
        "(a): the transport verb inflections collapse to 'channels' and 'pumps'. "
        "If this survives, the inflection SET is untested and a missing "
        "inflection cannot be caught by any test.",
        '        r"|channels|channelled|channeling|channelling|pumps|pumped|pumping"\n',
        '        r"|channels|pumps"  # MUTATION R3\n',
    ),
    (
        "R4",
        "(d) F3: only 'of' and the participial heads remain a target head. "
        "'an inhibitor against P4X' / 'selective for P4X' stop refusing.",
        '    r"(?:of|for|against|on|upon|to|toward|towards"\n',
        '    r"(?:of"  # MUTATION R4\n',
    ),
]


def run_suite(root: Path):
    cmd = [PY, "-m", "pytest", "-q", "--basetemp=" + BASETEMP, "-p", "no:randomly"] + TESTS
    proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
    tail = [ln for ln in proc.stdout.splitlines() if ln.strip()][-1:]
    red = [ln.split(" ")[1] for ln in proc.stdout.splitlines() if ln.startswith("FAILED ")]
    return proc.returncode, (tail[0] if tail else ""), red


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    target = root / GUARD
    original = target.read_bytes()
    start_sha = sha256_of(original)
    start_crlf = crlf_count(original)
    print("target                       :", target)
    print("target sha256 at start       :", start_sha)
    print("target CRLF count            :", start_crlf)
    rc, tail, _red = run_suite(root)
    print("BASELINE (unmutated): exit=%d  %s" % (rc, tail))
    if rc != 0:
        print("!! baseline is not green; nothing below is interpretable")
        return 2

    survivors = []
    for name, why, old, new in MUTATIONS:
        print()
        print("=" * 92)
        print("%s  %s" % (name, why))
        print("=" * 92)
        saved = apply_mutation(target, old, new)
        try:
            rc, tail, red = run_suite(root)
            verdict = "RED  (guard is tested)" if rc != 0 else "GREEN  << SURVIVOR: NO TEST COVERS THIS"
            print("  result: %s  exit=%d  %s" % (verdict, rc, tail))
            for t in red[:8]:
                print("    RED:", t)
            if rc == 0:
                survivors.append((name, why))
        finally:
            restore_saved_bytes(target, saved)
        assert sha256_of(target.read_bytes()) == start_sha, "restore did not replay saved bytes"

    after = target.read_bytes()
    end_sha = sha256_of(after)
    print()
    print("=" * 92)
    print("target sha256 after everything:", end_sha)
    print("byte-identical to the start   :", end_sha == start_sha)
    print("CRLF count preserved          :", crlf_count(after) == start_crlf)
    print("REV108 OWN MUTATIONS: %d run, %d SURVIVED" % (len(MUTATIONS), len(survivors)))
    for name, why in survivors:
        print("  SURVIVOR %s -- %s" % (name, why))
    return 0


sys.exit(main())
