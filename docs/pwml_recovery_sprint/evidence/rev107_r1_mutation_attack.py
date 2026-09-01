"""REV-107 correction-round-1 mutation attack, by the NON-author.

Attacks the guards this ROUND introduced, at a finer grain than the author's
M10-M13. D-084 via C-106 primitives; sha256 and CRLF proved on every restore.

The last entry is a REVERSE mutation. It does not break a guard: it applies the
FIX for the F1 left-boundary defect REV-107 found this round. If the suite stays
GREEN under it, no committed test pins the current unanchored behaviour and the
repair is not test-blocked. That is the information the orchestrator needs.

Usage::  <python> rev107_r1_mutation_attack.py <worktree-root>
"""
from __future__ import annotations

import re
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
TESTS = ["tests/test_c107_actor_cue_calibration.py",
         "tests/test_c105_actor_role_evidence.py"]
BASETEMP = "C:/t/bt/rev107r1mut"

MUTATIONS = [
    (
        "V1", "F1: the of/in linker is made MANDATORY, so bare 'reduction NDM-1' escapes",
        '                _ATTENUATION_STEM_SRC + r"[a-z]*\\b(?:\\s+(?:of|in))?"\n',
        '                _ATTENUATION_STEM_SRC + r"[a-z]*\\b(?:\\s+(?:of|in))"  # MUTATION V1\n',
    ),
    (
        "V2", "F1: the modifier budget drops to zero -- one adjective evades the contra",
        '                r"(?:\\s+" + _PASSIVE_AGENT_MODIFIERS_SRC + r"){0,4}\\s+"\n',
        '                r"(?:\\s+" + _PASSIVE_AGENT_MODIFIERS_SRC + r"){0,0}\\s+"'
        '  # MUTATION V2\n',
    ),
    (
        "V3", "F1: the needle right boundary is dropped, so a PREFIX of a longer token matches",
        '                + escaped + r"(?![a-z0-9])"\n'
        '                # F2: "<actor> ... <activity noun> ... <stem>"\n',
        '                + escaped  # MUTATION V3\n'
        '                # F2: "<actor> ... <activity noun> ... <stem>"\n',
    ),
    (
        "V4", "F2: the attenuation OBJECT vocabulary is emptied",
        '_ATTENUATION_OBJECT_SRC = r"(?:activit|express|level|abundance|function)"\n',
        '_ATTENUATION_OBJECT_SRC = r"(?:(?!))"  # MUTATION V4\n',
    ),
    (
        "V5", "F2: the look distance collapses to zero",
        '_ATTENUATION_GAP = 40\n',
        '_ATTENUATION_GAP = 0  # MUTATION V5\n',
    ),
    (
        "V6", "1e: the cofactor dependence vocabulary is emptied",
        '_COFACTOR_DEPENDENCE_SRC = (\n'
        '    r"(?:requires|requiring|required|requirement"\n'
        '    r"|depends|depend|dependent|dependence"\n'
        '    r"|in the presence)"\n'
        ')\n',
        '_COFACTOR_DEPENDENCE_SRC = r"(?:(?!))"  # MUTATION V6\n',
    ),
    (
        "V7", "1e: the cofactor modifier budget drops to zero",
        '_COFACTOR_MAX_MODIFIERS = 4\n',
        '_COFACTOR_MAX_MODIFIERS = 0  # MUTATION V7\n',
    ),
    (
        "V8", "1e: the cofactor-specific modifier nouns are removed, leaving the passive list",
        '    + r"|cofactor|cofactors|coenzyme|coenzymes|metal|divalent|ion|ions"\n'
        '    + r"|essential|catalytic|added|exogenous|free)"\n',
        '    + r")"  # MUTATION V8\n',
    ),
    (
        "V9", "1e REGRESSION: the dependence route is made family-wide again -- the exact "
              "widening of the unmapped-role fallback REV-107 raised as blocking 2",
        '        dependence = None\n        if family == "cofactor":\n',
        '        dependence = None\n        if True:  # MUTATION V9\n',
    ),
    # V10 WITHDRAWN by REV-107: its substitution matched 2 times, not 1, and
    # apply_mutation correctly ABORTED the run. Preserved in
    # rev107_r1-rev-mut.log. V9 already covers that regression direction.
    (
        "V11-REVERSE", "NOT a break: this APPLIES REV-107's proposed fix for the F1 "
                       "left-boundary defect. GREEN here means no test pins the current "
                       "unanchored behaviour and the repair is not test-blocked.",
        '                _ATTENUATION_STEM_SRC + r"[a-z]*\\b(?:\\s+(?:of|in))?"\n',
        '                r"(?<![a-z])" + _ATTENUATION_STEM_SRC'
        ' + r"[a-z]*\\b(?:\\s+(?:of|in))?"  # MUTATION V11\n',
    ),
]


def run_suite(root: Path):
    proc = subprocess.run(
        [PY, "-m", "pytest", *TESTS, "-q", "--no-header", "-rf",
         "--basetemp=" + BASETEMP],
        cwd=str(root), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    out = proc.stdout + proc.stderr
    failed = sorted(set(re.findall(r"FAILED \S+::([A-Za-z_0-9]+)", out)))
    tail = [ln for ln in out.splitlines() if re.search(r"\d+ (passed|failed)", ln)]
    return proc.returncode, failed, tail


def main(root: Path) -> int:
    path = root / GUARD
    code, failed, tail = run_suite(root)
    print(f"=== BASELINE (unmutated r1 tip) === exit={code}")
    for line in tail:
        print(f"    {line}")
    if code != 0:
        print("=== BASELINE PRECONDITION FAILED ===")
        for f in failed:
            print(f"    {f}")
        return 2

    survivors = []
    for name, what, old, new in MUTATIONS:
        before = path.read_bytes()
        print(f"\n=== {name}: {what}")
        print(f"    before: bytes={len(before)} crlf={crlf_count(before)} "
              f"sha256={sha256_of(before)[:16]}")
        try:
            saved = apply_mutation(path, old, new)
        except ValueError as exc:
            print(f"=== {name}: ABORT -- the substitution did not apply: {exc}")
            return 3
        try:
            mcode, mfailed, mtail = run_suite(root)
        finally:
            restore_saved_bytes(path, saved)
        after = path.read_bytes()
        print(f"    after : bytes={len(after)} crlf={crlf_count(after)} "
              f"sha256={sha256_of(after)[:16]}")
        assert sha256_of(after) == sha256_of(before), f"{name}: restore not byte-exact"
        assert crlf_count(after) == crlf_count(before), f"{name}: restore changed CRLF"
        if mcode == 0:
            survivors.append(name)
        print(f"    suite : exit={mcode}  "
              f"{'RED (guard is tested)' if mcode else 'GREEN'}")
        for line in mtail:
            print(f"        {line}")
        for f in mfailed[:14]:
            print(f"        RED: {f}")
        if len(mfailed) > 14:
            print(f"        ... and {len(mfailed) - 14} more distinct test functions")

    breaks = [n for n in survivors if not n.startswith("V11")]
    print()
    print("=" * 78)
    print(f"REV-107 r1 MUTATIONS: {len(MUTATIONS)}")
    print(f"BREAKING mutations that SURVIVED: {len(breaks)} {breaks}")
    print(f"V11-REVERSE (the proposed fix) was: "
          f"{'GREEN -- repair NOT test-blocked' if 'V11-REVERSE' in survivors else 'RED -- a test pins the current behaviour'}")
    print("=" * 78)
    final = path.read_bytes()
    print(f"final target: bytes={len(final)} crlf={crlf_count(final)} "
          f"sha256={sha256_of(final)}")
    porcelain = subprocess.run(
        ["git", "status", "--porcelain", "--", GUARD],
        cwd=str(root), capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()
    print(f"porcelain: {porcelain!r} (secondary signal only)")
    return 0 if not breaks else 1


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]).resolve()))
