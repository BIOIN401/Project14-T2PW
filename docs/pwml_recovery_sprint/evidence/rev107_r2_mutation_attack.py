"""REV-107 correction-round-2 mutation attack, by the NON-author.

Two groups.

V-series: REV-107's round-1 mutations, RE-POINTED where this round rewrote the
line they targeted. V1 and V2 aimed at the F1 line, which now reads
_ATTENUATION_WORD_SRC; their round-1 text no longer matches and apply_mutation
would abort, so they are re-pointed here exactly as the author re-pointed M3.
V1/V2/V7/V8 were REGISTERED as coverage gaps and authorised not-fixed, so they
are EXPECTED GREEN; V9 was made blocking and must now be RED.

W-series: NEW attacks on the guards THIS round added -- the two anchors on
_ATTENUATION_WORD_SRC, the two anchors on the six inhibition additions, the
inflection list itself, and the companion assertion that the cofactor scope pin
cannot be satisfied by disabling the route.

D-084 via C-106 primitives; sha256 and CRLF proved on every restore.

Usage::  <python> rev107_r2_mutation_attack.py <worktree-root>
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
BASETEMP = "C:/t/bt/rev107r2mut"

EXPECTED_GREEN = {"V1", "V2", "V7", "V8"}   # registered gaps, authorised not-fixed

MUTATIONS = [
    # ---- V-series, re-pointed ------------------------------------------------
    (
        "V1", "F1: the of/in linker is made MANDATORY  [RE-POINTED: the round-1 "
              "text targeted _ATTENUATION_STEM_SRC, which this round replaced]",
        '                _ATTENUATION_WORD_SRC + r"(?:\\s+(?:of|in))?"\n',
        '                _ATTENUATION_WORD_SRC + r"(?:\\s+(?:of|in))"  # MUTATION V1\n',
    ),
    (
        "V2", "F1: the modifier budget drops to zero  [RE-POINTED]",
        '                r"(?:\\s+" + _PASSIVE_AGENT_MODIFIERS_SRC + r"){0,4}\\s+"\n',
        '                r"(?:\\s+" + _PASSIVE_AGENT_MODIFIERS_SRC + r"){0,0}\\s+"'
        '  # MUTATION V2\n',
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
        "V8", "1e: the cofactor-specific modifier nouns are removed",
        '    + r"|cofactor|cofactors|coenzyme|coenzymes|metal|divalent|ion|ions"\n'
        '    + r"|essential|catalytic|added|exogenous|free)"\n',
        '    + r")"  # MUTATION V8\n',
    ),
    (
        "V9", "1e SCOPE: the dependence route goes family-wide again. BLOCKING last "
              "round; must now be RED.",
        '        dependence = None\n        if family == "cofactor":\n',
        '        dependence = None\n        if True:  # MUTATION V9\n',
    ),
    # ---- W-series, new this round --------------------------------------------
    (
        "W1", "A: the LEFT anchor comes off _ATTENUATION_WORD_SRC "
              "-- 'reduce' matches inside 'oxidoreduce'",
        '_ATTENUATION_WORD_SRC = (\n    r"(?<![a-z])(?:"\n',
        '_ATTENUATION_WORD_SRC = (\n    r"(?:"  # MUTATION W1\n',
    ),
    (
        "W2", "A: the RIGHT anchor comes off _ATTENUATION_WORD_SRC "
              "-- 'reduce' matches inside 'reductase'",
        '    r")(?![a-z])"\n)\n',
        '    r")"  # MUTATION W2\n)\n',
    ),
    (
        "W3", "A: ONE inflection is dropped from the list -- does anything notice "
              "that 'reduction' is no longer an attenuation word?",
        '    r"reduce|reduces|reduced|reducing|reduction|reductions"\n',
        '    r"reduce|reduces|reduced|reducing|reductions"  # MUTATION W3\n',
    ),
    (
        "W4", "A: the LEFT anchor comes off this card's six inhibition additions",
        '        r"|(?<![a-z])(?:blockades?|impair(?:s|ed|ing|ment|ments)?"\n',
        '        r"|(?:blockades?|impair(?:s|ed|ing|ment|ments)?"  # MUTATION W4\n',
    ),
    (
        "W5", "A: the RIGHT anchor comes off this card's six inhibition additions "
              "-- 'silenc' matches inside 'silencer' again, the SECOND SITE",
        '        r"|interfer(?:e|es|ed|ing|ence))(?![a-z])"\n',
        '        r"|interfer(?:e|es|ed|ing|ence))"  # MUTATION W5\n',
    ),
    (
        "W6", "B: the cofactor dependence route is DISABLED outright. The scope pin "
              "alone would go green; the companion preservation test must go RED.",
        '        if family == "cofactor":\n            dependence = re.compile(\n',
        '        if False:  # MUTATION W6\n            dependence = re.compile(\n',
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
    print(f"=== BASELINE (unmutated r2 tip) === exit={code}")
    for line in tail:
        print(f"    {line}")
    if code != 0:
        print("=== BASELINE PRECONDITION FAILED ===")
        for f in failed:
            print(f"    {f}")
        return 2

    green, red = [], []
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
        (green if mcode == 0 else red).append(name)
        note = ""
        if mcode == 0 and name in EXPECTED_GREEN:
            note = "  (EXPECTED -- registered gap, authorised not-fixed)"
        elif mcode == 0:
            note = "  <<< UNEXPECTED SURVIVOR"
        print(f"    suite : exit={mcode}  "
              f"{'RED (guard is tested)' if mcode else 'GREEN'}{note}")
        for line in mtail:
            print(f"        {line}")
        for f in mfailed[:14]:
            print(f"        RED: {f}")
        if len(mfailed) > 14:
            print(f"        ... and {len(mfailed) - 14} more distinct test functions")

    unexpected = [n for n in green if n not in EXPECTED_GREEN]
    print()
    print("=" * 78)
    print(f"REV-107 r2 MUTATIONS: {len(MUTATIONS)}")
    print(f"RED   : {len(red)}  {red}")
    print(f"GREEN : {len(green)}  {green}")
    print(f"EXPECTED-GREEN (registered gaps): "
          f"{sorted(n for n in green if n in EXPECTED_GREEN)}")
    print(f"UNEXPECTED SURVIVORS: {len(unexpected)}  {unexpected}")
    print("=" * 78)
    final = path.read_bytes()
    print(f"final target: bytes={len(final)} crlf={crlf_count(final)} "
          f"sha256={sha256_of(final)}")
    porcelain = subprocess.run(
        ["git", "status", "--porcelain", "--", GUARD],
        cwd=str(root), capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()
    print(f"porcelain: {porcelain!r} (secondary signal only)")
    return 0 if not unexpected else 1


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]).resolve()))
