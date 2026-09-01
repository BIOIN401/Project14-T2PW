"""C-107: attack every load-bearing guard this card adds, then restore.

F-144 / D-078. A guard that has not been shown to go RED when it is broken is not
evidence. Each mutation below forces exactly one C-107 guard false and the
focused suite is re-run against it; a mutation that leaves the suite green means
the guard has no test, not that the guard is fine.

BOTH DIRECTIONS ARE MUTATED, because this card moves the guard both ways:

* M1, M2, M3, M4 break a REFUSAL and must turn a rejection test red.
* M5, M6, M7 break an ADMISSION and must turn a preservation test red. Those
  matter just as much here -- C-105 round 1's defect was refusing too much, and a
  preservation battery nobody has mutated is exactly the battery that passed
  while 12 of 29 legitimate cases were refused.
* M8 and M9 attack the two halves of 1a separately, so "the contra fires" and
  "redox still licenses" are shown to be independently load-bearing rather than
  one assertion wearing two hats.

M6b IS A RESULT, NOT A DESIGN. M6 -- deleting the ``cofactor`` entry from
_ROLE_FAMILY_BY_ROLE -- SURVIVED on the first run of this harness, green, and the
run is preserved at ``c107_mutation_attack.attempt1-m6-survivor.log``. The reason
is that _ANY_ROLE_CUE_RE is rebuilt from every _ROLE_CUE_RES value, so once the
cofactor VOCABULARY exists the "other" fallback licenses the same spans and the
map entry changes nothing a licensing assertion can see. What the map entry does
change is the reason string batch tooling greps, which is now pinned, so M6 bites;
M6b attacks the vocabulary, which is what the fix actually turned on.

RESTORE DISCIPLINE -- D-084. Nothing here hand-rolls a restore. It imports
``apply_mutation`` and ``restore_saved_bytes`` from C-106's repaired
``c102_mutation_attack``, which write bytes throughout, translate the pattern to
the target's newline rather than the target to the pattern's, and prove the
restore by sha256 AND CRLF count. ``git checkout --`` reverts more; a text-mode
round trip reverts less; neither is used.

Usage::  <python> c107_mutation_attack.py <worktree-root>
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
BASETEMP = "C:/t/bt/c107mut"

MUTATIONS = [
    (
        "M1", "1f: the 'mediat' anchor is reverted, so it matches inside 'intermediate'",
        '        r"|(?<![a-z])mediat"\n',
        '        r"|mediat"  # MUTATION M1\n',
    ),
    (
        "M2", "1a: the contra-cue is switched off entirely for the catalysis family",
        '    contra = _CATALYSIS_CONTRA_RE if family == "catalysis" else None\n',
        '    contra = None  # MUTATION M2\n',
    ),
    (
        "M3", "1a: only the ACTIVITY-DIRECTED half of the contra is removed",
        '_CATALYSIS_CONTRA_RE = re.compile(\n'
        '    _ROLE_CUE_RES["inhibition"].pattern + r"|" + _ACTIVITY_ATTENUATION_SRC\n'
        ')\n',
        '_CATALYSIS_CONTRA_RE = _ROLE_CUE_RES["inhibition"]  # MUTATION M3\n',
    ),
    (
        "M4", "1c: the stoplist exclusion loses its optional plural, reopening the bypass",
        '    r"(?:(?<![a-z])(?!(?:" + "|".join(_NON_ENZYME_ASE_WORDS) + r")s?(?![a-z]))"\n',
        '    r"(?:(?<![a-z])(?!(?:" + "|".join(_NON_ENZYME_ASE_WORDS) + r")(?![a-z]))"  # MUTATION M4\n',
    ),
    (
        "M5", "1d: the transport family loses the enzyme-family noun rule again",
        '        # which is the inversion C-105 round 1 was rejected for.\n'
        '        r"|" + _ENZYME_NOUN_RE_SRC\n',
        '        # which is the inversion C-105 round 1 was rejected for.\n'
        '        r"|(?!)"  # MUTATION M5\n',
    ),
    (
        "M6", "1e: the cofactor role loses its family entry and falls back to 'other'",
        '    "cofactor": "cofactor",\n',
        '    "cofactor_MUTATED": "cofactor",  # MUTATION M6\n',
    ),
    (
        "M6b", "1e: the cofactor VOCABULARY is emptied -- the guard M6 turned out not to be",
        '    "cofactor": re.compile(\n'
        '        r"is a cofactor|is the cofactor|cofactor for|cofactor of|cofactor in this"\n'
        '        r"|is a coenzyme|is the coenzyme|coenzyme for|coenzyme of"\n'
        '        r"|prosthetic group"\n'
        '        r"|requires|required for|requirement for"\n'
        '        r"|depends on|dependent on|dependence on"\n'
        '        r"|in the presence of"\n'
        '    ),\n',
        '    "cofactor": re.compile(  # MUTATION M6b\n'
        '        r"(?!)"\n'
        '    ),\n',
    ),
    (
        "M7", "1b: the passive-agent gap accepts ANY token, so a bystander inherits agency",
        '_PASSIVE_AGENT_MODIFIERS_SRC = (\n'
        '    r"(?:the|a|an|its|their|his|her|this|that|these|those|of"\n',
        '_PASSIVE_AGENT_MODIFIERS_SRC = (  # MUTATION M7\n'
        '    r"(?:[a-z0-9]+|the|a|an|its|their|his|her|this|that|these|those|of"\n',
    ),
    (
        "M8", "1b: the passive-agent route is removed outright -- a PRESERVATION attack",
        '        passive = (\n            _PASSIVE_AGENT_VERBS_SRC\n',
        '        passive = (\n            r"(?!)" + _PASSIVE_AGENT_VERBS_SRC  # MUTATION M8\n',
    ),
    (
        "M9", "1a: 'reduction of' is deleted from catalysis -- the trap the card names",
        '        r"|reduces|reducing|reduction of"\n',
        '        r"|(?!)"  # MUTATION M9\n',
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
    print(f"=== BASELINE (unmutated tip) === exit={code}")
    for line in tail:
        print(f"    {line}")
    if code != 0:
        print("=== BASELINE PRECONDITION FAILED -- nothing below would mean anything ===")
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
            print(f"=== {name}: ABORT -- {exc}")
            return 3
        try:
            mcode, mfailed, mtail = run_suite(root)
        finally:
            restore_saved_bytes(path, saved)
        after = path.read_bytes()
        print(f"    after : bytes={len(after)} crlf={crlf_count(after)} "
              f"sha256={sha256_of(after)[:16]}")
        assert sha256_of(after) == sha256_of(before), f"{name}: restore not byte-exact"
        assert crlf_count(after) == crlf_count(before), f"{name}: restore changed line endings"
        verdict = "RED (guard is tested)" if mcode != 0 else "GREEN  <<< SURVIVOR"
        if mcode == 0:
            survivors.append(name)
        print(f"    suite : exit={mcode}  {verdict}")
        for line in mtail:
            print(f"        {line}")
        for f in mfailed[:12]:
            print(f"        RED: {f}")
        if len(mfailed) > 12:
            print(f"        ... and {len(mfailed) - 12} more distinct test functions")

    print()
    print("=" * 78)
    print(f"MUTATIONS: {len(MUTATIONS)}   SURVIVORS: {len(survivors)} {survivors}")
    print("=" * 78)
    final = path.read_bytes()
    print(f"final target: bytes={len(final)} crlf={crlf_count(final)} "
          f"sha256={sha256_of(final)}")
    porcelain = subprocess.run(
        ["git", "status", "--porcelain", "--", GUARD],
        cwd=str(root), capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()
    print(f"git status --porcelain -- {GUARD}: {porcelain!r} "
          f"(additional signal only -- sha256 and CRLF above are the check)")
    return 0 if not survivors else 1


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]).resolve()))
