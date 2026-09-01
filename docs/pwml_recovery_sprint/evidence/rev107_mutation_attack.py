"""REV-107's OWN mutation attack -- R7: every load-bearing guard mutated by the
NON-author. These are mutations the author's c107_mutation_attack.py does NOT
contain; they attack the sub-parts of the new guards that the author's coarser
mutations could mask.

D-084: restores go through C-106's repaired primitives (apply_mutation /
restore_saved_bytes), never git checkout and never a text-mode round trip. Each
restore is proved by sha256 AND CRLF count.

Usage::  <python> rev107_mutation_attack.py <worktree-root>
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
BASETEMP = "C:/t/bt/rev107mut"

MUTATIONS = [
    (
        "R1", "1a: the ATTENUATION OBJECT noun list is emptied -- the phrase rule "
              "can no longer find activity/level/expression",
        '_ATTENUATION_OBJECT_SRC = r"(?:activit|express|level|abundance|function)"\n',
        '_ATTENUATION_OBJECT_SRC = r"(?:(?!))"  # MUTATION R1\n',
    ),
    (
        "R2", "1a: the ATTENUATION STEM list is emptied -- no stem can start the phrase",
        '_ATTENUATION_STEM_SRC = (\n'
        '    r"(?:reduc|loss|deplet|disrupt|quench|blockade|block|impair|silenc"\n'
        '    r"|sequestr|ablat|interfer)"\n'
        ')\n',
        '_ATTENUATION_STEM_SRC = r"(?:(?!))"  # MUTATION R2\n',
    ),
    (
        "R3", "1a: the SIX new bare inhibition stems are removed, leaving only the "
              "activity-directed phrase",
        '        r"|blockade|impair|silenc|sequestr|ablat|interfer(?:e|i)"\n',
        '        r"|(?!)"  # MUTATION R3\n',
    ),
    (
        "R4", "1a: the 40-character gap between stem and object is closed to zero",
        '    _ATTENUATION_STEM_SRC + r"[a-z]*\\b[^.]{0,40}?\\b" + _ATTENUATION_OBJECT_SRC\n',
        '    _ATTENUATION_STEM_SRC + r"[a-z]*\\b[^.]{0,0}?\\b" + _ATTENUATION_OBJECT_SRC'
        '  # MUTATION R4\n',
    ),
    (
        "R5", "1b: the modifier budget between 'by' and the actor drops to zero",
        '_PASSIVE_AGENT_MAX_MODIFIERS = 4\n',
        '_PASSIVE_AGENT_MAX_MODIFIERS = 0  # MUTATION R5\n',
    ),
    (
        "R6", "1c: the SEVENTEEN newly added stoplist words are removed -- the list "
              "reverts to its C-106 content while the s? rule stays",
        '    "appease", "bookcase", "crease", "debase", "decease", "displease",\n'
        '    "encase", "grease", "lowercase", "nutcase", "paraphrase", "pillowcase",\n'
        '    "rebase", "rephrase", "suitcase", "surcease", "uppercase",\n',
        '    # MUTATION R6 -- the seventeen new words are gone\n',
    ),
    (
        "R7", "1a: the contra is applied to EVERY family, not only catalysis -- the "
              "over-refusal direction C-105 round 1 was rejected for",
        '    contra = _CATALYSIS_CONTRA_RE if family == "catalysis" else None\n',
        '    contra = _CATALYSIS_CONTRA_RE  # MUTATION R7\n',
    ),
    (
        "R8", "1e: the cofactor family is made to license the BARE SCHEMA NOUN, the "
              "single line the author says separates it from F-146",
        '        r"is a cofactor|is the cofactor|cofactor for|cofactor of|cofactor in this"\n',
        '        r"cofactor"  # MUTATION R8\n',
    ),
    (
        "R9", "control (pre-existing C-105 guard): _SHORT_ENZYME_NOUNS is emptied",
        '_SHORT_ENZYME_NOUNS = ("lyase", "lyases", "dnase", "dnases", "rnase", "rnases")\n',
        '_SHORT_ENZYME_NOUNS = ("zzzzzzq",)  # MUTATION R9\n',
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
              f"{'RED (guard is tested)' if mcode else 'GREEN  <<< SURVIVOR'}")
        for line in mtail:
            print(f"        {line}")
        for f in mfailed[:14]:
            print(f"        RED: {f}")
        if len(mfailed) > 14:
            print(f"        ... and {len(mfailed) - 14} more distinct test functions")

    print()
    print("=" * 78)
    print(f"REV-107 MUTATIONS: {len(MUTATIONS)}   SURVIVORS: {len(survivors)} {survivors}")
    print("=" * 78)
    final = path.read_bytes()
    print(f"final target: bytes={len(final)} crlf={crlf_count(final)} "
          f"sha256={sha256_of(final)}")
    porcelain = subprocess.run(
        ["git", "status", "--porcelain", "--", GUARD],
        cwd=str(root), capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()
    print(f"porcelain: {porcelain!r} (secondary signal only)")
    return 0 if not survivors else 1


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]).resolve()))
