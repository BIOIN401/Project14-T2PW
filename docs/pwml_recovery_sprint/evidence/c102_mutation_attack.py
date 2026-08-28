"""C-102: attack every load-bearing C-102 test by mutation, then restore.

F-144 is why this file exists. C-101 shipped two non-vacuity guards that guarded
the wrong emptiness -- a reviewer deleted one outright and all 38 tests stayed
green. A test that claims to detect something and has not been shown to go RED
when that something is broken is not evidence, so each mutation below breaks
exactly one thing a C-102 test claims to detect and the suite is re-run against
it.

M7 is the mutation this file was missing for a round. The numerator half of the
exclusion is a DEVIATION from D-072's literal text -- the ruling says
"denominator", this code removes forbidden terms from both sides -- and it was
escalated rather than taken silently. It still shipped with nothing asserting it:
reverting that one line left all eleven tests green. A deviation is exactly the
line most in need of a mutation, because no existing test was written with it in
mind. Tests 12 and 13 are what M7 now bites.

Every mutation is applied by exact text substitution to a COMMITTED file and
reverted with ``git checkout --`` afterwards, and the tree is verified clean at
the end. A mutation whose substitution does not apply aborts the run: a mutation
that silently did nothing would produce a green suite and read as a pass.

Usage::

    <python> c102_mutation_attack.py <worktree-root>
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
PY = sys.executable
ACCEPTANCE = "src/t2pw/bench/acceptance.py"
SEMANTIC = "src/t2pw/bench/semantic.py"
TESTS = "tests/test_c102_coverage_denominator.py"

MUTATIONS = [
    (
        "M1",
        "the exclusion never happens -- the contradictory denominator is restored",
        ACCEPTANCE,
        "\n    hit = case.forbidden_match(term)\n",
        "\n    return None  # MUTATION M1\n    hit = case.forbidden_match(term)\n",
    ),
    (
        "M2",
        "the parenthetical-gloss head retry is dropped",
        ACCEPTANCE,
        '    text = str(term)\n    head = text.split("(")[0].strip()\n',
        '    text = str(term)\n    head = ""  # MUTATION M2\n',
    ),
    (
        "M3",
        "guard rail 1 is weakened from exact matching to containment",
        ACCEPTANCE,
        "    hit = case.forbidden_match(term)\n    if hit is not None:\n        return hit\n",
        "    hit = case.forbidden_match(term)\n"
        "    if hit is None:  # MUTATION M3\n"
        "        from t2pw.bench.goldset import normalize_name as _nn\n"
        "        for _entry in case.forbidden_identifiers:\n"
        "            if _nn(_entry.name) and _nn(_entry.name) in _nn(term):\n"
        "                hit = _entry\n"
        "                break\n"
        "    if hit is not None:\n        return hit\n",
    ),
    (
        "M4",
        "an empty accepted denominator is reported as a coverage success",
        ACCEPTANCE,
        "        state = COVERAGE_UNDEFINED_ALL_FORBIDDEN\n        accepted_ratio = None\n",
        "        state = COVERAGE_MEASURED  # MUTATION M4\n        accepted_ratio = 1.0\n",
    ),
    (
        "M5",
        "guard rail 3 is silenced -- withheld terms vanish from the record",
        ACCEPTANCE,
        '        "excluded_terms": excluded,\n',
        '        "excluded_terms": [],  # MUTATION M5\n',
    ),
    (
        "M7",
        "the NUMERATOR half is reverted to D-072's literal denominator-only text",
        ACCEPTANCE,
        "    accepted_matched = [str(t) for t in matched if str(t) not in excluded_terms]\n",
        "    accepted_matched = [str(t) for t in matched]  # MUTATION M7\n",
    ),
    (
        "M6",
        "the coverage exemption leaks into Priority 1 and stops scoring a forbidden export",
        SEMANTIC,
        "            forbidden = case.forbidden_match(name)\n"
        "            if forbidden is not None:\n"
        "                if ids:\n",
        "            forbidden = case.forbidden_match(name)\n"
        "            if forbidden is not None:\n"
        "                if ids and False:  # MUTATION M6\n",
    ),
]


def run_suite() -> tuple[int, list[str]]:
    proc = subprocess.run(
        [PY, "-m", "pytest", TESTS, "-q", "--no-header", "-rf",
         "--basetemp=" + str(Path("C:/t/bt/c102mut"))],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    out = proc.stdout + proc.stderr
    failed = sorted(set(re.findall(r"FAILED [^:]+::(\w+)", out)))
    tail = [line for line in out.splitlines() if re.search(r"\d+ (passed|failed)", line)]
    return proc.returncode, [*failed, *tail]


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()


# The two mutated files must be clean; the attack driver and its own log are
# untracked while it runs, so the check is scoped to what it will revert.
assert git("status", "--porcelain", "--", ACCEPTANCE, SEMANTIC) == "", "mutated files must be clean"
code, summary = run_suite()
print(f"=== BASELINE (unmutated tip) === exit={code}")
for line in summary:
    print(f"    {line}")
assert code == 0, "the unmutated suite must be green before any mutation means anything"

failures = 0
for name, what, rel, old, new in MUTATIONS:
    path = ROOT / rel
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        print(f"=== {name}: ABORT -- the substitution matched {text.count(old)} times, not 1")
        raise SystemExit(2)
    path.write_text(text.replace(old, new, 1), encoding="utf-8", newline="")
    code, summary = run_suite()
    print(f"\n=== {name}: {what}")
    print(f"    file={rel}  exit={code}  {'RED (detected)' if code else 'GREEN -- NOT DETECTED'}")
    for line in summary:
        print(f"    {line}")
    git("checkout", "--", rel)
    assert git("status", "--porcelain", "--", rel) == "", f"{name} did not revert"
    if code == 0:
        failures += 1

restored = git("status", "--porcelain", "--", ACCEPTANCE, SEMANTIC)
print(f"\n=== tree after restore: {restored!r}  (must be empty)")
code, summary = run_suite()
print(f"=== SUITE AFTER RESTORE === exit={code}")
for line in summary:
    print(f"    {line}")
if failures or restored or code:
    print(f"\nATTACK FAILED: {failures} mutation(s) went undetected")
    raise SystemExit(1)
print(f"\nATTACK PASSED: all {len(MUTATIONS)} mutations detected, tree clean, suite green")
