"""REV-110 -- B18: mutations the AUTHOR DID NOT SUPPLY.

The author's ``test_every_guard_is_load_bearing`` flips NINE fields of the
passing fixture. But for the condition-3 guards it flips TWO ROW FIELDS AT ONCE
-- ``timed_out`` sets ``status`` AND ``failure_kind``; ``operational`` sets
``termination_reason`` AND ``operational_failure``; ``crashed`` sets ``status``
AND ``failure_kind``. The author's docstring claims FOUR INDEPENDENT READINGS
that must ALL clear, "redundant on purpose".

Redundancy is exactly what a coarse mutation cannot see: if two guards each
catch the same fixture, deleting EITHER ONE leaves the suite green.

So these mutations delete ONE reading at a time from the ``casualty``
expression and re-run the author's own test file. A mutation that stays GREEN
is a guard the author's suite does not pin.

D-084: restores replay SAVED BYTES via ``restore_saved_bytes``. Never
``git checkout --``.

Usage: rev110_my_mutations.py <tree> <basetemp-parent>
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TREE = Path(sys.argv[1])
BASETEMP = Path(sys.argv[2])
EVID = TREE / "docs" / "pwml_recovery_sprint" / "evidence"
sys.path.insert(0, str(EVID))

from c102_mutation_attack import (  # noqa: E402
    apply_mutation, find_occurrences, restore_saved_bytes, sha256_of,
)

TARGET = TREE / "src" / "t2pw" / "bench" / "acceptance.py"
TESTS = "tests/test_c110_negative_control_status.py"
PY = sys.executable

#: (name, old, new, what it removes, is this a CONTROL we expect to be caught?)
MUTATIONS = [
    ("control_artifacts",
     "    preserved = int(leg.artifacts_recorded or 0) > 0",
     "    preserved = True",
     "the >=1-preserved-artifact condition", True),
    ("only_operational_failure",
     "        bool(leg.operational_failure)\n        or termination in",
     "        False\n        or termination in",
     "the row's own operational_failure boolean", False),
    ("only_termination_reason",
     "        or termination in OPERATIONAL_TERMINATION_REASONS\n",
     "        or False\n",
     "D-005's OPERATIONAL_TERMINATION_REASONS reading", False),
    ("only_status",
     "        or status in _NC_CASUALTY_STATUSES\n",
     "        or False\n",
     "the row status timeout/error reading", False),
    ("only_failure_kind",
     "        or kind in _NC_CASUALTY_KINDS\n",
     "        or False\n",
     "the failure_kind timeout/crash/network/llm reading", False),
    ("only_boundary",
     "        or leg.boundary in _NC_CASUALTY_BOUNDARIES\n",
     "        or False\n",
     "classify_strict_boundary's reading", False),
]

ORIGINAL = sha256_of(TARGET.read_bytes())
print(f"target            : {TARGET}")
print(f"sha256 BEFORE all : {ORIGINAL}")
print()

BASETEMP.mkdir(parents=True, exist_ok=True)


def run_tests(tag: str):
    proc = subprocess.run(
        [PY, "-m", "pytest", "-q", f"--basetemp={BASETEMP / tag}", TESTS],
        cwd=str(TREE), capture_output=True, text=True,
    )
    tail = [ln for ln in proc.stdout.splitlines() if ln.strip()][-1:]
    return proc.returncode, (tail[0] if tail else "(no output)")


code, tail = run_tests("baseline")
print(f"BASELINE (unmutated): exit={code}  {tail}")
if code != 0:
    print("!! baseline is not green -- every mutation below would be vacuous. STOP.")
    raise SystemExit(2)
print()

results = []
for name, old, new, removes, is_control in MUTATIONS:
    occurrences = find_occurrences(TARGET, old)
    if occurrences != 1:
        print(f"!! {name}: substitution matched {occurrences} times, skipped")
        results.append((name, removes, is_control, None, "NOT APPLIED"))
        continue
    saved = apply_mutation(TARGET, old, new)
    try:
        code, tail = run_tests(name)
    finally:
        restore_saved_bytes(TARGET, saved)
    caught = code != 0
    results.append((name, removes, is_control, caught, tail))
    flag = "CAUGHT " if caught else "GREEN  "
    print(f"  {flag} {name:<26} removes {removes}")
    print(f"          -> exit={code}  {tail}")

print()
print(f"sha256 AFTER  all : {sha256_of(TARGET.read_bytes())}")
print(f"byte-identical restore : {sha256_of(TARGET.read_bytes()) == ORIGINAL}")

print()
print("=" * 78)
print("VERDICT")
print("=" * 78)
control = [r for r in results if r[2]]
real = [r for r in results if not r[2]]
print(f"  control mutation caught (harness works) : "
      f"{all(r[3] for r in control)}")
unpinned = [r for r in real if r[3] is False]
print(f"  condition-3 readings deleted one at a time : {len(real)}")
print(f"  of those, NOT caught by the author's suite : {len(unpinned)}")
for name, removes, _c, _caught, _tail in unpinned:
    print(f"      GREEN after deleting {removes}")
print("=" * 78)
