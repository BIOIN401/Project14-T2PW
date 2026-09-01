"""C-110 round 1 -- prove each guard is PINNED BY A TEST, not merely present.

REV-110's second finding: deleting **any one** of the five condition-3 readings
left the 22-test suite fully green, because the nine row-level mutations flipped
two fields at once and the redundancy hid every individual guard. The guards were
real; no test held them.

This is the answer to that, and it is a claim about the TESTS, not the code. For
each guard it deletes exactly that guard from ``acceptance.py`` and runs the new
test file. **A guard that is pinned turns the suite RED.** A guard that survives
green is unpinned and is reported as such.

Mutations, twelve:

* **five condition-3 readings** -- ``operational_failure``, ``termination_reason``,
  ``status``, ``failure_kind``, ``boundary``, each deleted alone;
* **the artifact condition**;
* **condition 2's two halves** -- the ``declared`` narrowing and the
  ``classified`` check -- which are round 1's blocking finding;
* **the round-1 regressions themselves**: put ``contract`` back into
  ``_NC_DECLINE_KINDS``, and put ``or bool(codes)`` back into ``declared``, and
  put ``and not codes`` back onto the indeterminate branch. Those three restore
  the exact defect REV-110 found, and each must turn the suite red.
* **one CONTROL** -- a comment-only edit that changes no behaviour and must stay
  GREEN. Without it a harness that reported red for everything would look
  perfect.

D-084: mutations are applied and reverted with :func:`apply_mutation` /
:func:`restore_saved_bytes` from ``c102_mutation_attack.py`` -- byte-exact
restore, replayed from saved bytes and proved by sha256 and CRLF count. No
``git checkout --``, no text-mode rewrite.

Usage::

    <venv-python> c110_r1_guard_mutations.py <worktree-root> <basetemp-parent>
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
BASETEMP = Path(sys.argv[2])
BASETEMP.mkdir(parents=True, exist_ok=True)
PY = sys.executable

TARGET = ROOT / "src" / "t2pw" / "bench" / "acceptance.py"
TESTS = "tests/test_c110_negative_control_status.py"

_spec = importlib.util.spec_from_file_location(
    "_c102_mutation_attack",
    ROOT / "docs" / "pwml_recovery_sprint" / "evidence" / "c102_mutation_attack.py",
)
_harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_harness)
apply_mutation = _harness.apply_mutation
restore_saved_bytes = _harness.restore_saved_bytes
sha256_of = _harness.sha256_of
crlf_count = _harness.crlf_count


#: ``(name, old, new, expectation)``. ``expectation`` is "RED" for a guard that
#: must be pinned and "GREEN" for the control.
MUTATIONS = [
    (
        "cond3-1-operational_failure",
        "        bool(leg.operational_failure)\n        or termination in OPERATIONAL_TERMINATION_REASONS\n",
        "        termination in OPERATIONAL_TERMINATION_REASONS\n",
        "RED",
    ),
    (
        "cond3-2-termination_reason",
        "        or termination in OPERATIONAL_TERMINATION_REASONS\n        or status in _NC_CASUALTY_STATUSES\n",
        "        or status in _NC_CASUALTY_STATUSES\n",
        "RED",
    ),
    (
        "cond3-3-status",
        "        or status in _NC_CASUALTY_STATUSES\n        or kind in _NC_CASUALTY_KINDS\n",
        "        or kind in _NC_CASUALTY_KINDS\n",
        "RED",
    ),
    (
        "cond3-4-failure_kind",
        "        or kind in _NC_CASUALTY_KINDS\n        or leg.boundary in _NC_CASUALTY_BOUNDARIES\n",
        "        or leg.boundary in _NC_CASUALTY_BOUNDARIES\n",
        "RED",
    ),
    (
        "cond3-5-boundary",
        "        or kind in _NC_CASUALTY_KINDS\n        or leg.boundary in _NC_CASUALTY_BOUNDARIES\n    )\n",
        "        or kind in _NC_CASUALTY_KINDS\n    )\n",
        "RED",
    ),
    (
        "cond3-artifacts",
        "    if not preserved:\n        blocked.append(NC_BLOCK_NO_ARTIFACTS)\n",
        "    if False:\n        blocked.append(NC_BLOCK_NO_ARTIFACTS)\n",
        "RED",
    ),
    (
        "cond2-declared-removed",
        "    stated = bool(message) and declared\n",
        "    stated = bool(message)\n",
        "RED",
    ),
    (
        "cond2-classified-removed",
        "    if not classified:\n        blocked.append(NC_BLOCK_INDETERMINATE)\n",
        "    if False:\n        blocked.append(NC_BLOCK_INDETERMINATE)\n",
        "RED",
    ),
    (
        "regression-contract-back-in",
        '_NC_DECLINE_KINDS: Tuple[str, ...] = ("no_reactions",)\n',
        '_NC_DECLINE_KINDS: Tuple[str, ...] = ("no_reactions", "contract")\n',
        "RED",
    ),
    (
        "regression-codes-satisfy-declared",
        "    declared = kind in _NC_DECLINE_KINDS or termination == SCIENTIFICALLY_UNRECOVERABLE\n",
        "    declared = kind in _NC_DECLINE_KINDS or termination == SCIENTIFICALLY_UNRECOVERABLE or bool(codes)\n",
        "RED",
    ),
    (
        "regression-codes-suppress-indeterminate",
        "    if not classified:\n",
        "    if not classified and not codes:\n",
        "RED",
    ),
    (
        "CONTROL-comment-only",
        "    if not _empty_is_correct(case):\n        return None\n",
        "    if not _empty_is_correct(case):  # control mutation, no behaviour change\n        return None\n",
        "GREEN",
    ),
]


def _run_tests(tag: str):
    return subprocess.run(
        [PY, "-m", "pytest", "-q", TESTS, f"--basetemp={BASETEMP / tag}"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def main() -> int:
    original = TARGET.read_bytes()
    print("=" * 78)
    print("C-110 round 1 -- is each guard PINNED BY A TEST?")
    print("=" * 78)
    print(f"target      : {TARGET.relative_to(ROOT)}")
    print(f"sha256      : {sha256_of(original)}")
    print(f"crlf lines  : {crlf_count(original)}")
    print()

    baseline = _run_tests("baseline")
    print(f"BASELINE (unmutated): exit={baseline.returncode}")
    print(f"  {baseline.stdout.strip().splitlines()[-1] if baseline.stdout.strip() else ''}")
    if baseline.returncode != 0:
        print("BASELINE IS RED -- every mutation below would be vacuous. Stopping.")
        print(baseline.stdout[-3000:])
        return 1
    print()

    results = []
    for name, old, new, expectation in MUTATIONS:
        try:
            saved = apply_mutation(TARGET, old, new)
        except ValueError as exc:
            print(f"{name:<38} ANCHOR FAILED: {exc}")
            results.append((name, expectation, "ANCHOR", False))
            continue
        try:
            result = _run_tests(name)
            observed = "GREEN" if result.returncode == 0 else "RED"
            tail = ""
            if observed == "RED":
                failed = [
                    line.split("::")[-1].split()[0]
                    for line in result.stdout.splitlines()
                    if line.startswith("FAILED")
                ]
                tail = f"  killed by: {', '.join(sorted(set(failed))[:3])}"
                if len(set(failed)) > 3:
                    tail += f" (+{len(set(failed)) - 3} more)"
            ok = observed == expectation
            print(f"{name:<38} want={expectation:<6} got={observed:<6} "
                  f"{'OK' if ok else '*** UNPINNED ***'}{tail}")
            results.append((name, expectation, observed, ok))
        finally:
            restore_saved_bytes(TARGET, saved)

    after = TARGET.read_bytes()
    print()
    print("=" * 78)
    print(f"restored sha256 : {sha256_of(after)}")
    print(f"identical       : {sha256_of(after) == sha256_of(original)}")
    print(f"crlf preserved  : {crlf_count(after) == crlf_count(original)}")
    unpinned = [name for name, _, _, ok in results if not ok]
    print(f"mutations       : {len(results)}")
    print(f"unpinned        : {len(unpinned)}  {unpinned}")
    print("RESULT          : " + ("ALL PINNED" if not unpinned else "UNPINNED GUARDS REMAIN"))
    return 1 if unpinned or sha256_of(after) != sha256_of(original) else 0


if __name__ == "__main__":
    raise SystemExit(main())
