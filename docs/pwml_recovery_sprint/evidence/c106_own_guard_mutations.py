"""C-106: attack C-106's OWN guards -- F-144 / R7, and REV-106 A13.

A guard is not evidence until someone has tried to defeat it. Every load-bearing
assertion C-106 adds is forced false here, one at a time, and the test that
claims to detect it must go RED. A guard that stays green under its own mutation
is decorative and the card has not discharged its purpose.

Six mutations, one per guard:

  own-1  reintroduce ``write_text(newline="")`` in the harness restore path --
         REV-106 A13 by name. Tests 03, 04 and 05 must go RED.
  own-2  corrupt one mutation's ``old`` string so it matches ZERO times. This is
         the dead-mutation case: at base it would produce a green suite that
         reads as a pass. Test 02[M2] must go RED.
  own-3  delete the harness's baseline precondition. Test 06 must go RED. This
         is the change that would satisfy C-106's headline and destroy its
         purpose, so it gets its own mutation.
  own-4  put ``git checkout --`` back in the restore path. Test 07 must go RED.
  own-5  relax a derived census pin from ``==`` to ``>=`` -- the change F-151
         proposed and REV-104 refused. Test 09 must go RED.
  own-6  make the census pin stale by one, as a committed benchmark run would.
         Test 08 must go RED. This is F-151 itself, replayed.

Applied and restored with the harness's OWN byte-exact primitives, so this file
dogfoods the thing it is testing. The harness module is imported BEFORE any
mutation, so own-1 cannot sabotage this driver's own restores.

`src/t2pw/bench/acceptance.py` is snapshotted and force-restored after every
round: own-1 deliberately breaks the restore path, and test 04 mutates the real
module, so that round can and does leave it damaged inside the child process.

Usage::

    <python> c106_own_guard_mutations.py <worktree-root>
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
PY = sys.executable
HARNESS_REL = "docs/pwml_recovery_sprint/evidence/c102_mutation_attack.py"
NEWTEST_REL = "tests/test_c106_mutation_harness_executable.py"
C102_REL = "tests/test_c102_coverage_denominator.py"
ACCEPTANCE_REL = "src/t2pw/bench/acceptance.py"

spec = importlib.util.spec_from_file_location("c102_mutation_attack", ROOT / HARNESS_REL)
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def crlf(data: bytes) -> int:
    return data.count(b"\r\n")


OWN_MUTATIONS = [
    (
        "own-1-d084-textmode-restore",
        "the D-084 defect reintroduced: read_text + write_text(newline='')",
        HARNESS_REL,
        '    path.write_bytes(saved)\n    after = path.read_bytes()\n',
        '    path.write_bytes(saved)  # MUTATION own-1\n'
        '    _t = path.read_text(encoding="utf-8")\n'
        '    path.write_text(_t, encoding="utf-8", newline="")\n'
        '    after = path.read_bytes()\n',
        ["test_03", "test_04", "test_05"],
    ),
    (
        "own-2-dead-mutation",
        "M2's substitution is corrupted so it matches ZERO times",
        HARNESS_REL,
        '        \'    text = str(term)\\n    head = text.split("(")[0].strip()\\n\',\n',
        '        \'    text = str(term)\\n    head = text.split("(")[0].strip()  # own-2\\n\',\n',
        ["test_02"],
    ),
    (
        "own-3-baseline-precondition-deleted",
        "the harness stops refusing a red baseline",
        HARNESS_REL,
        "    assert code == 0, (\n",
        "    assert True, (  # MUTATION own-3 -- precondition deleted\n",
        ["test_06"],
    ),
    (
        "own-4-git-checkout-restored",
        "git checkout -- is back in the restore path",
        HARNESS_REL,
        '        after = path.read_bytes()\n',
        '        git(root, "checkout", "--", rel)  # MUTATION own-4\n'
        '        after = path.read_bytes()\n',
        ["test_07"],
    ),
    (
        "own-5-derived-pin-relaxed",
        "a derived census pin is relaxed from == to >=",
        C102_REL,
        "    assert legs == 72\n",
        "    assert legs >= 72  # MUTATION own-5\n",
        ["test_09"],
    ),
    (
        "own-6-census-pin-stale",
        "the census pin goes stale by one, exactly as a committed run makes it",
        C102_REL,
        "    assert len(paths) >= 72,",
        "    assert len(paths) >= 71,  # MUTATION own-6",
        ["test_08"],
    ),
]


def run_newtest() -> tuple[int, str]:
    proc = subprocess.run(
        [PY, "-m", "pytest", NEWTEST_REL, "-q", "--no-header", "-rf",
         "--basetemp=C:/t/bt/c106own"],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return proc.returncode, proc.stdout + proc.stderr


def main() -> int:
    acceptance = ROOT / ACCEPTANCE_REL
    # Byte snapshots of everything this driver can touch, taken BEFORE anything
    # is mutated. Restores are verified against these, not against HEAD.
    snapshots = {
        rel: (ROOT / rel).read_bytes()
        for rel in (HARNESS_REL, NEWTEST_REL, C102_REL, ACCEPTANCE_REL)
    }
    acceptance_snapshot = snapshots[ACCEPTANCE_REL]
    print("byte snapshots taken before any mutation:")
    for rel, snap in snapshots.items():
        print(f"    {rel:58s} bytes={len(snap):>6} crlf={crlf(snap):>5} "
              f"sha256={sha(snap)[:16]}")

    code, out = run_newtest()
    tail = [ln for ln in out.splitlines() if re.search(r"\d+ (passed|failed)", ln)]
    print(f"\n=== BASELINE (unmutated tip) === exit={code}  {tail}")
    if code != 0:
        print("the new test file is not green before mutation; nothing below means anything")
        return 3

    bad = 0
    for name, what, rel, old, new, expect_red in OWN_MUTATIONS:
        path = ROOT / rel
        before = path.read_bytes()
        print(f"\n=== {name}: {what}")
        print(f"    target={rel}  before: bytes={len(before)} crlf={crlf(before)} "
              f"sha256={sha(before)[:16]}")
        try:
            saved = H.apply_mutation(path, old, new)
        except ValueError as exc:
            print(f"    ABORT -- {exc}")
            bad += 1
            continue
        try:
            code, out = run_newtest()
        finally:
            H.restore_saved_bytes(path, saved)
            # own-1 breaks the restore path itself, and test 04 mutates the real
            # module, so the child can leave acceptance.py damaged. Force it back
            # from the snapshot regardless of what the child did.
            if acceptance.read_bytes() != acceptance_snapshot:
                print("    !! acceptance.py was left damaged by the child -- "
                      "force-restoring from snapshot")
                acceptance.write_bytes(acceptance_snapshot)

        failed_names = sorted(set(re.findall(r"FAILED [^:]+::(\w+)", out)))
        tail = [ln for ln in out.splitlines() if re.search(r"\d+ (passed|failed)", ln)]
        went_red = code != 0
        hit = [e for e in expect_red if any(f.startswith(e) for f in failed_names)]
        missed = [e for e in expect_red if e not in hit]
        ok = went_red and not missed

        print(f"    exit={code}  {'RED (guard held)' if went_red else 'GREEN -- GUARD IS VACUOUS'}")
        print(f"    {tail}")
        print(f"    expected red: {expect_red}")
        print(f"    actually red: {failed_names}")
        if missed:
            print(f"    MISSED      : {missed}")

        after = path.read_bytes()
        print(f"    restored: bytes={len(after)} crlf={crlf(after)} sha256={sha(after)[:16]}")
        print(f"    byte-exact={after == before}  crlf-preserved={crlf(after) == crlf(before)}")
        if after != before:
            print("    RESTORE WAS NOT BYTE-EXACT -- D-084 failure in this driver")
            bad += 1
        if not ok:
            bad += 1
        print(f"    VERDICT: {'OK' if ok else 'FAIL'}")

    # Final state: everything back, and the suite green again.
    #
    # ATTEMPT 1 OF THIS CHECK WAS WRONG and its log is preserved beside this one
    # as `c106_own_guard_mutations.attempt1-porcelain-vs-head.log`. It asserted
    # `git status --porcelain` was EMPTY for all four files -- but three of them
    # are C-106's own in-flight changes on an uncommitted branch, so a clean
    # porcelain there was never achievable and the driver reported FAILED while
    # all six guards had in fact gone RED correctly. The bug was in the check,
    # not in the guards.
    #
    # The right question is not "does this file match HEAD" but "does this file
    # match the BYTES IT HAD BEFORE THIS DRIVER TOUCHED IT" -- which is D-084's
    # own question, and the one the sha256 comparison below asks. `acceptance.py`
    # is the single file that must ALSO be clean against HEAD, because C-106
    # changes no production line and the mutations only borrow it.
    print(f"\n=== AFTER ALL MUTATIONS ===")
    restored_ok = True
    for rel, snap in snapshots.items():
        now = (ROOT / rel).read_bytes()
        same = now == snap
        restored_ok &= same
        print(f"    {rel:58s} byte-identical={same!s:<5} "
              f"crlf={crlf(now)} sha256={sha(now)[:16]}")
    porcelain = subprocess.run(
        ["git", "status", "--porcelain", "--", ACCEPTANCE_REL],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()
    print(f"    git status --porcelain on {ACCEPTANCE_REL}: {porcelain!r}")
    print("      ^ this one, and only this one, must also be clean against HEAD:")
    print("        C-106 changes no production line, it only borrows the file.")
    code, out = run_newtest()
    tail = [ln for ln in out.splitlines() if re.search(r"\d+ (passed|failed)", ln)]
    print(f"    new test file after restore: exit={code}  {tail}")
    if code or porcelain or not restored_ok:
        bad += 1

    if bad:
        print(f"\nOWN-GUARD ATTACK FAILED: {bad} problem(s)")
        return 1
    print(f"\nOWN-GUARD ATTACK PASSED: all {len(OWN_MUTATIONS)} guards went RED under "
          f"their own mutation, every restore byte-exact, tree clean, suite green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
