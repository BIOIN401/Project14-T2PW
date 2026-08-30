"""C-104 half 2: measure the split-gate abort guard, BASE against TIP, five ways.

D-083 follow-on 2 says a setup error is never a legitimate outcome of the
gold-readers split gate. The guard as C-102 shipped it fires on an unexpected
exit code but NOT on the original F-114 condition, because it was specified as
"nonzero exit with nothing failed AND NOTHING ERRORED" and in that scenario
errors are present. This probe measures that, rather than asserting it.

It runs the REAL driver -- ``c102_goldreaders_split.py``, unmodified, both the
committed BASE revision and the working-tree TIP -- against a synthetic tree
whose ``tests/`` directory carries the driver's own 22 filenames, with file 1
crafted per scenario. Only ``sys.argv[1]`` changes; no driver is edited and no
guard is reimplemented here.

**The literal F-114 condition is unreachable through this driver and that is
deliberate**: the driver calls ``BASETEMP.mkdir(parents=True, exist_ok=True)``
itself, which is the C-102 fix. Scenario 1 therefore reproduces the *shape and
the error*, not the cause -- a ``tmp_path`` fixture that raises the same
``FileNotFoundError`` the missing parent raised, so tests error in SETUP and the
file exits 1 with ``errors>0`` and ``failed=0``. That is exactly the state the
committed
``c102_goldreaders_split_r1.attempt1-missing-basetemp-parent.log`` was in when it
lost 71 tests and still exited 0.

The last two scenarios are the preservation half and they are not optional. A
guard that aborts a healthy run, or one that aborts on a genuine red test
instead of folding it into the totals, is worse than the gap it closes -- the
committed C-102 split run carried two real failures and had to keep going.

Usage::

    <python> c104_split_guard_probe.py <worktree-root> [<base-sha>]
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
BASE_SHA = sys.argv[2] if len(sys.argv) > 2 else "36f773c"
DRIVER_REL = "docs/pwml_recovery_sprint/evidence/c102_goldreaders_split.py"
PY = sys.executable

TIP_DRIVER = ROOT / DRIVER_REL

# The driver's own file list, parsed out of its source so this probe cannot
# drift from it.
_source = TIP_DRIVER.read_text(encoding="utf-8")
_match = re.search(r'FILES = """(.*?)"""', _source, re.S)
assert _match, "could not find the FILES literal in the driver"
FILES = _match.group(1).split()
assert len(FILES) == 22, f"expected 22 files, parsed {len(FILES)}"

PASSING = """def test_one():
    assert True


def test_two():
    assert True
"""

# Scenario 1. The F-114 SHAPE: tests error in SETUP, at exit code 1, with
# nothing failed. `tmp_path` is overridden with a fixture raising the very error
# a missing --basetemp parent raised, so two tests pass and three are lost.
F114_SETUP_ERRORS = """import pytest


@pytest.fixture
def tmp_path():
    raise FileNotFoundError(
        "[Errno 2] No such file or directory: "
        "'C:\\\\t\\\\bt\\\\does-not-exist\\\\g01' -- the F-114 condition"
    )


def test_survives_one():
    assert True


def test_survives_two():
    assert True


def test_lost_one(tmp_path):
    assert tmp_path


def test_lost_two(tmp_path):
    assert tmp_path


def test_lost_three(tmp_path):
    assert tmp_path
"""

# Scenario 2. REV-102's planted bad import: a collection error, which pytest
# reports at an exit code outside (0, 1). This must keep firing.
BAD_IMPORT = """import t2pw_c104_no_such_module_planted_by_the_probe  # noqa: F401


def test_never_runs():
    assert True
"""

# Scenario 3. An unexpected exit code with ZERO errors -- pytest's exit 5, "no
# tests collected". This isolates the exit-code disjunct so the new `errors`
# clause cannot be what is doing the work, and proves the old condition is still
# live rather than subsumed.
NO_TESTS = """# A module with no tests at all: pytest exits 5, errors=0, failed=0.
VALUE = 1
"""

# Scenario 4. A GENUINE red test: exit 1, failed=1, errors=0. The gate must fold
# this into its totals and keep going -- the committed C-102 split run had two.
GENUINE_FAILURE = """def test_passes():
    assert True


def test_really_fails():
    assert 1 == 2


def test_also_passes():
    assert True
"""

SCENARIOS = [
    # name, file-1 content, expect_abort_on_base, expect_abort_on_tip, why
    ("f114_setup_errors", F114_SETUP_ERRORS, False, True,
     "CASE 1: fires on setup errors -- the gap D-083 follow-on 2 closes"),
    ("bad_import", BAD_IMPORT, True, True,
     "CASE 2: still fires on an unexpected exit code (planted bad import)"),
    ("exit_code_no_errors", NO_TESTS, True, True,
     "CASE 2b: the exit-code disjunct still fires with errors=0 (not subsumed)"),
    ("genuine_failure", GENUINE_FAILURE, False, False,
     "CASE 3a: a real red test still folds into the totals"),
    ("all_clean", PASSING, False, False,
     "CASE 3b: a clean run still completes -- the preservation case"),
]


def build_tree(work: Path, name: str, first_file: str) -> Path:
    root = work / ("tree_" + name)
    (root / "tests").mkdir(parents=True, exist_ok=True)
    for index, rel in enumerate(FILES):
        body = first_file if index == 0 else PASSING
        (root / rel).write_text(body, encoding="utf-8")
    return root


def run_driver(driver: Path, tree: Path, basetemp: Path) -> dict:
    proc = subprocess.run(
        [PY, str(driver), str(tree), str(basetemp)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    out = proc.stdout + proc.stderr
    lines = [line for line in out.splitlines() if re.match(r"\s*\d+\. ", line)]
    totals = [line for line in out.splitlines() if line.startswith("split totals")]
    infra = [line for line in out.splitlines() if "INFRASTRUCTURE FAILURE" in line]
    return {
        "exit": proc.returncode,
        "aborted": bool(infra),
        "message": infra[0].strip() if infra else "",
        "files_reported": len(lines),
        "first_line": lines[0] if lines else "",
        "totals": totals[0] if totals else "",
        "out": out,
    }


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="c104probe_"))
    base_driver = work / "base_c102_goldreaders_split.py"
    blob = subprocess.run(
        ["git", "show", f"{BASE_SHA}:{DRIVER_REL}"],
        cwd=str(ROOT), capture_output=True, check=True,
    ).stdout
    base_driver.write_bytes(blob)

    print(f"worktree      : {ROOT}")
    print(f"base sha      : {BASE_SHA}")
    print(f"base driver   : extracted to {base_driver} ({len(blob)} bytes)")
    print(f"tip driver    : {TIP_DRIVER}")
    print(f"files per tree: {len(FILES)}")
    print(f"scratch       : {work}")
    print()

    rows = []
    mismatches = 0
    try:
        for name, first, want_base, want_tip, why in SCENARIOS:
            tree = build_tree(work, name, first)
            print("=" * 78)
            print(f"SCENARIO {name} -- {why}")
            print("=" * 78)
            for which, driver, want in (("BASE", base_driver, want_base),
                                        ("TIP ", TIP_DRIVER, want_tip)):
                basetemp = work / "bt" / f"{name}_{which.strip()}"
                got = run_driver(driver, tree, basetemp)
                ok = got["aborted"] == want
                mismatches += 0 if ok else 1
                print(f"  {which}  exit={got['exit']}  aborted={got['aborted']}  "
                      f"expected_abort={want}  files_reported={got['files_reported']}  "
                      f"{'OK' if ok else '*** MISMATCH ***'}")
                if got["first_line"]:
                    print(f"        file 1 : {got['first_line'].strip()}")
                if got["totals"]:
                    print(f"        totals : {got['totals'].strip()}")
                if got["message"]:
                    print(f"        abort  : {got['message']}")
                rows.append((name, which.strip(), got["exit"], got["aborted"], want, ok))
                if name == "f114_setup_errors":
                    print("        ---- full driver output (the before/after pair) ----")
                    for line in got["out"].splitlines():
                        print("        | " + line)
            print()
    finally:
        shutil.rmtree(work, ignore_errors=True)
        print(f"scratch removed: {not work.exists()}")

    print()
    print(f"{'scenario':22s} {'driver':6s} {'exit':>4s} {'aborted':>8s} "
          f"{'expected':>9s}  verdict")
    for name, which, code, aborted, want, ok in rows:
        print(f"{name:22s} {which:6s} {code:>4d} {str(aborted):>8s} "
              f"{str(want):>9s}  {'OK' if ok else 'MISMATCH'}")
    print()
    if mismatches:
        print(f"PROBE FAILED: {mismatches} mismatch(es)")
        return 1
    print(f"PROBE PASSED: {len(rows)} runs, every verdict as specified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
