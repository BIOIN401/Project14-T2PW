"""C-106 / F-152: does the count parse read prose, or pytest's summary line?

G9 proof for C-106 section 3.3, and REV-106 A8's "both directions".

**This runs the REAL `c102_goldreaders_split.py`, end to end, twice** -- once as
it stands at base `c7fb5c5` and once at the C-106 tip -- over synthetic scenario
files built for the purpose. It is a behavioural proof, not a symbol check: what
is compared is the exit code, the abort, and the counts each version records.

The only surgery applied to either version is swapping the hard-coded 22-file
`FILES` block for the scenario files, and **the identical surgery is applied to
both arms**, so the comparison is fair. Nothing else in either script is edited.

Three scenarios, because fixing the false positive by counting nothing would be
a worse defect than the one being fixed (REV-106 A8):

* ``green_with_warning_text`` -- every test PASSES, and the output carries the
  prose "3 errors". This is F-152 exactly. Base must record ``errors=3`` and
  ABORT; tip must record ``errors=0`` and not abort.
* ``genuine_red`` -- a real failing test and no prose at all. **Both** arms must
  record ``failed=1`` and fold it into the totals. A tip that loses this has
  replaced a false positive with a false negative.
* ``red_with_errors_prose`` -- a real failing test whose FAILURE MESSAGE contains
  "3 errors". F-152 names this one: at base the genuine red is reported as an
  infrastructure failure and the gate stops early instead of counting it.

Usage::

    <python> c106_f152_scenarios.py <worktree-root> <base-sha>
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
BASE_SHA = sys.argv[2]
SPLIT = "docs/pwml_recovery_sprint/evidence/c102_goldreaders_split.py"
PY = sys.executable

SCEN_DIR = ROOT / "c106_scenarios"
WORK = ROOT / "c106_scenario_work"

GREEN_WITH_WARNING = '''"""Scenario: a GREEN file whose output contains the prose "3 errors"."""
import warnings


def test_one_passes():
    warnings.warn(UserWarning("upstream payload reported 3 errors and recovered"))
    assert True


def test_two_passes():
    assert True
'''

GENUINE_RED = '''"""Scenario: a genuine red, with no prose anywhere near the word errors."""


def test_one_passes():
    assert True


def test_two_really_fails():
    assert 1 == 2, "a real failure with no interesting words in it"
'''

RED_WITH_ERRORS_PROSE = '''"""Scenario: a genuine red whose FAILURE MESSAGE contains "3 errors"."""


def test_one_passes():
    assert True


def test_two_fails_and_says_errors():
    assert False, "validation of the payload found 3 errors in the manifest"
'''

SCENARIOS = {
    "green_with_warning_text": GREEN_WITH_WARNING,
    "genuine_red": GENUINE_RED,
    "red_with_errors_prose": RED_WITH_ERRORS_PROSE,
}

FILES_BLOCK = re.compile(r'FILES = """.*?""".split\(\)', re.DOTALL)


def variant(source: str, rel_test: str) -> str:
    """Swap ONLY the FILES block. Applied identically to base and to tip."""
    replacement = f'FILES = """\n{rel_test}\n""".split()'
    new, count = FILES_BLOCK.subn(lambda _m: replacement, source, count=1)
    assert count == 1, "the FILES block did not match -- refusing to guess"
    return new


def base_source() -> str:
    proc = subprocess.run(
        ["git", "show", f"{BASE_SHA}:{SPLIT}"],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", check=True,
    )
    return proc.stdout


def run(script: Path, basetemp: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [PY, "-u", str(script), str(ROOT), str(basetemp)],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return proc.returncode, proc.stdout + proc.stderr


def summarise(out: str) -> str:
    row = [ln.strip() for ln in out.splitlines() if re.match(r"^\s*\d+\.\s", ln)]
    aborted = "INFRASTRUCTURE FAILURE" in out
    files = ""
    for line in out.splitlines():
        if line.startswith("files run"):
            files = line.split(":")[1].strip()
    return (f"aborted={aborted!s:<5}  files_reported={files or '0 (aborted)':<12}  "
            f"row={row[0] if row else '(none)'}")


def main() -> int:
    SCEN_DIR.mkdir(exist_ok=True)
    WORK.mkdir(exist_ok=True)
    (SCEN_DIR / "__init__.py").unlink(missing_ok=True)

    tip_src = (ROOT / SPLIT).read_text(encoding="utf-8")
    base_src = base_source()
    print(f"base sha : {BASE_SHA}")
    base_scans_everything = "summary_counts" not in base_src
    tip_scopes = "summary_counts" in tip_src
    print(f"base scans the whole stream  : {base_scans_everything}")
    print(f"tip  scopes to a summary line: {tip_scopes}")
    # Stated for orientation only. NEITHER of these is the proof -- G9 refuses
    # symbol presence and absence alike. The proof is the six behavioural rows.

    bad = 0
    for name, body in SCENARIOS.items():
        test_file = SCEN_DIR / f"test_{name}.py"
        test_file.write_text(body, encoding="utf-8")
        rel = f"c106_scenarios/test_{name}.py"

        print(f"\nSCENARIO {name}")
        for arm, src in (("BASE", base_src), ("TIP ", tip_src)):
            script = WORK / f"split_{arm.strip().lower()}_{name}.py"
            script.write_text(variant(src, rel), encoding="utf-8")
            code, out = run(script, WORK / f"bt_{arm.strip().lower()}_{name}")
            print(f"  {arm} exit={code}  {summarise(out)}")
            if "INFRASTRUCTURE FAILURE" in out:
                for line in out.splitlines():
                    if "INFRASTRUCTURE FAILURE" in line:
                        print(f"       abort : {line.strip()[:150]}")

    # ---- the verdicts, asserted rather than eyeballed ----------------------
    print("\n---- VERDICTS ----")
    expectations = [
        # scenario, arm, must_abort, must_show
        ("green_with_warning_text", "BASE", True, "errors=3"),
        ("green_with_warning_text", "TIP", False, "errors=0"),
        ("genuine_red", "BASE", False, "failed=1"),
        ("genuine_red", "TIP", False, "failed=1"),
        ("red_with_errors_prose", "BASE", True, "errors=3"),
        ("red_with_errors_prose", "TIP", False, "failed=1"),
    ]
    for scen, arm, must_abort, must_show in expectations:
        src = base_src if arm == "BASE" else tip_src
        script = WORK / f"v_{arm.lower()}_{scen}.py"
        script.write_text(variant(src, f"c106_scenarios/test_{scen}.py"), encoding="utf-8")
        code, out = run(script, WORK / f"vb_{arm.lower()}_{scen}")
        aborted = "INFRASTRUCTURE FAILURE" in out
        shown = must_show in out
        ok = (aborted == must_abort) and shown
        bad += 0 if ok else 1
        print(f"  {'OK  ' if ok else 'FAIL'} {scen:24s} {arm:4s} "
              f"aborted={aborted!s:<5}(want {must_abort!s:<5}) contains {must_show!r}={shown}")

    shutil.rmtree(SCEN_DIR, ignore_errors=True)
    shutil.rmtree(WORK, ignore_errors=True)
    if bad:
        print(f"\n{bad} expectation(s) not met -- this is a finding, report it")
        return 1
    print("\nAll six expectations met: the false positive is gone in BOTH directions,")
    print("and a genuine red is still counted and still folded into the totals.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        shutil.rmtree(SCEN_DIR, ignore_errors=True)
        shutil.rmtree(WORK, ignore_errors=True)
