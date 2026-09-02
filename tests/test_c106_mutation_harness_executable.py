"""C-106 -- the sprint's mutation-attack harness is executable and non-vacuous.

**G9 LABEL: NEW ACCEPTANCE TEST, NEW CAPABILITY. No base failure is claimed and
none is fabricated.** Nothing in this file corrects a pre-existing observable
behaviour, so there is no behaviour to fail at base `c7fb5c5`. The two
corrections C-106 does make -- the census re-pin (section 3.1) and the F-152
count parse (section 3.3) -- carry their own behavioural base failures elsewhere:

  * `evidence/c106_c102_base_red.log`   (base: 2 failed, 12 passed)
    against `evidence/c106_c102_tip_green.log` (tip: 14 passed)
  * `evidence/c106_f152_scenarios.log`  (base aborts a GREEN file; tip does not,
    and both arms still count a genuine red)

Mislabelling either of those as new functionality would be a reject, so they are
NOT in this file. What IS here is a capability the sprint never had.

Why it exists
-------------
F-151 found that `evidence/c102_mutation_attack.py` **could not run at all**, and
had not been able to since `e77ad3d`: its baseline precondition asserts the
unmutated c102 suite is green, and a committed benchmark run had turned two
census pins red. D-078 and F-144 make mutation testing a required practice on
every card, so for a whole card the sprint's own instrument for proving guards
non-vacuous was itself unexercised -- and nothing anywhere said so. C-104's R5
was registered and statically verified but had never been driven.

REV-104 then measured that the harness violated **both** rows of D-084 in one
loop: a text-mode `write_text(newline="")` reverted LESS than it took (it rewrote
every CRLF in the target), and `git checkout --` reverted MORE -- and, worse,
MASKED the first, because it repaired the line endings as a side effect and left
`git status --porcelain` clean. A clean porcelain is exactly what a broken
restore produced, which is why the tests below assert sha256 and a CRLF count.

This file pins all of that as tests rather than as a review habit. Everything
here is STRUCTURAL and fast: it never runs the attack end to end, never runs
pytest, and never runs a benchmark. The end-to-end run is
`evidence/c106_mutation_attack.log` (all 8 mutations RED, including R5's first
ever pass through the driver).
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "docs" / "pwml_recovery_sprint" / "evidence"
HARNESS_PATH = EVIDENCE / "c102_mutation_attack.py"
C102_TESTS = REPO / "tests" / "test_c102_coverage_denominator.py"


def _load_harness():
    """Import the harness BY PATH, without running the attack.

    That the import is possible at all is part of what is being tested: at base
    the module read `sys.argv[1]` at import time, so importing it raised
    IndexError and no test could inspect the mutation set.
    """
    spec = importlib.util.spec_from_file_location("c102_mutation_attack", HARNESS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HARNESS = _load_harness()


# ---------------------------------------------------------------------------
# 1. The mutation set is intact and every substitution still applies.
#
#    THIS IS THE NON-VACUITY GUARD. A mutation whose substitution silently
#    matched zero times would leave the suite green and read as a pass -- the
#    exact F-144 failure the harness exists to prevent, occurring inside the
#    harness. This is also what catches a future refactor renaming a mutated
#    line out from under the attack set.
# ---------------------------------------------------------------------------
EXPECTED_MUTATIONS = ("M1", "M2", "M3", "M4", "M5", "M7", "R5", "M6")


def test_01_the_harness_imports_without_running_the_attack():
    assert [m[0] for m in HARNESS.MUTATIONS] == list(EXPECTED_MUTATIONS), \
        "the attack set changed; a mutation must never be dropped silently"
    # R5 is C-104's, carried in under D-083 follow-on 1 and kept under its
    # reviewer number. It is the one that had never been driven before C-106.
    assert "R5" in EXPECTED_MUTATIONS
    for name, what, rel, old, new in HARNESS.MUTATIONS:
        assert what.strip(), f"{name} has no description"
        assert (REPO / rel).is_file(), f"{name} targets a missing file: {rel}"
        assert old != new, f"{name} is the identity and would prove nothing"


@pytest.mark.parametrize("name,rel,old", [(m[0], m[2], m[3]) for m in HARNESS.MUTATIONS])
def test_02_every_mutation_substitution_matches_exactly_once(name, rel, old):
    """Exactly once -- not 'at least once'.

    Zero means the mutation is dead and the attack that 'passed' proved nothing.
    More than one means the attack changes more than it claims to and its RED is
    not attributable to the thing being tested.
    """
    found = HARNESS.find_occurrences(REPO / rel, old)
    assert found == 1, (
        f"mutation {name} matches {found} times in {rel}, not 1 -- the attack set "
        f"has drifted from the code it attacks and every 'RED' it reports is suspect"
    )


# ---------------------------------------------------------------------------
# 2. The restore replays SAVED BYTES -- D-084, pinned as a test.
# ---------------------------------------------------------------------------
def _census(data: bytes) -> tuple[int, int, int, str]:
    crlf = data.count(b"\r\n")
    return (len(data), crlf, data.count(b"\n") - crlf, hashlib.sha256(data).hexdigest())


def _crlf_fixture(target: Path) -> bytes:
    """120 CRLF lines with the marker appearing EXACTLY once.

    Exactly once matters: `apply_mutation` refuses a substitution that matches
    any other number of times, which is the harness own non-vacuity rule and
    the first thing a fixture for it must not violate. The first draft of this
    fixture repeated the marker 40 times and `apply_mutation` correctly refused
    it -- recorded here rather than quietly fixed, because the refusal was the
    guard working.
    """
    lines = [f"line {i:03d}" for i in range(119)]
    lines.insert(60, "MARKER HERE")
    body = "\r\n".join(lines) + "\r\n"
    target.write_bytes(body.encode("utf-8"))
    return target.read_bytes()


def test_03_restore_replays_saved_bytes_on_a_crlf_fixture(tmp_path):
    """A CRLF fixture, because CRLF is what this working tree actually has.

    `core.autocrlf=true` is set globally and `.gitattributes` carries no rule for
    these paths, so every tracked text file is LF in the object store and CRLF on
    disk. A restore that "works" on an LF fixture proves nothing about this tree.
    """
    target = tmp_path / "crlf_target.py"
    _crlf_fixture(target)
    before = _census(target.read_bytes())
    assert before[1] == 120 and before[2] == 0, "the fixture itself must be pure CRLF"

    saved = HARNESS.apply_mutation(target, "MARKER HERE\n", "MARKER MUTATED\n")
    mutated = target.read_bytes()
    assert b"MARKER MUTATED" in mutated, "the mutation did not apply"
    # The mutation changed ONE line. Every other line ending is untouched --
    # this is the half a text-mode round trip gets wrong.
    assert _census(mutated)[1] == 120, "applying the mutation changed line endings"
    assert _census(mutated)[2] == 0, "applying the mutation introduced bare LFs"

    HARNESS.restore_saved_bytes(target, saved)
    after = _census(target.read_bytes())
    assert after[3] == before[3], f"sha256 moved: {before[3]} -> {after[3]}"
    assert after[1] == before[1], f"CRLF count moved: {before[1]} -> {after[1]}"
    assert after[0] == before[0], f"byte count moved: {before[0]} -> {after[0]}"


def test_04_restore_is_byte_exact_on_the_real_mutated_module():
    """R9 in spirit: the REAL artifact the harness mutates, not a stand-in.

    Applies and restores one real mutation through the harness's own restore
    path. It runs no pytest and takes milliseconds; the `finally` guarantees the
    tree is put back even if an assertion fires mid-test.
    """
    name, _what, rel, old, new = HARNESS.MUTATIONS[0]
    assert name == "M1"
    target = REPO / rel
    before = target.read_bytes()
    before_census = _census(before)
    # The measured shape of this file. If it changes, that is a real event and
    # the number is here so somebody has to look at it.
    assert before_census[2] == 0, "acceptance.py has bare LFs on disk; investigate"

    saved = None
    try:
        saved = HARNESS.apply_mutation(target, old, new)
        assert target.read_bytes() != before, "the mutation did not change the file"
    finally:
        if saved is not None:
            HARNESS.restore_saved_bytes(target, saved)

    after = target.read_bytes()
    after_census = _census(after)
    assert after_census[3] == before_census[3], "restore was not byte-exact"
    assert after_census[1] == before_census[1], "restore changed the CRLF count"
    assert after == before

    # `git status --porcelain` is checked LAST and on purpose: it is the weakest
    # of these signals, not the strongest. A clean porcelain is precisely what
    # the broken `git checkout --` restore produced while it was silently
    # rewriting all 1673 line endings in this file.
    porcelain = subprocess.run(
        ["git", "status", "--porcelain", "--", rel],
        cwd=str(REPO), capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()
    assert porcelain == "", f"the real module was left dirty: {porcelain!r}"


def test_05_a_text_mode_round_trip_is_actually_caught_by_this_guard(tmp_path):
    """F-144 / R7: the guard above, mutated, and shown to go RED.

    If someone reintroduces `write_text(..., newline="")` -- the exact D-084
    defect C-106 removed -- test 03 and test 04 must fail. This test forces that
    defect and proves the assertion catches it, so the byte-exactness guard is
    non-vacuous by construction rather than by review habit.
    """
    target = tmp_path / "crlf_target.py"
    saved = _crlf_fixture(target)
    before = _census(saved)

    # THE DEFECT, reproduced exactly: read_text collapses CRLF to \n (universal
    # newlines), and write_text(newline="") then writes those bare \n straight
    # out. The content is held IDENTICAL, so every byte of the delta below is
    # line endings alone.
    text = target.read_text(encoding="utf-8")
    target.write_text(text, encoding="utf-8", newline="")
    damaged = _census(target.read_bytes())

    assert damaged[3] != before[3], (
        "a text-mode round trip did NOT change this file, so test 03's sha256 "
        "assertion could not detect the D-084 defect and is decorative"
    )
    assert damaged[1] == 0 and before[1] == 120, "the CRLF collapse is the defect"
    assert damaged[2] == before[1], "every CRLF became a bare LF"
    assert damaged[0] == before[0] - before[1], "the delta is exactly the CR bytes"

    # And the harness's own restore puts it back, which the broken one could not
    # do without shelling out to git.
    HARNESS.restore_saved_bytes(target, saved)
    assert _census(target.read_bytes())[3] == before[3]


# ---------------------------------------------------------------------------
# 3. The baseline precondition is present, load-bearing, and satisfiable.
# ---------------------------------------------------------------------------
def test_06_the_harness_still_refuses_to_certify_against_a_red_baseline():
    """The precondition must never be deleted to make the harness 'runnable'.

    Removing it is the single change that would satisfy C-106's headline while
    destroying its purpose: against a red suite every mutation goes RED for free
    and the harness certifies guards it never exercised. That would be a
    strictly worse instrument than the broken one, because it would certify.
    """
    source = HARNESS_PATH.read_text(encoding="utf-8")
    assert re.search(r"assert\s+code\s*==\s*0", source), (
        "the harness's baseline precondition is GONE. It is what makes every "
        "mutation result mean something. Fix the census pin instead -- see C-106"
    )
    # And it must still be reached before the mutation loop, not after it.
    precondition = source.index("assert code == 0")
    loop = source.index("for name, what, rel, old, new in MUTATIONS")
    assert precondition < loop, "the precondition no longer guards the mutation loop"


def test_07_the_restore_path_does_not_shell_out_to_git_checkout():
    """`git checkout --` reverts MORE than the mutation, and masked the damage.

    Prose about it in the docstring is fine and deliberate; a CALL is not. This
    inspects executable lines only.
    """
    code_lines = [
        line for line in HARNESS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    offenders = [
        line.strip() for line in code_lines
        if re.search(r"""["']checkout["']""", line)
    ]
    assert offenders == [], (
        f"the restore path shells out to git again: {offenders}. D-084: a restore "
        f"replays SAVED BYTES. git checkout reverts more, text mode reverts less"
    )


def test_08_the_census_pins_still_match_the_committed_corpus():
    """The root cause of F-151, guarded at its source.

    Committing a benchmark run grows the tracked `quarantine_report.json`
    population and turns the c102 census pins red -- which is what made the
    harness's baseline unsatisfiable for a whole card. This asserts the floor
    pin still equals the live census, so the drift is reported HERE, with an
    actionable message, rather than as an unexplained red in a file nobody runs.
    """
    listed = subprocess.run(
        ["git", "ls-files", "*quarantine_report.json"],
        cwd=str(REPO), capture_output=True, text=True, encoding="utf-8", check=True,
    )
    live = len([ln for ln in listed.stdout.splitlines() if ln.strip()])

    source = C102_TESTS.read_text(encoding="utf-8")
    floor = re.search(r"assert len\(paths\) >= (\d+)", source)
    assert floor, "the census floor assertion vanished from the c102 suite"
    pinned = int(floor.group(1))

    assert live == pinned, (
        f"the committed quarantine_report.json population is {live} but the c102 "
        f"suite pins {pinned}. A run was committed and the census pins are now "
        f"stale -- this is F-151 recurring. FIVE pins move together, not one: the "
        f"floor, test 10 `legs`, test 10 `withheld`, test 13 `checked` and test 13 "
        f"`with_matched_forbidden`. Re-measure with "
        f"evidence/orch717_census_probe.py, move each pin to its measured value "
        f"WITH the per-run attribution recorded beside it, and re-run "
        f"evidence/c102_mutation_attack.py. Do NOT relax a pin to >= and do NOT "
        f"delete the harness's baseline precondition to route around it."
    )


def test_09_the_derived_census_pins_are_equality_not_a_floor():
    """REV-104's argument, pinned as a test rather than left in a findings file.

    The floor at `len(paths)` is `>=` on purpose: it catches the corpus
    SHRINKING. The DERIVED pins are `==` on purpose: they assert their loop
    visited every leg it should have, and the census is how they know how many
    that is. Relaxing one to `>=` would let ten new legs enter the population
    unremarked and go unvisited -- a quiet vacuity of exactly the F-144 shape,
    and the opposite of what the guard is for.
    """
    source = C102_TESTS.read_text(encoding="utf-8")
    for name in ("legs", "checked", "withheld", "with_matched_forbidden"):
        equality = re.search(rf"^\s*assert {name} == \d+", source, re.MULTILINE)
        relaxed = re.search(rf"^\s*assert {name} >= \d+", source, re.MULTILINE)
        assert equality, f"the `{name}` census pin is gone"
        assert not relaxed, (
            f"`{name}` was relaxed from == to >=. F-151 proposed exactly this and "
            f"REV-104 refused it: a >= here stops the suite noticing that new legs "
            f"entered the population unremarked. Move the pin, do not widen it"
        )
