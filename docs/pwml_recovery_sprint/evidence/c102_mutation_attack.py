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

R5 is REV-102's own mutation, carried in by D-083 follow-on 1 and kept under its
reviewer number rather than renumbered, so the finding and the mutation that
proves it share a name. It reverts `to_dict`'s deep copy of
``coverage_reconciliation`` to a shallow `dict(...)`; it went GREEN when REV-102
ran it, and test 4's identity and mutation-consequence assertions are what now
make it RED.

RESTORE DISCIPLINE -- D-084, and C-106 is the card that fixed it
----------------------------------------------------------------
Every mutation is applied by exact text substitution to a COMMITTED file and
restored by writing back the **bytes that were read before the mutation**. A
mutation whose substitution does not apply aborts the run: a mutation that
silently did nothing would produce a green suite and read as a pass.

This file previously did neither half of that correctly, and hit **both rows of
D-084's table in one loop** (measured in ``c106_d084_probe.log``):

* it wrote the mutant with ``write_text(..., newline="")`` after a ``read_text``,
  which rewrote every CRLF in the whole file to a bare LF. On the real target
  that is ``bytes=79745 crlf=1673`` becoming ``bytes=78072 crlf=0`` **with the
  mutation content held identical** -- a text-mode round trip reverts LESS than
  it took;
* and it restored with ``git checkout -- <path>``, which reverts MORE: it
  discards anything else in the working tree for that path. It is also what
  MASKED the damage above for an entire card, because it repaired the line
  endings as a side effect and left ``git status --porcelain`` clean.

So ``git checkout --`` is gone from the restore path entirely, and a clean
``git status`` is no longer accepted as proof of a restore: each mutation asserts
the target's **sha256 and its CRLF count** are unchanged, with the porcelain
check kept only as an additional, insufficient-on-its-own signal.

Substitutions are written with ``\\n`` but the working tree is CRLF
(``core.autocrlf=true``, and ``.gitattributes`` carries no rule for these paths).
``find_occurrences`` and ``apply_mutation`` translate the pattern to the target's
own newline rather than translating the target to the pattern's -- which is what
made the old text-mode round trip destructive in the first place.

This module is IMPORTABLE. Nothing runs at import time, so
``tests/test_c106_mutation_harness_executable.py`` can check every substitution
still matches, and exercise the restore path, without running the attack.

Usage::

    <python> c102_mutation_attack.py <worktree-root>
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

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
        "R5",
        "the to_dict copy goes shallow again -- excluded_terms aliases the scored leg",
        ACCEPTANCE,
        '            data["coverage_reconciliation"] = deepcopy(dict(self.coverage_reconciliation))\n',
        '            data["coverage_reconciliation"] = dict(self.coverage_reconciliation)'
        '  # MUTATION R5\n',
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


# ---------------------------------------------------------------------------
# Byte-exact mutation primitives -- D-084. Used by the driver below AND by
# tests/test_c106_mutation_harness_executable.py, so the test exercises the
# harness's real restore path rather than a reimplementation of it.
# ---------------------------------------------------------------------------

def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def crlf_count(data: bytes) -> int:
    return data.count(b"\r\n")


def newline_of(text: str) -> str:
    """The target's OWN newline. We translate the pattern to the file, never the
    file to the pattern -- translating the file is precisely the D-084 defect."""
    return "\r\n" if "\r\n" in text else "\n"


def find_occurrences(path: Path, old: str) -> int:
    """How many times this substitution matches, newline-aware."""
    text = path.read_bytes().decode("utf-8")
    return text.count(old.replace("\n", newline_of(text)))


def apply_mutation(path: Path, old: str, new: str) -> bytes:
    """Apply one substitution and return the SAVED BYTES to restore with.

    Reads and writes bytes throughout. The only bytes that change are the ones
    inside the substituted region; every line ending outside it is untouched.
    """
    saved = path.read_bytes()
    text = saved.decode("utf-8")
    newline = newline_of(text)
    old_nl = old.replace("\n", newline)
    new_nl = new.replace("\n", newline)
    count = text.count(old_nl)
    if count != 1:
        raise ValueError(f"the substitution matched {count} times, not 1")
    path.write_bytes(text.replace(old_nl, new_nl, 1).encode("utf-8"))
    return saved


def restore_saved_bytes(path: Path, saved: bytes) -> None:
    """Replay the saved bytes and PROVE it, by sha256 and CRLF count.

    D-084: `git checkout --` reverts more, a text-mode write reverts less.
    Neither is used here. `git status --porcelain` is deliberately NOT the
    check -- a clean porcelain is exactly what the broken loop produced while
    it was rewriting every line ending in the file.
    """
    path.write_bytes(saved)
    after = path.read_bytes()
    if sha256_of(after) != sha256_of(saved):
        raise AssertionError(
            f"restore was not byte-exact: {sha256_of(saved)} -> {sha256_of(after)}"
        )
    if crlf_count(after) != crlf_count(saved):
        raise AssertionError(
            f"restore changed line endings: crlf {crlf_count(saved)} -> {crlf_count(after)}"
        )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_suite(root: Path) -> tuple[int, list[str]]:
    proc = subprocess.run(
        [PY, "-m", "pytest", TESTS, "-q", "--no-header", "-rf",
         "--basetemp=" + str(Path("C:/t/bt/c102mut"))],
        cwd=str(root), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    out = proc.stdout + proc.stderr
    failed = sorted(set(re.findall(r"FAILED [^:]+::(\w+)", out)))
    tail = [line for line in out.splitlines() if re.search(r"\d+ (passed|failed)", line)]
    return proc.returncode, [*failed, *tail]


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(root), capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()


def main(root: Path) -> int:
    # The two mutated files must be clean; the attack driver and its own log are
    # untracked while it runs, so the check is scoped to what it will revert.
    assert git(root, "status", "--porcelain", "--", ACCEPTANCE, SEMANTIC) == "", \
        "mutated files must be clean"

    code, summary = run_suite(root)
    print(f"=== BASELINE (unmutated tip) === exit={code}")
    for line in summary:
        print(f"    {line}")

    # ------------------------------------------------------------------
    # THE BASELINE PRECONDITION. Do not delete it, and do not weaken it.
    #
    # It is the thing that makes every mutation result below mean anything:
    # against a red suite, every mutation "goes RED" for free and the harness
    # certifies guards it never exercised. F-151 left this unsatisfiable for a
    # whole card, and the correct repair -- C-106 -- was to fix the census pin
    # in tests/test_c102_coverage_denominator.py so the baseline is honestly
    # green, NOT to remove this check so the harness would run.
    #
    # The diagnostic below exists so the failure names the actionable thing
    # rather than only the disappointed condition. The assertion itself is
    # unchanged.
    # ------------------------------------------------------------------
    if code != 0:
        print()
        print("=== BASELINE PRECONDITION FAILED -- NOTHING BELOW WOULD MEAN ANYTHING ===")
        print("    The unmutated c102 suite is RED, so every mutation would 'go RED'")
        print("    for free and this harness would certify guards it never exercised.")
        print()
        print("    MOST LIKELY CAUSE: the artifact-census pin is stale -- see C-106.")
        print("    tests/test_c102_coverage_denominator.py pins four census-derived")
        print("    quantities with `==` against the committed quarantine_report.json")
        print("    population. Committing a benchmark run grows that population and")
        print("    turns tests 10 and 13 red. FIVE pins move together, not one --")
        print("    those four `==` pins plus the `>=` floor they are derived under:")
        print("        len(paths) floor       >=")
        print("        test 10 legs           ==")
        print("        test 10 withheld       ==")
        print("        test 13 checked        ==")
        print("        test 13 matched-forb   ==")
        print()
        print("    NO VALUES OR LINE NUMBERS HERE, DELIBERATELY (C-117 / F-171): the")
        print("    ones this block used to carry went three re-pins stale while every")
        print("    assertion beside them stayed correct. Re-measure with")
        print("    evidence/orch717_census_probe.py, which prints each quantity beside")
        print("    the pin it is compared against, READ FROM the suite. Move the pins")
        print("    to the measured values WITH the per-run attribution recorded beside")
        print("    them, and re-run this harness. DO NOT delete this precondition.")
        print()
    assert code == 0, (
        "the unmutated suite must be green before any mutation means anything -- "
        "the census pin is probably stale, see C-106 and the diagnostic above"
    )

    failures = 0
    greens: list[str] = []
    for name, what, rel, old, new in MUTATIONS:
        path = root / rel
        before = path.read_bytes()
        print(f"\n=== {name}: {what}")
        print(f"    file={rel}  before: bytes={len(before)} crlf={crlf_count(before)} "
              f"sha256={sha256_of(before)[:16]}")
        try:
            saved = apply_mutation(path, old, new)
        except ValueError as exc:
            # A substitution that silently matched zero times would produce a
            # green suite that reads as a pass. Abort instead.
            print(f"=== {name}: ABORT -- {exc}")
            return 2
        try:
            code, summary = run_suite(root)
        finally:
            # Saved bytes, always, on every exit path from the suite run.
            restore_saved_bytes(path, saved)
        after = path.read_bytes()
        print(f"    exit={code}  {'RED (detected)' if code else 'GREEN -- NOT DETECTED'}")
        for line in summary:
            print(f"    {line}")
        print(f"    restored: bytes={len(after)} crlf={crlf_count(after)} "
              f"sha256={sha256_of(after)[:16]}")
        print(f"    byte-exact={after == before}  crlf-preserved="
              f"{crlf_count(after) == crlf_count(before)}")
        # Additional, and NOT sufficient on its own -- see the module docstring.
        porcelain = git(root, "status", "--porcelain", "--", rel)
        print(f"    git status --porcelain (secondary): {porcelain!r}")
        assert after == before, f"{name} did not restore byte-exactly"
        assert porcelain == "", f"{name} left {rel} dirty"
        if code == 0:
            failures += 1
            greens.append(name)

    restored = git(root, "status", "--porcelain", "--", ACCEPTANCE, SEMANTIC)
    print(f"\n=== tree after restore: {restored!r}  (must be empty)")
    code, summary = run_suite(root)
    print(f"=== SUITE AFTER RESTORE === exit={code}")
    for line in summary:
        print(f"    {line}")
    if failures or restored or code:
        print(f"\nATTACK FAILED: {failures} mutation(s) went undetected: {greens}")
        return 1
    print(f"\nATTACK PASSED: all {len(MUTATIONS)} mutations detected, tree clean, suite green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]).resolve()))
