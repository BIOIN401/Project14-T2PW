"""C-106: does the harness STILL refuse to certify against a red baseline?

C-106 section 4 "Preservation", and REV-106 A7 -- described there as the most
important item in the review, and the one that requires running something rather
than reading something.

The trap this exists to catch: making the mutation harness "runnable" by deleting
its `assert code == 0` baseline precondition satisfies C-106's headline and
destroys its purpose. Against a red suite EVERY mutation "goes RED" for free, so
a harness without that precondition does not merely fail to help -- it actively
CERTIFIES guards it never exercised. That is a strictly worse instrument than the
broken one, because the broken one at least refused.

So: break one c102 test deliberately, run the real harness, and require that it

  1. ABORTS on the baseline precondition,
  2. applies ZERO mutations -- no `=== M1:` line, nothing further,
  3. leaves `src/t2pw/bench/acceptance.py` byte-identical, and
  4. prints the actionable diagnostic naming the census pin and C-106,

then restore the c102 file byte-exactly and require the harness to certify again.

Usage::

    <python> c106_preservation_red_baseline.py <worktree-root>
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
PY = sys.executable
C102_REL = "tests/test_c102_coverage_denominator.py"
ACCEPTANCE_REL = "src/t2pw/bench/acceptance.py"
HARNESS_REL = "docs/pwml_recovery_sprint/evidence/c102_mutation_attack.py"

# One deliberate break, in test 10, chosen because it is NOT one of the census
# pins: this must prove the harness refuses ANY red baseline, not merely a stale
# census. If it only noticed census drift it would be a narrower guard than the
# one C-106 claims to preserve.
BREAK_OLD = "    assert cleared == []\n"
BREAK_NEW = '    assert cleared == ["DELIBERATE BREAK -- C-106 preservation case"]\n'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def crlf(data: bytes) -> int:
    return data.count(b"\r\n")


def run_harness() -> tuple[int, str]:
    proc = subprocess.run(
        [PY, "-u", "docs/pwml_recovery_sprint/evidence/c102_mutation_attack.py", str(ROOT)],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return proc.returncode, proc.stdout + proc.stderr


def main() -> int:
    c102 = ROOT / C102_REL
    acceptance = ROOT / ACCEPTANCE_REL
    saved = c102.read_bytes()
    acceptance_before = acceptance.read_bytes()

    harness_src = (ROOT / HARNESS_REL).read_text(encoding="utf-8")
    print("precondition present in the harness source: "
          f"{bool(re.search(r'assert[ ]+code[ ]*==[ ]*0', harness_src))}")
    print("  ^ orientation only. The proof is the behaviour below, not this line.")
    print(f"\nc102 before : bytes={len(saved)} crlf={crlf(saved)} sha256={sha(saved)[:16]}")
    print(f"acceptance  : bytes={len(acceptance_before)} crlf={crlf(acceptance_before)} "
          f"sha256={sha(acceptance_before)[:16]}")

    bad = 0
    text = saved.decode("utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    old_nl = BREAK_OLD.replace("\n", newline)
    new_nl = BREAK_NEW.replace("\n", newline)
    if text.count(old_nl) != 1:
        print(f"the break target matched {text.count(old_nl)} times, not 1 -- refusing to guess")
        return 3

    try:
        c102.write_bytes(text.replace(old_nl, new_nl, 1).encode("utf-8"))
        print("\n=== c102 DELIBERATELY BROKEN (test 10 `cleared`, not a census pin) ===")
        code, out = run_harness()

        aborted = "BASELINE PRECONDITION FAILED" in out
        applied = re.findall(r"^=== (M\d|R5):", out, re.MULTILINE)
        names_census = "census pin is stale" in out or "census pin" in out
        names_card = "C-106" in out
        acceptance_now = acceptance.read_bytes()
        untouched = acceptance_now == acceptance_before

        print(f"    harness exit           : {code}   (must be nonzero)")
        print(f"    aborted on precondition: {aborted}   (must be True)")
        print(f"    mutations APPLIED      : {applied}   (must be [])")
        print(f"    diagnostic names census: {names_census}")
        print(f"    diagnostic names C-106 : {names_card}")
        print(f"    acceptance.py untouched: {untouched}  sha256={sha(acceptance_now)[:16]}")
        for line in out.splitlines():
            if "assert code == 0" in line or "AssertionError" in line:
                print(f"    raised: {line.strip()[:160]}")

        if code == 0:
            print("    !! THE HARNESS CERTIFIED AGAINST A RED BASELINE. "
                  "The precondition is gone or weakened. This is a REJECT condition.")
            bad += 1
        if not aborted:
            bad += 1
        if applied:
            print("    !! mutations were applied despite a red baseline -- "
                  "every one of their RED results is meaningless")
            bad += 1
        if not untouched:
            print("    !! the mutated module was modified during an aborted run")
            bad += 1
        if not (names_census and names_card):
            print("    !! the abort does not name the actionable thing")
            bad += 1
    finally:
        # Saved bytes. D-084 applies to this driver too.
        c102.write_bytes(saved)

    after = c102.read_bytes()
    print(f"\nc102 restored: bytes={len(after)} crlf={crlf(after)} sha256={sha(after)[:16]}")
    print(f"    byte-exact={after == saved}  crlf-preserved={crlf(after) == crlf(saved)}")
    if after != saved:
        bad += 1

    print("\n=== AND THE REJECTION CASE MUST NOT BE THE ONLY ONE IT KNOWS ===")
    print("A guard that refuses everything is a defect, not a fix (REV-106 R8/A7).")
    code, out = run_harness()
    applied = re.findall(r"^=== (M\d|R5):", out, re.MULTILINE)
    passed = "ATTACK PASSED" in out
    print(f"    harness on the RESTORED tree: exit={code} (must be 0)")
    print(f"    mutations applied           : {applied}")
    print(f"    ATTACK PASSED               : {passed}")
    if code != 0 or len(applied) != 8 or not passed:
        bad += 1

    if bad:
        print(f"\nPRESERVATION FAILED: {bad} problem(s)")
        return 1
    print("\nPRESERVATION PASSED: the harness refuses a red baseline and applies no")
    print("mutation against it, and still certifies normally once the tree is restored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
