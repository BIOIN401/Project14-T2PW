"""C-106: measure what the BASE harness's restore path does to line endings.

D-084 says a mutation restore must replay SAVED BYTES. The harness at
`c7fb5c5` does neither half of that:

* it writes the mutant with ``write_text(..., newline="")`` after a
  ``read_text``, which converts every CRLF in the whole file to a bare LF --
  a text-mode round trip reverts LESS than it took;
* and it restores with ``git checkout -- <path>``, which reverts MORE: it
  discards whatever else was in the working tree for that file, and -- the
  part that mattered -- it silently repaired the line-ending damage above, so
  ``git status --porcelain`` came back clean and nobody saw it for a card.

This probe reproduces both rows of D-084's table on the real file, WITHOUT
running pytest and WITHOUT leaving the tree modified. It asserts nothing about
the fix; it records the defect so the correction has its measurement beside it.

Usage::

    <python> c106_d084_probe.py <worktree-root>
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
ACCEPTANCE = "src/t2pw/bench/acceptance.py"
target = ROOT / ACCEPTANCE


def census(data: bytes) -> str:
    crlf = data.count(b"\r\n")
    return (f"bytes={len(data)}  crlf={crlf}  "
            f"bare_lf={data.count(chr(10).encode()) - crlf}  "
            f"sha256={hashlib.sha256(data).hexdigest()[:16]}")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()


original = target.read_bytes()
print(f"ON DISK          : {census(original)}")
print(f"git status       : {git('status', '--porcelain', '--', ACCEPTANCE)!r}")

# --- Row 1 of D-084: text-mode write reverts LESS -------------------------
# Exactly what c102_mutation_attack.py:150+154 did at base. A no-op mutation:
# the substitution is the identity, so ANY difference below is line endings
# alone and cannot be blamed on the mutation content.
text = target.read_text(encoding="utf-8")
target.write_text(text, encoding="utf-8", newline="")
damaged = target.read_bytes()
print(f"\nAFTER TEXT WRITE : {census(damaged)}")
print(f"  identical to disk? {damaged == original}")
print(f"  git status       : {git('status', '--porcelain', '--', ACCEPTANCE)!r}")
print("  ^ the mutation content was the IDENTITY. Every byte of this delta is")
print("    line endings, introduced by the harness on a file it did not change.")

# --- Row 2 of D-084: git checkout -- reverts MORE -------------------------
# It is also what MASKED row 1: the tree comes back clean, so `git status`
# certified a restore that had in fact rewritten the whole file.
git("checkout", "--", ACCEPTANCE)
recovered = target.read_bytes()
print(f"\nAFTER git checkout --: {census(recovered)}")
print(f"  git status       : {git('status', '--porcelain', '--', ACCEPTANCE)!r}")
print(f"  byte-identical to the pre-probe file? {recovered == original}")
print("  ^ THIS is why the damage survived a card. `git status --porcelain == \"\"`")
print("    is exactly what the broken loop produced, which is why C-106's new")
print("    guard asserts sha256 and a CRLF count instead.")

# --- The correct restore, for contrast ------------------------------------
saved = target.read_bytes()
target.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))  # deliberate damage
print(f"\nDELIBERATE DAMAGE: {census(target.read_bytes())}")
target.write_bytes(saved)                                        # saved-bytes restore
final = target.read_bytes()
print(f"SAVED-BYTES RESTORE: {census(final)}")
print(f"  byte-identical: {final == original}   crlf preserved: "
      f"{final.count(chr(13).encode() + chr(10).encode()) == original.count(chr(13).encode() + chr(10).encode())}")
print(f"  git status     : {git('status', '--porcelain', '--', ACCEPTANCE)!r}")

if final != original:
    raise SystemExit("PROBE LEFT THE TREE DIRTY -- this is itself a D-084 failure")
print("\nPROBE CLEAN: tree byte-identical to where it started.")
