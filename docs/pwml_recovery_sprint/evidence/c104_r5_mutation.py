"""C-104 half 1: apply and restore REV-102's mutation R5, under D-084.

R5 reverts ``ModeResult.to_dict``'s deep copy of ``coverage_reconciliation`` to a
shallow ``dict(...)``. REV-102 ran it and it came back GREEN -- a shipped fix
with no proof -- which is what D-083 follow-on 1 carries and what test 4 now
answers.

**D-084 is LOCKED and this file is written to it.** The restore replays SAVED
BYTES: binary read before the mutation, binary write after it. It never runs
``git checkout --`` (which reverts more than it mutated, and would discard the
card's own uncommitted work on the same file) and never a text-mode write (which
silently rewrites CRLF as LF and leaves the path modified). ``sha256`` is printed
at every step so the restore is *verified* rather than assumed; a restore that
has not been verified is not a restore.

This is file I/O and hashing only. It spawns no child process, runs no test and
issues no LLM call, so it is outside the four job classes ``[S8]`` item 1 names.
The pytest runs it brackets go through ``bounded_run.py`` like everything else,
and ``git status --porcelain`` is checked by the operator between steps.

Usage::

    <python> c104_r5_mutation.py save    <root> <save-path>
    <python> c104_r5_mutation.py apply   <root> <save-path>
    <python> c104_r5_mutation.py restore <root> <save-path>
    <python> c104_r5_mutation.py digest  <root> <save-path>
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

TARGET_REL = "src/t2pw/bench/acceptance.py"

OLD = (
    b'            data["coverage_reconciliation"] ='
    b" deepcopy(dict(self.coverage_reconciliation))\r\n"
)
NEW = (
    b'            data["coverage_reconciliation"] ='
    b" dict(self.coverage_reconciliation)  # MUTATION R5\r\n"
)


def sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    action, root, save = sys.argv[1], Path(sys.argv[2]).resolve(), Path(sys.argv[3])
    target = root / TARGET_REL
    crlf = b"\r\n"
    lf = b"\n"

    if action == "save":
        blob = target.read_bytes()
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_bytes(blob)
        print(f"SAVED  {target}")
        print(f"  bytes  {len(blob)}   crlf {blob.count(crlf)}   "
              f"bare-lf {blob.count(lf) - blob.count(crlf)}")
        print(f"  sha256 {sha(blob)}")
        print(f"  -> {save}  sha256 {sha(save.read_bytes())}")
        return 0

    if action == "apply":
        saved = save.read_bytes()
        blob = target.read_bytes()
        if sha(blob) != sha(saved):
            print("ABORT: the target no longer matches the saved bytes; save first")
            return 2
        if blob.count(OLD) != 1:
            print(f"ABORT: R5 substitution matched {blob.count(OLD)} times, not 1")
            return 2
        mutated = blob.replace(OLD, NEW, 1)
        target.write_bytes(mutated)
        print(f"MUTATED (R5) {target}")
        print(f"  pre  sha256 {sha(blob)}")
        print(f"  post sha256 {sha(mutated)}")
        print(f"  bytes {len(blob)} -> {len(mutated)}   "
              f"crlf {mutated.count(crlf)}   "
              f"bare-lf {mutated.count(lf) - mutated.count(crlf)}")
        return 0

    if action == "restore":
        saved = save.read_bytes()
        before = target.read_bytes()
        target.write_bytes(saved)
        after = target.read_bytes()
        print(f"RESTORED {target}")
        print(f"  mutated  sha256 {sha(before)}")
        print(f"  saved    sha256 {sha(saved)}")
        print(f"  on disk  sha256 {sha(after)}")
        print(f"  byte-exact: {sha(after) == sha(saved)}")
        print(f"  crlf {after.count(crlf)}   bare-lf {after.count(lf) - after.count(crlf)}")
        return 0 if sha(after) == sha(saved) else 1

    if action == "digest":
        blob = target.read_bytes()
        print(f"TARGET   {target}")
        print(f"  sha256 {sha(blob)}")
        print(f"  crlf {blob.count(crlf)}   bare-lf {blob.count(lf) - blob.count(crlf)}")
        print(f"  contains deepcopy form : {blob.count(OLD)}")
        print(f"  contains R5 form       : {blob.count(NEW)}")
        if save.exists():
            print(f"SAVED    {save}")
            print(f"  sha256 {sha(save.read_bytes())}")
        return 0

    print(f"unknown action {action!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
