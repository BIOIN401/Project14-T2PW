"""ORCH-717 / F-156: is the graph-delta enforcement LOAD-BEARING, or merely present?

A peer session refuted the standing MASTER_PLAN claim that graph-delta validation
is partial, by static reading. Static reading is not behavioural proof under G9,
and "52 tests pass" only shows the tests are green -- not that any of them would
NOTICE if the enforcement were removed. This mutates the enforcement and requires
a test to go red.

The enforcement under test, ``rag/controller.py``::

    if verdict.admissible:
        graph = candidate

Remove the guard so a REFUSED delta advances the canonical graph anyway -- which
is precisely the fail-open the claim worried about -- and a test must fail.

D-084: the restore replays SAVED BYTES. The original bytes are captured before
the write and written back in a ``finally``; ``git checkout --`` is never used.

Usage::  <python> orch717_f156_mutation.py <mutation-worktree> <venv-python>
Exit 0 if the mutation is CAUGHT (>=1 test fails), 1 if it survives.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TREE = Path(sys.argv[1]).resolve()
PY = sys.argv[2]
TARGET = TREE / "src" / "t2pw" / "rag" / "controller.py"
TESTS = ["tests/test_rag_graph_delta.py", "tests/test_c055_rag_loop_wiring.py"]

# Line endings are NOT assumed. The repo checks out CRLF on Windows, and the FIRST
# run of this probe correctly REFUSED to mutate rather than guess, because its LF
# anchor did not match the file. That refusal is the behaviour to keep: a probe
# that "helpfully" normalises here would have silently mutated the wrong bytes.
def _eol(blob: bytes) -> bytes:
    return b"\r\n" if blob.count(b"\r\n") * 2 > blob.count(b"\n") else b"\n"


def _original(eol: bytes) -> bytes:
    return (b"        if verdict.admissible:" + eol
            + b"            graph = candidate" + eol)


def _mutated(eol: bytes) -> bytes:
    return (b"        if True:  # MUTANT: enforcement removed" + eol
            + b"            graph = candidate" + eol)


def run_tests(label: str) -> tuple[int, str]:
    proc = subprocess.run(
        [PY, "-m", "pytest", "-q", "--basetemp=C:/t/bt/f156mut", *TESTS],
        cwd=str(TREE), capture_output=True, text=True,
        env={**__import__("os").environ,
             "PYTHONPATH": str(TREE / "src"),
             "T2PW_OFFLINE_CURATOR": "1",
             "PYTHONIOENCODING": "utf-8"},
    )
    tail = "\n".join(proc.stdout.strip().splitlines()[-6:])
    print(f"--- {label}: exit={proc.returncode}")
    print(tail)
    return proc.returncode, tail


saved = TARGET.read_bytes()
EOL = _eol(saved)
ORIGINAL = _original(EOL)
MUTATED = _mutated(EOL)
print(f"line endings detected : {EOL!r}")
if ORIGINAL not in saved:
    print("FATAL: enforcement bytes not found verbatim; refusing to guess at a mutation")
    print("       (this is the anchor the probe is written against)")
    raise SystemExit(2)

print("=" * 78)
print("F-156: does removing the graph-delta enforcement break a test?")
print("=" * 78)
print(f"target : {TARGET}")
print(f"bytes  : {len(saved)}  saved for restore (D-084)")
print()

baseline_rc, _ = run_tests("BASELINE (unmutated)")
if baseline_rc != 0:
    print("FATAL: baseline is not green; a mutation result would be meaningless")
    raise SystemExit(2)

caught = False
try:
    TARGET.write_bytes(saved.replace(ORIGINAL, MUTATED))
    print()
    mutant_rc, mutant_tail = run_tests("MUTANT (enforcement removed)")
    caught = mutant_rc != 0
finally:
    TARGET.write_bytes(saved)
    restored_ok = TARGET.read_bytes() == saved
    print()
    print(f"restore replayed saved bytes : {restored_ok}")
    if not restored_ok:
        print("FATAL: restore did not reproduce the saved bytes")
        raise SystemExit(2)

print()
print("=" * 78)
if caught:
    print("MUTATION CAUGHT -- the enforcement is LOAD-BEARING and covered.")
else:
    print("MUTATION SURVIVED -- the enforcement is present but NOT covered by these")
    print("tests. The peer's static read still stands on the code, but the property")
    print("is UNPINNED and a refactor could remove it with the suite green.")
print("=" * 78)
raise SystemExit(0 if caught else 1)
