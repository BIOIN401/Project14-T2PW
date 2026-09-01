"""ORCH-717: confirm the .pyc staleness hazard REV-108 found, because it can
turn EVERY same-length mutation in this sprint into a silent false GREEN.

CPython's default bytecode invalidation (PEP 552 "timestamp" mode) keys a .pyc on
(source mtime truncated to seconds, source size). A mutation that changes NEITHER
-- i.e. any same-length edit landing in the same wall-clock second as the write
that preceded it -- leaves the cached .pyc looking valid, and the interpreter
runs the OLD bytecode.

Consequence for a mutation harness: the mutant is never executed, the suite
passes, and the harness reports MUTATION SURVIVED. That reads as "this guard is
not covered by any test" when the truth is "the guard was never removed".

**A false GREEN here is the dangerous direction**: it invents a coverage gap that
does not exist, and the natural response is to weaken or delete the guard.

Usage::  <python> orch717_pyc_staleness.py <scratch-dir>
Exit 0 if the hazard reproduces (and is therefore real), 1 if it does not.
"""

from __future__ import annotations

import importlib
import importlib.util
import shutil
import sys
from pathlib import Path

SCRATCH = Path(sys.argv[1]).resolve()
if SCRATCH.exists():
    shutil.rmtree(SCRATCH)
SCRATCH.mkdir(parents=True)
sys.path.insert(0, str(SCRATCH))

MOD = SCRATCH / "victim.py"

# Same length by construction: "AAA" -> "BBB".
ORIGINAL = 'VALUE = "AAA"\n'
MUTANT = 'VALUE = "BBB"\n'
assert len(ORIGINAL) == len(MUTANT)

print("=" * 78)
print("Does a SAME-LENGTH mutation get executed, or silently ignored?")
print("=" * 78)

MOD.write_text(ORIGINAL, encoding="utf-8")
import victim  # noqa: E402
print(f"  1. wrote ORIGINAL, imported            -> VALUE = {victim.VALUE!r}")

pyc = Path(importlib.util.cache_from_source(str(MOD)))
print(f"  2. bytecode cached                     -> {pyc.name} exists={pyc.exists()}")

st_before = MOD.stat()
MOD.write_text(MUTANT, encoding="utf-8")
st_after = MOD.stat()
print(f"  3. wrote MUTANT (same length)          -> size {st_before.st_size} -> "
      f"{st_after.st_size}, mtime_s {int(st_before.st_mtime)} -> {int(st_after.st_mtime)}")

same_key = (int(st_before.st_mtime) == int(st_after.st_mtime)
            and st_before.st_size == st_after.st_size)
print(f"  4. (mtime_s, size) UNCHANGED?          -> {same_key}")

importlib.reload(victim)
print(f"  5. reloaded                            -> VALUE = {victim.VALUE!r}")

stale = victim.VALUE == "AAA"
print()
print("=" * 78)
if stale:
    print("HAZARD REPRODUCED -- the mutant was NOT executed; stale bytecode ran.")
    print("A harness would report MUTATION SURVIVED for a guard it never removed.")
else:
    print("not reproduced in this run -- the write crossed a second boundary, which")
    print("is exactly why the hazard is INTERMITTENT and therefore worse: it fails")
    print("only sometimes, and a passing re-run looks like confirmation.")
print("=" * 78)
print()
print("MITIGATION: clear __pycache__ between mutation and test, or make every")
print("mutation change the file SIZE (a marker comment does this incidentally --")
print("which is why some mutations in this sprint escaped the hazard by luck).")

raise SystemExit(0 if stale else 1)
