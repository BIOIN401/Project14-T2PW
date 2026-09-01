"""C-108 correction round 2: F-160, reproduced and bounded on this card target.

CPython keys a cached .pyc on (source mtime TRUNCATED TO WHOLE SECONDS, source
size). A mutation that changes NEITHER leaves the cache valid, so the OLD
bytecode runs, the mutant never executes, and a harness prints MUTATION SURVIVED
for a mutant that was never in the interpreter. That reads as "this guard has no
test", and the natural response is to weaken or delete the guard -- which makes a
false green here more dangerous than a false red.

Reproduced DETERMINISTICALLY rather than waiting for the same-second race: the
mutation is SAME LENGTH and the mtime is restored with os.utime, so the cache key
is provably unchanged. Three arms, and the third is the one that surprised:

  ARM 0  plain import, no purge      -> FALSE GREEN. The mutant never ran.
  ARM 1  pytest, no purge            -> FALSE GREEN. 220 passed, exit 0, on a
                                        tree whose source says otherwise.
  ARM 2  pytest, caches purged       -> RED, 2 failed. The mutant executes.

BOTH ARMS BITE. An earlier ordering of this script reported ARM 1 as RED and I
nearly recorded "pytest is immune" as a finding; it was not immune, the
measurement was contaminated. The (e) mutation tests in
tests/test_c108_f155_class.py apply and restore mutations to THIS FILE, so a
pytest warm-up churns its mtime and the later os.utime moves the source AWAY from
the cached key -- the cache then misses for a reason that has nothing to do with
the defect. The failed run is kept beside this one as
c108_r2_f160_demo.attempt1-pytest-warmup-churned-mtime.log. ARM 0 is now run
first and alone.

WHAT IS EXPOSED: everything. The pytest-driven mutation harness
(c108_own_mutations.py) AND the probe scripts -- c108_frames.py,
c108_guard_attribution.py, c108_r1_blocking.py, c107_battery.py and
rev107_corpus.py all import the guard module directly. Both the harness and
c108_job.sh now purge first.

THE PURGE IS SCOPED TO src/t2pw AND tests. This repo TRACKS __pycache__ at
__pycache__/, scripts/__pycache__/, src/__pycache__/ and src/tools/__pycache__/,
and the first version of the purge was unscoped and DELETED 56 TRACKED FILES.
They were restored from HEAD and the scope is now stated in all three purges.

Restores through C-106 primitives (D-084), byte-exact, and proves it.

Usage::  <python> c108_r2_f160_demo.py <worktree-root>
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from c102_mutation_attack import (  # noqa: E402
    apply_mutation, crlf_count, restore_saved_bytes, sha256_of,
)

PY = "c:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe"
ROOT = Path(sys.argv[1]).resolve()
TARGET = ROOT / "src" / "t2pw" / "curation" / "apply_audit_patch.py"
TESTS = "tests/test_c108_f155_class.py"
BASETEMP = "C:/t/bt/c108f160"

DET = chr(95) + "APPOSITIVE_DETERMINER_SRC"
OLD = DET + ' = r"(?:the|a|an)"'
NEW = DET + ' = r"(?:xhe|a|an)"'
assert len(OLD) == len(NEW), "the demonstration requires a SAME-LENGTH mutation"


def purge(root: Path) -> int:
    """Scoped to src/t2pw and tests -- this repo TRACKS __pycache__ elsewhere."""

    removed = 0
    for scope in ("src/t2pw", "tests"):
        base = root / scope
        if not base.is_dir():
            continue
        for cache in base.rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)
            removed += 1
    return removed


def import_value() -> str:
    """What a PLAIN IMPORT of the guard module sees. ARM 0."""

    probe = ("import t2pw.curation.apply_audit_patch as m;"
             "print(m." + DET + ")")
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    proc = subprocess.run([PY, "-c", probe], capture_output=True, text=True, env=env)
    return proc.stdout.strip() or proc.stderr.strip()[:200]


def run_pytest() -> tuple:
    proc = subprocess.run(
        [PY, "-m", "pytest", TESTS, "-q", "--no-header", "--basetemp=" + BASETEMP],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    lines = proc.stdout.strip().splitlines()
    return proc.returncode, (lines[-1] if lines else "(no output)")


Path(BASETEMP).mkdir(parents=True, exist_ok=True)
before = TARGET.read_bytes()
stat = TARGET.stat()
print("target sha256 :", sha256_of(before))
print("target size   :", len(before))

# ARM 0 IS RUN FIRST AND ALONE, and that ordering is load-bearing. The (e)
# mutation tests in tests/test_c108_f155_class.py apply and restore mutations to
# this very file, which CHURNS ITS MTIME. A pytest warm-up before ARM 0 therefore
# leaves the .pyc keyed to a mtime that os.utime then moves away from, the cache
# misses for an unrelated reason, and the demonstration silently measures nothing.
# The first version of this script did exactly that -- see
# c108_r2_f160_demo.attempt1-pytest-warmup-churned-mtime.log.
arm0 = arm1 = arm2 = None
purge(ROOT)
print("ARM 0 warm-up (cold cache, unmutated) :", import_value())

saved = apply_mutation(TARGET, OLD, NEW)
try:
    after = TARGET.read_bytes()
    assert len(after) == len(before), "mutation changed the file size"
    os.utime(TARGET, (stat.st_atime, stat.st_mtime))
    print("MUTATED. size unchanged: %s   mtime restored: %s"
          % (len(after) == len(before), TARGET.stat().st_mtime == stat.st_mtime))

    arm0 = import_value()
    stale = "xhe" not in arm0
    print()
    print("ARM 0 -- plain import, NO purge : %s" % arm0)
    print("   %s" % ("FALSE GREEN -- F-160 REPRODUCED: the mutant never ran"
                     if stale else "the mutant executed"))
finally:
    restore_saved_bytes(TARGET, saved)
    purge(ROOT)

# Now the pytest arms, with their own warm-up.
code, summary = run_pytest()
print()
print("pytest warm-up (unmutated, caches primed) : exit=%d  %s" % (code, summary))
if code != 0:
    print("BASELINE NOT GREEN -- uninterpretable. Stopping.")
    raise SystemExit(2)

stat2 = TARGET.stat()
saved = apply_mutation(TARGET, OLD, NEW)
try:
    os.utime(TARGET, (stat2.st_atime, stat2.st_mtime))
    arm1 = run_pytest()
    print()
    print("ARM 1 -- pytest, NO purge : exit=%d  %s" % arm1)
    print("   %s" % ("FALSE GREEN -- F-160 reproduced under pytest too"
                     if arm1[0] == 0 else
                     "RED -- pytest import machinery did not serve stale bytecode here"))

    n = purge(ROOT)
    arm2 = run_pytest()
    print()
    print("ARM 2 -- pytest, purged %d caches : exit=%d  %s" % ((n,) + arm2))
    print("   %s" % ("RED -- the mutant executes"
                     if arm2[0] != 0 else "STILL GREEN -- investigate"))
finally:
    restore_saved_bytes(TARGET, saved)
    purge(ROOT)

back = TARGET.read_bytes()
print()
print("restored byte-identical:", sha256_of(back) == sha256_of(before))
print("CRLF count preserved   :", crlf_count(back) == crlf_count(before))
print()
print("F-160 REPRODUCED ON A PLAIN IMPORT :", "xhe" not in (arm0 or "xhe"))
print("F-160 REPRODUCED UNDER PYTEST      :", arm1 is not None and arm1[0] == 0)
print("PURGE MAKES THE MUTANT EXECUTE     :", arm2 is not None and arm2[0] != 0)
