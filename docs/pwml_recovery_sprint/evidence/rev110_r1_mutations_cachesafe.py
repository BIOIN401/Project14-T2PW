"""REV-110 -- guard mutations, F-160 SAFE, run against EITHER round.

F-160: CPython keys a .pyc on (source mtime truncated to whole seconds, source
size). A same-length mutation landing in the same second as the write before it
leaves the cache valid, the OLD bytecode runs, the mutant never executes, and
the harness prints a FALSE GREEN. That reads as "this guard has no test".

Two defences, both applied, neither trusted alone:
  1. every __pycache__ under src/ and tests/ is REMOVED before every suite run;
  2. each mutation's SIZE DELTA is printed, so a same-length mutation is
     visible rather than assumed absent.

Purpose: my round-0 B18 finding (deleting any ONE of the five condition-3
readings left the suite green) was obtained by mutation. This re-confirms it
with caches cleared, and runs the SAME five mutations against round 1 to check
the author's new isolation tests actually kill them.

D-084: restores replay SAVED BYTES. Never `git checkout --`.

Usage: rev110_r1_mutations_cachesafe.py <tree> <basetemp-parent> <label>
"""
from __future__ import annotations
import shutil, subprocess, sys
from pathlib import Path

TREE, BASETEMP, LABEL = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(TREE / "docs" / "pwml_recovery_sprint" / "evidence"))
from c102_mutation_attack import apply_mutation, find_occurrences, restore_saved_bytes, sha256_of

TARGET = TREE / "src" / "t2pw" / "bench" / "acceptance.py"
TESTS = "tests/test_c110_negative_control_status.py"
PY = sys.executable

MUTATIONS = [
    ("control_artifacts",
     "    preserved = int(leg.artifacts_recorded or 0) > 0",
     "    preserved = True", "the >=1-preserved-artifact condition", True),
    ("only_operational_failure",
     "        bool(leg.operational_failure)\n        or termination in",
     "        False\n        or termination in",
     "reading 1: the row's own operational_failure boolean", False),
    ("only_termination_reason",
     "        or termination in OPERATIONAL_TERMINATION_REASONS\n",
     "        or False\n", "reading 2: OPERATIONAL_TERMINATION_REASONS", False),
    ("only_status",
     "        or status in _NC_CASUALTY_STATUSES\n",
     "        or False\n", "reading 3: status timeout/error", False),
    ("only_failure_kind",
     "        or kind in _NC_CASUALTY_KINDS\n",
     "        or False\n", "reading 4: failure_kind timeout/crash/network/llm", False),
    ("only_boundary",
     "        or leg.boundary in _NC_CASUALTY_BOUNDARIES\n",
     "        or False\n", "reading 5: classify_strict_boundary", False),
]

def purge_caches():
    n = 0
    for root in (TREE / "src", TREE / "tests"):
        for cache in root.rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True); n += 1
    return n

def run_tests(tag):
    purged = purge_caches()
    proc = subprocess.run([PY, "-m", "pytest", "-q", f"--basetemp={BASETEMP/tag}", TESTS],
                          cwd=str(TREE), capture_output=True, text=True)
    tail = [ln for ln in proc.stdout.splitlines() if ln.strip()][-1:]
    return proc.returncode, (tail[0] if tail else "(no output)"), purged

ORIGINAL_BYTES = TARGET.read_bytes()
ORIGINAL = sha256_of(ORIGINAL_BYTES)
print(f"LABEL             : {LABEL}")
print(f"tree              : {TREE}")
print(f"target            : {TARGET}")
print(f"sha256 BEFORE all : {ORIGINAL}")
print(f"source size       : {len(ORIGINAL_BYTES)} bytes")
BASETEMP.mkdir(parents=True, exist_ok=True)

code, tail, purged = run_tests("baseline")
print(f"BASELINE          : exit={code}  {tail}   (__pycache__ dirs purged: {purged})")
if code != 0:
    print("!! baseline not green -- every mutation would be vacuous. STOP."); raise SystemExit(2)
print()

results = []
for name, old, new, removes, is_control in MUTATIONS:
    occ = find_occurrences(TARGET, old)
    if occ != 1:
        print(f"!! {name}: matched {occ} times, not 1 -- SKIPPED"); continue
    saved = apply_mutation(TARGET, old, new)
    delta = len(TARGET.read_bytes()) - len(saved)
    try:
        code, tail, purged = run_tests(name)
    finally:
        restore_saved_bytes(TARGET, saved)
    caught = code != 0
    results.append((name, removes, is_control, caught, delta))
    same_len = " *** SAME LENGTH -- F-160 RISK ***" if delta == 0 else ""
    print(f"  {'RED  ' if caught else 'GREEN'} {name:<26} size delta {delta:+d}{same_len}")
    print(f"          removes {removes}")
    print(f"          exit={code}  {tail}   (caches purged: {purged})")

print()
print(f"sha256 AFTER  all      : {sha256_of(TARGET.read_bytes())}")
print(f"byte-identical restore : {sha256_of(TARGET.read_bytes()) == ORIGINAL}")
print()
print("=" * 78)
same_length = [r for r in results if r[4] == 0]
control = [r for r in results if r[2]]
real = [r for r in results if not r[2]]
print(f"same-length mutations (F-160 exposure) : {len(same_length)}  -> {[r[0] for r in same_length]}")
print(f"control caught (harness works)         : {all(r[3] for r in control)}")
print(f"condition-3 readings mutated           : {len(real)}")
print(f"  RED   (a test kills it)              : {sum(1 for r in real if r[3])}")
print(f"  GREEN (no test pins it)              : {sum(1 for r in real if not r[3])}")
for name, removes, _c, caught, _d in real:
    if not caught: print(f"      UNPINNED: {removes}")
print("=" * 78)
