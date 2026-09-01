"""REV-106 / R7 + A13 -- I mutate every guard in the new test file myself.

F-144: a guard is not evidence until someone who did NOT write it has failed to
defeat it. Six mutations, each a byte-level edit restored by replaying SAVED
BYTES and proved by sha256 + CRLF count (D-084).

A13 is mutation G-a: reintroduce `write_text(newline="")` in the harness and
confirm the byte-exactness guard goes RED. If it stays green the test is
decorative.
"""
from __future__ import annotations
import hashlib, subprocess, sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
PY = sys.executable
HARNESS = ROOT / "docs/pwml_recovery_sprint/evidence/c102_mutation_attack.py"
C102 = ROOT / "tests/test_c102_coverage_denominator.py"
NEWTEST = "tests/test_c106_mutation_harness_executable.py"

def cen(b: bytes):
    crlf = b.count(b"\r\n")
    return dict(n=len(b), crlf=crlf, lf=b.count(b"\n") - crlf,
                sha=hashlib.sha256(b).hexdigest())

def nl_of(t): return "\r\n" if "\r\n" in t else "\n"

# (id, target, what it defeats, old, new, tests that MUST go red)
MUTS = [
 ("G-a", HARNESS,
  "A13: reintroduce the exact D-084 defect -- read_text (universal newlines) "
  "+ write_text(newline='') in apply_mutation",
  '    saved = path.read_bytes()\n    text = saved.decode("utf-8")\n',
  '    saved = path.read_bytes()\n    text = path.read_text(encoding="utf-8")\n',
  ["test_03", "test_04"]),
 ("G-a2", HARNESS,
  "A13 second half: the write side alone",
  '    path.write_bytes(text.replace(old_nl, new_nl, 1).encode("utf-8"))\n',
  '    path.write_text(text.replace(old_nl, new_nl, 1), encoding="utf-8", newline="")\n',
  ["test_03", "test_04"]),
 ("G-b", HARNESS,
  "delete the baseline precondition -- the C-106 headline-satisfying wrecking change",
  '    assert code == 0, (\n',
  '    assert code >= 0, (\n',
  ["test_06"]),
 ("G-c", HARNESS,
  "put `git checkout --` back into the restore path",
  '    path.write_bytes(saved)\n',
  '    path.write_bytes(saved)\n    git(path.parent, "checkout", "--", str(path))\n',
  ["test_07"]),
 ("G-d", C102,
  "relax a DERIVED census pin from == to >= (F-151's original proposal)",
  "    assert legs == 72\n",
  "    assert legs >= 72\n",
  ["test_09"]),
 ("G-e", C102,
  "let the census floor go stale again (72 -> 62), i.e. F-151 recurring",
  "    assert len(paths) >= 72,",
  "    assert len(paths) >= 62,",
  ["test_08"]),
 ("G-f", HARNESS,
  "make one mutation's `old` string match ZERO times -- the non-vacuity guard",
  'ACCEPTANCE = "src/t2pw/bench/acceptance.py"\n',
  'ACCEPTANCE = "src/t2pw/bench/acceptance.py"\nZZZ_UNUSED = 1\n',
  []),  # expectation recorded after measurement; see note below
]

def run_focused():
    p = subprocess.run(
        [PY, "-m", "pytest", NEWTEST, "-q", "--no-header", "-rf",
         "--basetemp=C:/t/bt/rev106mut"],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = p.stdout + p.stderr
    summary = [l for l in out.splitlines() if " in " in l and ("passed" in l or "failed" in l or "error" in l)]
    reds = sorted(set(__import__("re").findall(r"FAILED \S+::(\w+)", out)))
    return p.returncode, (summary[-1] if summary else "<no summary>"), reds

print("=== CONTROL: unmutated tip ===")
code, summ, reds = run_focused()
print(f"  exit={code}  {summ}  reds={reds}")
assert code == 0, "control must be green"

overall = 0
for mid, target, what, old, new, expect in MUTS:
    saved = target.read_bytes()
    b4 = cen(saved)
    text = saved.decode("utf-8")
    nl = nl_of(text)
    o, n = old.replace("\n", nl), new.replace("\n", nl)
    hits = text.count(o)
    print(f"\n=== {mid}: {what}")
    print(f"    target={target.name}  matches={hits}")
    if hits != 1:
        print(f"    !! MUTATION DID NOT APPLY (matched {hits}); recorded, not hidden")
        overall = max(overall, 2); continue
    try:
        target.write_bytes(text.replace(o, n, 1).encode("utf-8"))
        mid_c = cen(target.read_bytes())
        print(f"    mutated: bytes={mid_c['n']} crlf={mid_c['crlf']} bare_lf={mid_c['lf']}")
        code, summ, reds = run_focused()
        print(f"    exit={code}  {summ}")
        print(f"    RED tests: {reds}")
        hit = all(any(e in r for r in reds) for e in expect) if expect else (code != 0)
        print(f"    expected red: {expect or '(non-zero exit)'}   -> {'DEFEATED-BY-GUARD (good)' if hit and code != 0 else 'GUARD FAILED TO CATCH IT'}")
        if not (hit and code != 0):
            overall = 1
    finally:
        target.write_bytes(saved)
        aft = cen(target.read_bytes())
        ok = aft['sha'] == b4['sha'] and aft['crlf'] == b4['crlf']
        print(f"    restored: sha {b4['sha'][:16]} -> {aft['sha'][:16]}  crlf {b4['crlf']} -> {aft['crlf']}  byte-exact={ok}")
        if not ok: overall = 3

print("\n=== CONTROL AFTER: tree restored ===")
code, summ, reds = run_focused()
print(f"  exit={code}  {summ}")
porc = subprocess.run(["git", "status", "--porcelain", "--", "tests/", "src/",
                       "docs/pwml_recovery_sprint/evidence/c102_mutation_attack.py"],
                      cwd=str(ROOT), capture_output=True, text=True).stdout.strip()
print(f"  git status --porcelain (secondary): {porc!r}")
if code != 0 or porc: overall = 4
print(f"\nR7 VERDICT: {'ALL GUARDS HELD' if overall == 0 else 'PROBLEM rc=' + str(overall)}")
raise SystemExit(overall)
