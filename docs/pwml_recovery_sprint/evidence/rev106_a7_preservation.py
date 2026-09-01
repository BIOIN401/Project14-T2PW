"""REV-106 / A7 -- the preservation case, run by the NON-author.

Breaks one c102 test by a BYTE-LEVEL edit, runs the mutation harness, and proves
it ABORTS rather than certifying mutations against a red suite. Then restores by
replaying SAVED BYTES and proves it with sha256 AND CRLF count (D-084).

Zero-mutations-applied is proved three ways: no `=== M<n>:` banner in the output,
the two mutated modules' sha256 unchanged, and `git status --porcelain` clean.
"""
from __future__ import annotations
import hashlib, subprocess, sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
PY = sys.executable
C102 = ROOT / "tests/test_c102_coverage_denominator.py"
ACC = ROOT / "src/t2pw/bench/acceptance.py"
SEM = ROOT / "src/t2pw/bench/semantic.py"
HARNESS = ROOT / "docs/pwml_recovery_sprint/evidence/c102_mutation_attack.py"

def cen(p: Path):
    b = p.read_bytes(); crlf = b.count(b"\r\n")
    return dict(bytes=len(b), crlf=crlf, bare_lf=b.count(b"\n") - crlf,
                sha=hashlib.sha256(b).hexdigest())

def show(tag, p): 
    c = cen(p); print(f"  {tag:<10} {p.name:<42} bytes={c['bytes']} crlf={c['crlf']} bare_lf={c['bare_lf']} sha={c['sha'][:16]}")

print("=== BEFORE ===")
for p in (C102, ACC, SEM): show("before", p)
c102_before, acc_before, sem_before = cen(C102), cen(ACC), cen(SEM)
saved = C102.read_bytes()

rc = 99
try:
    # --- break it, byte-level, one line, CRLF preserved -------------------
    text = saved.decode("utf-8")
    nl = "\r\n" if "\r\n" in text else "\n"
    old = f"    assert legs == 72{nl}"
    assert text.count(old) == 1, f"marker matched {text.count(old)} times"
    C102.write_bytes(text.replace(old, f"    assert legs == 999{nl}", 1).encode("utf-8"))
    print("\n=== c102 BROKEN (assert legs == 72 -> == 999) ===")
    show("broken", C102)
    print(f"  CRLF preserved by my own edit: {cen(C102)['crlf'] == c102_before['crlf']}")

    # --- run the harness --------------------------------------------------
    proc = subprocess.run([PY, str(HARNESS), str(ROOT)], cwd=str(ROOT),
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = proc.stdout + proc.stderr
    print("\n=== HARNESS OUTPUT (verbatim) ===")
    print(out)
    print(f"=== harness exit code: {proc.returncode}")

    aborted = "BASELINE PRECONDITION FAILED" in out
    assertion = "AssertionError" in out and "must be green before any mutation" in out
    banners = [ln for ln in out.splitlines() if ln.startswith("=== M") or ln.startswith("=== R")]
    print(f"\nA7 CHECK 1  harness printed the abort diagnostic : {aborted}")
    print(f"A7 CHECK 2  precondition AssertionError raised    : {assertion}")
    print(f"A7 CHECK 3  mutation banners printed (must be []) : {banners}")
    print(f"A7 CHECK 4  harness exit non-zero                 : {proc.returncode != 0}")
    print(f"A7 CHECK 5  acceptance.py sha unchanged           : {cen(ACC)['sha'] == acc_before['sha']}")
    print(f"A7 CHECK 6  semantic.py   sha unchanged           : {cen(SEM)['sha'] == sem_before['sha']}")
    porc = subprocess.run(["git", "status", "--porcelain", "--",
                           "src/t2pw/bench/acceptance.py", "src/t2pw/bench/semantic.py"],
                          cwd=str(ROOT), capture_output=True, text=True).stdout.strip()
    print(f"A7 CHECK 7  porcelain on mutated modules (must be ''): {porc!r}")
    rc = 0 if (aborted and assertion and not banners and proc.returncode != 0
               and cen(ACC)['sha'] == acc_before['sha'] and cen(SEM)['sha'] == sem_before['sha']
               and porc == "") else 1
finally:
    # --- D-084 restore: replay SAVED BYTES -------------------------------
    C102.write_bytes(saved)
    after = cen(C102)
    print("\n=== RESTORE (saved bytes replayed) ===")
    show("after", C102)
    print(f"  sha256 equal : {after['sha'] == c102_before['sha']}  ({c102_before['sha'][:16]} -> {after['sha'][:16]})")
    print(f"  CRLF equal   : {after['crlf'] == c102_before['crlf']}  ({c102_before['crlf']} -> {after['crlf']})")
    print(f"  bytes equal  : {after['bytes'] == c102_before['bytes']}")
    porc2 = subprocess.run(["git", "status", "--porcelain", "--", "tests/", "src/"],
                           cwd=str(ROOT), capture_output=True, text=True).stdout.strip()
    print(f"  git status --porcelain tests/ src/ (secondary): {porc2!r}")
    if after['sha'] != c102_before['sha'] or after['crlf'] != c102_before['crlf']:
        rc = 3

print(f"\nA7 VERDICT: {'PASS -- the harness refuses a red baseline' if rc == 0 else 'FAIL rc=' + str(rc)}")
raise SystemExit(rc)
