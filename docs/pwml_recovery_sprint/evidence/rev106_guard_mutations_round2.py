"""REV-106 / R7 round 2 -- CORRECTIONS to two of my own round-1 mutations.

Round 1 (rev106_guard_mutations.log) is preserved unedited beside this. Two of
its seven were MY errors, not guard holes:

  * G-a2 changed `write_bytes(text.encode())` to
    `write_text(text, newline="")` where `text` came from
    `read_bytes().decode()`. That string still CONTAINS "\r\n" and newline=""
    means "no translation", so the two forms emit IDENTICAL bytes. It is an
    EQUIVALENT MUTANT, not an undetected defect. Round 2 replaces it with a
    mutation that actually damages bytes.
  * G-f inserted an unused constant instead of altering a mutation's `old`
    string, so nothing about the attack set changed. Round 2 does it properly.

Round 2 also adds two guard-of-the-guard probes that round 1 did not have.
"""
from __future__ import annotations
import hashlib, re, subprocess, sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
PY = sys.executable
HARNESS = ROOT / "docs/pwml_recovery_sprint/evidence/c102_mutation_attack.py"
NEWTEST = "tests/test_c106_mutation_harness_executable.py"

def cen(b: bytes):
    crlf = b.count(b"\r\n")
    return dict(n=len(b), crlf=crlf, lf=b.count(b"\n") - crlf, sha=hashlib.sha256(b).hexdigest())
def nl_of(t): return "\r\n" if "\r\n" in t else "\n"

MUTS = [
 ("G-a2R", "the WRITE side genuinely damaged: emit LF-only bytes from apply_mutation",
  '    path.write_bytes(text.replace(old_nl, new_nl, 1).encode("utf-8"))\n',
  '    path.write_bytes(text.replace(old_nl, new_nl, 1).replace("\r\n", "\n").encode("utf-8"))\n'),
 ("G-fR", "M5's `old` string altered so the substitution matches ZERO times",
  '        \'        "excluded_terms": excluded,\n\',\n',
  '        \'        "excluded_terms": excluded_NOPE,\n\',\n'),
 ("G-g", "drop the sha256 assertion from restore_saved_bytes",
  '    if sha256_of(after) != sha256_of(saved):\n',
  '    if False and sha256_of(after) != sha256_of(saved):\n'),
 ("G-h", "make find_occurrences always report 1 -- would make test_02 vacuous",
  '    text = path.read_bytes().decode("utf-8")\n    return text.count(old.replace("\n", newline_of(text)))\n',
  '    return 1  # forced\n'),
]

def run_focused():
    p = subprocess.run([PY, "-m", "pytest", NEWTEST, "-q", "--no-header", "-rf",
                        "--basetemp=C:/t/bt/rev106mut"], cwd=str(ROOT),
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = p.stdout + p.stderr
    s = [l for l in out.splitlines() if " in " in l and ("passed" in l or "failed" in l or "error" in l)]
    return p.returncode, (s[-1] if s else "<none>"), sorted(set(re.findall(r"FAILED \S+::(\w+)", out)))

code, summ, _ = run_focused()
print(f"=== CONTROL: exit={code}  {summ}")
rc = 0
for mid, what, old, new in MUTS:
    saved = HARNESS.read_bytes(); b4 = cen(saved)
    text = saved.decode("utf-8"); nl = nl_of(text)
    o, n = old.replace("\n", nl), new.replace("\n", nl)
    hits = text.count(o)
    print(f"\n=== {mid}: {what}\n    matches={hits}")
    if hits != 1:
        print(f"    !! DID NOT APPLY (matched {hits}) -- recorded, not hidden"); rc = max(rc, 2); continue
    try:
        HARNESS.write_bytes(text.replace(o, n, 1).encode("utf-8"))
        c, s2, reds = run_focused()
        print(f"    exit={c}  {s2}\n    RED: {reds}")
        print(f"    -> {'CAUGHT by the new test file' if c != 0 else 'NOT CAUGHT'}")
    finally:
        HARNESS.write_bytes(saved); a = cen(HARNESS.read_bytes())
        print(f"    restored: sha {b4['sha'][:16]} -> {a['sha'][:16]}  crlf {b4['crlf']} -> {a['crlf']}  byte-exact={a['sha']==b4['sha'] and a['crlf']==b4['crlf']}")
        if a['sha'] != b4['sha']: rc = 3
c, s2, _ = run_focused()
porc = subprocess.run(["git","status","--porcelain","--","tests/","src/","docs/pwml_recovery_sprint/evidence/c102_mutation_attack.py"],
                      cwd=str(ROOT), capture_output=True, text=True).stdout.strip()
print(f"\n=== CONTROL AFTER: exit={c}  {s2}   porcelain={porc!r}")
raise SystemExit(rc if c == 0 and not porc else 4)
