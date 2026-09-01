"""REV-106 / R7 round 3 -- LINE-INDEXED mutations.

Round 2's G-fR and G-h did not apply: I built the patterns as Python string
literals containing backslash escapes and my own `replace("\n", newline)` step
mangled them. That failure is preserved in rev106_guard_mutations_round2.log
and corrected here by addressing lines by INDEX instead of by content, which
removes the escaping question entirely.

Round 3 also captures the FULL pytest output for each mutation, because round 2
reported "1 error" for G-a2R without saying which node errored.
"""
from __future__ import annotations
import hashlib, re, subprocess, sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
PY = sys.executable
HARNESS = ROOT / "docs/pwml_recovery_sprint/evidence/c102_mutation_attack.py"
NEWTEST = "tests/test_c106_mutation_harness_executable.py"

def cen(b):
    crlf = b.count(b"\r\n")
    return dict(n=len(b), crlf=crlf, lf=b.count(b"\n") - crlf, sha=hashlib.sha256(b).hexdigest())

saved = HARNESS.read_bytes()
BASE = cen(saved)
text = saved.decode("utf-8")
NL = "\r\n" if "\r\n" in text else "\n"
lines = text.split(NL)
print(f"harness: {len(lines)} split-parts, crlf={BASE['crlf']}, sha={BASE['sha'][:16]}")

def find(pred, label):
    hits = [i for i, l in enumerate(lines) if pred(l)]
    print(f"  locate {label}: line indices {hits}")
    assert len(hits) == 1, f"{label} matched {len(hits)}"
    return hits[0]

I_M5OLD = find(lambda l: l.strip().startswith('\'        "excluded_terms": excluded,'), "M5 old-string")
I_FINDOCC = find(lambda l: l.strip().startswith("return text.count(old.replace("), "find_occurrences body")
I_WRITE = find(lambda l: "path.write_bytes(text.replace(old_nl" in l, "apply_mutation write")

MUTS = [
 ("G-fR2", I_M5OLD, "M5's `old` string altered so its substitution matches ZERO times "
                    "-- the exact non-vacuity failure the harness exists to prevent",
  lambda l: l.replace("excluded,", "excluded_NOPE,")),
 ("G-h2", I_FINDOCC, "find_occurrences forced to always report 1 -- would make test_02 vacuous",
  lambda l: "    return 1"),
 ("G-a2R2", I_WRITE, "apply_mutation's WRITE genuinely damaged: emit LF-only bytes",
  lambda l: l.replace('.encode("utf-8"))', '.replace(chr(13)+chr(10), chr(10)).encode("utf-8"))')),
]

def run_focused():
    p = subprocess.run([PY, "-m", "pytest", NEWTEST, "-q", "--no-header", "-rf", "-rE",
                        "--basetemp=C:/t/bt/rev106mut"], cwd=str(ROOT),
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.returncode, p.stdout + p.stderr

code, out = run_focused()
print(f"\n=== CONTROL: exit={code}  {out.strip().splitlines()[-1]}")

rc = 0
for mid, idx, what, fn in MUTS:
    new_lines = list(lines)
    old_line = new_lines[idx]
    new_lines[idx] = fn(old_line)
    print(f"\n=== {mid}: {what}")
    print(f"    line[{idx}] OLD: {old_line!r}")
    print(f"    line[{idx}] NEW: {new_lines[idx]!r}")
    if new_lines[idx] == old_line:
        print("    !! MUTATION IS THE IDENTITY -- recorded, not hidden"); rc = 2; continue
    try:
        HARNESS.write_bytes(NL.join(new_lines).encode("utf-8"))
        m = cen(HARNESS.read_bytes())
        print(f"    mutated: bytes={m['n']} crlf={m['crlf']} bare_lf={m['lf']}")
        c, o = run_focused()
        print(f"    exit={c}")
        for line in o.strip().splitlines():
            if re.match(r"^(FAILED|ERROR|E\s|\d+ (passed|failed|error))", line.strip()) or " in 0." in line:
                print(f"      | {line.rstrip()}")
        print(f"    -> {'CAUGHT (non-zero)' if c != 0 else '*** NOT CAUGHT ***'}")
        if c == 0: rc = max(rc, 1)
    finally:
        HARNESS.write_bytes(saved)
        a = cen(HARNESS.read_bytes())
        ok = a['sha'] == BASE['sha'] and a['crlf'] == BASE['crlf']
        print(f"    restored: sha {BASE['sha'][:16]} -> {a['sha'][:16]}  crlf {BASE['crlf']} -> {a['crlf']}  byte-exact={ok}")
        if not ok: rc = 3

c, o = run_focused()
porc = subprocess.run(["git","status","--porcelain","--","tests/","src/","docs/pwml_recovery_sprint/evidence/c102_mutation_attack.py"],
                      cwd=str(ROOT), capture_output=True, text=True).stdout.strip()
print(f"\n=== CONTROL AFTER: exit={c}  {o.strip().splitlines()[-1]}   porcelain={porc!r}")
raise SystemExit(rc if (c == 0 and not porc) else 4)
