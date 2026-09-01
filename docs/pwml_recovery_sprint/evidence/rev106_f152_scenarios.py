"""REV-106 / A8 -- the F-152 parse, BOTH directions, on REAL pytest output.

Generates real pytest output from throwaway test files, then feeds each output
to the BASE parser and the TIP parser, both lifted verbatim from the two
committed files rather than retyped.

Direction 1 (the false positive it must kill): a GREEN file whose output merely
CONTAINS "3 errors".
Direction 2 (the preservation case): a GENUINE red must still count failed=1 and
still fold into the totals. A parse that killed the false positive by counting
nothing would be a worse defect.
Direction 3: a GENUINE collection error must still count as an error, or the
D-083 abort it feeds becomes vacuous.
"""
from __future__ import annotations
import re, subprocess, sys, tempfile
from pathlib import Path

TIP = Path(sys.argv[1]).resolve()
BASE = Path(sys.argv[2]).resolve()
PY = sys.executable
SPLIT = "docs/pwml_recovery_sprint/evidence/c102_goldreaders_split.py"

# ---- BASE parser, lifted verbatim from the base file --------------------
base_src = (BASE / SPLIT).read_text(encoding="utf-8")
m = re.search(r'for value, key in re\.findall\(r"(.*?)", out\):', base_src)
assert m, "could not lift the base parse pattern"
BASE_PATTERN = m.group(1)
print(f"BASE parse pattern lifted from {BASE.name}: {BASE_PATTERN!r}")

def base_counts(out: str) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "error": 0, "errors": 0}
    for value, key in re.findall(BASE_PATTERN, out):
        counts[key] = int(value)
    return counts

# ---- TIP parser, exec'd verbatim out of the tip file --------------------
tip_src = (TIP / SPLIT).read_text(encoding="utf-8")
start = tip_src.index("_SUMMARY_LINE = re.compile(")
end = tip_src.index("totals = {", start)
block = tip_src[start:end]
print(f"TIP parse block lifted from {TIP.name}: {len(block)} chars, "
      f"defines summary_counts={'def summary_counts' in block}")
ns: dict = {"re": re}
exec(compile(block, "tip_parse", "exec"), ns)
tip_counts = ns["summary_counts"]

# ---- real pytest output --------------------------------------------------
SCENARIOS = {
 "green_with_warning_text": '''
import warnings
def test_ok():
    warnings.warn(UserWarning("upstream reported 3 errors in the payload"))
    assert True
''',
 "genuine_red": '''
def test_bad():
    assert 1 == 2
def test_ok():
    assert True
''',
 "genuine_collection_error": '''
import a_module_that_does_not_exist_anywhere
def test_never_runs():
    assert True
''',
 "green_with_failure_prose": '''
import warnings
def test_ok():
    warnings.warn(UserWarning("historical note: 7 failed and 3 errors in run 2026-08-01"))
    assert True
''',
 "red_plus_prose": '''
import warnings
def test_bad():
    warnings.warn(UserWarning("upstream reported 9 failed and 3 errors"))
    assert 1 == 2
def test_ok():
    assert True
''',
 "plain_green": '''
def test_a(): assert True
def test_b(): assert True
''',
}

tmp = Path(tempfile.mkdtemp())
bt = Path("C:/t/bt/rev106f152"); bt.mkdir(parents=True, exist_ok=True)
bad = 0
for name, body in SCENARIOS.items():
    f = tmp / f"test_{name}.py"
    f.write_text(body, encoding="utf-8")
    p = subprocess.run([PY, "-m", "pytest", str(f), "-q", "--no-header", "-rf", "-s",
                        f"--basetemp={bt}"], cwd=str(tmp), capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    out = p.stdout + p.stderr
    b, t = base_counts(out), tip_counts(out)
    summary = [l for l in out.splitlines() if l.strip()][-1]
    print(f"\n=== SCENARIO {name}   pytest exit={p.returncode}")
    print(f"    real summary line : {summary.strip()!r}")
    print(f"    output contains '3 errors' : {'3 errors' in out}")
    print(f"    BASE parse : {b}")
    print(f"    TIP  parse : {t}")
    be = b['error'] + b['errors']; te = t['error'] + t['errors']
    # The C-104 abort predicate, applied to each parse's numbers.
    def aborts(c, rc):
        e = c['error'] + c['errors']
        return bool(e or rc not in (0, 1) or (rc == 1 and not c['failed'] and not e))
    print(f"    BASE errors={be} aborts={aborts(b, p.returncode)}   "
          f"TIP errors={te} aborts={aborts(t, p.returncode)}")
    if name in ("green_with_warning_text", "green_with_failure_prose"):
        ok = (not aborts(t, p.returncode)) and t['failed'] == 0 and te == 0 and t['passed'] >= 1
        print(f"    EXPECT tip: no abort, failed=0, errors=0, passed>=1  -> {'OK' if ok else 'VIOLATED'}")
        print(f"    (base aborts spuriously: {aborts(b, p.returncode)})")
    elif name == "red_plus_prose":
        ok = t['failed'] == 1 and t['passed'] == 1 and te == 0 and not aborts(t, p.returncode)
        print(f"    EXPECT tip: SHARPEST CASE -- a genuine red whose output ALSO says '3 errors': failed=1, passed=1, errors=0, no abort -> {'OK' if ok else 'VIOLATED'}")
    elif name == "genuine_red":
        ok = t['failed'] == 1 and t['passed'] == 1 and te == 0 and not aborts(t, p.returncode)
        print(f"    EXPECT tip: failed=1, passed=1, errors=0, folds into totals (no abort) -> {'OK' if ok else 'VIOLATED'}")
    elif name == "genuine_collection_error":
        ok = te >= 1 and aborts(t, p.returncode)
        print(f"    EXPECT tip: errors>=1 and the D-083 abort STILL fires -> {'OK' if ok else 'VIOLATED'}")
    else:
        ok = t['passed'] == 2 and t['failed'] == 0 and te == 0
        print(f"    EXPECT tip: passed=2, clean -> {'OK' if ok else 'VIOLATED'}")
    if not ok: bad += 1

print(f"\nA8 VERDICT: {'BOTH DIRECTIONS HOLD' if bad == 0 else str(bad) + ' scenario(s) VIOLATED'}")
raise SystemExit(1 if bad else 0)
