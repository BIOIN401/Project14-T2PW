"""Run each gold-reader file ALONE in a fresh process; report per-file counts.

REV-101 ran the 22-file selection split as well as combined and found zero
per-file shifts. This reproduces that at correction-round-1 tip. Interpreter and
env are inherited from the bounded wrapper; ASCII-only output.
"""
import io
import os
import re
import subprocess
import sys

TREE = sys.argv[1]
SEL = io.open(sys.argv[2], encoding="utf-8").read().split()
BT = sys.argv[3]
os.chdir(TREE)
env = dict(os.environ, PYTHONPATH=os.path.join(TREE, "src"), T2PW_OFFLINE_CURATOR="1")
SUMMARY = re.compile(r"^\d+ (passed|failed).*$|^.*\d+ (passed|failed).*in [\d.]+s.*$")

total_p = total_f = total_s = 0
print("PER-FILE, each in a FRESH process (%d files)" % len(SEL))
for i, f in enumerate(SEL, 1):
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:randomly",
         "--basetemp=%s/sp%d" % (BT, i), f],
        capture_output=True, text=True, env=env,
    )
    line = ""
    for ln in reversed(r.stdout.splitlines()):
        if " passed" in ln or " failed" in ln or "no tests ran" in ln:
            line = ln.strip()
            break
    for key, pat in (("p", r"(\d+) passed"), ("f", r"(\d+) failed"), ("s", r"(\d+) skipped")):
        m = re.search(pat, line)
        n = int(m.group(1)) if m else 0
        if key == "p":
            total_p += n
        elif key == "f":
            total_f += n
        else:
            total_s += n
    print("  %-56s exit=%d  %s" % (f, r.returncode, line))

print()
print("SPLIT TOTALS: %d passed, %d failed, %d skipped" % (total_p, total_f, total_s))
print("COMBINED was: 453 passed, 2 failed, 8 skipped")
print("PER-FILE SHIFT:", "NONE" if (total_p, total_f, total_s) == (453, 2, 8) else "PRESENT")
