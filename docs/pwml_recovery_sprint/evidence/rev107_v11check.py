"""REV-107: does the proposed one-line F1 left-boundary fix actually close the
finding? Apply it, re-run probe4, restore with D-084 proof.

attempt 1 is preserved as rev107_v11check.attempt1-heredoc-ate-backslashes.log:
written through a bash heredoc, which collapsed the doubled backslashes, so the
substitution matched 0 times and apply_mutation ABORTED. That abort is the
F-144 property working -- a silently non-matching mutation would have produced a
clean run and a false result.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

sys.path.insert(0, "docs/pwml_recovery_sprint/evidence")
from c102_mutation_attack import (  # noqa: E402
    apply_mutation, crlf_count, restore_saved_bytes, sha256_of,
)

PY = "c:/Users/Angad/Desktop/SummerBIOIN/Project14-T2PW/.venv/Scripts/python.exe"
p = pathlib.Path("src/t2pw/curation/apply_audit_patch.py")

OLD = '                _ATTENUATION_STEM_SRC + r"[a-z]*\\b(?:\\s+(?:of|in))?"\n'
NEW = ('                r"(?<![a-z])" + _ATTENUATION_STEM_SRC'
       ' + r"[a-z]*\\b(?:\\s+(?:of|in))?"\n')

before = p.read_bytes()
print(f"before bytes={len(before)} crlf={crlf_count(before)} sha={sha256_of(before)[:16]}")
saved = apply_mutation(p, OLD, NEW)
try:
    env = {**os.environ, "PYTHONPATH": "C:/t/rev107/src"}
    r = subprocess.run(
        [PY, "-u", "docs/pwml_recovery_sprint/evidence/rev107_probe4.py", "V11-APPLIED"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    out = r.stdout + r.stderr
finally:
    restore_saved_bytes(p, saved)
after = p.read_bytes()
print(f"after  bytes={len(after)} crlf={crlf_count(after)} sha={sha256_of(after)[:16]}")
assert sha256_of(after) == sha256_of(before), "restore not byte-exact"
assert crlf_count(after) == crlf_count(before), "restore changed CRLF"
print("RESTORE PROVED byte-exact by sha256 AND CRLF count\n")

WANT = ("MATRIX OPEN CELLS", "EXTRA-FRAME OPEN CELLS", "FALSE REFUSAL",
        "REDOX/CATALYSIS REFUSED", "COFACTOR PRESERVATION REFUSED", "OPEN ")
for ln in out.split("\n"):
    if any(w in ln for w in WANT):
        print(ln.rstrip()[:124])
