"""REV-109 -- mutation proof that the B9 probe is load-bearing.

``rev109_route_adversarial.py`` S3 asserts that a file with the RIGHT NAME and
DIFFERENT BYTES is reported unreachable. That assertion is only worth anything
if it would FAIL against a decorative, filename-keyed implementation.

So: copy ``reviewer_evidence_route.py``, mutate exactly one decision -- let a
matching filename count as reachable -- and re-run the same probe. S3 must flip
from exit 1 to exit 0. If it does not, S3 is vacuous and proves nothing about
the real script.

usage: rev109_route_mutation.py <tree> <venv-python> <workdir>
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

EVID = "docs/pwml_recovery_sprint/evidence"

OLD = """        found_at = integration_index.get(oid, [])
        if found_at:
            verdict = REACHABLE"""

NEW = """        found_at = integration_index.get(oid, [])
        if found_at or rel in integration_paths:   # MUTANT: filename counts
            verdict = REACHABLE
            found_at = found_at or [rel]"""


def s3_exit(log: str):
    m = re.search(r"^--- S3 .*?observed exit : (\S+)", log, re.S | re.M)
    return m.group(1) if m else "?"


def main():
    tree, py, work = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    probe = tree / EVID / "rev109_route_adversarial.py"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    mut_tree = work / "mutant"
    (mut_tree / EVID).mkdir(parents=True, exist_ok=True)
    src = (tree / EVID / "reviewer_evidence_route.py").read_text(encoding="utf-8")
    if OLD not in src:
        print("MUTATION TARGET NOT FOUND -- the script changed shape; "
              "this proof must be rewritten, not skipped")
        return 3
    (mut_tree / EVID / "reviewer_evidence_route.py").write_text(
        src.replace(OLD, NEW), encoding="utf-8", newline="\n")
    print("mutation applied: reachability decided by FILENAME, not content")

    print("\n--- baseline: the REAL script ---")
    real = subprocess.run([py, str(probe), str(tree), str(work / "advreal")],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    print(f"S3 observed exit against the real script  : {s3_exit(real.stdout)}")

    print("\n--- mutant ---")
    mut = subprocess.run([py, str(probe), str(mut_tree), str(work / "advmut")],
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace")
    print(f"S3 observed exit against the mutant       : {s3_exit(mut.stdout)}")
    for line in mut.stdout.splitlines():
        if line.startswith("FAIL") or "decorative" in line:
            print("   " + line.strip())

    ok = s3_exit(real.stdout) == "1" and s3_exit(mut.stdout) == "0"
    print("\nVERDICT: " + ("S3 IS LOAD-BEARING -- it passes on the real script "
                           "and fails on a filename-keyed mutant"
                           if ok else
                           "S3 IS VACUOUS -- it does not distinguish the two"))
    # S13/S13b are expected to error (exit 2) against the mutant tree, which is
    # not a git repo. That is not part of this proof.
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
