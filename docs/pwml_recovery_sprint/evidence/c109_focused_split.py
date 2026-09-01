"""C-109: run every node of `tests/test_c109_reviewer_evidence_route.py` ALONE.

A combined green run does not prove the tests are independent. If one case leaves
state another depends on -- a temp tree, an imported module object, a cached git
config -- the file is green together and red apart, and the failing half is the
NEGATIVE half, which is the half that makes this a check at all. So each node is
collected, then run in its own fresh process, serially, and its counts printed.

Run under `bounded_run.py`: every child it spawns is inside the wrapper's job object
and is cleaned up with it.

Usage::

    <python> c109_focused_split.py <worktree-root> <basetemp-parent>
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
BASETEMP = Path(sys.argv[2])
# The parent MUST exist before pytest is handed a --basetemp under it (F-114).
BASETEMP.mkdir(parents=True, exist_ok=True)
PY = sys.executable
TARGET = "tests/test_c109_reviewer_evidence_route.py"

# Scoped to pytest's OWN summary line (F-152): anchored on the count at the start and
# the duration at the end, so prose containing "3 errors" cannot be parsed as a result.
SUMMARY = re.compile(
    r"^=*\s*(?P<body>\d+ \w+(?:, \d+ \w+)*)\s+in\s+[\d.]+s.*$", re.M
)


def counts(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for m in SUMMARY.finditer(text):
        for part in m.group("body").split(", "):
            n, _, word = part.partition(" ")
            out[word] = out.get(word, 0) + int(n)
    return out


def run(args: list[str], basetemp: str) -> tuple[int, dict[str, int], str]:
    proc = subprocess.run(
        [PY, "-m", "pytest", "-q", f"--basetemp={BASETEMP / basetemp}", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode, counts(proc.stdout + proc.stderr), proc.stdout + proc.stderr


collect = subprocess.run(
    [PY, "-m", "pytest", "-q", "--collect-only", "--no-header", TARGET],
    cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
)
nodes = [ln.strip() for ln in collect.stdout.splitlines() if "::" in ln and ln.startswith(TARGET)]
print(f"collected {len(nodes)} node(s) from {TARGET}\n")
if not nodes:
    print("COLLECTED NOTHING -- infrastructure failure, not a result", file=sys.stderr)
    print(collect.stdout[-2000:], file=sys.stderr)
    sys.exit(2)

print("--- COMBINED (all nodes, one process) ---")
combined_rc, combined, _ = run([TARGET], "combined")
print(f"  exit={combined_rc}  {combined}\n")

print("--- SPLIT (one process per node) ---")
split_total: dict[str, int] = {}
bad: list[str] = []
for i, node in enumerate(nodes, start=1):
    rc, c, out = run([node], f"n{i:02d}")
    name = node.split("::", 1)[1]
    print(f"  [{i:02d}] exit={rc} {c}  {name}")
    for k, v in c.items():
        split_total[k] = split_total.get(k, 0) + v
    if rc != 0:
        bad.append(name)
        print(out[-1500:], file=sys.stderr)

print(f"\ncombined : exit={combined_rc}  {combined}")
print(f"split    : {len(nodes)} processes  {split_total}")

ok = (
    combined_rc == 0
    and not bad
    and combined.get("passed") == len(nodes)
    and split_total.get("passed") == len(nodes)
    and not combined.get("failed")
    and not split_total.get("failed")
)
print(f"VERDICT  : {'SPLIT AND COMBINED AGREE, all green' if ok else 'DISAGREEMENT'}")
if not ok:
    print(f"  failing nodes: {bad}", file=sys.stderr)
sys.exit(0 if ok else 1)
