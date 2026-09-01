"""REV-109 -- run C-109's new acceptance test ONE NODE PER PROCESS.

The card requires the new test green "split as well as combined". Split means a
fresh interpreter per node id, so a test that only passes because a sibling left
state behind is exposed. Each node gets its own --basetemp.

usage: rev109_split_newtest.py <tree> <venv-python> <basetemp-root>
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TARGET = "tests/test_c109_reviewer_evidence_route.py"


def main():
    tree, py, btroot = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    btroot.mkdir(parents=True, exist_ok=True)

    coll = subprocess.run(
        [py, "-m", "pytest", "-q", "--collect-only", "--no-header",
         f"--basetemp={btroot / 'collect'}", TARGET],
        cwd=tree, capture_output=True, text=True,
    )
    nodes = [l.strip() for l in coll.stdout.splitlines()
             if l.strip().startswith(TARGET) and "::" in l]
    print(f"collected {len(nodes)} node ids")
    if not nodes:
        print("COLLECTED NOTHING -- refusing to report a pass")
        print(coll.stdout[-2000:])
        return 3

    results = []
    for i, node in enumerate(nodes, 1):
        bt = btroot / f"n{i:02d}"
        bt.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [py, "-m", "pytest", "-q", "--no-header", f"--basetemp={bt}", node],
            cwd=tree, capture_output=True, text=True,
        )
        last = [l for l in proc.stdout.strip().splitlines() if l.strip()]
        summary = last[-1] if last else "<no output>"
        results.append((node, proc.returncode, summary))
        print(f"  [{ 'OK ' if proc.returncode == 0 else 'FAIL'}] "
              f"rc={proc.returncode}  {node.split('::')[-1]}  |  {summary}")

    bad = [r for r in results if r[1] != 0]
    print(f"\nsplit: {len(results)} nodes, {len(results) - len(bad)} passed, "
          f"{len(bad)} failed")
    for node, rc, summary in bad:
        print(f"  FAILED rc={rc}  {node}\n    {summary}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
