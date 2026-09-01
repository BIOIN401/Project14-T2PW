"""C-110 -- run every test in the new file ALONE, in a fresh process.

The charter requires the new tests "split as well as combined". A combined green
run cannot show that a test passes on its own: shared module state, a fixture
that leaked, or an assertion that only holds after an earlier test ran would all
survive it. This collects the node IDs and runs each one in its own process.

Usage::

    <venv-python> c110_split_tests.py <worktree-root> <basetemp-parent>
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
BASETEMP = Path(sys.argv[2])
# The parent MUST exist before pytest is handed a --basetemp under it, or every
# test errors in setup and the file reports 0 passed -- an infrastructure
# failure that reads exactly like a wiped test file.
BASETEMP.mkdir(parents=True, exist_ok=True)
PY = sys.executable
TARGET = "tests/test_c110_negative_control_status.py"

# Anchored on BOTH ends of pytest's own summary line, per F-152: prose that
# merely CONTAINS "3 errors" must not be parsed as a count.
SUMMARY = re.compile(
    r"^=*\s*(?:\d+ \w+(?:, )?)*\d+ (?:passed|failed|error|errors|skipped)"
    r"[^\n]*? in [\d.]+s",
    re.MULTILINE,
)
COUNT = re.compile(r"(\d+) (passed|failed|skipped|errors|error)")


def _run(args):
    return subprocess.run(
        [PY, "-m", "pytest", "-q", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def main() -> int:
    # NOT a second `-q`: `_run` already passes one, and `-qq` makes pytest print
    # a per-file COUNT ("<file>: 22") instead of the node ids. That line starts
    # with the target path, so it survives the filter below and is then handed
    # back to pytest as a node id that does not exist -- exit 4, zero tests run,
    # and a RED that says nothing about the tests. Measured; the log of that run
    # is kept beside this fix.
    collected = _run(["--collect-only", TARGET, f"--basetemp={BASETEMP / 'collect'}"])
    nodes = [
        line.strip()
        for line in collected.stdout.splitlines()
        if "::" in line
        and (
            line.strip().startswith(TARGET.replace("/", "\\"))
            or line.strip().startswith(TARGET)
        )
    ]
    if not nodes:
        print("COLLECTION FAILED -- no node ids")
        print(collected.stdout[-4000:])
        print(collected.stderr[-4000:])
        return 1

    print(f"collected {len(nodes)} node id(s) from {TARGET}")
    print("=" * 78)

    failures = []
    total = 0
    for index, node in enumerate(nodes, start=1):
        result = _run([node, f"--basetemp={BASETEMP / f'n{index}'}"])
        blob = result.stdout + result.stderr
        match = SUMMARY.search(blob)
        counts = dict(
            (kind, int(number)) for number, kind in COUNT.findall(match.group(0))
        ) if match else {}
        passed = counts.get("passed", 0)
        total += passed
        state = "OK  " if result.returncode == 0 else "FAIL"
        print(f"{state} exit={result.returncode} passed={passed:<3} {node}")
        if result.returncode != 0:
            failures.append((node, blob[-2500:]))

    print("=" * 78)
    print(f"nodes run           : {len(nodes)}")
    print(f"passed (summed)     : {total}")
    print(f"failing nodes       : {len(failures)}")
    for node, tail in failures:
        print("-" * 78)
        print(node)
        print(tail)
    print("SPLIT RESULT        : " + ("GREEN" if not failures else "RED"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
