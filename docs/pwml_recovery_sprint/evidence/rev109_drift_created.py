"""REV-109 -- REVIEWER ADDITION (not anticipated by criteria B1-B15).

C-109 fixes F-154 (citations that drift). This probe asks the symmetric
question the criteria did not: **does the C-109 diff itself create new drift?**

It edits two files whose line addresses other committed documents cite:

* ``docs/pwml_recovery_sprint/MASTER_PLAN.md`` -- the F-153 note grows in place,
  pushing everything below it down;
* ``.claude/agents/pwml-test-runner.md`` -- the SMOKE bullet and the chunk
  bullet both grow.

``TEST_MATRIX.md`` was protected by an explicit line-neutrality constraint and
``FINDINGS.md`` was edited line-neutrally. Neither of these two was, and the
card imposed no such constraint on them.

This is NOT a boundary violation -- both files are in C-109's boundary and the
card set no line-neutrality rule for them. It is a registerable consequence.

usage: rev109_drift_created.py <tree> <base-sha>
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

TREE = None


def show(ref, path):
    p = subprocess.run(["git", "-C", TREE, "show", f"{ref}:{path}"],
                       capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"git show {ref}:{path}")
    return p.stdout.decode("utf-8").split("\n")


def citers(target_basename):
    """Every committed line that cites ``<target>:<n>``, and the n it cites."""
    p = subprocess.run(
        ["git", "-C", TREE, "grep", "-n", "-E",
         re.escape(target_basename) + r":[0-9]+", "HEAD", "--",
         "docs/", ".claude/", "*.md"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    out = []
    for line in p.stdout.splitlines():
        try:
            _head, path, lineno, text = line.split(":", 3)
        except ValueError:
            continue
        for n in re.findall(re.escape(target_basename) + r":(\d+)", text):
            out.append((path, int(lineno), int(n), text.strip()))
    return out


def analyse(path, base):
    base_lines = show(base, path)
    tip_lines = show("HEAD", path)
    first = next((i + 1 for i in range(min(len(base_lines), len(tip_lines)))
                  if base_lines[i] != tip_lines[i]), None)
    delta = len(tip_lines) - len(base_lines)
    name = Path(path).name
    print(f"\n===== {path}")
    print(f"  base {len(base_lines) - 1} lines -> tip {len(tip_lines) - 1} "
          f"lines   (net {delta:+d})")
    print(f"  first differing line: {first}")

    cites = citers(name)
    if not cites:
        print("  no committed document cites this file by line number")
        return 0
    stale = []
    for src, srcline, n, text in sorted(set(cites)):
        if first is None or n < first:
            continue
        b = base_lines[n - 1] if n <= len(base_lines) else "<eof>"
        t = tip_lines[n - 1] if n <= len(tip_lines) else "<eof>"
        if b == t:
            continue
        try:
            newn = tip_lines.index(b) + 1
            moved = f"now at :{newn} (shift {newn - n:+d})"
        except ValueError:
            moved = "base content rewritten, no verbatim match at tip"
        stale.append((src, srcline, n, b, t, moved))

    print(f"  citations at or below the first change: {len(stale)} now STALE")
    for src, srcline, n, b, t, moved in stale:
        print(f"\n    {src}:{srcline} cites {name}:{n}")
        print(f"      base  :{n} | {b[:100]}")
        print(f"      tip   :{n} | {t[:100]}")
        print(f"      -> {moved}")
    return len(stale)


def main():
    global TREE
    TREE, base = sys.argv[1], sys.argv[2]
    total = 0
    for path in ("docs/pwml_recovery_sprint/MASTER_PLAN.md",
                 ".claude/agents/pwml-test-runner.md"):
        total += analyse(path, base)
    # the ones edited line-neutrally, for contrast
    for path in ("docs/pwml_recovery_sprint/FINDINGS.md",
                 "docs/pwml_recovery_sprint/TEST_MATRIX.md"):
        b, t = show(base, path), show("HEAD", path)
        first = next((i + 1 for i in range(min(len(b), len(t)))
                      if b[i] != t[i]), None)
        print(f"\n===== {path}  (contrast)")
        print(f"  base {len(b) - 1} -> tip {len(t) - 1} lines; "
              f"first differing line {first}; "
              f"{'line-neutral / append-only: creates NO drift' if first is None or len(b) == len(t) else 'CHECK'}")
    print(f"\n================ TOTAL newly-stale committed citations: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
