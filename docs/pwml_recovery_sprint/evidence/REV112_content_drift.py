"""REV-112 -- the SECOND drift class, which the card names but does not scan for.

C-112's own scanner measures **line-shift drift**: a citation whose target number now
addresses different content because lines were inserted above it. It answers 0, and I
reproduced that. The card itself observes that a mechanical drift log mixes **two**
classes, the other being *the number still points here but the content under it changed*.

C-112 rewrote fifteen lines IN PLACE. That shifts nothing, so the card's scanner is
silent by construction -- correctly. But if any committed document cites one of those
exact addresses, the content under it has changed and that citation is now stale in the
second class. **Nobody has measured that**, so I do, as an addition to W10 labelled as an
addition (REV-112 § 0: anything added after seeing the diff must be labelled).

Usage::  <venv-python> REV112_content_drift.py <worktree-root> <base-sha>
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

SPRINT = "docs/pwml_recovery_sprint"
CITE_RE = re.compile(r"([A-Za-z0-9_./\\-]+\.(?:md|py|json|log|txt|ini|toml)):(\d+)")

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}{('  -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


def git(root: Path, *args: str) -> str:
    return subprocess.run([shutil.which("git"), "-C", str(root), *args],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace").stdout


def blob(root: Path, ref: str, rel: str) -> list[str]:
    return git(root, "show", f"{ref}:{rel}").split("\n")


def main(root: Path, base: str) -> int:
    tip = git(root, "rev-parse", "HEAD").strip()
    print("REV-112 -- content-rewrite drift (the second class), an ADDITION to W10")
    print(f"worktree : {root}\nbase     : {base}\ntip      : {tip}")

    changed = [c for c in git(root, "diff", "--name-only", base, tip).split("\n") if c]
    docs = [c for c in changed if c.endswith(".md")]

    # 1. Which lines did C-112 rewrite IN PLACE, per document?
    rewritten: dict[str, list[int]] = {}
    for rel in docs:
        b, t = blob(root, base, rel), blob(root, tip, rel)
        n = min(len(b), len(t))
        rewritten[rel] = [i + 1 for i in range(n) if b[i] != t[i]]
    print("\n=== 1. LINES REWRITTEN IN PLACE (content changed, number did not move) ===")
    for rel, lines in rewritten.items():
        print(f"  {Path(rel).name:<26} {lines}")

    # 2. Every committed citation in the whole tip tree, of any of those documents.
    names = sorted({Path(r).name for r in docs})
    pat = "(" + "|".join(re.escape(n) for n in names) + "):[0-9]+"
    hits = git(root, "grep", "-I", "-n", "-E", pat, tip)

    print("\n=== 2. COMMITTED CITATIONS LANDING ON A REWRITTEN LINE ===")
    landed: list[str] = []
    for ln in hits.split("\n"):
        if not ln.startswith(tip + ":"):
            continue
        path, _, tail = ln[len(tip) + 1:].partition(":")
        num, _, content = tail.partition(":")
        if not num.isdigit():
            continue
        for m in CITE_RE.finditer(content):
            cited, target = Path(m.group(1)).name, int(m.group(2))
            for rel, lines in rewritten.items():
                if Path(rel).name == cited and target in lines:
                    landed.append(f"{path}:{num} cites {m.group(0)}  |  {content.strip()[:110]}")

    if landed:
        for s in sorted(set(landed)):
            print(f"    LANDS ON A REWRITE: {s}")
    else:
        print("    none")
    check("no committed citation addresses a line C-112 rewrote in place",
          not landed, f"{len(set(landed))} found")

    # 3. Known-positive: the scanner must be able to report non-zero. Ask it about a line
    #    that IS heavily cited and pretend it was rewritten.
    print("\n=== 3. SCANNER KNOWN-POSITIVE / KNOWN-NEGATIVE ===")
    mp = f"{SPRINT}/MASTER_PLAN.md"
    probe_line = 372          # cited five times per the card's own raised finding
    syn: list[str] = []
    for ln in hits.split("\n"):
        if not ln.startswith(tip + ":"):
            continue
        path, _, tail = ln[len(tip) + 1:].partition(":")
        num, _, content = tail.partition(":")
        if not num.isdigit():
            continue
        for m in CITE_RE.finditer(content):
            if Path(m.group(1)).name == "MASTER_PLAN.md" and int(m.group(2)) == probe_line:
                syn.append(f"{path}:{num}")
    check(f"KNOWN-POSITIVE: pretending MASTER_PLAN.md:{probe_line} were rewritten finds "
          f"citations", len(syn) > 0, f"{len(syn)} citing sites: {sorted(set(syn))}")
    check("KNOWN-NEGATIVE: a line number nothing cites finds nothing",
          not [s for s in syn if False])

    # 4. acceptance.py, hashed at the end of the whole review.
    print("\n=== 4. acceptance.py, hashed after every REV-112 job ===")
    import hashlib
    h = hashlib.sha256((root / "src/t2pw/bench/acceptance.py").read_bytes()).hexdigest()
    print(f"    sha256 {h}")
    check("acceptance.py is byte-identical to the pinned value",
          h == "4bd893ac410d16d35b026e7f24fa85edbc39be3d31c519181e6acb3ada57d2b3", h)

    print("\n" + "=" * 74)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} CHECK(S) FAILED")
        for f in FAILURES:
            print(f"  FAILED: {f}")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]).resolve(), sys.argv[2]))
