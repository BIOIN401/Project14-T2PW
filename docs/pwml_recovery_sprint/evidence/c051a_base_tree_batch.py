"""Hash-verified base tree through ONE ``git cat-file --batch``. Reusable.

``c045b_base_tree.py`` spawns three ``git.exe`` per blob -- ~5,500 for 1842 --
and ``bounded_run``'s PID enumeration then wrote an 81,566-byte report, over G11
RULE 5's 65,536-byte cap. ``git archive`` is NOT the alternative: ``REV-045a``
measured ``.gitattributes`` EOL conversion mismatching 326 of 379 blob hashes,
destroying this verification silently. So verification is in-process, on git's
own blob id ``sha1(b"blob <len>\\0" + data)``, and ``PATHSPEC`` is imported
rather than restated so the two exporters cannot drift.

Usage::  <python> c051a_base_tree_batch.py --rev <sha> --dest <dir>
"""

from __future__ import annotations

import argparse, hashlib, subprocess, sys  # noqa: E401
from pathlib import Path
from typing import Dict, List

from _repo_root import REPO_ROOT
from c045b_base_tree import PATHSPEC


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rev", required=True)
    parser.add_argument("--dest", required=True)
    args = parser.parse_args(argv)
    listing = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-tree", "-r", args.rev, "--", *PATHSPEC],
        check=True, stdout=subprocess.PIPE).stdout.decode("utf-8", "replace")
    entries: Dict[str, str] = {}
    for line in listing.splitlines():
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) == 3 and parts[1] == "blob":
            entries[path.strip('"')] = parts[2]
    assert entries, f"FAILED: {args.rev} has no blobs under {PATHSPEC}"
    dest, mismatched = Path(args.dest), []
    child = subprocess.Popen(["git", "-C", str(REPO_ROOT), "cat-file", "--batch"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert child.stdin is not None and child.stdout is not None
    try:
        for path, sha in entries.items():
            child.stdin.write(f"{sha}\n".encode())
            child.stdin.flush()
            data = child.stdout.read(int(child.stdout.readline().split()[2]))
            child.stdout.read(1)  # git terminates each payload with a newline
            target = dest / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            blob = target.read_bytes()  # re-read: catches a filtered write too
            if hashlib.sha1(b"blob %d\0" % len(blob) + blob).hexdigest() != sha:
                mismatched.append(path)
    finally:
        child.stdin.close()
        child.wait()
    print(f"rev / dest     : {args.rev} -> {dest}\n"
          f"blobs exported : {len(entries)}\n"
          f"blobs verified : {len(entries) - len(mismatched)}")
    if mismatched:
        print(f"MISMATCHED     : {mismatched[:10]}\n"
              "FAILED: the exported tree is not byte-identical to the revision")
        return 1
    print("VERIFIED: every exported file re-hashes to its blob sha at this rev")
    return 0


if __name__ == "__main__":
    sys.exit(main())
