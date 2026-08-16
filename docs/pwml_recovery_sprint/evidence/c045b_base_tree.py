"""Materialize a hash-verified base tree for C-045b's G9 behavioural proof.

The trap this exists to close (charter A7). ``tests/test_pwml_writer.py`` does
``sys.path.insert(0, Path(__file__).resolve().parents[1] / "src")`` from its own
location, and ``docs/.../c045_pinned_pytest.py`` imports ``t2pw`` before pytest
starts. So a "base" leg run from the tip worktree, or from a tree whose ``src``
is not the base's, silently measures **tip** code and passes. The only reliable
answer is a real base tree on disk whose sources are proven to be the base's.

``git archive`` would be the obvious way and is what C-045 used, but it applies
``.gitattributes`` EOL conversion, so the bytes on disk need not be the blob's.
This writes blob content directly with ``git cat-file`` -- no filters, no
conversion -- and then **verifies** every exported file by re-hashing it with
``git hash-object`` and comparing against ``git ls-tree``. A single mismatch
aborts: an unverified base tree is not a base proof.

Usage::

    <python> c045b_base_tree.py --rev d146be48 --dest C:/t/c045b/base
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from _repo_root import REPO_ROOT

#: Enough of the tree to run the focused suite: the package, the tests, the
#: reference PWML, the offline name index, pytest config, and the evidence
#: helpers the pinned runner imports.
PATHSPEC: List[str] = [
    "src", "tests", "reference", "pytest.ini", "pyproject.toml",
    "data/pathwhiz_id_db.json", "docs/pwml_recovery_sprint/evidence",
]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=True, stdout=subprocess.PIPE,
    ).stdout.decode("utf-8", "replace")


def _tree(rev: str) -> Dict[str, str]:
    entries: Dict[str, str] = {}
    for line in _git("ls-tree", "-r", rev, "--", *PATHSPEC).splitlines():
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) == 3 and parts[1] == "blob":
            entries[path.strip('"')] = parts[2]
    return entries


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rev", required=True)
    parser.add_argument("--dest", required=True)
    args = parser.parse_args(argv)

    dest = Path(args.dest)
    entries = _tree(args.rev)
    if not entries:
        print(f"FAILED: {args.rev} has no blobs under {PATHSPEC}")
        return 1

    for path, sha in entries.items():
        target = dest / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            subprocess.run(
                ["git", "-C", str(REPO_ROOT), "cat-file", "blob", sha],
                check=True, stdout=subprocess.PIPE,
            ).stdout
        )

    mismatched = [
        path for path, sha in entries.items()
        if _git("hash-object", "--", str((dest / path).resolve())).strip() != sha
    ]
    print(f"rev              : {_git('rev-parse', args.rev).strip()}")
    print(f"dest             : {dest}")
    print(f"blobs exported   : {len(entries)}")
    print(f"blobs verified   : {len(entries) - len(mismatched)}")
    if mismatched:
        print(f"MISMATCHED       : {mismatched[:10]}")
        print("FAILED: the exported tree is not byte-identical to the revision")
        return 1
    print("VERIFIED: every exported file re-hashes to its blob sha at this rev")
    return 0


if __name__ == "__main__":
    sys.exit(main())
