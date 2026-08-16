"""C-051b: pin the 32 protected leg fixtures at a rev, hash-verified.

``c045b_base_tree.PATHSPEC`` carries code and ``data/pathwhiz_id_db.json``, not
fixtures, so a base tree exported with it cannot answer ``--mode digest``: every
key of ``GOLDEN`` resolves to a missing file. This rebinds that shared list --
before ``c051a_base_tree_batch`` does its ``from ... import PATHSPEC`` -- to the
leg fixtures discovered by exactly the rule
``test_the_golden_covers_every_committed_leg_fixture`` uses, every
``final_mapped.json`` under ``runs/`` and ``runs_verify/``, and then defers to
C-051a's exporter **unchanged**. So there is no second, drifting implementation
of the export, the same in-process ``sha1(b"blob <len>\\0" + data)`` check
re-read from disk covers the fixtures, and ``git archive`` is still avoided
(``REV-045a`` measured ``.gitattributes`` EOL conversion mismatching 326 of 379
blob hashes, destroying that verification while appearing to work).

``runs/`` and ``runs_verify/`` are PROTECTED: this only ever reads them through
``git``, and writes exclusively under ``--dest``.

Usage::  <python> c051b_leg_fixture_export.py --rev <sha> --dest <dir>
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import c045b_base_tree  # noqa: E402
from _repo_root import REPO_ROOT  # noqa: E402

_rev = sys.argv[sys.argv.index("--rev") + 1]
_listing = subprocess.run(
    ["git", "-C", str(REPO_ROOT), "ls-tree", "-r", "--name-only", _rev, "--",
     "runs", "runs_verify"],
    check=True, stdout=subprocess.PIPE).stdout.decode("utf-8", "replace")
_legs = sorted(
    line.strip('"') for line in _listing.splitlines()
    if line.strip('"').endswith("final_mapped.json")
)
assert _legs, f"FAILED: {_rev} has no committed leg fixtures"
print(f"leg fixtures at {_rev}: {len(_legs)}", flush=True)
c045b_base_tree.PATHSPEC = _legs

import c051a_base_tree_batch  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(c051a_base_tree_batch.main())
