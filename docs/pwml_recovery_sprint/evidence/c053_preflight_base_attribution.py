"""Was ``test_batch_preflight.py``'s deferred-import guard already red at the base?

C-053's collateral sweep found ``tests/test_batch_preflight.py`` failing. The file
is in **no chunk** (F-049's class), so nothing in the mandated gate set would have
surfaced it and a later card could easily mis-attribute it. This measures the
question directly instead of arguing it.

The test's predicate is entirely STATIC -- ``_deferred_imports`` parses a file with
``ast`` and ``_covered`` compares names against ``runner.CHILD_IMPORTS`` -- so it
can be evaluated against ANY revision of ``driver.py`` in the same interpreter. The
base file is taken from git (``git show <base>:src/t2pw/batch/driver.py``) into a
temporary path; nothing in the worktree is touched and no checkout, stash or
worktree command is run.

Output path is resolved before anything else (F-045).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "src", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _missed(path: Path) -> List[str]:
    from test_batch_preflight import _covered, _deferred_imports

    return [
        name
        for name in _deferred_imports(path)
        if name.split(".")[0] not in sys.stdlib_module_names and not _covered(name)
    ]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base", required=True, help="the base SHA")
    args = parser.parse_args(argv)

    out_path = Path(args.out).resolve()  # BEFORE anything runs (F-045)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tip_path = ROOT / "src" / "t2pw" / "batch" / "driver.py"
    blob = subprocess.run(
        ["git", "show", f"{args.base}:src/t2pw/batch/driver.py"],
        cwd=str(ROOT), capture_output=True, text=True, check=True,
    ).stdout

    with tempfile.TemporaryDirectory(prefix="c053preflight") as tmp:
        base_path = Path(tmp) / "driver_base.py"
        base_path.write_text(blob, encoding="utf-8")
        base_missed = _missed(base_path)
        tip_missed = _missed(tip_path)

    record: Dict[str, Any] = {
        "question": "is tests/test_batch_preflight.py's deferred-import guard red at the base?",
        "base_sha": args.base,
        "base_missed": base_missed,
        "tip_missed": tip_missed,
        "base_already_red": bool(base_missed),
        "modules_newly_uncovered_at_tip": sorted(set(tip_missed) - set(base_missed)),
    }
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    print(f"wrote {out_path}  (exists={out_path.is_file()})")
    # Exit 0 only when the tip introduces NO module the base did not already miss.
    return 0 if not record["modules_newly_uncovered_at_tip"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
