"""C-092 probe: the C-010 pre-prune moved-leg set over the CURRENT corpus.

``tests/test_strict_quarantine_real_artifact_replay.py`` pins
``EXPECTED_P01_DELTAS`` and asserts ``sorted(moved) == sorted(expected)``. That
equality is a corpus census pin of the same class as C-074's. This probe prints
every moved leg and its tuple so the unlisted ones can be judged on their shape
rather than on their membership.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "tests"))

import test_strict_quarantine_real_artifact_replay as R  # noqa: E402
from t2pw.pipeline import strict_quarantine as SQ  # noqa: E402


def main() -> int:
    legs = []
    for base in ("runs", "runs_verify"):
        if (ROOT / base).is_dir():
            legs.extend(sorted((ROOT / base).glob("*/papers/*/*/final_mapped.json")))

    shipped_dz = SQ._degree_zero_exports
    shipped_sv = SQ._surviving_processes
    moved = {}
    for path in legs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        SQ._degree_zero_exports = R._pre_c010_degree_zero
        SQ._surviving_processes = R._pre_c010_surviving
        try:
            before = R._p01_observables(payload)
        finally:
            SQ._degree_zero_exports = shipped_dz
            SQ._surviving_processes = shipped_sv
        after = R._p01_observables(payload)
        if before != after:
            rel = path.parent.relative_to(ROOT).as_posix()
            moved[rel] = (before[0], after[0], before[1], after[1], before[2], after[2])

    json.dump(
        {
            "legs_measured": len(legs),
            "moved": moved,
            "listed": sorted(R.EXPECTED_P01_DELTAS),
            "unlisted": sorted(set(moved) - set(R.EXPECTED_P01_DELTAS)),
        },
        sys.stdout,
        indent=2,
        sort_keys=True,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
