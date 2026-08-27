"""C-093: measure ``_leg_digest`` over the WHOLE committed corpus, under pytest.

Not a suite test. ``pytest.ini`` sets ``testpaths = tests``, so nothing here is
collected by a normal run; it is collected only when this file is named on the
command line. It exists because **F-047 makes pytest the authority** for these
digests -- ``evidence/c045a_golden_rebaseline.py --mode digest`` runs OUTSIDE
pytest and is documented to report different values for every IR-BUILDING leg,
so it cannot be used to derive a new golden entry.

It calls ``tests/test_compound_resolution_extraction.py``'s own ``_leg_digest``
and reinstalls that module's autouse ``_no_live_db`` guard, so there is no second
implementation to trust and no path to the live PathBank DB.

Writes ``$C093_SWEEP_OUT`` -- ``{leg: {digest, stops, in_golden, golden_digest}}``
plus a corpus summary. Asserts nothing about the values: it is an instrument.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import test_compound_resolution_extraction as tm  # noqa: E402


@pytest.fixture(autouse=True)
def _no_live_db(monkeypatch: Any) -> None:
    """The measured module's own guard, reinstalled here.

    ``tm``'s autouse fixture belongs to ITS module and does not apply to a node
    collected from this file, so without this the sweep would call
    ``PathBankDbResolver.from_env()`` for real under configuration E and measure
    a network answer into a digest.
    """
    import t2pw.mapping.map_ids as map_ids

    class _ForcedUnavailable:
        @classmethod
        def from_env(cls) -> Any:
            raise RuntimeError("harvest_forced_unavailable")

    monkeypatch.setattr(map_ids, "PathBankDbResolver", _ForcedUnavailable)


def _corpus() -> list[str]:
    listed = subprocess.run(["git", "ls-files", "*final_mapped.json"], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    return sorted(listed.stdout.split())


def test_c093_leg_digest_sweep() -> None:
    out = os.environ.get("C093_SWEEP_OUT")
    assert out, "set C093_SWEEP_OUT to the file this instrument should write"
    rows: Dict[str, Any] = {}
    for leg in _corpus():
        payload = json.loads((ROOT / leg).read_text(encoding="utf-8"))
        digest, stops = tm._leg_digest(payload)
        rows[leg] = {
            "digest": digest,
            "stops": stops,
            "in_golden": leg in tm.GOLDEN,
            "golden_digest": tm.GOLDEN.get(leg),
            "matches_golden": tm.GOLDEN.get(leg) == digest if leg in tm.GOLDEN else None,
            "in_excluded": leg in tm.EXCLUDED,
        }
    moved = sorted(k for k, v in rows.items() if v["matches_golden"] is False)
    Path(out).write_text(json.dumps({
        "corpus_size": len(rows),
        "golden_legs": sum(1 for v in rows.values() if v["in_golden"]),
        "golden_digests_that_moved": moved,
        "legs_with_stops": sorted(k for k, v in rows.items() if v["stops"]),
        "rows": rows,
    }, indent=1, sort_keys=True), encoding="utf-8")
    print(f"C093_SWEEP wrote {out}: {len(rows)} legs, "
          f"{len(moved)} golden digests moved", flush=True)
