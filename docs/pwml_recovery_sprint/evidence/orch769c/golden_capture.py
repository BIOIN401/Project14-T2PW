"""ORCH-769c: capture the seven golden driver-observable tuples of the tree in the cwd, with
each leg's ``row["counts"]``, so the seam-golden move is derived, not asserted.

Same method as ``evidence/c083_golden_capture.py``: imports ``tests/test_batch_driver_seam_golden``
of the tree under measurement and reuses its own ``_legs`` / ``_observe`` / ``_observable``.

Usage (cwd = the tree root; PYTHONPATH = <tree>/src)::

    bounded_run.py ... -- python golden_capture.py <out.json> <scratch dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path.cwd()
for _p in (root / "src", root / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import t2pw  # noqa: E402
import test_batch_driver_seam_golden as golden  # noqa: E402

out, scratch = Path(sys.argv[1]), Path(sys.argv[2])
for sub in ("t", "r"):
    (scratch / sub).mkdir(parents=True, exist_ok=True)
assert Path(t2pw.__file__).resolve().parent == (root / "src" / "t2pw").resolve(), t2pw.__file__

tuples, rows = {}, {}
for leg in sorted(golden._legs()):
    tuples[leg] = list(golden._observe(scratch / "t", leg))
    # A second run of the same leg, kept whole, so the delta names the fields that moved.
    body, mode, app_timeout = golden._legs()[leg]
    app = golden._write_app(scratch / "r", leg, body)
    rows[leg] = golden._observable(golden.run_one(golden.PAPER, mode, app_path=app, timeout=120.0,
                                                   app_timeout=app_timeout))

out.write_text(json.dumps({"tree": str(root), "t2pw": t2pw.__file__, "tuples": tuples, "observables": rows,
                           "committed_GOLDEN": {k: list(v) for k, v in golden.GOLDEN.items()},
                           "matches_committed": {k: tuples[k] == list(golden.GOLDEN[k]) for k in tuples}},
                          indent=1, sort_keys=True, default=str), encoding="utf-8")
print("wrote", out)
print("matches committed GOLDEN:", json.dumps({k: tuples[k] == list(golden.GOLDEN[k]) for k in tuples}))
