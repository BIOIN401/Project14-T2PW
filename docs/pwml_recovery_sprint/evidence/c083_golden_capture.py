"""C-083: capture the seven golden driver-observable tuples of the tree it runs in.

Orchestration-side measurement, not pipeline code. It exists so the C-083 golden
baseline move is DERIVED and reproducible rather than asserted: run it once in a git
worktree at the base SHA and once at the tip, in the SAME interpreter with ``.env``
present on BOTH sides (F-051), and diff the two captures.

It imports ``tests/test_batch_driver_seam_golden`` and reuses that module's own
``_legs`` / ``_observe`` / ``_observable``, so the capture cannot drift from the
harness whose baseline it is measuring.

Beside the seven tuples it records the WHOLE row of ``input_timeout`` -- the only
leg C-083 can move, because it is the only fixture that reaches
``_finalize_timeout`` -- so the delta can name the fields that appeared instead of
reporting only that a digest moved.

Usage, from a worktree root::

    <python> docs/pwml_recovery_sprint/evidence/bounded_run.py \\
        --timeout 900 --label <label> --json <allocated> -- \\
        <python> docs/pwml_recovery_sprint/evidence/c083_golden_capture.py \\
        <out.json> <scratch-dir under C:/t/>

Produced ``c083_golden_base.json`` (at ``116c8fa``) and ``c083_golden_tip.json``;
``c083_golden_delta.json`` is the slot-by-slot difference of the two.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path.cwd()
for _p in (root / "src", root / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import test_batch_driver_seam_golden as golden  # noqa: E402

out = Path(sys.argv[1])
scratch = Path(sys.argv[2])
scratch.mkdir(parents=True, exist_ok=True)

captured = {leg: list(golden._observe(scratch, leg)) for leg in sorted(golden._legs())}

# The whole observable of the ONE leg that can move, so the delta names fields and
# not just digests. ``seconds`` is popped by ``_observable`` -- wall clock is not
# behaviour -- so the row here is already comparable across the two trees.
body, mode, app_timeout = golden._legs()["input_timeout"]
app = golden._write_app(scratch, "detail_input_timeout", body)
outcome = golden.run_one(golden.PAPER, mode, app_path=app, timeout=120.0, app_timeout=app_timeout)
observable = golden._observable(outcome)

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(
    json.dumps(
        {
            "tuples": captured,
            "input_timeout_row": observable["row"],
            "input_timeout_row_keys": sorted(observable["row"]),
            "committed_GOLDEN": {leg: list(v) for leg, v in golden.GOLDEN.items()},
        },
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)
print(f"wrote {out}")
