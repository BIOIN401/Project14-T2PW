"""C-056d: capture the seven golden driver-observable tuples of the tree it runs in.

Orchestration-side measurement, not pipeline code. It exists so the C-056d golden
baseline move is DERIVED and reproducible rather than asserted: run it once in a git
worktree at the base SHA and once at the tip, in the SAME interpreter with `.env`
present on BOTH sides (F-051), and diff the two captures.

It imports ``tests/test_batch_driver_seam_golden`` and reuses that module's own
``_legs`` / ``_observe`` / ``_observable``, so the capture cannot drift from the
harness whose baseline it is measuring.

Usage, from a worktree root::

    .venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/bounded_run.py \\
        --timeout 600 --label <label> --json <allocated> -- \\
        .venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/c056d_golden_capture.py \\
        <out.json> <scratch-dir under C:/t/>

Produced ``c056d_golden_base.json`` (at ``40fdb23``) and ``c056d_golden_tip.json``;
``c056d_golden_delta.json`` is the slot-by-slot difference of the two.
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

# The whole ``release_status`` for the two legs that carry one, so the delta can name
# the FIELD that moved rather than only reporting that a digest did.
detail = {}
for leg in ("strict_gate_failure", "strict_contract_failure"):
    body, mode, app_timeout = golden._legs()[leg]
    app = golden._write_app(scratch, f"detail_{leg}", body)
    outcome = golden.run_one(golden.PAPER, mode, app_path=app, timeout=120.0, app_timeout=app_timeout)
    detail[leg] = golden._observable(outcome)["release_status"]

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(
    json.dumps({"tuples": captured, "release_status": detail}, indent=2, sort_keys=True),
    encoding="utf-8",
)
print(f"wrote {out}")
