"""C-083: capture every observable of the OUTER timeout path in the tree it runs in.

Charter step 6.2: prove the OUTER path (``runner._timeout_row``, defect 2, REFUTED)
is byte-identical to the base. C-083 does not touch ``runner.py``, and this is the
behavioural statement of that rather than a diff-level one.

The sample is a grid over the two inputs that decide a killed-child row --
``seconds`` (elapsed) and ``timeout`` (the kill ceiling) -- crossed with the three
``reason=`` channels (unset, and both operational reasons stated explicitly),
plus the ceiling values the real runs used. Every row is serialized whole, so a
single changed byte anywhere in the record shows up.

Usage, from a worktree root::

    <python> docs/pwml_recovery_sprint/evidence/bounded_run.py \\
        --timeout 300 --label <label> --json <allocated> -- \\
        <python> docs/pwml_recovery_sprint/evidence/c083_outer_capture.py <out.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path.cwd()
if str(root / "src") not in sys.path:
    sys.path.insert(0, str(root / "src"))

from t2pw.batch import runner  # noqa: E402

PAPER = {"paper_id": "PMC12444477", "title": "a paper", "source_uri": "u", "topic": "t"}

SECONDS = [0.0, 9.0, 400.0, 471.99, 1672.86, 3479.0, 3480.0, 3599.0, 3600.0, 3600.4, 7200.0]
TIMEOUTS = [60.0, 120.0, 1234.0, 3600.0]
REASONS = ["", "budget_exhausted", "operation_timeout"]

rows = {}
for secs in SECONDS:
    for tmo in TIMEOUTS:
        for reason in REASONS:
            key = f"{secs}|{tmo}|{reason or '-'}"
            try:
                rows[key] = runner._timeout_row(
                    slug="PMC12444477__a", mode="strict", paper=PAPER,
                    seconds=secs, timeout=tmo, tail="the tail", reason=reason,
                )
            except Exception as exc:  # noqa: BLE001 - a raise is an observable too
                rows[key] = {"_raised": f"{type(exc).__name__}: {exc}"}

out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"sample_size": len(rows), "rows": rows}, indent=2, sort_keys=True),
               encoding="utf-8")
print(f"wrote {out} ({len(rows)} rows)")
