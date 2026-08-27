"""C-092 census probe: what the committed corpus actually holds NOW.

Read-only. Reproduces, from outside the test modules, the four quantities that
``tests/test_c074_strict_core_floor.py`` pinned as literals at C-074 time, plus
the moved-leg set that ``tests/test_strict_quarantine_real_artifact_replay.py``
pins in ``EXPECTED_P01_DELTAS``. Emits JSON on stdout so the design decision in
the C-092 report rests on a measurement rather than an argument.
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

import test_c074_strict_core_floor as M  # noqa: E402

from t2pw.pipeline.release_status import (  # noqa: E402
    DIAGNOSTIC_ONLY,
    RELEASE_READY,
    REVIEW_REQUIRED,
)


def main() -> int:
    legs = M._committed_legs()
    order = {RELEASE_READY: 2, REVIEW_REQUIRED: 1, DIAGNOSTIC_ONLY: 0}
    out = {
        "legs_listed": len(legs),
        "with_release": 0,
        "recorded_release_ready": 0,
        "control_release_ready": 0,
        "demoted": {},
        "untouched": {},
        "armb_hits_via_verdict": [],
        "armb_hits_via_raw_json": [],
        "core_sizes_release_ready": {},
        "cap_violations": [],
        "detail": {},
    }
    for label, leg in legs:
        report, payload = M._load(leg)
        raw = report.get("coverage") or {}
        ctx = raw.get("requested_context") or {}
        declared = bool(
            (ctx.get("key_compounds") or [])
            or (ctx.get("key_proteins") or [])
            or (ctx.get("subprocesses") or [])
        )
        if M.CoverageVerdict(raw).declares_core_without_stating_a_pathway:
            out["armb_hits_via_verdict"].append(label)
            out["detail"].setdefault(label, {})["armb_recorded"] = str(
                (report.get("release") or {}).get("status") or ""
            )
        if declared and not str(ctx.get("pathway_name") or "").strip():
            out["armb_hits_via_raw_json"].append(label)

        record = report.get("release") or {}
        recorded = str(record.get("status") or "")
        if recorded not in order or payload is None:
            continue
        out["with_release"] += 1
        control = M._replay(report, payload, arms=False)
        applied = M._replay(report, payload, arms=True)
        if order[applied.status] > order[control.status]:
            out["cap_violations"].append(label)
        if control.status == RELEASE_READY:
            out["control_release_ready"] += 1
        if recorded != RELEASE_READY:
            continue
        out["recorded_release_ready"] += 1
        out["core_sizes_release_ready"][label] = M._connected_core_size(report, payload)
        out["detail"][label] = {
            "recorded": recorded,
            "control_status": control.status,
            "control_reasons": [str(r) for r in control.reasons],
            "applied_status": applied.status,
            "applied_reasons": [str(r) for r in applied.reasons],
            "pathway_name": str(ctx.get("pathway_name") or ""),
            "declared_core": declared,
            "core_size": M._connected_core_size(report, payload),
        }
        arms = M._arms_that_fired(applied)
        if arms:
            out["demoted"][label] = " + ".join(arms)
        else:
            out["untouched"][label] = M._connected_core_size(report, payload)
    json.dump(out, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
