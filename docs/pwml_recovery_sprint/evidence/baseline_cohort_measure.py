"""Measure the pre-implementation baseline cohort for the H-001 manifest freeze.

Product-owner ruling (2026-08-05): the replay merge gate must stop globbing the
filesystem. This script records **what the cohort and its expectations actually
are today**, on the integration branch before any C-branch lands, so that:

* H-001's implementer has a verified target rather than a guess, and
* H-001's reviewer has an independent reference to check the frozen manifest
  against.

It imports the test module's OWN helpers (``_legs``, ``_full_stack``) rather than
re-implementing them, so the population is identical by construction -- including
the ``_MAX_PAYLOAD_BYTES = 2_000_000`` filter, which is why a naive glob reports a
different count.

This script measures. It freezes nothing and changes no test.

::

    .venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/baseline_cohort_measure.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _repo_root import REPO_ROOT  # noqa: E402

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

import test_strict_quarantine_real_artifact_replay as replay  # noqa: E402


def main() -> int:
    legs = replay._legs()
    leg_ids = replay._leg_ids()
    print(f"cohort size: {len(legs)} legs "
          f"(filter: final_mapped.json or merged_payload.json, "
          f"<= {replay._MAX_PAYLOAD_BYTES:,} bytes)\n")

    rows: List[Dict[str, Any]] = []
    for leg, leg_id in zip(legs, leg_ids):
        row = replay._full_stack(leg)
        rel = leg.relative_to(ROOT).as_posix()
        rows.append({"leg_id": leg_id, "path": rel, **{
            k: v for k, v in row.items() if k != "leg"
        }})

    admitted = [r for r in rows if r["quarantine"] == "ok"]
    measured = {
        "legs_examined": len(rows),
        "quarantine_admitted": len(admitted),
        "quarantine_refused": len(rows) - len(admitted),
        "stage3_after_pass": sum(1 for r in admitted if r["stage3_after"] == "pass"),
        "required_contract_pass": sum(1 for r in admitted if r["required_contract"] == "pass"),
        "reached_ir": sum(1 for r in admitted if r["ir"] is not None),
        "ir_pass": sum(1 for r in admitted if r["ir"] == "pass"),
        "exportable": sum(1 for r in rows if r["exportable"]),
    }
    by_leg = Counter(c for r in admitted for c in set(r["required_codes"]))
    by_row = Counter(c for r in admitted for c in r["required_codes"])

    print("MEASURED FULL_STACK_BASELINE (fixed cohort):")
    print(json.dumps(measured, indent=4))
    print("\nRESIDUAL_CODES_BY_LEG:")
    print(json.dumps(dict(sorted(by_leg.items())), indent=4))
    print("\nRESIDUAL_CODES_BY_ROW:")
    print(json.dumps(dict(sorted(by_row.items())), indent=4))

    print(f"\n{'#':>3}  {'run dir':<34}{'paper':<14}{'mode':<9}"
          f"{'quar':<6}{'st3':<6}{'req':<6}{'ir':<6}{'exp':<4}")
    print("-" * 96)
    for i, r in enumerate(rows, 1):
        parts = r["path"].split("/")
        print(f"{i:>3}  {parts[1]:<34}{parts[3]:<14}{parts[4]:<9}"
              f"{str(r['quarantine']):<6}{str(r['stage3_after']):<6}"
              f"{str(r['required_contract']):<6}{str(r['ir']):<6}"
              f"{'Y' if r['exportable'] else '-':<4}")

    per_run = Counter(r["path"].split("/")[1] for r in rows)
    print(f"\nlegs per run directory: {dict(per_run)}")

    # C-010's six expected changes, kept SEPARATE from this repair so the two
    # deltas can never be conflated (product-owner ruling, item 1).
    c010 = {
        "runs/2026-08-02_2130/papers/PMC12096016/strict",
        "runs/2026-08-02_2130/papers/PMC12856317/research",
        "runs_verify/2026-08-04_1207/papers/PMC12452463/strict",
        "runs_verify/2026-08-04_1234/papers/PMC12096016/strict",
        "runs_verify/2026-08-04_1754/papers/PMC12452463/research",
        "runs_verify/2026-08-04_1754/papers/PMC12452463/strict",
    }
    in_cohort = sorted(c010 & {r["path"] for r in rows})
    print(f"\nC-010 allowlist legs that fall INSIDE this cohort: "
          f"{len(in_cohort)} of 6")
    for path in in_cohort:
        print(f"  {path}")
    print("  (the rest live under runs_verify/, which this gate does not read)")

    out = ROOT / "docs" / "pwml_recovery_sprint" / "evidence" / "baseline_cohort_measured.json"
    payload = {
        "measured_on": "2026-08-05",
        "cohort_source": "tests/test_strict_quarantine_real_artifact_replay.py::_legs "
                         "(filesystem glob of runs/, TO BE REPLACED by H-001)",
        "max_payload_bytes": replay._MAX_PAYLOAD_BYTES,
        "full_stack_measured": measured,
        "residual_codes_by_leg": dict(sorted(by_leg.items())),
        "residual_codes_by_row": dict(sorted(by_row.items())),
        "c010_allowlist_legs_inside_cohort": in_cohort,
        "legs": rows,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
