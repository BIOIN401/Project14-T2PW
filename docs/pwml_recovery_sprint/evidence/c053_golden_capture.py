"""Capture ``tests/test_batch_driver_seam_golden.py``'s observable, whole, per leg.

MASTER_PLAN SS3 hotspot 10 grants C-053 test-function-level ownership of
``_observable`` and ``GOLDEN`` on the condition that the move ships as an EXACT
SLOT-BY-SLOT DELTA with the base capture committed under ``evidence/``. Seven
digests move at once, and a digest on its own says nothing about WHY -- so this
dumps the whole ``_observable`` dict beside the three-slot tuple, at the base and
again at the tip, and the two files are diffable field by field.

It calls the test module's OWN ``_legs``/``_observable``/``run_one``, so the capture
cannot drift from what the test asserts. Run it at the base SHA first:

    <py> c053_golden_capture.py --out .../c053_golden_base.json

Output path is resolved before anything else runs (F-045) and every leg runs under
a throwaway temporary directory, exactly as ``tmp_path`` gives the test.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "src", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


#: The ONE removal hotspot 10 authorizes by name -- ``release_status_absent`` is
#: "replaced, not deleted" -- and the two fields that replace it. Any other name
#: leaving the list, or the list getting shorter, is the reject the guard exists
#: for: a digest stabilised by dropping a field.
_REPLACED = "release_status_absent"
_REPLACED_BY = ("release_status", "pwml_artifact")


def _delta(base_path: Path, tip_path: Path, out_path: Path) -> int:
    """The slot-by-slot move hotspot 10 requires, derived rather than narrated."""

    base = json.loads(base_path.read_text(encoding="utf-8"))["legs"]
    tip = json.loads(tip_path.read_text(encoding="utf-8"))["legs"]
    report: Dict[str, Any] = {
        "base_capture": str(base_path),
        "tip_capture": str(tip_path),
        "observable_fields_base": base[sorted(base)[0]]["observable_fields"],
        "observable_fields_tip": tip[sorted(tip)[0]]["observable_fields"],
        "legs": {},
    }
    dropped = set(report["observable_fields_base"]) - set(report["observable_fields_tip"])
    report["fields_added"] = sorted(
        set(report["observable_fields_tip"]) - set(report["observable_fields_base"])
    )
    report["fields_dropped"] = sorted(dropped)
    for leg in sorted(set(base) | set(tip)):
        before, after = base[leg]["golden_slots"], tip[leg]["golden_slots"]
        seen_before, seen_after = base[leg]["observable"], tip[leg]["observable"]
        report["legs"][leg] = {
            "slot_0_status_stage_kind": {"base": before[0], "tip": after[0], "moved": before[0] != after[0]},
            "slot_1_message": {"base": before[1], "tip": after[1], "moved": before[1] != after[1]},
            "slot_2_digest": {"base": before[2], "tip": after[2], "moved": before[2] != after[2]},
            "observable_fields_that_differ": sorted(
                key
                for key in set(seen_before) | set(seen_after)
                if seen_before.get(key) != seen_after.get(key)
            ),
        }
    unauthorized = sorted(dropped - {_REPLACED})
    report["unauthorized_removals"] = unauthorized
    report["growth_only"] = (
        not unauthorized
        and len(report["observable_fields_tip"]) >= len(report["observable_fields_base"])
        and set(_REPLACED_BY) <= set(report["observable_fields_tip"])
    )
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out_path}  (exists={out_path.is_file()})")
    print(f"fields added         : {report['fields_added']}")
    print(f"fields removed       : {report['fields_dropped']}")
    print(f"unauthorized removals: {unauthorized}  (hotspot 10: must be [])")
    print(f"GROWTH-ONLY          : {report['growth_only']}")
    for leg, data in sorted(report["legs"].items()):
        print(f"  {leg}: differ={data['observable_fields_that_differ']}")
    return 0 if report["growth_only"] else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--delta", nargs=2, metavar=("BASE", "TIP"))
    args = parser.parse_args(argv)

    out_path = Path(args.out).resolve()  # BEFORE anything runs (F-045)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.delta:
        return _delta(Path(args.delta[0]).resolve(), Path(args.delta[1]).resolve(), out_path)

    import t2pw
    from t2pw.batch.driver import run_one
    from test_batch_driver import PAPER, _write_app
    from test_batch_driver_seam_golden import _legs, _observable, _sha

    record: Dict[str, Any] = {
        "capture": "tests/test_batch_driver_seam_golden.py :: _observable, per leg",
        "t2pw_file": t2pw.__file__,
        "legs": {},
    }
    with tempfile.TemporaryDirectory(prefix="c053golden") as tmp:
        for leg in sorted(_legs()):
            body, mode, app_timeout = _legs()[leg]
            app = _write_app(Path(tmp), leg, body)
            outcome = run_one(PAPER, mode, app_path=app, timeout=120.0, app_timeout=app_timeout)
            seen = _observable(outcome)
            row = seen["row"]
            record["legs"][leg] = {
                "golden_slots": [
                    f"{row['status']}|{row['stage']}|{row['failure_kind']}",
                    row["message"],
                    _sha(json.dumps(seen, sort_keys=True)),
                ],
                "observable_fields": sorted(seen),
                "observable": seen,
            }

    out_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(f"t2pw.__file__ = {t2pw.__file__}")
    print(f"wrote {out_path}  (exists={out_path.is_file()})")
    for leg, data in sorted(record["legs"].items()):
        print(f'    "{leg}": {tuple(data["golden_slots"])!r},')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
