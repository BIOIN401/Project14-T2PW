"""Generate the C-010 per-leg expected-delta allowlist.

WHAT THIS PROVES
----------------
``quarantine_and_close`` calls ``_drop_quarantined_processes`` (compacting every
process bucket) and THEN calls ``_degree_zero_exports``, which resolves ORIGINAL
admission indices against the now-compacted lists via ``_surviving_processes``.
Out-of-range indices are silently skipped; shifted in-range indices resolve to
the wrong row. Entities referenced only by skipped rows are falsely reported
degree-zero and the run refuses export.

This script replays every archived leg twice -- once as shipped, once with
degree-zero resolved against an immutable pre-prune snapshot -- and prints the
exact set of legs whose verdict changes.

WHY IT IS COMMITTED
-------------------
``BASELINE.md`` section 5 carries the resulting table, and ``C-010``'s regression
test pins it. The table is the acceptance criterion; this is the thing that
produces it. Gate G9 requires the implementer's test to fail on the base SHA, and
a reviewer verifying that needs to be able to regenerate the table independently
rather than trusting a number in a document.

The monkeypatching here is a MEASUREMENT DEVICE, not a proposed patch. C-010
implements the fix properly inside ``strict_quarantine.py``; this script only has
to answer "what would change?" without modifying the module under test.

INVOCATION
----------
    .venv/Scripts/python.exe docs/pwml_recovery_sprint/evidence/allowlist_generator.py

No network, no LLM, no database. Reads committed run artifacts only. Runtime is
seconds.

INPUTS
------
Every ``runs*/<stamp>/papers/<paper>/<mode>/final_mapped.json`` in the checkout.
``pathway_context=None`` is passed deliberately: no archived leg carries Stage-0
context (``test_strict_quarantine_real_artifact_replay`` pins this), so the
replay runs in the same undeclared-core regime the harness itself uses.

EXPECTED OUTPUT once ``runs_verify/2026-08-04_1754/`` is committed by INIT-001::

    legs examined : 32      unchanged : 26      CHANGED : 6      errors : 0

Before that commit only 30 legs are visible and 4 changes are found; the two
PMC12452463 rows from the 1754 run are missing. That is a checkout difference,
not a disagreement -- see BASELINE.md section 5.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from _repo_root import REPO_ROOT, add_src_to_path

add_src_to_path()

import t2pw.pipeline.strict_quarantine as SQ  # noqa: E402


def _patched_degree_zero(snapshot: dict):
    """``_degree_zero_exports`` computed against the PRE-PRUNE process lists.

    Entity rows still come from the post-drop payload -- only the REFERENCE set
    is taken from the snapshot, which is exactly the change C-010 must make.
    """

    def patched(payload, admissions):
        referenced = SQ._referenced_entity_norms(snapshot["pre"], admissions)
        complexes = SQ._safe_list(SQ._safe_dict(payload.get("entities")).get("protein_complexes"))
        surviving_complex_norms = {
            SQ._normalize(SQ._row_name(row))
            for row in complexes
            if isinstance(row, dict) and SQ._normalize(SQ._row_name(row)) in referenced
        }
        exempt = SQ._complex_component_norms(payload, surviving_complex_norms)
        out = []
        for bucket in SQ._DEGREE_ZERO_BUCKETS:
            for row in SQ._safe_list(SQ._safe_dict(payload.get("entities")).get(bucket)):
                if not isinstance(row, dict):
                    continue
                name = SQ._row_name(row)
                norm = SQ._normalize(name)
                if not norm or norm in referenced or norm in exempt:
                    continue
                out.append({"bucket": bucket, "name": name})
        return out

    return patched


def main() -> int:
    legs = []
    for base in ("runs", "runs_verify"):
        base_dir = REPO_ROOT / base
        if base_dir.is_dir():
            legs.extend(sorted(base_dir.glob("*/papers/*/*/final_mapped.json")))

    original_drop = SQ._drop_quarantined_processes
    original_degree_zero = SQ._degree_zero_exports
    snapshot: dict = {}

    def snapshotting_drop(working, admissions):
        snapshot["pre"] = copy.deepcopy(working)
        return original_drop(working, admissions)

    changed, unchanged, errors = [], 0, []

    for path in legs:
        rel = str(path.relative_to(REPO_ROOT)).replace("\\", "/").rsplit("/", 1)[0]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - a bad artifact is a finding, not a crash
            errors.append((rel, f"unreadable: {exc}"))
            continue
        try:
            SQ._drop_quarantined_processes = original_drop
            SQ._degree_zero_exports = original_degree_zero
            shipped = SQ.quarantine_and_close(copy.deepcopy(payload), strict_db=True)

            SQ._drop_quarantined_processes = snapshotting_drop
            SQ._degree_zero_exports = _patched_degree_zero(snapshot)
            fixed = SQ.quarantine_and_close(copy.deepcopy(payload), strict_db=True)
        except Exception as exc:  # noqa: BLE001
            errors.append((rel, f"{type(exc).__name__}: {exc}"))
            continue
        finally:
            SQ._drop_quarantined_processes = original_drop
            SQ._degree_zero_exports = original_degree_zero

        before = sorted(r["name"] for r in shipped.quarantine_report["strict_invariants"]["degree_zero_exports"])
        after = sorted(r["name"] for r in fixed.quarantine_report["strict_invariants"]["degree_zero_exports"])
        if before != after or shipped.ok != fixed.ok or shipped.refusal_reasons != fixed.refusal_reasons:
            changed.append(
                (rel, before, after, shipped.ok, fixed.ok,
                 list(shipped.refusal_reasons), list(fixed.refusal_reasons))
            )
        else:
            unchanged += 1

    print(f"legs examined : {len(legs)}")
    print(f"unchanged     : {unchanged}")
    print(f"CHANGED       : {len(changed)}")
    print(f"errors        : {len(errors)}")
    print()
    for rel, before, after, ok_before, ok_after, refusals_before, refusals_after in changed:
        print(f"  {rel}")
        print(f"      degree_zero : {before}  ->  {after}")
        print(f"      ok          : {ok_before}  ->  {ok_after}")
        print(f"      refusals    : {refusals_before}  ->  {refusals_after}")
    for rel, message in errors:
        print(f"  ERR {rel}: {message}")

    print()
    print("INVARIANTS the C-010 test must assert:")
    print("  1. every listed leg transitions exactly as tabulated")
    print("  2. set(changed legs) == set(allowlist) -- no unlisted leg moved")
    print("  3. every changed leg's degree_zero AFTER is [] -- the fix never ADDS one")
    adds = [rel for rel, _b, after, *_rest in changed if after]
    print(f"  invariant 3 holds: {not adds}" + (f"  violated by {adds}" if adds else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
