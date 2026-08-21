"""C-067 replay-cohort census: BOTH columns of the C-010 attribution table.

REV-067 rejected the first version of this instrument. It was sound but aimed one
layer too low: it measured only the ``after`` column, so it reported an empty
delta across all 35 legs while
``test_the_pre_prune_reference_set_moves_exactly_the_allowlisted_legs`` was
breaking on the ``before`` column. That test's ``before`` is produced by
``_pre_c010_degree_zero`` (``test_strict_quarantine_real_artifact_replay.py:1163-1172``),
which calls the SHIPPED ``_degree_zero_exports`` -- so a change to that function
moves the ``before`` column too.

This version measures what the test measures: for every committed leg, ``before``
under the two pre-C-010 substitutions and ``after`` on the shipped code, and the
same six-tuple the test tabulates in ``EXPECTED_P01_DELTAS``.

``--src`` is prepended to ``sys.path`` BEFORE ``t2pw`` is imported, and the run
aborts if the module did not come from there. The base arm therefore runs the base
bytes of ``strict_quarantine.py``, not a local reimplementation of them.

The two measurement devices below are the test's, reproduced as substitutions into
the SHIPPED code rather than copies of an old body, exactly as the test does it, so
they cannot drift from what they claim to reproduce.

**What this measures and what it does not.** These are the committed
``final_mapped.json`` files -- the REPLAY cohort. They are not the payload the
production quarantine judged: ``FINDINGS.md:1580`` records that the quarantine
input payload is never persisted, and the run report's ``admitted_payload_hash``
does not match any committed file. A leg that does not move here has not been shown
to be unmoved in production, and a leg that does move here has not been shown to
move in production.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from typing import Any, Dict, List, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

SQ: Any = None


def _bind_source(src: str) -> str:
    global SQ
    src = os.path.abspath(src)
    sys.path.insert(0, src)
    from t2pw.pipeline import strict_quarantine as module

    loaded_from = os.path.abspath(module.__file__)
    if not loaded_from.startswith(src):
        raise SystemExit(f"refusing to measure: t2pw loaded from {loaded_from}, not {src}")
    SQ = module
    return loaded_from


def _legs(root: str) -> List[str]:
    out: List[str] = []
    for base in ("runs", "runs_verify"):
        top = os.path.join(root, base)
        if not os.path.isdir(top):
            continue
        for dirpath, _dirnames, filenames in os.walk(top):
            if "final_mapped.json" in filenames:
                out.append(os.path.join(dirpath, "final_mapped.json"))
    return sorted(out)


def _observables(payload: Dict[str, Any]) -> Tuple[List[str], bool, List[str]]:
    """``_p01_observables`` (``test_..._real_artifact_replay.py:1179-1182``)."""

    result = SQ.quarantine_and_close(deepcopy(payload), strict_db=True, pathway_context=None)
    degree_zero = result.quarantine_report["strict_invariants"]["degree_zero_exports"]
    return sorted(row["name"] for row in degree_zero), result.ok, list(result.refusal_reasons)


def _release_observables(payload: Dict[str, Any]) -> Dict[str, Any]:
    """``ok`` beside the classifier verdict, to test one specific contradiction.

    ``ok`` is the PWML production switch (``app/streamlit_app.py:4717``) and
    ``PRODUCT_CONTRACT.md:343`` forbids a final PWML on ``diagnostic_only``. A run
    that reports both is a shipped export on a refused graph. Measured rather than
    argued, because ``ok`` (``:2500``, folds in coverage, always True in research
    mode) and ``strict_gates_passed`` (``:2384-2386``, four structural gates) are
    genuinely different values and do diverge on real legs.
    """

    result = SQ.quarantine_and_close(deepcopy(payload), strict_db=True, pathway_context=None)
    release = result.quarantine_report.get("release") or {}
    invariants = result.quarantine_report.get("strict_invariants") or {}
    return {
        "ok": bool(result.ok),
        "invariants_ok": bool(invariants.get("ok")),
        "release_status": release.get("status"),
        "strict_gates_passed": release.get("strict_gates_passed"),
        "contradiction": bool(result.ok) and release.get("status") == "diagnostic_only",
    }


def _measure_leg(payload: Dict[str, Any]) -> Dict[str, Any]:
    shipped_dz = SQ._degree_zero_exports
    shipped_surviving = SQ._surviving_processes

    def _pre_c010_degree_zero(payload_, admissions, **_ignored):
        return shipped_dz(payload_, admissions, process_snapshot=payload_.get("processes") or {})

    def _pre_c010_surviving(payload_, admissions, **_ignored):
        return shipped_surviving(payload_, admissions)

    SQ._degree_zero_exports = _pre_c010_degree_zero
    SQ._surviving_processes = _pre_c010_surviving
    try:
        before = _observables(payload)
    finally:
        SQ._degree_zero_exports = shipped_dz
        SQ._surviving_processes = shipped_surviving

    after = _observables(payload)
    return {
        "before": {"degree_zero": before[0], "ok": before[1], "refusal_reasons": before[2]},
        "after": {"degree_zero": after[0], "ok": after[1], "refusal_reasons": after[2]},
        "moved": before != after,
        "tuple": [before[0], after[0], before[1], after[1], before[2], after[2]],
        "release": _release_observables(payload),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", required=True, help="source tree to import t2pw from")
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)

    loaded_from = _bind_source(args.src)

    legs: Dict[str, Any] = {}
    for path in _legs(ROOT):
        rel = os.path.relpath(os.path.dirname(path), ROOT).replace("\\", "/")
        try:
            payload = json.loads(open(path, encoding="utf-8").read())
            legs[rel] = _measure_leg(payload)
        except Exception as exc:  # a corrupt or unreplayable leg is a fact, not a crash
            legs[rel] = {"error": f"{type(exc).__name__}: {exc}"}

    moved = {rel: row["tuple"] for rel, row in legs.items() if row.get("moved")}
    report = {
        "probe": "c067_replay_census",
        "label": args.label,
        "src": os.path.abspath(args.src).replace("\\", "/"),
        "module": loaded_from.replace("\\", "/"),
        "leg_count": len(legs),
        "moved_count": len(moved),
        "moved": moved,
        "legs": legs,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"legs measured : {len(legs)}  from {loaded_from}")
    print(f"legs MOVED    : {len(moved)}")
    for rel in sorted(moved):
        print(f"  {rel}\n      {json.dumps(moved[rel], ensure_ascii=False)}")
    errors = {rel: row["error"] for rel, row in legs.items() if "error" in row}
    print(f"legs that raised: {len(errors)}")
    for rel, err in sorted(errors.items()):
        print(f"  {rel}: {err}")
    gained = [rel for rel, row in sorted(moved.items()) if row[1]]
    print(f"legs that GAIN a degree-zero entity (assertion 3): {gained}")
    contradictions = [
        rel for rel, row in sorted(legs.items())
        if isinstance(row.get("release"), dict) and row["release"]["contradiction"]
    ]
    print(f"legs reporting ok=True AND diagnostic_only: {len(contradictions)} {contradictions}")
    split = [
        rel for rel, row in sorted(legs.items())
        if isinstance(row.get("release"), dict)
        and row["release"]["ok"] != row["release"]["strict_gates_passed"]
    ]
    print(f"legs where ok != strict_gates_passed (expected, they are different values): {len(split)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
