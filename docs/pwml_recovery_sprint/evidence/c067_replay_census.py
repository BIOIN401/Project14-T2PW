"""C-067 replay-cohort census: what does the degree-zero change move on real legs?

Chunk E (``tests/test_strict_quarantine_real_artifact_replay.py``) pins
``degree_zero_exports``, ``ok`` and ``refusal_reasons`` per committed leg, and this
branch may not run a chunk gate. So the same observables are measured directly,
once per leg, at whichever source tree ``--src`` names -- run it against the base
tree and against the branch tree and diff the two reports.

``--src`` is prepended to ``sys.path`` BEFORE ``t2pw`` is imported, which is the
whole point: the base arm runs the base bytes of ``strict_quarantine.py``, not a
local reimplementation of what they used to do.

Observables match ``test_strict_quarantine_real_artifact_replay._p01_observables``
(``strict_db=True``, ``pathway_context=None``) so a delta here is a delta there.

**What this measures and what it does not.** These are the committed
``final_mapped.json`` files -- the REPLAY cohort. They are not the payload the
production quarantine judged: ``FINDINGS.md:1580`` records that the quarantine
input payload is never persisted, and the run report's ``admitted_payload_hash``
does not match any committed file. A leg that does not move here has not been
shown to be unmoved in production, and a leg that does move here has not been
shown to move in production.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from typing import Any, Dict, List, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))


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


def _observe(payload: Dict[str, Any]) -> Dict[str, Any]:
    from t2pw.pipeline.strict_quarantine import quarantine_and_close

    result = quarantine_and_close(deepcopy(payload), strict_db=True, pathway_context=None)
    invariants = result.quarantine_report.get("strict_invariants") or {}
    release = result.quarantine_report.get("release") or {}
    return {
        "degree_zero_exports": sorted(
            str(row.get("name")) for row in (invariants.get("degree_zero_exports") or [])
        ),
        "closure_converged": bool(invariants.get("closure_converged")),
        "ok": bool(result.ok),
        "refusal_reasons": list(result.refusal_reasons),
        "review_reasons": list(getattr(result, "review_reasons", []) or []),
        "release_status": release.get("status"),
        "strict_acceptance_eligible": release.get("strict_acceptance_eligible"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", required=True, help="source tree to import t2pw from")
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)

    src = os.path.abspath(args.src)
    sys.path.insert(0, src)
    import t2pw.pipeline.strict_quarantine as SQ

    loaded_from = os.path.abspath(SQ.__file__)
    if not loaded_from.startswith(src):
        raise SystemExit(f"refusing to measure: t2pw loaded from {loaded_from}, not {src}")

    legs: Dict[str, Any] = {}
    for path in _legs(ROOT):
        rel = os.path.relpath(os.path.dirname(path), ROOT).replace("\\", "/")
        try:
            payload = json.loads(open(path, encoding="utf-8").read())
        except Exception as exc:  # a corrupt committed leg is a fact, not a crash
            legs[rel] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        try:
            legs[rel] = _observe(payload)
        except Exception as exc:
            legs[rel] = {"error": f"{type(exc).__name__}: {exc}"}

    report = {
        "probe": "c067_replay_census",
        "label": args.label,
        "src": src.replace("\\", "/"),
        "module": loaded_from.replace("\\", "/"),
        "leg_count": len(legs),
        "legs": legs,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(f"legs measured: {len(legs)}  from {loaded_from}")
    flagged = {rel: row.get("degree_zero_exports") for rel, row in legs.items() if row.get("degree_zero_exports")}
    print(f"legs with a non-empty degree_zero_exports: {len(flagged)}")
    for rel, names in sorted(flagged.items()):
        print(f"  {rel}: {names}")
    errors = {rel: row["error"] for rel, row in legs.items() if "error" in row}
    print(f"legs that raised: {len(errors)}")
    for rel, err in sorted(errors.items()):
        print(f"  {rel}: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
