"""`F-192` census — read-only, over committed and preserved run artifacts.

Answers, per leg, from artifacts only. Nothing is re-run, no LLM is called, no
network is touched, no run directory is written.

1. Did ``ensure_autostates`` create a state (``n_autostate_created``)?
2. Did a biological state later disappear (``removed_biological_states``)?
3. Did the leg end with **zero** states, and did the required-field gate fail on
   ``no_biological_states`` / ``visible_entity_missing_location_state``?
4. Would the leg otherwise have had a viable reaction core?
5. Is the ordering always autostate -> audit repair -> missing state -> sweep?
6. Which element-location entity types are affected?

**Populations are never summed.** Each run family is reported separately and the
per-leg rows carry their family, so no denominator is invented.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parents[3]
FAMILIES = ("runs", "runs_verify", "runs_smoke", "runs_validation")

_AUTOSTATE_NAME = "__auto_state__"


def _load(path: Path) -> Optional[Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:  # noqa: BLE001 - a missing or truncated artifact is data
        return None


def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""


def _autostate_counters(leg: Path) -> Dict[str, Optional[int]]:
    """``n_autostate_created`` as recorded by any report that carries it.

    The normalization summary is copied into several reports; the first one that
    carries the key is taken, and which file it came from is recorded so the
    number is traceable.
    """
    for name in (
        "contract_reports.json",
        "initial_stage3_gate_report.json",
        "final_stage3_gate_report.json",
        "gate_fail_report.json",
    ):
        path = leg / name
        if not path.exists():
            continue
        blob = _text(path)
        created = re.search(r'"n_autostate_created"\s*:\s*(\d+)', blob)
        assigned = re.search(r'"n_entities_assigned_to_autostate"\s*:\s*(\d+)', blob)
        if created or assigned:
            return {
                "source": name,
                "created": int(created.group(1)) if created else None,
                "assigned": int(assigned.group(1)) if assigned else None,
            }
    return {"source": None, "created": None, "assigned": None}


def _gate_errors(leg: Path) -> Dict[str, Any]:
    report = _load(leg / "pwml_required_field_gate_report.json")
    if not isinstance(report, dict):
        return {"present": False, "total": 0, "no_states": 0, "missing_location_state": 0, "buckets": {}}
    errors = [row for row in (report.get("errors") or []) if isinstance(row, dict)]
    buckets: Counter = Counter()
    for row in errors:
        if row.get("code") != "visible_entity_missing_location_state":
            continue
        pointer = str(row.get("pointer") or "")
        match = re.search(r"/element_locations/([a-z_]+)/", pointer)
        buckets[match.group(1) if match else "unknown"] += 1
    return {
        "present": True,
        "ok": bool(report.get("ok")),
        "total": len(errors),
        "no_states": sum(1 for row in errors if row.get("code") == "no_biological_states"),
        "missing_location_state": sum(
            1 for row in errors if row.get("code") == "visible_entity_missing_location_state"
        ),
        "buckets": dict(buckets),
        "other_codes": sorted(
            {
                str(row.get("code"))
                for row in errors
                if row.get("code")
                not in {"no_biological_states", "visible_entity_missing_location_state"}
            }
        )[:8],
    }


def _location_lineage(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Which pass last wrote each element-location row, and did it carry a state?"""
    element_locations = payload.get("element_locations")
    if not isinstance(element_locations, dict):
        return {"buckets": {}, "stages": {}, "rows_without_state": 0, "rows": 0}
    stages: Counter = Counter()
    buckets: Counter = Counter()
    without = 0
    total = 0
    for bucket, rows in element_locations.items():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            total += 1
            state = str(row.get("biological_state") or "").strip()
            if not state:
                without += 1
                buckets[bucket] += 1
            for entry in row.get("provenance_lineage") or []:
                if isinstance(entry, dict) and entry.get("stage"):
                    stages[str(entry["stage"])] += 1
    return {
        "buckets": dict(buckets),
        "stages": dict(stages),
        "rows_without_state": without,
        "rows": total,
    }


def _result_facts(leg: Path) -> Dict[str, Any]:
    text = _text(leg / "RESULT.txt")
    facts: Dict[str, Any] = {"status": "", "stage": "", "issue_codes": ""}
    for key, pattern in (
        ("status", r"^status\s*:\s*(.+)$"),
        ("stage", r"^stage\s*:\s*(.+)$"),
        ("issue_codes", r"^issue codes\s*:\s*(.+)$"),
    ):
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            facts[key] = match.group(1).strip()
    for key in ("reactions", "blocking_issues", "gate_errors", "proteins", "compounds"):
        match = re.search(rf"^\s*{key}\s*=\s*(\d+)\s*$", text, re.MULTILINE)
        facts[key] = int(match.group(1)) if match else None
    return facts


def scan_leg(leg: Path) -> Optional[Dict[str, Any]]:
    removed = _load(leg / "removed_entity_report.json")
    final_mapped = _load(leg / "final_mapped.json")
    stage1 = _load(leg / "stage1_payload.json")
    if removed is None and final_mapped is None:
        return None

    removed_states: List[Dict[str, Any]] = []
    if isinstance(removed, dict):
        removed_states = [
            row for row in (removed.get("removed_biological_states") or []) if isinstance(row, dict)
        ]

    final_states = None
    lineage: Dict[str, Any] = {}
    if isinstance(final_mapped, dict):
        final_states = len(final_mapped.get("biological_states") or [])
        lineage = _location_lineage(final_mapped)

    stage1_states = None
    if isinstance(stage1, dict):
        stage1_states = len(stage1.get("biological_states") or [])

    parts = leg.parts
    pwml = sorted(path.name for path in leg.glob("*.pwml"))

    return {
        "family": parts[-5] if len(parts) >= 5 else "",
        "run": parts[-4] if len(parts) >= 4 else "",
        "paper": parts[-2],
        "mode": parts[-1],
        "autostate": _autostate_counters(leg),
        "removed_state_names": [str(row.get("name")) for row in removed_states],
        "removed_state_reasons": sorted({str(row.get("reason")) for row in removed_states}),
        "removed_autostate": any(
            str(row.get("name") or "").strip().casefold() == _AUTOSTATE_NAME for row in removed_states
        ),
        "removed_count": len(removed_states),
        "stage1_biological_states": stage1_states,
        "final_biological_states": final_states,
        "location_lineage": lineage,
        "gate": _gate_errors(leg),
        "result": _result_facts(leg),
        "pwml_files": pwml,
        "path": str(leg.relative_to(_REPO)).replace("\\", "/"),
    }


def main() -> int:
    legs: List[Dict[str, Any]] = []
    for family in FAMILIES:
        root = _REPO / family
        if not root.is_dir():
            continue
        # Depth-agnostic on purpose. A fixed ``*/papers/*/*`` glob silently
        # MISSED ``runs_validation/c120/<stamp>/papers/…`` -- one extra directory
        # level -- and with it the only leg in the whole corpus that exhibits the
        # defect, turning the census into a confident "never happens". Match on
        # the shape (``…/papers/<paper>/<mode>``) rather than on the depth.
        for leg in sorted(root.rglob("papers/*/*")):
            if not leg.is_dir() or leg.name not in {"strict", "research"}:
                continue
            row = scan_leg(leg)
            if row is not None:
                legs.append(row)

    # ── the four populations, kept SEPARATE ───────────────────────────────────
    blocked = [
        row
        for row in legs
        if row["gate"].get("present") and row["gate"].get("no_states", 0) > 0
    ]
    removed_any = [row for row in legs if row["removed_count"] > 0]
    removed_auto = [row for row in legs if row["removed_autostate"]]
    zero_final = [row for row in legs if row["final_biological_states"] == 0]

    def by_family(rows: List[Dict[str, Any]]) -> Dict[str, int]:
        counter: Counter = Counter(row["family"] for row in rows)
        return dict(sorted(counter.items()))

    # Q4 -- is the ordering always the same?
    ordering: Counter = Counter()
    for row in blocked:
        created = row["autostate"].get("created")
        ordering[
            (
                f"created={created}",
                f"removed_autostate={row['removed_autostate']}",
                f"stage1_states={row['stage1_biological_states']}",
                f"final_states={row['final_biological_states']}",
            )
        ] += 1

    # Q5 -- which entity types.
    buckets: Counter = Counter()
    for row in blocked:
        for bucket, count in (row["gate"].get("buckets") or {}).items():
            buckets[bucket] += count

    # Q3 -- viable reaction core among the blocked.
    viable = [
        row
        for row in blocked
        if (row["result"].get("reactions") or 0) >= 2
        and (row["result"].get("gate_errors") or 0) == 0
    ]

    # Cross-tab: does a removal always block?
    harmless = [row for row in removed_auto if row not in blocked]

    summary = {
        "legs_scanned": len(legs),
        "legs_by_family": by_family(legs),
        "q1_autostate_removed": {
            "legs_with_any_state_removed": len(removed_any),
            "legs_with___auto_state___removed": len(removed_auto),
            "by_family": by_family(removed_auto),
            "removal_reasons_seen": sorted(
                {reason for row in removed_any for reason in row["removed_state_reasons"]}
            ),
        },
        "q2_blocked": {
            "legs_failing_no_biological_states": len(blocked),
            "by_family": by_family(blocked),
            "distinct_papers": len({row["paper"] for row in blocked}),
            "papers": sorted({row["paper"] for row in blocked}),
        },
        "q3_viable_core_among_blocked": {
            "count": len(viable),
            "legs": [
                {
                    "paper": row["paper"],
                    "mode": row["mode"],
                    "run": row["run"],
                    "reactions": row["result"].get("reactions"),
                    "gate_errors": row["result"].get("gate_errors"),
                    "blocking_issues": row["result"].get("blocking_issues"),
                    "status": row["result"].get("status"),
                }
                for row in viable
            ],
        },
        "q4_ordering_signature": {" | ".join(key): value for key, value in ordering.most_common()},
        "q5_entity_types": dict(buckets.most_common()),
        "removal_that_did_NOT_block": {
            "count": len(harmless),
            "note": "auto-state removed but a real state remained -- the defect is masked",
            "legs": [
                {
                    "paper": row["paper"],
                    "mode": row["mode"],
                    "final_states": row["final_biological_states"],
                    "stage1_states": row["stage1_biological_states"],
                }
                for row in harmless
            ][:40],
        },
        "zero_final_states": {
            "count": len(zero_final),
            "of_which_gate_report_present": sum(
                1 for row in zero_final if row["gate"].get("present")
            ),
        },
        # EXPOSURE, not occurrence. The defect needs Stage 1 to emit no state of
        # its own; everything else is masked. This is the population a charter
        # would actually be sized against.
        "exposure_stage1_zero_states": _exposure(legs),
    }

    sys.stdout.write(
        json.dumps({"summary": summary, "blocked_legs": blocked}, indent=1, default=str) + "\n"
    )
    return 0


def _exposure(legs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Legs whose Stage 1 emitted no biological state -- the at-risk population."""
    known = [row for row in legs if row["stage1_biological_states"] is not None]
    at_risk = [row for row in known if row["stage1_biological_states"] == 0]
    survived = [row for row in at_risk if (row["final_biological_states"] or 0) > 0]
    died = [row for row in at_risk if (row["final_biological_states"] or 0) == 0]
    lineage: Counter = Counter()
    for row in at_risk:
        stages = (row["location_lineage"] or {}).get("stages") or {}
        lineage["audit_repair" if "audit_repair" in stages else "no_audit_repair_lineage"] += 1
    survivor_lineage: Counter = Counter()
    for row in survived:
        stages = (row["location_lineage"] or {}).get("stages") or {}
        survivor_lineage[
            "audit_repair" if "audit_repair" in stages else "no_audit_repair_lineage"
        ] += 1
    return {
        "legs_with_stage1_payload_readable": len(known),
        "legs_with_stage1_zero_states": len(at_risk),
        "of_those_survived_with_a_state": len(survived),
        "of_those_ended_with_zero": len(died),
        "at_risk_location_lineage": dict(lineage),
        "survivor_location_lineage": dict(survivor_lineage),
        "survivors_final_state_counts": dict(
            Counter(row["final_biological_states"] for row in survived).most_common()
        ),
        "at_risk_legs": [
            {
                "family": row["family"],
                "run": row["run"],
                "paper": row["paper"],
                "mode": row["mode"],
                "final_states": row["final_biological_states"],
                "autostate_removed": row["removed_autostate"],
                "audit_repair": "audit_repair"
                in ((row["location_lineage"] or {}).get("stages") or {}),
                "status": row["result"].get("status"),
                "reactions": row["result"].get("reactions"),
            }
            for row in at_risk
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
