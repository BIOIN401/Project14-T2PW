"""C-097 read-only probe: what the ``ref``/``id`` narrowing moves in the committed corpus.

F-131's registration records REV-089's measurement of C-089's widening as **0**.
This probe measures the reverse move -- ``bench.semantic._names`` reading
``PARTICIPANT_SCHEMA_NAME_KEYS`` instead of the full ``PARTICIPANT_NAME_KEYS``
union -- over ``git ls-files "*final_mapped.json"``, and must agree.

Two independent readouts:

1. **Mechanism (by construction).** Every dict-shaped participant row in every
   participant slot of every process bucket is resolved twice -- first-key-present
   under the UNION and under the SCHEMA keys. A row whose two answers agree cannot
   move any ``_names`` caller. ``rows_that_move`` is therefore the whole exposure,
   at every one of the reader's call sites at once.
2. **Observables.** ``_orphaned_references`` findings (acceptance priority 3) and
   ``_connected_core`` largest-core size, computed per leg under both regimes.

The base regime is emulated in-process by rebinding
``participant_schema.PARTICIPANT_SCHEMA_NAME_KEYS`` to the union -- the attribute
the tip reader actually reads, so the switch under measurement is the real one.
Nothing is written outside ``--out``; no payload is mutated.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src")]

from t2pw.bench import semantic as S  # noqa: E402
from t2pw.pipeline import participant_schema as PS  # noqa: E402


def _corpus() -> List[str]:
    listed = subprocess.run(["git", "ls-files", "*final_mapped.json"], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    return sorted(listed.stdout.split())


def _first(row: Dict[str, Any], keys) -> Any:
    for key in keys:
        candidate = row.get(key)
        if candidate:
            return candidate
    return None


def _row_census(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Rows whose first-present key differs between the two regimes."""

    union = PS.PARTICIPANT_NAME_KEYS
    schema = PS.PARTICIPANT_SCHEMA_NAME_KEYS
    seen = 0
    legacy = 0
    moved: List[Dict[str, Any]] = []
    processes = payload.get("processes")
    if not isinstance(processes, dict):
        return {"dict_rows": 0, "rows_with_legacy_key": 0, "rows_that_move": moved}
    for bucket in PS.PARTICIPANT_SLOTS:
        rows = processes.get(bucket)
        if not isinstance(rows, list):
            continue
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            for slot in PS.participant_slots(bucket):
                value = row.get(slot)
                items = value if isinstance(value, list) else [value]
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    seen += 1
                    if any(item.get(key) for key in PS.PARTICIPANT_LEGACY_NAME_KEYS):
                        legacy += 1
                    before, after = _first(item, union), _first(item, schema)
                    if before != after:
                        moved.append({"pointer": f"/processes/{bucket}/{index}/{slot}",
                                      "under_union": before, "under_schema": after,
                                      "keys_present": sorted(item)})
    return {"dict_rows": seen, "rows_with_legacy_key": legacy, "rows_that_move": moved}


def _observables(payload: Dict[str, Any]) -> Dict[str, Any]:
    core = S._connected_core(S._processes(payload), S._cofactor_names())
    return {
        "orphans": sorted(json.dumps(f, sort_keys=True)
                          for f in S._orphaned_references(payload)),
        "largest_core_size": core.get("largest_core_size"),
    }


def _instrument_control() -> Dict[str, Any]:
    """The comparator must DISCRIMINATE, or a corpus total of 0 proves nothing.

    A synthetic ``id``-only input is exactly the shape the narrowing moves, and a
    ``name``-plus-``id`` input is exactly the shape it must not. Both are checked
    against the same two readouts the corpus legs use.
    """

    schema_keys = PS.PARTICIPANT_SCHEMA_NAME_KEYS

    def _leg(row: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"entities": {"compounds": []},
                   "processes": {"reactions": [{"name": "R", "inputs": [row]}]}}
        tip = _observables(payload)
        try:
            PS.PARTICIPANT_SCHEMA_NAME_KEYS = PS.PARTICIPANT_NAME_KEYS
            base = _observables(payload)
        finally:
            PS.PARTICIPANT_SCHEMA_NAME_KEYS = schema_keys
        return {"rows_that_move": len(_row_census(payload)["rows_that_move"]),
                "orphans_base": len(base["orphans"]), "orphans_tip": len(tip["orphans"])}

    positive = _leg({"id": "CHEBI:4167"})
    negative = _leg({"name": "Glucose 6-phosphate", "id": "CHEBI:4167"})
    return {
        "positive_id_only": positive,
        "negative_name_and_id": negative,
        "discriminates": (positive["rows_that_move"] == 1
                          and positive["orphans_base"] == 1
                          and positive["orphans_tip"] == 0
                          and negative["rows_that_move"] == 0
                          and negative["orphans_base"] == negative["orphans_tip"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    args = parser.parse_args()

    control = _instrument_control()
    legs: List[Dict[str, Any]] = []
    totals = {"legs": 0, "unreadable": 0, "dict_rows": 0, "rows_with_legacy_key": 0,
              "rows_that_move": 0, "orphan_delta_legs": 0, "core_delta_legs": 0}
    schema_keys = PS.PARTICIPANT_SCHEMA_NAME_KEYS

    for relative in _corpus():
        try:
            payload = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - a corrupt artifact is a fact
            legs.append({"leg": relative, "unreadable": repr(exc)})
            totals["unreadable"] += 1
            continue
        if not isinstance(payload, dict):
            legs.append({"leg": relative, "unreadable": "not an object"})
            totals["unreadable"] += 1
            continue

        rows = _row_census(payload)
        tip = _observables(payload)
        try:
            PS.PARTICIPANT_SCHEMA_NAME_KEYS = PS.PARTICIPANT_NAME_KEYS
            base = _observables(payload)
        finally:
            PS.PARTICIPANT_SCHEMA_NAME_KEYS = schema_keys

        orphan_delta = tip["orphans"] != base["orphans"]
        core_delta = tip["largest_core_size"] != base["largest_core_size"]
        totals["legs"] += 1
        totals["dict_rows"] += rows["dict_rows"]
        totals["rows_with_legacy_key"] += rows["rows_with_legacy_key"]
        totals["rows_that_move"] += len(rows["rows_that_move"])
        totals["orphan_delta_legs"] += int(orphan_delta)
        totals["core_delta_legs"] += int(core_delta)
        legs.append({
            "leg": relative,
            "dict_rows": rows["dict_rows"],
            "rows_with_legacy_key": rows["rows_with_legacy_key"],
            "rows_that_move": rows["rows_that_move"],
            "orphans_base": len(base["orphans"]),
            "orphans_tip": len(tip["orphans"]),
            "orphan_delta": orphan_delta,
            "core_base": base["largest_core_size"],
            "core_tip": tip["largest_core_size"],
            "core_delta": core_delta,
        })

    report = {
        "probe": "c097_corpus_census",
        "union_keys": list(PS.PARTICIPANT_NAME_KEYS),
        "schema_keys": list(PS.PARTICIPANT_SCHEMA_NAME_KEYS),
        "instrument_control": control,
        "totals": totals,
        "verdict": ("no corpus number moves" if control["discriminates"]
                    and totals["rows_that_move"] == 0
                    and totals["orphan_delta_legs"] == 0
                    and totals["core_delta_legs"] == 0 else "A CORPUS NUMBER MOVES"),
        "legs": legs,
    }
    text = json.dumps(report, indent=2, sort_keys=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(json.dumps({"instrument_control": control, "totals": totals,
                      "verdict": report["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
