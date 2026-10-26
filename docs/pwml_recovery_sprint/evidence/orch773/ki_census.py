"""ORCH-773 phase 2: the known-issue label of every reason a replayed corpus carries, ORCH-773's rows apart.

Walks ``<out>/<leg>/outputs/{quarantine_report,removed_entity_report}.json`` of an ORCH-768 corpus run and
labels each recorded reason with the measured tree's ``reaction_evidence.known_issue_id``. A row is
ORCH-773's when its ``detail.omission`` is set (per-row pass); the rest are counted apart.

    python ki_census.py --src <tree>/src --out <corpus run --out dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.path.insert(0, str(Path(args.src).resolve()))
    from t2pw.app import reaction_evidence as E

    print(f"reaction_evidence: {E.__file__}")
    mine, other = Counter(), Counter()
    for leg in sorted(p for p in Path(args.out).iterdir() if p.is_dir()):
        for name, key in (("quarantine_report.json", "quarantined"), ("removed_entity_report.json", "removed_entities")):
            path = leg / "outputs" / name
            if not path.is_file():
                continue
            for row in json.loads(path.read_text(encoding="utf-8")).get(key) or []:
                detail = row.get("detail") or {}
                reason = str(row.get("reason") or "")
                ours = bool(detail.get("omission"))
                (mine if ours else other)[(name, reason[:110], E.known_issue_id(reason))] += 1
    for title, tally in (("ORCH-773 rows (detail.omission set)", mine), ("other rows", other)):
        print(f"\n== {title}: {sum(tally.values())} rows, {len(tally)} distinct ==")
        for (name, reason, ki), n in sorted(tally.items(), key=lambda kv: (kv[0][2] != "unclassified", -kv[1], kv[0])):
            print(f"  {n:4d}  {ki:12s}  {name:28s}  {reason}")
    unlabelled = sum(n for (_, _, ki), n in mine.items() if ki == "unclassified")
    print(f"\nORCH-773 rows labelled unclassified: {unlabelled}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
