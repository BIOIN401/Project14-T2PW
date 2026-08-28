"""C-102 / D-072: the G9 behavioural proof, and the Priority-4/5 baseline reading.

Runs the PUBLIC acceptance surface -- ``score_run(...).to_dict()`` -- over a
committed run directory and asks it one question about a real leg:

    On ``PMC12782028/strict``, whose coverage block reads
    ``requested_core_coverage_below_minimum:0.222<0.500``, does the instrument
    report a requested-core denominator that leaves out the four identifiers
    THIS SAME CASE forbids exporting -- ``LIPA``, ``LBR``, ``SREBF1``,
    ``SREBF2``?

It is a question about the VALUES in the report, not about whether a symbol
exists. On the base tree the instrument's only answer is 27 drawn terms, four of
which it simultaneously forbids, so the probe exits 1 having found no permitted
denominator at all. At the tip it finds 23 beside the preserved 27.

The base leg is run with ``PYTHONPATH`` pointing at a verified base tree; the
run artifacts are read from the repo either way, since no artifact changes.

Usage::

    <python> c102_g9_denominator_proof.py <repo-root> [run-dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
RUN = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "runs_verify" / "2026-08-24_1428"

from t2pw.bench.acceptance import score_run  # noqa: E402

PAPER, MODE = "PMC12782028", "strict"
#: The four gold-forbidden identifiers the same draw supplied as requested core.
#: They are the exact four Priority-1 survivors on this paper -- the sharpest
#: instance of F-132, and the reason this leg is the proof target.
FORBIDDEN = ("LIPA", "LBR", "SREBF1", "SREBF2")

report = score_run(RUN).to_dict()

leg = None
for paper in report["papers"]:
    if paper["paper_id"] == PAPER:
        leg = paper["legs"].get(MODE)
print(f"run dir      : {RUN}")
print(f"leg          : {PAPER}/{MODE}   found={leg is not None}")
if leg is None:
    print("FAIL: the run directory does not carry that leg")
    raise SystemExit(2)

frozen = (leg.get("release_status") or {}).get("reasons")
print(f"frozen reason: {frozen}")

# Every requested-core denominator the report states anywhere on this leg. A
# value hunt, not a symbol hunt: the base tree states none, the tip states two.
found: list[int] = []
names: list[str] = []


def walk(node) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key.endswith("_denominator") and isinstance(value, int):
                found.append(value)
            if key == "excluded_terms" and isinstance(value, list):
                names.extend(str(e.get("term")) for e in value if isinstance(e, dict))
            walk(value)
    elif isinstance(node, list):
        for item in node:
            walk(item)


walk(leg)
print(f"requested-core denominators the instrument reports for this leg: {sorted(set(found))}")
print(f"terms it names as withheld-but-recorded                        : {names}")

for entry in report["acceptance_priorities"]:
    if entry["rank"] in (4, 5):
        recon = entry.get("requested_core_coverage")
        print(
            f"priority {entry['rank']} : ok={entry['ok']}  observed={entry['observed']!r}"
            f"  reconciliation={'present' if recon else 'ABSENT'}"
        )
        if recon:
            print(
                f"             legs_with_forbidden_terms={recon['legs_with_forbidden_terms']}"
                f"/{recon['legs_with_coverage']}"
                f"  terms_withheld={recon['forbidden_terms_excluded']}"
                f"  cleared={recon['legs_cleared_by_reconciliation']}"
                f"  still_below={recon['legs_still_below_minimum']}"
            )

problems: list[str] = []
if 27 not in found:
    problems.append("the RAW denominator of 27 drawn terms is not preserved in the report")
if 23 not in found:
    problems.append(
        "the instrument reports NO requested-core denominator that excludes this case's own "
        "forbidden identifiers: every coverage figure it states is computed over a term list "
        "that includes " + ", ".join(FORBIDDEN)
    )
missing = [f for f in FORBIDDEN if f not in names]
if missing:
    problems.append(f"withheld terms are not named in the record (guard rail 3): {missing}")

print()
if problems:
    print("PRE-CHANGE BEHAVIOUR -- the denominator is contradictory and unreconciled:")
    for problem in problems:
        print(f"  - {problem}")
    raise SystemExit(1)
print("POST-CHANGE BEHAVIOUR: raw 27 preserved, accepted 23 reported beside it,")
print(f"and all four withheld terms still named: {names}")
print(json.dumps({"raw_denominator": 27, "accepted_denominator": 23}, sort_keys=True))
