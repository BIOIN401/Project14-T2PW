"""ORCH-749 -- what the extended interaction-type table does to the committed corpus.

The corpus is every committed ``final_mapped.json`` under ``runs/`` and ``runs_verify/``
(``git ls-files``, read-only): the payload the PWML exporter is handed. For every
interaction row it records the relationship as written, the table verdict under the
ORCH-747 table (the two nouns) and under the ORCH-749 table (this tree's
``PWML_INTERACTION_TYPES``), and -- for every leg that builds offline under the R3 test's
DB-down configuration -- the exporter's own verdict from ``build_pwml_ir``: ``exportable``,
``untyped_interaction`` or ``bare_protein_endpoint``. Legs that refuse offline are listed
with their stop; their rows still carry the table verdicts, which need no DB.

Writes a JSON record and a plain-text summary beside this script. Read-only everywhere.

Usage::

    python docs/pwml_recovery_sprint/evidence/orch749_interaction_spelling_census.py [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUT = Path(__file__).resolve().parent / "orch749_interaction_spelling_census.json"
ORCH_747_TABLE = {"activation": "Activation", "inhibition": "Inhibition"}


class _DownDb:
    last_error = "harvest_db_down"

    def available(self) -> bool:
        return False


def _committed_legs() -> List[str]:
    out = subprocess.run(["git", "ls-files", "runs", "runs_verify"], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout
    return sorted(p for p in out.split("\n") if p.endswith("/final_mapped.json"))


def _raw_type(row: Dict[str, Any]) -> str:
    # The exporter's own precedence (ir.py, interaction loop), canonicalised the same way.
    import re
    value = row.get("interaction_type") or row.get("relationship") or row.get("class") or "interaction"
    return re.sub(r"\s+", " ", str(value or "").strip())


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    logging.disable(logging.CRITICAL)
    import t2pw
    from t2pw.pwml.ir import PWML_INTERACTION_TYPES, _interaction_type_key, build_pwml_ir

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                          text=True).stdout.strip()
    legs = _committed_legs()

    rows_total = 0
    legs_with_rows = 0
    spelling: Counter = Counter()
    table_747: Counter = Counter()
    table_749: Counter = Counter()
    table_749_by_type: Counter = Counter()
    mapped_spellings: Dict[str, Dict[str, Any]] = {}
    withheld_spellings: Counter = Counter()
    exporter: Counter = Counter()
    exporter_by_spelling: Dict[str, Counter] = defaultdict(Counter)
    stops: Counter = Counter()
    per_leg: Dict[str, Any] = {}
    by_root: Dict[str, Counter] = defaultdict(Counter)

    for leg in legs:
        payload = json.loads((REPO / leg).read_text(encoding="utf-8"))
        rows = [r for r in ((payload.get("processes") or {}).get("interactions") or []) if isinstance(r, dict)]
        if not rows:
            continue
        legs_with_rows += 1
        root = leg.split("/", 1)[0]
        leg_entry: Dict[str, Any] = {"rows": len(rows), "table_749": Counter(), "exporter": None}
        for row in rows:
            rows_total += 1
            raw = _raw_type(row)
            key = _interaction_type_key(raw)
            spelling[raw] += 1
            v747 = ORCH_747_TABLE.get(raw.casefold())
            v749 = PWML_INTERACTION_TYPES.get(key)
            table_747[v747 or "withheld"] += 1
            table_749[v749 or "withheld"] += 1
            by_root[root][v749 or "withheld"] += 1
            leg_entry["table_749"][v749 or "withheld"] += 1
            if v749:
                table_749_by_type[v749] += 1
                entry = mapped_spellings.setdefault(raw, {"type": v749, "key": key, "rows": 0})
                entry["rows"] += 1
            else:
                withheld_spellings[raw] += 1

        try:
            ir, _report = build_pwml_ir(payload, db_resolver=_DownDb(), strict_db=True, name_index=None)
        except Exception as exc:  # noqa: BLE001 -- recorded as the leg's stop
            stop = f"{type(exc).__name__}"
            stops[stop] += 1
            leg_entry["exporter"] = {"stop": stop}
        else:
            outcomes: Counter = Counter()
            for row, ir_row in zip(rows, ir["processes"]["interactions"]):
                verdict = (ir_row.get("withheld") or {}).get("reason", "exportable")
                outcomes[verdict] += 1
                exporter[verdict] += 1
                exporter_by_spelling[_raw_type(row)][verdict] += 1
            leg_entry["exporter"] = dict(outcomes)
        leg_entry["table_749"] = dict(leg_entry["table_749"])
        per_leg[leg] = leg_entry

    built = sum(1 for e in per_leg.values() if "stop" not in (e["exporter"] or {}))
    rows_built = sum(exporter.values())
    record = {
        "task": "ORCH-749",
        "head_sha": head,
        "t2pw_file": str(Path(t2pw.__file__).resolve()),
        "corpus": "every committed final_mapped.json under runs/ and runs_verify/ (git ls-files) with at least one interaction row",
        "exporter_configuration": "build_pwml_ir(payload, db_resolver=_DownDb(), strict_db=True, name_index=None) -- the R3 test's DB-down configuration",
        "legs_committed_with_final_mapped": len(legs),
        "legs_with_interactions": legs_with_rows,
        "interaction_rows": rows_total,
        "table_verdicts": {
            "orch_747_nouns_only": dict(table_747),
            "orch_749_this_tree": dict(table_749),
            "orch_749_by_pathbank_type": dict(table_749_by_type),
            "by_run_root": {root: dict(c) for root, c in by_root.items()},
        },
        "mapped_spellings": dict(sorted(mapped_spellings.items(), key=lambda kv: (-kv[1]["rows"], kv[0]))),
        "withheld_spellings": dict(withheld_spellings.most_common()),
        "exporter_verdicts_on_legs_that_build_offline": {
            "legs_built": built,
            "legs_stopped": dict(stops),
            "rows_covered": rows_built,
            "outcomes": dict(exporter),
            "by_spelling": {k: dict(v) for k, v in sorted(exporter_by_spelling.items(), key=lambda kv: -sum(kv[1].values()))},
        },
        "all_spellings": dict(spelling.most_common()),
        "per_leg": per_leg,
    }
    out = Path(args.out)
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")

    lines = [
        f"ORCH-749 interaction spelling census  (HEAD {head[:8]}, t2pw from {record['t2pw_file']})",
        f"corpus: {legs_with_rows} committed final_mapped.json legs with interactions (of {len(legs)}), {rows_total} interaction rows",
        "",
        f"table verdict, ORCH-747 nouns only : {dict(table_747)}",
        f"table verdict, ORCH-749 this tree  : {dict(table_749)}   by type {dict(table_749_by_type)}",
        f"  by run root                      : {{{', '.join(f'{r}: {dict(c)}' for r, c in by_root.items())}}}",
        "",
        "mapped spellings (rows, type):",
    ]
    for word, entry in record["mapped_spellings"].items():
        lines.append(f"  {entry['rows']:4d}  {word!r:28s} -> {entry['type']}")
    lines += ["", "still withheld (rows, word as written):"]
    for word, n in withheld_spellings.most_common():
        lines.append(f"  {n:4d}  {word!r}")
    lines += [
        "",
        f"exporter verdicts on the {built} legs that build offline ({rows_built} rows; stops: {dict(stops)}):",
        f"  {dict(exporter)}",
    ]
    text = "\n".join(lines) + "\n"
    out.with_suffix(".txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"written: {out} and {out.with_suffix('.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
