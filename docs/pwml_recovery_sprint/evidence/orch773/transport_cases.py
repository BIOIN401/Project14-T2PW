"""ORCH-773 round 1: where each transport whose transporter failed lands (D-114 AMENDMENT 1 s2).

Usage: python transport_cases.py <corpus out-dir> <leg slug> [<leg slug> ...]

For each leg: every transport the per-row pass touched -- a ``removed_entities`` row of bucket ``transporters``
(transporter dropped, transport kept) or a ``quarantined[]`` transport carrying ``detail.omission`` (omitted
whole) -- and, for a kept one, whether the replay's canonical payload and the written PWML's IR still carry it
and which transporters it keeps. Read-only; stdlib only.
"""

import json
import sys
from pathlib import Path


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    out = Path(sys.argv[1])
    for slug in sys.argv[2:]:
        work = out / slug
        q = load(work / "outputs" / "quarantine_report.json")
        rem = load(work / "outputs" / "removed_entity_report.json")
        canonical = load(work / "canonical.json") if (work / "canonical.json").exists() else {}
        ir = load(work / "pwml_ir.json") if (work / "pwml_ir.json").exists() else {}
        kept = {t.get("name"): t for t in (canonical.get("processes") or {}).get("transports") or []}
        ir_names = {t.get("name") for t in (ir.get("processes") or {}).get("transports") or [] if isinstance(t, dict)}
        print(f"## {slug}  pwml={'yes' if ir else 'no'}  release={(q.get('release') or {}).get('status')}")
        for row in rem.get("removed_entities") or []:
            if row.get("bucket") != "transporters":
                continue
            for entry in row["detail"]["actor_removed_from"]:
                name = entry["process_name"]
                t = kept.get(name)
                print(f"   KEPT, transporter dropped: {name!r} lost {row['name']!r} ({row['reason']}); "
                      f"in canonical={t is not None} transporters left={[a.get('entity') or a.get('protein') for a in (t or {}).get('transporters') or []]} "
                      f"in PWML IR={name in ir_names}")
        for row in q.get("quarantined") or []:
            if row.get("process_kind") == "transport" and "omission" in (row.get("detail") or {}):
                print(f"   OMITTED WHOLE: {row['name']!r} ({row['reason']})")


if __name__ == "__main__":
    main()
