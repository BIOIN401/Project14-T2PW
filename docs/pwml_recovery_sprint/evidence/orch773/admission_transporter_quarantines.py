"""ORCH-773 round 1: price extending D-114 AMENDMENT 1 s2 to transports quarantined AT ADMISSION.

Usage: python admission_transporter_quarantines.py <manifest.json> <primary checkout> <tip corpus.json>

The per-row pass keeps a transport and drops its failing transporter. Admission (``_actor_verdict``) still
quarantines a whole transport when its transporter fails (pre-existing, ORCH-769b and earlier). Reads every
corpus leg's OWN recorded ``quarantine_report.json`` (the live run's admission decisions; a canonical replay
re-quarantines an already reduced payload and sees none of them) and lists each quarantined transport whose
failure is its transporter: a reason ending ``_in_transporters``, or a representability reason whose
``essential_participant`` is one of the original row's transporters. Read-only; stdlib only.
"""

import json
import sys
from pathlib import Path


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def actor_names(row):
    out = []
    for a in (row or {}).get("transporters") or []:
        if isinstance(a, str):
            out.append(a)
        elif isinstance(a, dict):
            out.append(str(a.get("entity") or a.get("name") or a.get("protein") or ""))
    return {n.strip().casefold() for n in out if n.strip()}


def main():
    manifest, primary, tip = load(sys.argv[1]), Path(sys.argv[2]), load(sys.argv[3])
    passing = {r["id"] for r in tip["legs"] if (r.get("replay") or {}).get("pwml_written")}
    legs = rows = 0
    for leg in manifest["legs"]:
        if leg.get("mode") not in (None, "strict") or not leg.get("checkpoint"):
            continue
        meta = leg["inputs"].get("quarantine_report.json") or {}
        if not meta.get("path"):
            continue
        q = load(primary / meta["path"])
        hits = []
        for row in q.get("quarantined") or []:
            if row.get("process_kind") != "transport":
                continue
            reason = str(row.get("reason") or "")
            participant = str(row.get("essential_participant") or "").strip().casefold()
            if reason.endswith("_in_transporters") or (participant and participant in actor_names(row.get("original_process"))):
                hits.append((row.get("name"), reason, row.get("essential_participant")))
        if hits:
            legs += 1
            rows += len(hits)
            print(f"{leg['id']}  (PWML at tip: {leg['id'] in passing})")
            for h in hits:
                print(f"   {h}")
    print(f"\n{rows} transports quarantined at admission for a transporter, in {legs} legs")


if __name__ == "__main__":
    main()
