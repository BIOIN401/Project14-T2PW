"""ORCH-804: for one replayed leg, every empty-state reaction/interaction with the
element_locations states of each participant (why the member rule did or did not fire).

usage: explain_leg.py <leg out dir>   (reads final.canonical.json)
"""
import json
import sys
from pathlib import Path

d = json.loads((Path(sys.argv[1]) / "final.canonical.json").read_text(encoding="utf-8"))
locs = {}
for bucket, rows in (d.get("element_locations") or {}).items():
    for r in rows or []:
        if isinstance(r, dict):
            name = r.get(bucket.replace("_locations", "")) or r.get("entity")
            locs.setdefault(str(name).lower(), set()).add(r.get("biological_state") or "<empty>")
for kind in ("reactions", "interactions"):
    for r in (d.get("processes") or {}).get(kind) or []:
        if str(r.get("biological_state") or "").strip():
            continue
        names = [p.get("name") if isinstance(p, dict) else p for k in ("inputs", "outputs") for p in r.get(k) or []]
        names += [a.get("name") or a.get("protein") if isinstance(a, dict) else a for a in r.get("enzymes") or []]
        names += [r.get(k) for k in ("entity_1", "entity_2", "source", "target") if r.get(k)]
        print(f"{kind}: {r.get('name')}")
        for n in names:
            print(f"    {n}: {sorted(locs.get(str(n).lower(), {'<no element_location>'}))}")
