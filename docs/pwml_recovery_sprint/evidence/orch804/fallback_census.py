"""ORCH-804: per replayed leg, empty reaction/interaction state references, and whether
__auto_state__ exists (if not, the unchanged first-state fallback applies).

usage: fallback_census.py REPLAY_OUT_DIR   (reads <leg>/final.canonical.json)
"""
import json
import sys
from pathlib import Path

for leg in sorted(Path(sys.argv[1]).iterdir()):
    f = leg / "final.canonical.json"
    if not f.exists():
        continue
    d = json.loads(f.read_text(encoding="utf-8"))
    states = [s.get("name") for s in d.get("biological_states") or [] if isinstance(s, dict)]
    procs = d.get("processes") or {}
    empty = sum(1 for k in ("reactions", "interactions") for r in procs.get(k) or []
                if isinstance(r, dict) and not str(r.get("biological_state") or "").strip())
    if empty:
        print(f"{leg.name}: empty_refs={empty} states={len(states)} auto_state={'__auto_state__' in states}")
