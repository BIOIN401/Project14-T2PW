"""ORCH-804: per leg, entities drawn in MORE THAN ONE biological state (a duplicate drawn
location), base vs tip replay IRs.

usage: dup_count.py BASE_OUT_DIR TIP_OUT_DIR [leg_slug ...]   (default: every leg dir present in both)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def dups(ir_path):
    if not ir_path.exists():
        return None
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    names = {}
    for rows in (ir.get("entities") or {}).values():
        for r in rows if isinstance(rows, list) else []:
            names[r.get("key")] = r.get("name")
    states = {s["key"]: s["name"] for s in ir.get("biological_states") or []}
    by_entity = defaultdict(set)
    for r in ir.get("locations") or []:
        by_entity[(r.get("entity_type"), r.get("entity_key"))].add(r.get("biological_state_key"))
    return {f"{t}:{names.get(k, k)}": sorted(states.get(s, s) for s in v) for (t, k), v in by_entity.items() if len(v) > 1}


bdir, tdir = Path(sys.argv[1]), Path(sys.argv[2])
slugs = sys.argv[3:] or sorted(p.name for p in tdir.iterdir() if (bdir / p.name).is_dir())
tot_new = 0
for slug in slugs:
    b, t = dups(bdir / slug / "pwml_ir.json"), dups(tdir / slug / "pwml_ir.json")
    if b is None or t is None:
        continue
    new = sorted(set(t) - set(b))
    gone = sorted(set(b) - set(t))
    tot_new += len(new)
    if new or gone or len(b) != len(t):
        print(f"{slug}: multi-state entities base={len(b)} tip={len(t)} new={len(new)} removed={len(gone)}")
        for e in new:
            print(f"   new dup: {e} {t[e]}")
        for e in gone:
            print(f"   dup gone: {e} {b[e]}")
print(f"total new duplicate entities: {tot_new}")
