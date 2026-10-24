"""ORCH-769c: slot-by-slot and field-by-field difference of two ``golden_capture.py`` outputs.

Usage: python golden_delta.py <base.json> <tip.json> <out.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

base, tip = (json.loads(Path(p).read_text(encoding="utf-8")) for p in sys.argv[1:3])
legs = {}
for leg in sorted(base["tuples"]):
    b, t = base["tuples"][leg], tip["tuples"][leg]
    bo, to = base["observables"][leg], tip["observables"][leg]
    fields = sorted(k for k in set(bo) | set(to) if bo.get(k) != to.get(k))
    row_keys = sorted(k for k in set(bo["row"]) | set(to["row"]) if bo["row"].get(k) != to["row"].get(k))
    legs[leg] = {
        "slot0_moved": b[0] != t[0], "slot1_moved": b[1] != t[1], "slot2_moved": b[2] != t[2],
        "digest": [b[2], t[2]],
        "observable_fields_differing": fields,
        "row_keys_differing": row_keys,
        "row_keys_added": sorted(set(to["row"]) - set(bo["row"])),
        "row_keys_dropped": sorted(set(bo["row"]) - set(to["row"])),
        "counts": {"base": bo["row"].get("counts"), "tip": to["row"].get("counts")},
    }
doc = {"base_tree": base["tree"], "tip_tree": tip["tree"], "legs": legs,
       "summary": {s: sum(1 for v in legs.values() if v[f"{s}_moved"]) for s in ("slot0", "slot1", "slot2")},
       "only_counts_moved": all(v["row_keys_differing"] in ([], ["counts"]) and v["observable_fields_differing"] in ([], ["row"])
                                for v in legs.values())}
Path(sys.argv[3]).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(doc["summary"]), "only_counts_moved:", doc["only_counts_moved"])
for leg, v in legs.items():
    print(f"{leg:24s} slot2 {v['digest'][0][:8]} -> {v['digest'][1][:8]}  fields {v['observable_fields_differing']} "
          f"row {v['row_keys_differing']}  counts {v['counts']['base']} -> {v['counts']['tip']}")
