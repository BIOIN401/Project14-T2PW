"""ORCH-804: per-leg diff of base vs tip corpus replay.

usage: state_diff.py BASE.json TIP.json BASE_OUT_DIR TIP_OUT_DIR
Prints, per leg, (1) which replay-record keys differ and (2) every IR process /
element location whose biological-state NAME differs, with the state's location.
"""
import json
import sys
from pathlib import Path


def flat(d, p=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from flat(v, f"{p}/{k}")
    else:
        yield p, d


def assignments(ir_path):
    if not ir_path.exists():
        return None
    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    locs = {s["key"]: s.get("name") for s in ir.get("subcellular_locations") or []}
    st = {s["key"]: f'{s["name"]} [{locs.get(s.get("subcellular_location_key"), s.get("subcellular_location_key"))}]'
          for s in ir.get("biological_states") or []}
    names = {}
    for bucket, rows in (ir.get("entities") or {}).items():
        for r in rows if isinstance(rows, list) else []:
            names[r.get("key")] = r.get("name")
    out = {}

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                if k.endswith("biological_state_key"):
                    label = node.get("name") or names.get(node.get("entity_key")) or ""
                    out[f"{path}/{k} ({label})"] = st.get(v, v)
                else:
                    walk(v, f"{path}/{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}/{i}")

    for top in ("processes", "process_visualizations"):
        walk(ir.get(top), "/" + top)
    # locations are positional and gain rows when a member moves state: compare as a set
    locset = {f"{r.get('entity_type')}:{names.get(r.get('entity_key'), r.get('entity_key'))} @ {st.get(r.get('biological_state_key'))}"
              for r in ir.get("locations") or []}
    return out, locset


def main():
    base, tip = (json.loads(Path(a).read_text(encoding="utf-8")) for a in sys.argv[1:3])
    bdir, tdir = Path(sys.argv[3]), Path(sys.argv[4])
    b = {l["id"]: l for l in base["legs"]}
    n = 0
    for leg in tip["legs"]:
        lid = leg["id"]
        rec = sorted(k for k, v in flat(leg.get("replay") or {}) if dict(flat(b[lid].get("replay") or {})).get(k) != v)
        slug = lid.replace("/", "_")
        bres, tres = assignments(bdir / slug / "pwml_ir.json"), assignments(tdir / slug / "pwml_ir.json")
        moved, gone, new = {}, set(), set()
        if bres and tres:
            (ba, bl), (ta, tl) = bres, tres
            moved = {k: (ba.get(k), ta.get(k)) for k in set(ba) | set(ta) if ba.get(k) != ta.get(k)}
            gone, new = bl - tl, tl - bl
        if not rec and not moved and not gone and not new:
            continue
        n += 1
        print(f"== {lid}\n   record keys differing: {rec}")
        for k, (x, y) in sorted(moved.items()):
            print(f"   {k}: {x}  ->  {y}")
        for x in sorted(gone):
            print(f"   location removed: {x}")
        for x in sorted(new):
            print(f"   location added:   {x}")
        if bres and tres:
            lost = {x.split(" @ ")[0] for x in bl} - {x.split(" @ ")[0] for x in tl}
            print(f"   entities with no location left at tip: {sorted(lost)}")
    print(f"legs changed: {n}/{len(tip['legs'])}")


if __name__ == "__main__":
    main()
