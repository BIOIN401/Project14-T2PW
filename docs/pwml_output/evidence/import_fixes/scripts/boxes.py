"""Real compartment-box duplicates per run: full label text per box, and drawn states sharing (species, location name).
usage: boxes.py <arm> [idx ...]"""
import collections, json, re, sys
import xml.etree.ElementTree as ET

ARM = sys.argv[1]
R = f"C:/t/pwml-fix-work/replay/{ARM}"
recs = [json.loads(l) for l in open(f"{R}/results.jsonl", encoding="utf-8")]
only = set(int(x) for x in sys.argv[2:])


def pts(path):
    return [(float(a), float(b)) for a, b in re.findall(r"(-?\d+(?:\.\d+)?)[ ,](-?\d+(?:\.\d+)?)", path or "")]


tot = collections.Counter()
out = {}
for rec in recs:
    i = rec["idx"]
    if not rec.get("pwml_written") or (only and i not in only):
        continue
    root = ET.parse(f"{R}/pwml/{i}.pwml").getroot()
    pv = root.find(".//pathway-visualization")
    boxes = []
    for m in pv.iter("membrane-visualization"):
        p = pts(m.findtext("path"))
        boxes.append((min(x for x, _ in p), min(y for _, y in p), max(x for x, _ in p), max(y for _, y in p)))
    labels = [[] for _ in boxes]
    for lb in pv.iter("label-location"):
        x, y = float(lb.findtext("x")), float(lb.findtext("y"))
        for j, b in enumerate(boxes):
            if b[0] <= x <= b[2] and b[1] <= y <= b[3]:
                labels[j].append((y, lb.findtext("label-text")))
                break
    full = [" ".join(t for _, t in sorted(l)).casefold() for l in labels]
    fc = collections.Counter(f for f in full if f)
    # drawn states: states with a location drawn (not hidden) in the PWML
    ir = json.load(open(f"{R}/ir/{i}.pwml_ir.json", encoding="utf-8"))
    nm = lambda sec, k: next((x.get("name") for x in ir.get(sec, []) if x["key"] == k), None)
    sl = collections.defaultdict(list)
    for s in ir["biological_states"]:
        sl[(nm("species", s["species_key"]), (nm("subcellular_locations", s["subcellular_location_key"]) or "").casefold())].append(
            (s["name"], nm("cell_types", s["cell_type_key"]), nm("tissues", s["tissue_key"])))
    same_loc = {f"{k[0]} | {k[1]}": v for k, v in sl.items() if len(v) > 1}
    r = {"boxes": len(boxes), "unlabeled": sum(1 for f in full if not f), "dup_full_labels": {k: n for k, n in fc.items() if n > 1},
         "states_sharing_location": same_loc}
    if r["dup_full_labels"] or same_loc or r["unlabeled"]:
        out[i] = r
    tot["boxes"] += len(boxes)
    tot["unlabeled"] += r["unlabeled"]
    tot["dup_full_labels"] += sum(n - 1 for n in fc.values() if n > 1)
    tot["runs_dup_full_labels"] += bool(r["dup_full_labels"])
    tot["state_groups_sharing_location"] += len(same_loc)
    tot["runs_states_sharing_location"] += bool(same_loc)
json.dump({"totals": tot, "runs": out}, open(f"C:/t/pwml-fix-work/boxes_{ARM}.json", "w", encoding="utf-8"), indent=1)
print(dict(tot))
for i, r in out.items():
    print(i, json.dumps(r)[:700])
