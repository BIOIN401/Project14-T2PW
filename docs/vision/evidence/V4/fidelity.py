"""Fidelity check for the round-4 sample: print each sampled step as its source states it.
PathBank: a read-only SELECT per reaction (run through a launcher that loads the primary .env into the
process, never printed; prints names only). WikiPathways: the cached GPML line. Reactome: the cached
ContentService record. Copy of C:/t/vis-runs/scratch/vis-extend-db/fidelity.py as run in round 4."""
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, "C:/t/vis-extend-db/src")
CACHE = Path("C:/t/vis-runs/cache/vis-extend-db")


def cached(src, url):
    p = CACHE / src / (hashlib.sha256(url.encode()).hexdigest()[:32] + ".txt")
    return p.read_text(encoding="utf-8") if p.exists() else None


PB = ["PW_R001193", "PW_R001218", "PW_R109721", "PW_R213421"]
WP = [("WP1422", "id3beb065f"), ("WP1264", "da314"), ("WP1264", "b964e"), ("WP1264", "b75f6")]
RC = ["R-MMU-6786239", "R-MMU-193845"]

from t2pw.bench.eval_v2.pathbank_gold import _Reader, _process, connect  # noqa: E402

conn = connect({"read_timeout": 60, "connect_timeout": 10})
try:
    r = _Reader(conn)
    for acc in PB:
        row = r.one("SELECT id, direction FROM reactions WHERE pwr_id=%s", (acc,))
        p = _process(r, "Reaction", int(row["id"]), None)
        left = [x.get("name") for x in p["participants"] if x["side"] == "left"]
        right = [x.get("name") for x in p["participants"] if x["side"] == "right"]
        cats = [c.get("name") for c in p["catalysts"]]
        print("PB", acc, left, row.get("direction"), right, "catalysts", cats)
finally:
    conn.close()

for wp, line in WP:
    text = cached("wikipathways", f"https://www.wikipathways.org/wikipathways-assets/pathways/{wp}/{wp}.gpml")
    root = ET.fromstring(text)
    labels = {n.get("GraphId"): n.get("TextLabel") for n in root.iter() if n.tag.endswith("DataNode")}
    for it in root.iter():
        if it.tag.endswith("Interaction") and it.get("GraphId") == line:
            pts = [p for p in it.iter() if p.tag.endswith("Point")]
            print("WP", wp, line, labels.get(pts[0].get("GraphRef")), "->", labels.get(pts[-1].get("GraphRef")),
                  "head", pts[-1].get("ArrowHead"), "first-point head", pts[0].get("ArrowHead"))
for st in RC:
    q = json.loads(cached("reactome", f"https://reactome.org/ContentService/data/query/{st}"))
    part = {p["peDbId"]: p["displayName"] for p in json.loads(cached("reactome", f"https://reactome.org/ContentService/data/participants/{st}"))}
    ids = lambda v: [x if isinstance(x, int) else x.get("dbId") for x in v or []]
    print("RC", st, q.get("displayName"), "| in", [part.get(i) for i in ids(q.get("input"))],
          "| out", [part.get(i) for i in ids(q.get("output"))])
