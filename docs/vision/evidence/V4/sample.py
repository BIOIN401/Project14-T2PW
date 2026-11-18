"""Seeded sample of added steps for the V4 hand checks: 4 PathBank, 4 WikiPathways, 2 Reactome
(Reactome has few hop-1 steps, so any hop), reaction steps only, excluding steps already checked.
Round 3 ran it with seed 3 (exclusions: the round-2 list only); round 4 with seed 4 (round-2 and
round-3 lists, as below). Copy of C:/t/vis-runs/scratch/vis-extend-db/sample.py as run in round 4.
usage: python sample.py <seed>"""
import json
import random
from pathlib import Path

ev = Path("C:/t/vis-extend-db/docs/vision/evidence/V4")
done = {"PW_R001168", "PW_R001169", "PW_R002453", "PW_R003029", "PW_R109709", "R-MMU-5693390",
        "PW_R213422", "PW_R050006", "PW_R213418", "PW_R213127",
        # round 3
        "PW_R1674708", "PW_R2146855", "PW_R2146839", "PW_R1998772", "idf29bfd2c", "ca982", "fe0df", "fa7c8",
        "R-HSA-9843721", "R-MMU-193068"}
SEED = int(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else 3
pool = {"pathbank": [], "wikipathways": [], "reactome": []}
for f in sorted(ev.glob("PMC*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    for s in d["added_steps"]:
        note = str(s.get("note") or "")
        db = note.split(" ", 1)[0]
        acc = note.split(" ")[3].rstrip(";") if len(note.split(" ")) > 3 else ""
        if db not in pool or not s.get("in") or acc in done:
            continue
        if db != "reactome" and s["hop"] != 1:
            continue
        pool[db].append((f.stem, s["hop"], s["name"], s["in"], s["out"], note[:200]))
rnd = random.Random(SEED)
for db, n in (("pathbank", 4), ("wikipathways", 4), ("reactome", 2)):
    print(f"== {db}: pool {len(pool[db])}")
    for x in rnd.sample(pool[db], min(n, len(pool[db]))):
        print("  ", x)
