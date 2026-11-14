"""What PathWhiz stores for reaction-member `side`: import each file with the harness, then read the DB.

usage: side_db.py LIST.txt OUT_DIR
For every reaction_compound_visualization / reaction_nucleic_acid_visualization / reaction_element_collection_visualization
row: its `side`, and the type of the reaction_elements row(s) for the same reaction and element
(ReactionLeftElement / ReactionRightElement: the reaction's stored role, read from the PWML's reaction element lists).
A swap = side names one list while the element is stored only in the other. Run while holding the heavy lock
(it drives the shared t2pw-pwi harness, which truncates the DB before each import).
"""
import json
import os
import subprocess
import sys

PY = sys.executable
H = "C:/t/pwi/t2pw/tools/pathwhiz_import"
lst, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
Q = """
SELECT rv.reaction_id, '{k}', v.id, v.side, l.{eid},
  (SELECT GROUP_CONCAT(DISTINCT re.type ORDER BY re.type) FROM reaction_elements re
   WHERE re.reaction_id = rv.reaction_id AND re.element_type = '{et}' AND re.element_id = l.{eid})
FROM {vt} v JOIN reaction_visualizations rv ON rv.id = v.reaction_visualization_id
JOIN {lt} l ON l.id = v.{lid};"""
PARTS = [("Compound", "reaction_compound_visualizations", "compound_locations", "compound_location_id", "compound_id"),
         ("NucleicAcid", "reaction_nucleic_acid_visualizations", "nucleic_acid_locations", "nucleic_acid_location_id", "nucleic_acid_id"),
         ("ElementCollection", "reaction_element_collection_visualizations", "element_collection_locations",
          "element_collection_location_id", "element_collection_id")]
tot = {"files": 0, "rows": 0, "consistent": 0, "swapped": 0, "both_lists": 0, "no_element": 0}
per = {}
for path in [l.strip() for l in open(lst) if l.strip()]:
    name = os.path.basename(path)
    o = os.path.join(out, name[:-5])
    r = subprocess.run([PY, "-X", "utf8", H + "/run_import.py", path, "--out", o], capture_output=True, text=True)
    env = dict(os.environ, SMPDB_SRC=H + "/work/smpdb", OUT_DIR=o, PWML_DIR=H + "/pwml_in")
    rows = []
    for et, vt, lt, lid, eid in PARTS:
        q = Q.format(k=et, et=et, vt=vt, lt=lt, lid=lid, eid=eid)
        m = subprocess.run(["docker", "compose", "-p", "t2pw-pwi", "-f", H + "/docker-compose.yml", "exec", "-T", "db",
                            "mysql", "-uroot", "-ppassword", "-N", "-B", "smpdb_development", "-e", q],
                           capture_output=True, text=True, env=env)
        rows += [line.split("\t") for line in m.stdout.splitlines() if line.strip()]
    c = {"rows": len(rows), "consistent": 0, "swapped": 0, "both_lists": 0, "no_element": 0, "import_rc": r.returncode, "swaps": []}
    for rx, et, vid, side, ent, types in rows:
        t = set((types or "").replace("NULL", "").split(",")) - {""}
        if not t:
            c["no_element"] += 1
        elif len(t) == 2:
            c["both_lists"] += 1
        elif t == {f"Reaction{side}Element"}:
            c["consistent"] += 1
        else:
            c["swapped"] += 1
            c["swaps"].append({"reaction": rx, "type": et, "vis_id": vid, "side": side, "element": ent, "stored_as": sorted(t)})
    per[name] = c
    tot["files"] += 1
    for k in ("rows", "consistent", "swapped", "both_lists", "no_element"):
        tot[k] += c[k]
    print(name, {k: v for k, v in c.items() if k != "swaps"}, flush=True)
json.dump({"totals": tot, "files": per}, open(os.path.join(out, "side_db.json"), "w"), indent=1)
print("TOTAL", json.dumps(tot))
