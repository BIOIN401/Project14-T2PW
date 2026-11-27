"""Every differing file of the VIS-7 content_check output must differ only by added <pwc-id> lines. usage: v7diffonly.py CC.json"""
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8")); bad = 0; n = 0
for row in r["rows"]:
    if row["same"]: continue
    n += 1
    for l in row["diff"].splitlines():
        if l.startswith(("---", "+++", "@@")) or not l[:1] in "+-": continue
        if not ("<pwc-id>" in l and l.startswith("+")): bad += 1; print(row["file"], l[:160])
print(sys.argv[1], "differing files", n, "non-pwc-id diff lines", bad)
