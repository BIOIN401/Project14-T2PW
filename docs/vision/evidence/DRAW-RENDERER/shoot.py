"""PNG of every <dir>/*.svg (natural width, capped) via the gallery's Playwright helper. usage: shoot.py <dir> [names...]"""
import glob
import json
import os
import re
import subprocess
import sys

D = sys.argv[1]
names = sys.argv[2:]
jobs = []
for p in sorted(glob.glob(os.path.join(D, "*.svg"))):
    n = os.path.splitext(os.path.basename(p))[0]
    if names and n not in names:
        continue
    w = float(re.search(r'viewBox="0 0 ([\d.]+) ', open(p, encoding="utf-8").read(4000)).group(1))
    jobs.append({"svg": p, "png": p[:-4] + ".png", "width": int(min(w, 3000))})
jf = os.path.join(D, "_jobs.json")
json.dump(jobs, open(jf, "w"))
g = "C:/t/lane-draw/src/t2pw/vision/checks/gallery.py"
r = subprocess.run(["C:/t/e2e3-venv/Scripts/python.exe", "-I", g, "--shoot", jf], capture_output=True, text=True, timeout=600)
print(r.returncode, (r.stdout + r.stderr)[-500:])
