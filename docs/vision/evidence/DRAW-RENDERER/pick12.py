"""Pick 12 stored runs for the judge: the 6 hardest by baseline (crossings + long edges) and 6 spread by index."""
import json
import os
import shutil

W = "C:/t/lane-draw-work"
rows = [r for r in json.load(open(W + "/out/metrics97_before.json"))["runs"] if "svg" in r]
hard = sorted(rows, key=lambda r: -(r["svg"]["crossings"] + r["svg"]["long_edges"]))[:6]
rest = [r for r in sorted(rows, key=lambda r: r["idx"]) if r not in hard]
spread = rest[:: max(1, len(rest) // 6)][:6]
os.makedirs(W + "/ir12", exist_ok=True)
for r in hard + spread:
    shutil.copy("%s/ir97/%03d.json" % (W, r["idx"]), "%s/ir12/%03d.json" % (W, r["idx"]))
print([r["idx"] for r in hard], [r["idx"] for r in spread])
