"""map the 60 triage-baseline files to runlist idx (by run directory) and to written status in off5/on5"""
import json, glob
from pathlib import Path
R = Path("C:/t/fp/docs/pwml_output/evidence/final_payload_p2/replay")
runs = [x for x in json.load(open("C:/t/ev-cm/runlist.json", encoding="utf-8")) if x["q"]]
norm = lambda p: p.replace("\\", "/").rstrip("/").lower()
bydir = {norm(r["dir"]): i for i, r in enumerate(runs)}
base = [json.loads(l)["file"] for l in open("C:/t/pwi/t2pw/docs/pwml_output/evidence/import/t2pw_results.jsonl", encoding="utf-8")]
w = {}
for L in ("off5", "on5"):
    d = {}
    for f in glob.glob(str(R / L / "results_*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l); d[r["idx"]] = r
    w[L] = d
m = []
for f in base:
    n = norm(f)
    hit = None
    for dn, i in bydir.items():
        if n.startswith(dn + "/"):
            hit = i
    m.append({"file": f, "idx": hit, "off5_written": bool(hit is not None and w["off5"][hit]["pwml_written"]), "on5_written": bool(hit is not None and w["on5"][hit]["pwml_written"])})
json.dump(m, open(R.parent / "import" / "baseline_map.json", "w"), indent=1)
print("mapped", sum(x["idx"] is not None for x in m), "of", len(m), "distinct idx", len({x["idx"] for x in m if x["idx"] is not None}),
      "off5 written", sum(x["off5_written"] for x in m), "on5 written", sum(x["on5_written"] for x in m))
for x in m:
    if x["idx"] is None: print("UNMAPPED", x["file"])
