import hashlib, json, sys
from pathlib import Path
from lxml import etree
R = Path("C:/t/corr-assess")
NEW = R / "replay/on_v2/pwml"
OLD = Path("C:/t/integ-reports/replay/on_v2/pwml")
h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
same, diff, miss = [], [], []
for i in range(152):
    a, b = NEW / f"{i}.pwml", OLD / f"{i}.pwml"
    if not a.exists() and not b.exists():
        continue
    if not a.exists() or not b.exists():
        miss.append(i)
    elif h(a) == h(b):
        same.append(i)
    else:
        diff.append(i)
print("identical", len(same), "different", diff, "missing", miss)

m = json.load(open(R / "metrics.json"))
files = m["files"]
for f in files:
    f["idx"] = int(Path(f["file"]).stem)
for k in ("edge_crossings", "long_edges", "unattached_nodes"):
    print(k, [(f["idx"], f[k]) for f in sorted(files, key=lambda f: -f[k])[:5]])

def cnt(root, tag):
    return len(root.findall(".//" + tag))

def info(i):
    meta = json.load(open(R / f"base/{i}/meta.json"))
    r = etree.parse(str(NEW / f"{i}.pwml")).getroot()
    tr = r.find(".//transports")
    return {
        "idx": i, "organism": meta["organism"], "pathway_name": meta["pathway_name"],
        "reactions": cnt(r, "reaction"), "compounds": cnt(r, "compound"), "proteins": cnt(r, "protein"),
        "transports": len(tr) if tr is not None else 0, "interactions": cnt(r, "interaction"),
        "compartments": sorted({(e.findtext("name") or "") for e in r.iter("subcellular-location")}),
        "run_dir": meta["run_dir"], "paper_text": meta["paper_text"],
    }

base = [7, 128, 9, 116, 5, 134, 135, 78, 125, 129, 138, 120, 148, 19, 144, 126, 122, 121, 142, 114, 14, 10, 58]
worst = []
for k in ("edge_crossings", "long_edges", "unattached_nodes"):
    for f in sorted(files, key=lambda f: -f[k]):
        if f["idx"] not in base and f["idx"] not in [w[0] for w in worst]:
            worst.append((f["idx"], f"worst metrics: {k}={f[k]}"))
            break
out = []
for i in base:
    d = info(i); d["why"] = "base sample"; out.append(d)
for i, why in worst:
    d = info(i); d["why"] = why; out.append(d)
json.dump(out, open(R / "sample.json", "w", encoding="utf-8"), indent=1)
for d in out:
    print(d["idx"], d["organism"], d["reactions"], d["transports"], d["why"])
nbase = len(list((R / "base").glob("*/final_payload.json")))
open(R / "base/DONE", "w").write(f"{__import__('datetime').datetime.now().isoformat()} runs={nbase} pwml={len(list(NEW.glob('*.pwml')))} identical={len(same)} different={len(diff)} missing={len(miss)}\n")
