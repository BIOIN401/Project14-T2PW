"""analyze.py LABEL... : per label, structural counts over written PWML (bound-viz, spontaneous, dangling compound-location-id, QA, gain/loss vs first label)."""
import collections, glob, json, sys
from pathlib import Path
from lxml import etree

R = Path("C:/t/fp/docs/pwml_output/evidence/final_payload_p2/replay6")


def load(label):
    d = {}
    for f in glob.glob(str(R / label / "results_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            d[r["idx"]] = r
    return d


def scan(path):
    t = etree.parse(str(path))
    ids = {(c.findtext("id") or "").strip() for c in t.iter("compound-location")}
    refs = [(e.text or "").strip() for e in t.iter("compound-location-id")]
    dang = [x for x in refs if x not in ids]
    sp = collections.Counter()
    for e in t.iter("spontaneous"):
        sp["nil" if e.get("nil") == "true" else (e.text or "").strip()] += 1
    return {"bound_viz": sum(1 for _ in t.iter("bound-visualization")), "sp": sp,
            "refs": len(refs), "dangling": len(dang), "dang_files": 1 if dang else 0}


out = {}
base = None
for label in sys.argv[1:]:
    d = load(label)
    w = {i for i, r in d.items() if r["pwml_written"]}
    tot = collections.Counter()
    sp = collections.Counter()
    for i in w:
        s = scan(R / label / "pwml" / ("%d.pwml" % i))
        tot["bound_viz"] += s["bound_viz"]
        tot["cl_refs"] += s["refs"]
        tot["dangling"] += s["dangling"]
        tot["files_with_dangling"] += s["dang_files"]
        sp.update(s["sp"])
    o = {"runs": len(d), "written": len(w), "bound_viz": tot["bound_viz"], "spontaneous": dict(sp),
         "cl_refs": tot["cl_refs"], "dangling_cl_refs": tot["dangling"], "files_with_dangling": tot["files_with_dangling"],
         "qa_errors": sum(d[i].get("qa_errors") or 0 for i in w), "qa_warnings": sum(d[i].get("qa_warnings") or 0 for i in w),
         "qa_ok_false": [i for i in w if d[i].get("qa_ok") is False]}
    if base is None:
        base = (label, w)
    else:
        o["vs_" + base[0]] = {"gained": sorted(w - base[1]), "lost": sorted(base[1] - w)}
    out[label] = o
print(json.dumps(out, indent=1))
