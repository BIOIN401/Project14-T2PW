import json, re, collections, sys
sys.path.insert(0, "C:/t/corr-assess/scripts")
import a3_models as M
B = M.B
sample = json.load(open(f"{B}/sample.json", encoding="utf-8"))
REV = re.compile(r"reversib|\u21cc|<->|<-->|\u2194|bidirection|interconver|\bback\b.*\bforward", re.I)
out = {}; tot = collections.Counter()
for s in sample:
    i = s["idx"]
    p = json.load(open(f"{B}/base/{i}/final_payload.json", encoding="utf-8"))
    ir = json.load(open(f"{B}/ir/{i}/pwml_ir.json", encoding="utf-8"))
    pw = M.from_pwml(f"{B}/replay/on_v2/pwml/{i}.pwml")
    dirs = collections.Counter(r.get("direction") for r in ir["processes"]["reactions"])
    wd = collections.Counter(r["dir"] for r in pw["rxn"])
    keys = [k for k in p["processes"]["reactions"][0].keys()] if p["processes"]["reactions"] else []
    rev = [r for r in p["processes"]["reactions"] if REV.search((r.get("evidence") or "") + " " + (r.get("name") or ""))]
    spont = sum(1 for r in p["processes"]["reactions"] if r.get("spontaneous"))
    tot["rxn"] += len(p["processes"]["reactions"]); tot["rev_text"] += len(rev); tot["spont_payload"] += spont
    out[i] = dict(ir_dir=dict(dirs), pwml_dir=dict(wd), payload_rev_text=[(r["name"], (r.get("evidence") or "")[:160]) for r in rev], payload_spont=spont,
                  pwml_spont=collections.Counter(r["spont"] for r in pw["rxn"]))
    print(i, dict(dirs), dict(wd), "rev-text", len(rev), "spont payload", spont, dict(out[i]["pwml_spont"]))
print(tot)
json.dump(out, open(f"{B}/a3/direction_scan.json", "w"), indent=1, default=str)
