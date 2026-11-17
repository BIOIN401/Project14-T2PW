import json, os, sys
sys.path.insert(0, "C:/t/integ/src")
from t2pw.clean_model.builder import build_from_payload, read_run_meta
from t2pw.clean_model.export.sbml import write_sbml
B = "C:/t/corr-assess"
sample = json.load(open(f"{B}/sample.json", encoding="utf-8"))
out = {}
for s in sample:
    i = s["idx"]
    payload = json.load(open(f"{B}/base/{i}/final_payload.json", encoding="utf-8"))
    rd = s["run_dir"]
    fm = os.path.join(rd, "final_mapped.json")
    same = None
    if os.path.exists(fm):
        same = json.load(open(fm, encoding="utf-8")) == payload
    meta = read_run_meta(rd) if os.path.isdir(rd) else {}
    meta.setdefault("organism", s["organism"])
    m = build_from_payload(payload, meta, run_dir=rd if os.path.isdir(rd) else None)
    write_sbml(m, f"{B}/a3/ours/{i}.sbml")
    out[i] = {"final_mapped_exists": os.path.exists(fm), "payload_equals_final_mapped": same}
json.dump(out, open(f"{B}/a3/ours_inputs.json", "w"), indent=1)
print("done", len(out), out)
