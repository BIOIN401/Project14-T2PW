"""A2: the patched proto with all OPTS off must give byte-identical SVGs to the unpatched copy (a2/svg/<idx>.svg)."""
import json, sys
sys.path.insert(0, "C:/t/corr-assess/a2/proto")
import t2pw_diagrams as D

S = json.load(open("C:/t/corr-assess/a2/before/render_summary.json", encoding="utf-8"))
same = diff = 0
for idx, r in S.items():
    if "error" in r:
        continue
    ir = D.load(f"C:/t/corr-assess/ir/{idx}/pwml_ir.json")
    svg, m, lay = D.render(ir, r["style"], f"d{idx}")
    old = open(f"C:/t/corr-assess/a2/svg/{idx}.svg", encoding="utf-8").read()
    if svg == old:
        same += 1
    else:
        diff += 1
        print("DIFF", idx)
print("same", same, "diff", diff)
