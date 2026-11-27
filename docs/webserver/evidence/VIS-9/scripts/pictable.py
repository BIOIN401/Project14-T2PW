"""Per-pathway table (markdown) of pics.py output: VIS-7 (a3 + a3b) vs VIS-9 (p1). usage: pictable.py"""
import json
from pathlib import Path
M = Path("C:/t/vis-pw9-work/metrics")
old = {}
for f in ("pics_vis7_a3.json", "pics_vis7_a3b.json"):
    for k, v in json.load(open(M / f, encoding="utf-8")).items():
        old[k.replace("v5_", "")] = v
new = json.load(open(M / "pics_p1.json", encoding="utf-8"))
def agg(v, pre):
    c = v["counts"]; g = lambda k: sum(n for kk, n in c.items() if kk.startswith(pre) and kk.split("/", 1)[1] == k)
    return g("our_pwc_picture"), g("our_pwc_no_structure"), g("dev_pwc_no_correct_picture"), g("merged_into"), g("dropped")
print("| pathway | compounds | VIS-7 right picture | VIS-9 right picture (no HMDB) | VIS-9 own pwc-id, MolDB has no structure | VIS-9 dev-DB pwc-id (no right picture) (no HMDB) | merged into a same-HMDB record | absent from PathWhiz's export (VIS-7 too) | errors VIS-7 -> VIS-9 |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
T = [0] * 9
for name in sorted(new):
    if name == "PW000779": continue
    v, o = new[name], old[name]
    a, an = agg(v, ""), agg(v, "no_hmdb")
    row = [len(v["rows"]), agg(o, "")[0], a[0], an[0], a[1], a[2], an[2], a[3], a[4]]
    T = [x + y for x, y in zip(T, row)]
    print(f"| {name} | {row[0]} | {row[1]} | {row[2]} ({row[3]}) | {row[4]} | {row[5]} ({row[6]}) | {row[7]} | {row[8]} | {o['error_count']} -> {v['error_count']} |")
print(f"| **total** | {T[0]} | {T[1]} | {T[2]} ({T[3]}) | {T[4]} | {T[5]} ({T[6]}) | {T[7]} | {T[8]} | |")
