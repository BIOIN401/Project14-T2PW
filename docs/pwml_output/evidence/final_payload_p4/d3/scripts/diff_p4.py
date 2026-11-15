import difflib, json, re, collections
from pathlib import Path
E = Path("C:/t/fp/docs/pwml_output/evidence")
idx = [3, 4, 7, 10, 17, 27, 68, 70, 76, 84, 105, 116, 119, 125, 126, 150, 151]
tot = collections.Counter(); out = []
for i in idx:
    a = (E/f"final_payload_p3/replay/on/pwml/{i}.pwml").read_text(encoding="utf-8").splitlines()
    b = (E/f"final_payload_p4/replay/on/pwml/{i}.pwml").read_text(encoding="utf-8").splitlines()
    add = [l for l in difflib.unified_diff(a, b, lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    ad = [l for l in add if l[0] == "+"]; rm = [l for l in add if l[0] == "-"]
    c = collections.Counter()
    for l in add:
        t = re.search(r"<(/?[a-z-]+)", l); c[(l[0], t.group(1) if t else "?")] += 1
    top = ", ".join("%s%s x%d" % (k[0], k[1], v) for k, v in c.most_common(6))
    out.append("%3d  +%d -%d   %s" % (i, len(ad), len(rm), top)); tot["add"] += len(ad); tot["rm"] += len(rm)
out.append("TOTAL +%d -%d" % (tot["add"], tot["rm"]))
open(E/"final_payload_p4/on_vs_p3_diff_summary.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out))
