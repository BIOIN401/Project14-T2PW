import sys, glob, os, collections, re
sys.path.insert(0, "C:/t/fp/src")
from t2pw.pwml.qa import run_pwml_qa  # noqa
off_dir = "C:/t/ev-fp/replay/off2/pwml"
on_dir = "C:/t/ev-fp/replay/on2/pwml"
co, cn = collections.Counter(), collections.Counter()
ex = {}
for f in sorted(glob.glob(on_dir + "/*.pwml")):
    b = os.path.basename(f)
    for d, c in ((off_dir, co), (on_dir, cn)):
        r = run_pwml_qa(open(os.path.join(d, b), "rb").read())
        for w in (r.get("warnings") or []):
            key = re.sub(r"[0-9]+", "N", str(w.get("code") or w)[:90]) if isinstance(w, dict) else re.sub(r"[0-9]+", "N", str(w)[:90])
            c[key] += 1
            if d == on_dir:
                ex.setdefault(key, str(w)[:250])
print("off", sum(co.values()), "on", sum(cn.values()))
for k in sorted(set(co) | set(cn)):
    if co[k] != cn[k]:
        print(k, co[k], "->", cn[k], "| e.g.", ex.get(k))
