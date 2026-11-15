"""driver: render every file via a subprocess with a timeout. Run with fmt-venv python under the bounded wrapper."""
import json, subprocess, sys, os
B = "C:/t/corr-assess"
PY = "C:/t/fmt-venv/Scripts/python.exe"
sample = [s["idx"] for s in json.load(open(f"{B}/sample.json", encoding="utf-8"))]
only = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else sample
log = []
for i in only:
    for tag, kind, src in (("pw_sbgn", "sbgn", f"{B}/pathwhiz_exports/{i}.sbgn"), ("ours_sbml", "sbml", f"{B}/a3/ours/{i}.sbml"), ("pw_sbml", "sbml", f"{B}/pathwhiz_exports/{i}.sbml")):
        out = f"{B}/a3/render/{i}_{tag}.png"
        if not os.path.exists(src):
            log.append({"idx": i, "tag": tag, "rendered": False, "error": "source missing"}); continue
        try:
            p = subprocess.run([PY, "-X", "utf8", f"{B}/scripts/a3_render_one.py", kind, src, out], capture_output=True, text=True, timeout=240)
            line = [l for l in p.stdout.splitlines() if l.startswith("{")]
            r = json.loads(line[-1]) if line else {"rendered": False, "error": "no output rc=%s %s" % (p.returncode, (p.stderr or "")[-300:])}
        except subprocess.TimeoutExpired:
            r = {"rendered": False, "error": "timeout 240s"}
        r.update(idx=i, tag=tag)
        log.append(r)
        print(i, tag, r.get("rendered"), r.get("full_size"), r.get("seconds"), (r.get("error") or "")[:100], flush=True)
        json.dump(log, open(f"{B}/a3/render_log.json", "w"), indent=1)
json.dump(log, open(f"{B}/a3/render_log.json", "w"), indent=1)
