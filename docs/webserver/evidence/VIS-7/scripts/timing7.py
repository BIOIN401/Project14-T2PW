"""usage: timing7.py SRC IDX|'' LABEL OUTDIR -- replay corpus items through SRC's export, timing layout.relayout; one line per
call in logs/relayout_times_LABEL.txt: seconds, edges, compound locations (F7)."""
import runpy
import sys
import time

src, idx, label, outdir = sys.argv[1:5]
sys.path.insert(0, src)
import t2pw.pwml.layout as lay  # noqa: E402

orig = lay.relayout
LOG = "C:/t/vis-pw7-work/logs/relayout_times_" + label + ".txt"


def timed(items, *a, **k):
    t = time.perf_counter()
    n = (len(items.get("edges", [])), len(items.get("compound-locations", [])))
    try:
        return orig(items, *a, **k)
    finally:
        with open(LOG, "a") as f:
            f.write(f"{round(time.perf_counter() - t, 3)} {n[0]} {n[1]}\n")


lay.relayout = timed
for m in list(sys.modules.values()):
    if m is not lay and getattr(m, "relayout", None) is orig:
        m.relayout = timed
sys.argv = ["replay97.py", "--src", src, "--label", label, "--out", outdir] + (["--idx", idx] if idx else [])
runpy.run_path("C:/t/vis-pw7-work/scripts/replay97.py", run_name="__main__")
