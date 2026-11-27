"""usage: timing8.py OUTDIR -- show_replay of the 8 showcases through C:/t/vis-pw8 with relayout timed; one line per call in logs/relayout_times_show.txt"""
import runpy, sys, time
src, out = "C:/t/vis-pw8-x/src", sys.argv[1]
sys.path.insert(0, src)
import t2pw.pwml.layout as lay
orig = lay.relayout
LOG = "C:/t/vis-pw8-work/logs/relayout_times_show.txt"
open(LOG, "w").close()
def timed(items, *a, **k):
    t = time.perf_counter()
    try:
        return orig(items, *a, **k)
    finally:
        open(LOG, "a").write(f"{round(time.perf_counter() - t, 3)} edges={len(items.get('edges', []))} cl={len(items.get('compound-locations', []))}\n")
lay.relayout = timed
for m in list(sys.modules.values()):
    if m is not lay and getattr(m, "relayout", None) is orig:
        m.relayout = timed
sys.argv = ["show_replay.py", "--src", src, "--spec", "C:/t/vis-pw8-work/spec.json", "--out", out]
runpy.run_path("C:/t/vis-pw8-work/scripts/show_replay.py", run_name="__main__")
