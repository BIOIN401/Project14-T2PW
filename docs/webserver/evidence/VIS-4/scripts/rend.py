"""Render PWMLs with the worktree's PathBank-style renderer, timing each.
usage: rend.py <src_tree> <outdir> <fmt> <name=pwml>...   (name= optional; default the file stem)"""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "src"))
from t2pw.render.pathbank import render_pwml

out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
fmt = sys.argv[3]
for arg in sys.argv[4:]:
    name, _, p = arg.rpartition("=")
    name = name or pathlib.Path(p).stem
    t0 = time.perf_counter()
    try:
        data = render_pwml(p, fmt)
    except Exception as e:  # report and go on
        print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)
        continue
    dt = time.perf_counter() - t0
    q = out / f"{name}.{fmt}"
    q.write_bytes(data)
    print(f"wrote {q} {len(data)} bytes {dt:.2f}s", flush=True)
