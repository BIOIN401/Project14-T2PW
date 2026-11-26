"""Render timing, cold vs warm structure cache. usage: timing.py <src_tree> <scratch_dir> <pwml>..."""
import os, sys, pathlib, time, shutil
sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / "src"))
from t2pw.render import structures
from t2pw.render.pathbank import render_pwml, _structure_cids, read_xml_with_repairs, find_first_visualization

scratch = pathlib.Path(sys.argv[2])
for p in sys.argv[3:]:
    name = pathlib.Path(p).stem
    cache = scratch / f"cache-{name}"
    if cache.exists():
        shutil.rmtree(cache)
    os.environ[structures.CACHE_ENV] = str(cache)
    root = read_xml_with_repairs(p)
    n = len(_structure_cids(root, find_first_visualization(root)))
    for label in ("no-fetch(budget 0)", "cold", "warm", "warm2"):
        t0 = time.perf_counter()
        data = render_pwml(p, "png", structure_budget_s=0 if label.startswith("no-fetch") else None)
        dt = time.perf_counter() - t0
        got = len(list(cache.glob("*.png"))) if cache.exists() else 0
        print(f"{name}: {label:18s} {dt:6.2f}s  png {len(data)//1024} KB  cids {n}  cached pictures {got}", flush=True)
