"""Render PWMLs with t2pw.render.pathbank: render_many.py OUT_DIR fmt file.pwml [...]"""
import sys
import time
from pathlib import Path

from t2pw.render import pathbank

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
fmt = sys.argv[2]
for f in sys.argv[3:]:
    p = Path(f)
    t = time.time()
    try:
        data = pathbank.render_pwml(p, fmt)
        name = (p.parent.name + "-" if p.parent.name == "pwml" else "") + p.stem + "." + fmt
        (out / name).write_bytes(data)
        print(f"{p.name} ok {len(data)} bytes {time.time() - t:.1f}s")
    except Exception as exc:  # report and go on
        print(f"{p.name} FAIL {type(exc).__name__}: {exc}")
