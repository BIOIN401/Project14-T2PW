"""Correction #2 safety check: will the installed runner continue EXACTLY the
directory we verified? Asked of the runner's own resolver, not inferred."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).resolve() / "src"))
from t2pw.batch.runner import find_resumable, load_plan, plan_pairs, pending_pairs, list_run_dirs, RESUME_MAX_AGE_HOURS  # noqa: E402

OUT = Path("runs_verify")
EXPECT = Path(sys.argv[2]).resolve()
print("expected staged dir :", EXPECT)
print("RESUME_MAX_AGE_HOURS:", RESUME_MAX_AGE_HOURS)
dirs = list_run_dirs(OUT)
print("newest 3 run dirs   :", [d.name for d in dirs[:3]])

lines = []
got = find_resumable(OUT, log=lines.append)
for l in lines: print("  resolver says:", l)
print()
print("resolver returned   :", got)
same = got is not None and Path(got).resolve() == EXPECT
print("SAME AS VERIFIED DIR:", same)

plan = load_plan(EXPECT)
pairs = plan_pairs(plan)
pend = pending_pairs(EXPECT, plan)
print()
print("plan pairs          :", len(pairs))
print("pending pairs       :", len(pend))
print("legs already present:", len(list(EXPECT.glob('papers/*/*/RESULT.txt'))))
ok = same and len(pairs) == 20 and len(pend) == 20
print()
print("CONTINUITY SAFE     :", ok)
sys.exit(0 if ok else 3)
