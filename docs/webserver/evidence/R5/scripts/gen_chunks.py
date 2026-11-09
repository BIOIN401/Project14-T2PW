import glob
import json
import os

EXTRA = {
    "second-fixture": "tests/test_equivalence_second_fixture.py",
    "export": "tests/test_service_export.py",
    "reasons": "tests/test_web_stage_reasons.py",
    "browser": "tests/test_web_browser_fixes.py",
    "post_run": "tests/test_service_post_run.py",
    "clobber": "tests/test_stage0_case_c_retry_clobber.py",
}
lines = []
for f in sorted(glob.glob("C:/t/ws-R5/docs/webserver/evidence/R4b/runs/t-*.json")):
    d = json.load(open(f, encoding="utf-8"))
    c = d.get("command", [])
    i = next((k for k, x in enumerate(c) if str(x).startswith("--basetemp")), None)
    if i is None:
        continue
    name = os.path.basename(f)[2:-5]
    if name in ("new", "at-bridge"):
        continue
    lines.append(f"{name}|{' '.join(c[i + 1:])}")
for name, path in EXTRA.items():
    lines.append(f"{name}|{path}")
open("C:/t/ws-R5-tmp/chunks.txt", "w", newline="\n").write("\n".join(lines) + "\n")
print(len(lines))
